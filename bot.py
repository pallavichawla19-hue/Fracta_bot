import os
import re
import decimal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
if not TOKEN or not DATABASE_URL:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and DATABASE_URL env vars")

AMT = decimal.Decimal

WEIGHT_WORDS = {
    "double": AMT("2"),
    "twice": AMT("2"),
    "triple": AMT("3"),
    "thrice": AMT("3"),
}

@dataclass
class ParsedExpense:
    payer: str
    amount: AMT
    currency: str
    note: str
    participants: List[str]  # includes Guest:xxx
    weights: Dict[str, AMT]
    confidence: float

AMOUNT_RE = re.compile(r"(\d+(?:\.\d{1,2})?)")
ALL_WORDS = re.compile(r"\b(all|everyone|everybody)\b", re.IGNORECASE)
EXCEPT_RE = re.compile(r"\b(except|besides|excluding)\b\s+(.+)$", re.IGNORECASE)
GUEST_RE = re.compile(r"\bguest\b\s+([A-Za-z0-9_-]{2,32})", re.IGNORECASE)

WEIGHT_PATTERNS = [
    re.compile(r"\b(?P<name>[A-Za-z]{2,5})\s*(?:=|:)\s*(?P<w>\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>[A-Za-z]{2,5})\s*(?P<w>\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>[A-Za-z]{2,5})\s*x\s*(?P<w>\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>[A-Za-z]{2,5})\s*(?P<word>double|twice|triple|thrice)\b", re.IGNORECASE),
    re.compile(r"\b(?P<word>double|twice|triple|thrice)\s*(?P<name>[A-Za-z]{2,5})\b", re.IGNORECASE),
]

def norm_code(s: str) -> str:
    return s.strip().upper()

def norm_guest(name: str) -> str:
    name = name.strip().replace(" ", "_").lower()
    return f"Guest:{name}"

def compute_shares(amount: AMT, participants: List[str], weights: Dict[str, AMT]) -> Dict[str, AMT]:
    total_w = sum(weights[p] for p in participants)
    raw = {p: (amount * weights[p] / total_w) for p in participants}
    rounded = {p: raw[p].quantize(AMT("0.01")) for p in participants}
    diff = amount - sum(rounded.values())
    cents = int((diff * 100).to_integral_value(rounding=decimal.ROUND_HALF_UP))
    if cents != 0:
        order = sorted(participants, key=lambda p: (raw[p] - rounded[p]), reverse=(cents > 0))
        step = AMT("0.01") if cents > 0 else AMT("-0.01")
        for i in range(abs(cents)):
            rounded[order[i % len(order)]] += step
    return rounded

def settlement_suggestions(net: Dict[str, AMT]) -> List[Tuple[str, str, AMT]]:
    creditors = [(n, v) for n, v in net.items() if v > 0]
    debtors = [(n, -v) for n, v in net.items() if v < 0]
    creditors.sort(key=lambda x: x[1], reverse=True)
    debtors.sort(key=lambda x: x[1], reverse=True)

    i = j = 0
    pays: List[Tuple[str, str, AMT]] = []
    while i < len(debtors) and j < len(creditors):
        d_name, d_amt = debtors[i]
        c_name, c_amt = creditors[j]
        x = min(d_amt, c_amt).quantize(AMT("0.01"))
        if x > 0:
            pays.append((d_name, c_name, x))
        d_amt -= x
        c_amt -= x
        debtors[i] = (d_name, d_amt)
        creditors[j] = (c_name, c_amt)
        if debtors[i][1] <= AMT("0.00001"):
            i += 1
        if creditors[j][1] <= AMT("0.00001"):
            j += 1
    return pays

async def ensure_group(conn: asyncpg.Connection, chat_id: int) -> int:
    row = await conn.fetchrow("select id from groups where tg_chat_id=$1", chat_id)
    if row:
        return row["id"]
    row = await conn.fetchrow("insert into groups(tg_chat_id) values($1) returning id", chat_id)
    return row["id"]

async def upsert_user_member(conn: asyncpg.Connection, group_id: int, tg_user_id: int, display_code: str) -> int:
    row = await conn.fetchrow(
        """
        insert into members(group_id, tg_user_id, display_code, is_guest)
        values($1,$2,$3,false)
        on conflict (group_id, tg_user_id)
        do update set display_code=excluded.display_code, is_active=true, is_guest=false
        returning id
        """,
        group_id, tg_user_id, display_code
    )
    return row["id"]

async def get_member_by_code(conn: asyncpg.Connection, group_id: int, code: str):
    return await conn.fetchrow(
        "select id, display_code, is_guest from members where group_id=$1 and lower(display_code)=lower($2) and is_active=true",
        group_id, code
    )

async def get_or_create_guest(conn: asyncpg.Connection, group_id: int, guest_code: str) -> int:
    row = await get_member_by_code(conn, group_id, guest_code)
    if row:
        return row["id"]
    row = await conn.fetchrow(
        """
        insert into members(group_id, tg_user_id, display_code, is_guest)
        values($1, null, $2, true)
        returning id
        """,
        group_id, guest_code
    )
    return row["id"]

async def list_member_codes(conn: asyncpg.Connection, group_id: int) -> List[str]:
    rows = await conn.fetch(
        "select display_code from members where group_id=$1 and is_active=true and is_guest=false order by id",
        group_id
    )
    return [r["display_code"] for r in rows]

async def compute_net_positions(conn: asyncpg.Connection, group_id: int) -> Dict[str, AMT]:
    rows = await conn.fetch(
        """
        with paid as (
          select payer_member_id as mid, sum(amount) as paid
          from expenses
          where group_id=$1 and currency='AED'
          group by payer_member_id
        ),
        owed as (
          select es.member_id as mid, sum(es.share_amount) as owed
          from expense_splits es
          join expenses e on e.id=es.expense_id
          where e.group_id=$1 and e.currency='AED'
          group by es.member_id
        )
        select m.display_code,
               coalesce(p.paid,0)::numeric as paid,
               coalesce(o.owed,0)::numeric as owed
        from members m
        left join paid p on p.mid=m.id
        left join owed o on o.mid=m.id
        where m.group_id=$1 and m.is_active=true
        """,
        group_id
    )
    out: Dict[str, AMT] = {}
    for r in rows:
        out[r["display_code"]] = AMT(str(r["paid"])) - AMT(str(r["owed"]))
    return out

def parse_loose_english(text: str, known_members: List[str]) -> ParsedExpense:
    t = text.strip()
    known = {norm_code(x) for x in known_members}

    # payer: first token matching known member
    payer = None
    for tok in re.split(r"\s+", t)[:6]:
        c = norm_code(re.sub(r"[^\w]", "", tok))
        if c in known:
            payer = c
            break
    if not payer:
        raise ValueError("Could not find payer. Start with one of: " + ", ".join(sorted(known)))

    m_amt = AMOUNT_RE.search(t)
    if not m_amt:
        raise ValueError("Could not find amount. Example: 'AVC paid 95 taxi split all'")
    amount = AMT(m_amt.group(1))

    currency = "AED"

    # guests
    guests = [norm_guest(g) for g in GUEST_RE.findall(t)]

    # exclusions
    excluded = set()
    m_exc = EXCEPT_RE.search(t)
    if m_exc:
        tail = m_exc.group(2)
        for code in re.findall(r"\b[A-Za-z]{2,5}\b", tail):
            c = norm_code(code)
            if c in known:
                excluded.add(c)
        for g in re.findall(r"\bguest\b\s+([A-Za-z0-9_-]{2,32})", tail, re.IGNORECASE):
            excluded.add(norm_guest(g))

    # participants base
    if ALL_WORDS.search(t):
        participants = list(sorted(known))
        confidence = 0.9
    else:
        mentioned = []
        for code in re.findall(r"\b[A-Za-z]{2,5}\b", t):
            c = norm_code(code)
            if c in known and c not in mentioned:
                mentioned.append(c)
        participants = mentioned if len(mentioned) >= 2 else list(sorted(known))
        confidence = 0.75

    # include payer by default
    if payer not in participants:
        participants.append(payer)

    # add guests
    for g in guests:
        if g not in participants:
            participants.append(g)

    # weights
    weights: Dict[str, AMT] = {p: AMT("1") for p in participants}
    weighted_explicit = set()

    for pat in WEIGHT_PATTERNS:
        for m in pat.finditer(t):
            name = m.groupdict().get("name")
            word = m.groupdict().get("word")
            wnum = m.groupdict().get("w")

            target = None
            if name:
                c = norm_code(name)
                if c in known:
                    target = c
            if target is None and name and name.lower().startswith("guest:"):
                target = name

            if target:
                if wnum:
                    weights[target] = AMT(wnum)
                    weighted_explicit.add(target)
                elif word and word.lower() in WEIGHT_WORDS:
                    weights[target] = WEIGHT_WORDS[word.lower()]
                    weighted_explicit.add(target)

    # apply exclusions, but keep explicitly weighted people
    final_participants = []
    for p in participants:
        if p in excluded and p not in weighted_explicit:
            continue
        final_participants.append(p)

    if not final_participants:
        raise ValueError("No participants after exclusions.")

    note = t
    return ParsedExpense(
        payer=payer,
        amount=amount,
        currency=currency,
        note=note,
        participants=final_participants,
        weights={p: weights.get(p, AMT("1")) for p in final_participants},
        confidence=float(confidence),
    )

async def main():
    bot = Bot(TOKEN)
    dp = Dispatcher()
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

    @dp.message(Command("help"))
    async def help_cmd(message: Message):
        await message.reply(
            "Commands:\n"
            "/join\n"
            "/me AVC (or PC/AS/SS/VH/KA)\n"
            "/guest Driver\n"
            "/list 10\n"
            "/delete last | /delete <id>\n"
            "/undo\n"
            "/balance\n\n"
            "Log expenses by typing:\n"
            "AVC paid 420 dinner split everyone VH double guest driver\n"
            "PC 95 taxi split all except KA\n"
        )

    @dp.message(Command("start"))
    async def start_cmd(message: Message):
        await help_cmd(message)

    @dp.message(Command("join"))
    async def join_cmd(message: Message):
        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)
            # default code until /me
            code = message.from_user.username or str(message.from_user.id)
            code = norm_code(code)[:5]
            await upsert_user_member(conn, gid, message.from_user.id, code)
        await message.reply("Joined. Now set your code: /me AVC (or PC/AS/SS/VH/KA)")

    @dp.message(Command("me"))
    async def me_cmd(message: Message):
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.reply("Usage: /me AVC")
            return
        code = norm_code(parts[1])
        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)
            await upsert_user_member(conn, gid, message.from_user.id, code)
        await message.reply(f"Code set to {code}")

    @dp.message(Command("guest"))
    async def guest_cmd(message: Message):
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.reply("Usage: /guest Driver")
            return
        guest_code = norm_guest(parts[1])
        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)
            mid = await get_or_create_guest(conn, gid, guest_code)
        await message.reply(f"Guest ready: {guest_code} (id {mid})")

    @dp.message(Command("list"))
    async def list_cmd(message: Message):
        n = 10
        parts = message.text.split()
        if len(parts) >= 2 and parts[1].isdigit():
            n = min(50, max(1, int(parts[1])))

        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)
            rows = await conn.fetch(
                """
                select id, amount, currency, note, created_at
                from expenses
                where group_id=$1
                order by created_at desc
                limit $2
                """,
                gid, n
            )

        if not rows:
            await message.reply("No expenses yet.")
            return

        lines = ["Last expenses:"]
        for r in rows:
            note = (r["note"] or "").strip()
            note = note[:60] + ("..." if len(note) > 60 else "")
            lines.append(f"- #{r['id']}: {r['amount']} {r['currency']} | {note} | {r['created_at']:%Y-%m-%d %H:%M}")
        await message.reply("\n".join(lines))

    @dp.message(Command("delete"))
    async def delete_cmd(message: Message):
        parts = message.text.split()
        if len(parts) < 2:
            await message.reply("Usage: /delete last  OR  /delete <expense_id>")
            return
        target = parts[1].lower()

        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)

            if target == "last":
                row = await conn.fetchrow(
                    """
                    select id, amount, currency, note, created_at
                    from expenses
                    where group_id=$1
                    order by created_at desc
                    limit 1
                    """,
                    gid
                )
            else:
                if not target.isdigit():
                    await message.reply("Usage: /delete last  OR  /delete <expense_id>")
                    return
                row = await conn.fetchrow(
                    """
                    select id, amount, currency, note, created_at
                    from expenses
                    where group_id=$1 and id=$2
                    """,
                    gid, int(target)
                )

            if not row:
                await message.reply("Nothing found to delete.")
                return

            exp_id = row["id"]
            await conn.execute("delete from expenses where group_id=$1 and id=$2", gid, exp_id)

        note = (row["note"] or "").strip()
        note = note[:60] + ("..." if len(note) > 60 else "")
        await message.reply(
            f"Deleted Expense #{exp_id}: {row['amount']} {row['currency']} | {note} | {row['created_at']:%Y-%m-%d %H:%M}"
        )

    @dp.message(Command("undo"))
    async def undo_cmd(message: Message):
        message.text = "/delete last"
        await delete_cmd(message)

    @dp.message(Command("balance"))
    async def balance_cmd(message: Message):
        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)
            net = await compute_net_positions(conn, gid)

        lines = ["Balances (net, AED):"]
        for n, v in sorted(net.items()):
            sign = "+" if v >= 0 else "-"
            lines.append(f"- {n}: {sign}{abs(v).quantize(AMT('0.01'))} AED")

        pays = settlement_suggestions(net)
        if pays:
            lines.append("Suggested settlements:")
            for d, c, x in pays:
                lines.append(f"- {d} -> {c}: {x} AED")
        else:
            lines.append("All settled.")
        await message.reply("\n".join(lines))

    @dp.message(F.text)
    async def handle_text(message: Message):
        text = (message.text or "").strip()
        if text.startswith("/"):
            return

        async with pool.acquire() as conn:
            gid = await ensure_group(conn, message.chat.id)
            known_members = await list_member_codes(conn, gid)
            if len(known_members) < 2:
                await message.reply("Not enough members. Everyone do /join then /me AVC (etc).")
                return

            try:
                parsed = parse_loose_english(text, known_members)
            except ValueError as e:
                await message.reply(str(e))
                return

            payer_row = await get_member_by_code(conn, gid, parsed.payer)
            if not payer_row or payer_row["is_guest"]:
                await message.reply(f"Payer '{parsed.payer}' must be a real member. They must /join and /me {parsed.payer}.")
                return
            payer_id = payer_row["id"]

            # ensure participants exist (create guests if needed)
            participant_ids: Dict[str, int] = {}
            for code in parsed.participants:
                if code.lower().startswith("guest:"):
                    mid = await get_or_create_guest(conn, gid, code)
                else:
                    row = await get_member_by_code(conn, gid, code)
                    if not row:
                        await message.reply(f"Member '{code}' not found. They must /join and /me {code}.")
                        return
                    mid = row["id"]
                participant_ids[code] = mid

            shares = compute_shares(parsed.amount, parsed.participants, parsed.weights)

            exp_row = await conn.fetchrow(
                """
                insert into expenses(group_id, created_by_tg_user_id, payer_member_id, amount, currency, note, raw_text)
                values($1,$2,$3,$4,'AED',$5,$6)
                returning id
                """,
                gid, message.from_user.id, payer_id, str(parsed.amount), parsed.note, text
            )
            exp_id = exp_row["id"]

            for code, share_amt in shares.items():
                await conn.execute(
                    """
                    insert into expense_splits(expense_id, member_id, weight, share_amount)
                    values($1,$2,$3,$4)
                    """,
                    exp_id, participant_ids[code], str(parsed.weights.get(code, AMT("1"))), str(share_amt)
                )

            net = await compute_net_positions(conn, gid)
            pays = settlement_suggestions(net)

        lines = [f"Expense #{exp_id} logged: {parsed.amount} AED"]
        lines.append(f"Paid by: {parsed.payer}")
        lines.append("Shares:")
        for n in parsed.participants:
            lines.append(f"- {n}: {shares[n]} AED")

        if pays:
            lines.append("Suggested settlements:")
            for d, c, x in pays[:6]:
                lines.append(f"- {d} -> {c}: {x} AED")
            if len(pays) > 6:
                lines.append(f"(+{len(pays)-6} more)")
        else:
            lines.append("All settled.")

        await message.reply("\n".join(lines))

    await dp.start_polling(Bot(TOKEN))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
