import os
import re
import json
import uuid
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import asyncpg
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# -----------------------------
# Config
# -----------------------------
DEFAULT_MEMBERS = ["AVC", "PC", "AS", "SS", "VH", "KA"]
DEFAULT_CURRENCY = "AED"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Global DB pool (optional)
DB_POOL: Optional[asyncpg.pool.Pool] = None


# -----------------------------
# Helpers: text parsing
# -----------------------------
AMOUNT_RE = re.compile(r"(?P<amt>\d+(\.\d{1,2})?)")
PAID_BY_RE = re.compile(r"\b(?P<payer>[A-Za-z]{1,10})\s+paid\b|\bpaid\s+by\s+(?P<payer2>[A-Za-z]{1,10})\b", re.IGNORECASE)

EXCLUDE_RE = re.compile(r"\b(except|excluding|besides)\b", re.IGNORECASE)
DOUBLE_RE = re.compile(r"\b(double|2x|two\s*x)\b", re.IGNORECASE)
TRIPLE_RE = re.compile(r"\b(triple|3x|three\s*x)\b", re.IGNORECASE)

SPLIT_ALL_RE = re.compile(r"\b(split|split it|split this|split by|split among|split between)\b", re.IGNORECASE)
EVERYONE_RE = re.compile(r"\b(everyone|all|all of us)\b", re.IGNORECASE)

UNDO_RE = re.compile(r"^\s*undo(\s+(?P<what>last|\d+))?\s*$", re.IGNORECASE)

# Clean member tokens like "PC," "pc." -> "PC"
def norm_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", s).upper()


def extract_amount(text: str) -> Optional[float]:
    # pick the first number as amount
    m = AMOUNT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group("amt"))
    except:
        return None


def extract_payer(text: str) -> Optional[str]:
    m = PAID_BY_RE.search(text)
    if not m:
        return None
    payer = m.group("payer") or m.group("payer2")
    if not payer:
        return None
    return norm_name(payer)


def extract_exclusions(text: str) -> List[str]:
    # Look for: "except D" / "excluding D and KA" / "besides PC"
    # Basic heuristic: after exclusion keyword, collect tokens that look like names
    m = EXCLUDE_RE.search(text)
    if not m:
        return []
    tail = text[m.end():]
    tokens = re.split(r"[,\s]+|and", tail, flags=re.IGNORECASE)
    names = []
    for t in tokens:
        n = norm_name(t)
        if n and len(n) <= 10:
            names.append(n)
    return list(dict.fromkeys(names))  # unique preserve order


def extract_multipliers(text: str) -> Dict[str, float]:
    """
    Examples:
    - "double the shares for D"
    - "double share for PC"
    - "triple share for SS"
    We'll scan for patterns "... for <NAME>" near 'double/triple'
    """
    out: Dict[str, float] = {}

    # double ... for X
    for mult_word, mult_val in [("double", 2.0), ("triple", 3.0)]:
        # capture "double ... for D" or "double share for D"
        pattern = re.compile(rf"\b{mult_word}\b.*?\bfor\s+([A-Za-z]{{1,10}})\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            name = norm_name(m.group(1))
            if name:
                out[name] = mult_val

    # "2x for D"
    pattern2 = re.compile(r"\b(2x|3x)\b.*?\bfor\s+([A-Za-z]{1,10})\b", re.IGNORECASE)
    for m in pattern2.finditer(text):
        mult = 2.0 if m.group(1).lower() == "2x" else 3.0
        name = norm_name(m.group(2))
        if name:
            out[name] = mult

    return out


def extract_participants_explicit(text: str) -> List[str]:
    """
    If user says "split between PC and AS" or "split among PC, AS, SS"
    We'll collect tokens after split keywords.
    """
    if not SPLIT_ALL_RE.search(text):
        return []

    # If "everyone" present, don't treat as explicit list
    if EVERYONE_RE.search(text):
        return []

    # Take tail after "split"
    m = SPLIT_ALL_RE.search(text)
    tail = text[m.end():] if m else text
    tokens = re.split(r"[,\s]+|and", tail, flags=re.IGNORECASE)
    names = []
    for t in tokens:
        n = norm_name(t)
        # heuristic: short tokens likely names
        if n and len(n) <= 10 and not n.isdigit():
            names.append(n)
    # de-dupe
    names = list(dict.fromkeys(names))
    return names


@dataclass
class ParsedExpense:
    amount: float
    payer: str
    participants: List[str]
    weights: Dict[str, float]
    note: str
    currency: str = DEFAULT_CURRENCY


def parse_expense_text(text: str, default_members: List[str]) -> Tuple[Optional[ParsedExpense], str]:
    """
    Returns (ParsedExpense or None, error_message_if_any)
    """
    raw = text.strip()

    amt = extract_amount(raw)
    if amt is None:
        return None, "I could not find an amount. Example: `PC paid 120 split by everyone`"

    payer = extract_payer(raw)
    if payer is None:
        # fallback: if message starts with a name token
        first = norm_name(raw.split()[0]) if raw.split() else ""
        payer = first if first else None

    if payer is None:
        return None, "I could not find who paid. Example: `PC paid 120 split by everyone`"

    # Determine base participants
    explicit = extract_participants_explicit(raw)
    exclusions = extract_exclusions(raw)
    multipliers = extract_multipliers(raw)

    everyone = bool(EVERYONE_RE.search(raw)) or bool(re.search(r"\beveryone\b", raw, re.IGNORECASE))
    if explicit:
        participants = explicit
    elif everyone:
        participants = list(default_members)
    else:
        # Default: split among default members
        participants = list(default_members)

    # Include payer by default (you requested). Ensure payer is in participants.
    if payer not in participants:
        participants.append(payer)

    # Apply exclusions
    participants = [p for p in participants if p not in exclusions]

    # Allow "guest" / non-members:
    # If text contains extra short names not in default list, treat them as additional participants.
    # Example: "split by everyone and GUEST1" or "with John"
    possible_names = re.findall(r"\b[A-Za-z]{1,10}\b", raw)
    for token in possible_names:
        n = norm_name(token)
        if not n:
            continue
        # ignore common words
        if n.lower() in {
            "PAID", "SPLIT", "BY", "EVERYONE", "ALL", "EXCEPT", "EXCLUDING", "BESIDES",
            "DOUBLE", "TRIPLE", "SHARE", "SHARES", "FOR", "AND", "AED"
        }:
            continue
        # keep if looks like an ID-ish name and not a number
        if n.isdigit():
            continue
        # If user explicitly mentions someone not in default list and not payer keyword etc, add as participant
        # Only add if the message contains "with" or "including" or explicit list was detected
        if n not in participants and (explicit or re.search(r"\b(with|including|include)\b", raw, re.IGNORECASE)):
            participants.append(n)

    if len(participants) < 1:
        return None, "Participants list became empty after exclusions. Try again."

    # Weights
    weights = {p: 1.0 for p in participants}
    for name, mult in multipliers.items():
        if name in weights:
            weights[name] = mult

    note = raw
    return ParsedExpense(amount=amt, payer=payer, participants=participants, weights=weights, note=note), ""


# -----------------------------
# DB layer (optional)
# -----------------------------
SCHEMA_SQL = """
create table if not exists groups (
  group_id bigint primary key,
  title text,
  currency text not null default 'AED',
  created_at timestamptz not null default now()
);

create table if not exists members (
  group_id bigint not null,
  name text not null,
  created_at timestamptz not null default now(),
  primary key (group_id, name),
  foreign key (group_id) references groups(group_id) on delete cascade
);

create table if not exists expenses (
  id bigserial primary key,
  group_id bigint not null,
  payer text not null,
  amount numeric not null,
  currency text not null default 'AED',
  note text,
  created_by bigint,
  created_at timestamptz not null default now(),
  is_deleted boolean not null default false,
  foreign key (group_id) references groups(group_id) on delete cascade
);

create table if not exists expense_splits (
  expense_id bigint not null,
  participant text not null,
  weight numeric not null default 1,
  share_amount numeric not null,
  created_at timestamptz not null default now(),
  primary key (expense_id, participant),
  foreign key (expense_id) references expenses(id) on delete cascade
);
"""


async def init_db_pool() -> None:
    global DB_POOL
    if not DATABASE_URL:
        print("DATABASE_URL not set. Running without DB.")
        DB_POOL = None
        return

    try:
        DB_POOL = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            timeout=10,
        )
        async with DB_POOL.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        print("DB connected and schema ensured.")
    except Exception as e:
        print(f"DB connection failed. Running without DB. Error: {e}")
        DB_POOL = None


async def ensure_group_defaults(group_id: int, title: str, members: List[str]) -> None:
    if DB_POOL is None:
        return
    async with DB_POOL.acquire() as conn:
        await conn.execute(
            "insert into groups(group_id, title, currency) values($1,$2,$3) on conflict (group_id) do update set title=excluded.title",
            group_id, title, DEFAULT_CURRENCY
        )
        for m in members:
            await conn.execute(
                "insert into members(group_id, name) values($1,$2) on conflict do nothing",
                group_id, m
            )


def compute_shares(amount: float, participants: List[str], weights: Dict[str, float]) -> Dict[str, float]:
    total_w = sum(weights.get(p, 1.0) for p in participants)
    if total_w <= 0:
        total_w = float(len(participants))
    raw = {p: (weights.get(p, 1.0) / total_w) * amount for p in participants}

    # Rounding to 2 decimals and adjust drift
    rounded = {p: round(v, 2) for p, v in raw.items()}
    drift = round(amount - sum(rounded.values()), 2)
    if abs(drift) >= 0.01:
        # add drift to payer if present, else first participant
        target = participants[0]
        rounded[target] = round(rounded[target] + drift, 2)
    return rounded


async def db_insert_expense(group_id: int, created_by: int, parsed: ParsedExpense) -> int:
    assert DB_POOL is not None
    shares = compute_shares(parsed.amount, parsed.participants, parsed.weights)
    async with DB_POOL.acquire() as conn:
        async with conn.transaction():
            exp_id = await conn.fetchval(
                "insert into expenses(group_id, payer, amount, currency, note, created_by) values($1,$2,$3,$4,$5,$6) returning id",
                group_id, parsed.payer, parsed.amount, parsed.currency, parsed.note, created_by
            )
            for p in parsed.participants:
                await conn.execute(
                    "insert into expense_splits(expense_id, participant, weight, share_amount) values($1,$2,$3,$4)",
                    exp_id, p, parsed.weights.get(p, 1.0), shares[p]
                )
            return int(exp_id)


async def db_undo(group_id: int, exp_id: int) -> bool:
    if DB_POOL is None:
        return False
    async with DB_POOL.acquire() as conn:
        res = await conn.execute(
            "update expenses set is_deleted=true where id=$1 and group_id=$2 and is_deleted=false",
            exp_id, group_id
        )
        # asyncpg returns "UPDATE <n>"
        return res.startswith("UPDATE ") and not res.endswith(" 0")


async def db_last_expense_id(group_id: int) -> Optional[int]:
    if DB_POOL is None:
        return None
    async with DB_POOL.acquire() as conn:
        return await conn.fetchval(
            "select id from expenses where group_id=$1 and is_deleted=false order by id desc limit 1",
            group_id
        )


async def db_balance(group_id: int) -> Dict[str, float]:
    """
    Net balance per person:
    + means they should receive money
    - means they owe money
    """
    if DB_POOL is None:
        return {}

    async with DB_POOL.acquire() as conn:
        rows = await conn.fetch("""
            select e.payer as name,
                   sum(e.amount)::numeric as paid
            from expenses e
            where e.group_id=$1 and e.is_deleted=false
            group by e.payer
        """, group_id)

        paid_map = {r["name"]: float(r["paid"]) for r in rows}

        rows2 = await conn.fetch("""
            select s.participant as name,
                   sum(s.share_amount)::numeric as owed
            from expenses e
            join expense_splits s on s.expense_id=e.id
            where e.group_id=$1 and e.is_deleted=false
            group by s.participant
        """, group_id)

        owed_map = {r["name"]: float(r["owed"]) for r in rows2}

    names = set(paid_map.keys()) | set(owed_map.keys())
    net = {}
    for n in names:
        net[n] = round(paid_map.get(n, 0.0) - owed_map.get(n, 0.0), 2)
    return dict(sorted(net.items(), key=lambda x: (-x[1], x[0])))


def settlement_suggestions(net: Dict[str, float]) -> List[Tuple[str, str, float]]:
    """
    Greedy settlement: debtors pay creditors.
    """
    creditors = [(n, amt) for n, amt in net.items() if amt > 0.01]
    debtors = [(n, -amt) for n, amt in net.items() if amt < -0.01]

    creditors.sort(key=lambda x: x[1], reverse=True)
    debtors.sort(key=lambda x: x[1], reverse=True)

    i = j = 0
    transfers = []
    while i < len(debtors) and j < len(creditors):
        d_name, d_amt = debtors[i]
        c_name, c_amt = creditors[j]
        x = min(d_amt, c_amt)
        x = round(x, 2)
        if x >= 0.01:
            transfers.append((d_name, c_name, x))
        d_amt = round(d_amt - x, 2)
        c_amt = round(c_amt - x, 2)
        debtors[i] = (d_name, d_amt)
        creditors[j] = (c_name, c_amt)
        if debtors[i][1] <= 0.01:
            i += 1
        if creditors[j][1] <= 0.01:
            j += 1
    return transfers


# -----------------------------
# Telegram handlers
# -----------------------------
def is_group_chat(update: Update) -> bool:
    return update.effective_chat and update.effective_chat.type in ("group", "supergroup")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat:
        return
    await ensure_group_defaults(chat.id, chat.title or "Untitled", DEFAULT_MEMBERS)

    db_status = "connected" if DB_POOL is not None else "NOT connected"
    members = ", ".join(DEFAULT_MEMBERS)

    msg = (
        f"Fracta is online.\n"
        f"DB: {db_status}\n\n"
        f"Default members: {members}\n\n"
        "Examples:\n"
        "- `PC paid 120 split by everyone`\n"
        "- `AVC paid 80 split by everyone except SS`\n"
        "- `AS paid 210 split by everyone besides VH and double the shares for PC`\n\n"
        "Commands:\n"
        "- `/balance`\n"
        "- `undo last` or `undo 123`\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat:
        return

    if DB_POOL is None:
        await update.message.reply_text("DB is down. I cannot compute balance yet.")
        return

    net = await db_balance(chat.id)
    if not net:
        await update.message.reply_text("No expenses recorded yet.")
        return

    lines = ["Balances ( + receive, - owe ):"] + [f"- {n}: {amt:.2f} {DEFAULT_CURRENCY}" for n, amt in net.items()]
    transfers = settlement_suggestions(net)
    if transfers:
        lines.append("\nSuggested settlements:")
        for d, c, x in transfers[:10]:
            lines.append(f"- {d} -> {c}: {x:.2f} {DEFAULT_CURRENCY}")
        if len(transfers) > 10:
            lines.append(f"(+{len(transfers)-10} more)")
    await update.message.reply_text("\n".join(lines))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    text = update.message.text or ""
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return

    # Ensure defaults in DB if available
    await ensure_group_defaults(chat.id, chat.title or "Untitled", DEFAULT_MEMBERS)

    # Undo handling
    um = UNDO_RE.match(text)
    if um:
        if DB_POOL is None:
            await update.message.reply_text("DB is down. Undo is unavailable.")
            return

        what = um.group("what")
        if not what or what.lower() == "last":
            exp_id = await db_last_expense_id(chat.id)
            if not exp_id:
                await update.message.reply_text("No expenses to undo.")
                return
        else:
            exp_id = int(what)

        ok = await db_undo(chat.id, exp_id)
        await update.message.reply_text("Undone." if ok else "Could not undo (not found or already undone).")
        return

    # Parse expense
    parsed, err = parse_expense_text(text, DEFAULT_MEMBERS)
    if not parsed:
        # keep it helpful, not spammy
        await update.message.reply_text(err)
        return

    shares = compute_shares(parsed.amount, parsed.participants, parsed.weights)

    if DB_POOL is None:
        # Respond but do not save
        lines = [
            f"Expense (NOT SAVED - DB down):",
            f"- Paid by: {parsed.payer}",
            f"- Amount: {parsed.amount:.2f} {parsed.currency}",
            "Shares:",
        ]
        for p in parsed.participants:
            lines.append(f"- {p}: {shares[p]:.2f} {parsed.currency}")
        await update.message.reply_text("\n".join(lines))
        return

    # Save to DB
    try:
        exp_id = await db_insert_expense(chat.id, user.id, parsed)
    except Exception as e:
        await update.message.reply_text(f"DB error. Could not save. Error: {e}")
        return

    # Reply summary
    lines = [
        f"Saved expense #{exp_id}",
        f"- Paid by: {parsed.payer}",
        f"- Amount: {parsed.amount:.2f} {parsed.currency}",
        "Shares:",
    ]
    for p in parsed.participants:
        w = parsed.weights.get(p, 1.0)
        w_txt = "" if abs(w - 1.0) < 1e-9 else f" (x{w:g})"
        lines.append(f"- {p}: {shares[p]:.2f} {parsed.currency}{w_txt}")

    lines.append("\nTip: `undo last` or `undo {id}`")
    await update.message.reply_text("\n".join(lines))


# -----------------------------
# Main
# -----------------------------
async def on_startup(app: Application):
    # DB init should not crash the bot
    await init_db_pool()


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("balance", cmd_balance))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Start DB init but do not block bot startup
    application.post_init = on_startup

    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
