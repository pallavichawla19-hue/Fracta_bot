import asyncio
import logging
import os
from urllib.parse import urlparse

import asyncpg
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fracta-bot")

# ----------------------------
# Env
# ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

DEFAULT_MEMBERS = ["AVC", "PC", "AS", "SS", "VH", "KA"]

# Global pool (lazy init)
_db_pool: asyncpg.Pool | None = None
_db_lock = asyncio.Lock()


def _redact_db_url(db_url: str) -> str:
    """Return a safe version for logs."""
    try:
        p = urlparse(db_url)
        if not p.scheme or not p.hostname:
            return "<invalid DATABASE_URL>"
        user = p.username or "?"
        host = p.hostname
        port = p.port or "?"
        db = (p.path or "").lstrip("/") or "?"
        return f"{p.scheme}://{user}:***@{host}:{port}/{db}"
    except Exception:
        return "<unparseable DATABASE_URL>"


async def get_db_pool() -> asyncpg.Pool | None:
    """
    Lazy-create a DB pool.
    If DB is unreachable, return None (do NOT crash the process).
    """
    global _db_pool

    if not DATABASE_URL:
        return None

    if _db_pool is not None:
        return _db_pool

    async with _db_lock:
        if _db_pool is not None:
            return _db_pool

        # Retry a few times, then give up (bot still runs)
        last_err = None
        for attempt in range(1, 6):
            try:
                logger.info("Connecting to DB (attempt %s/5): %s", attempt, _redact_db_url(DATABASE_URL))
                _db_pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=1,
                    max_size=5,
                    command_timeout=10,
                    timeout=10,
                )
                logger.info("DB pool created successfully.")
                return _db_pool
            except Exception as e:
                last_err = e
                logger.warning("DB connection attempt %s failed: %s", attempt, repr(e))
                await asyncio.sleep(min(2 * attempt, 8))

        logger.error("DB unreachable after retries. Bot will run without DB. Last error: %s", repr(last_err))
        return None


async def close_db_pool() -> None:
    global _db_pool
    if _db_pool is not None:
        try:
            await _db_pool.close()
        except Exception:
            pass
        _db_pool = None


# ----------------------------
# Telegram Handlers
# ----------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Fracta is online.\n"
        f"DB: {'connected' if (await get_db_pool()) else 'NOT connected'}\n\n"
        f"Default members: {', '.join(DEFAULT_MEMBERS)}\n\n"
        "Examples:\n"
        "- PC paid 120 split by everyone\n"
        "- AVC paid 80 split by everyone except SS\n"
        "- AS paid 210 split by everyone besides VH and double the shares for PC\n\n"
        "Commands:\n"
        "/balance\n"
        "undo last or undo 123"
    )
    await update.message.reply_text(msg)


async def balance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool = await get_db_pool()
    if pool is None:
        await update.message.reply_text("DB is not connected. Fix DATABASE_URL (use Supabase pooler) and restart.")
        return

    # Replace this with your real schema queries
    # For now, just a sanity ping
    try:
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1;")
        await update.message.reply_text(f"DB OK (ping={val}). Balance feature not wired yet.")
    except Exception as e:
        logger.exception("Balance query failed.")
        await update.message.reply_text(f"DB error: {type(e).__name__}. Check logs.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    lower = text.lower()

    # Quick commands without slash
    if lower.startswith("undo"):
        await update.message.reply_text("Undo feature not wired yet. (But bot is alive.)")
        return

    # This is where your natural language parsing will go
    await update.message.reply_text("Got it. (Parser not wired yet.) Send /balance to test DB.")


async def on_startup(app: Application) -> None:
    # Do NOT hard-fail if DB is down. Just log.
    pool = await get_db_pool()
    logger.info("Startup complete. DB connected: %s", bool(pool))


async def on_shutdown(app: Application) -> None:
    await close_db_pool()
    logger.info("Shutdown complete.")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("balance", balance_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Bot starting (polling)...")
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
