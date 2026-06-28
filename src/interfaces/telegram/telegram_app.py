"""
FastAPI application for Telegram webhook integration with the financial assistant bot.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.containers import get_async_database
from src.interfaces.telegram.telegram_handler import telegram_router
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: connect/disconnect the async database."""
    db = get_async_database()
    try:
        await db.connect()
        logger.info("Telegram app database connection pool initialized on startup.")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL on startup: {e}")
        logger.warning("The app will attempt to connect on the first request.")

    yield

    try:
        await db.disconnect()
        logger.info("PostgreSQL connection pool closed on shutdown.")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL pool: {e}")


app = FastAPI(
    title="Financial Assistant Telegram Bot",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(telegram_router)


@app.get("/")
async def root():
    return {"message": "Financial Assistant Telegram Bot is running!"}


@app.get("/health")
async def health_check():
    db = get_async_database()
    db_status = "connected" if db.pool is not None else "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "version": "2.0.0 (telegram)",
    }


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
