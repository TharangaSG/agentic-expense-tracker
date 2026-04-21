"""
FastAPI application for WhatsApp webhook integration with the financial assistance bot.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.utils.logging_config import get_logger
from src.interfaces.whatsapp.whatsapp_handler import whatsapp_router
from src.config.containers import get_async_database

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: connect/disconnect the async database."""
    # ── Startup ──
    db = get_async_database()
    try:
        await db.connect()
        logger.info("✅ PostgreSQL connection pool initialized on startup.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL on startup: {e}")
        logger.warning("The app will attempt to connect on the first request.")
    
    yield  # App runs here
    
    # ── Shutdown ──
    try:
        await db.disconnect()
        logger.info("PostgreSQL connection pool closed on shutdown.")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL pool: {e}")


# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Financial Assistant WhatsApp Bot",
    version="2.0.0",
    lifespan=lifespan,
)

# Include WhatsApp router
app.include_router(whatsapp_router)

@app.get("/")
async def root():
    return {"message": "Financial Assistant WhatsApp Bot is running!"}

@app.get("/health")
async def health_check():
    db = get_async_database()
    db_status = "connected" if db.pool is not None else "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "version": "2.0.0 (multi-agent)",
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)