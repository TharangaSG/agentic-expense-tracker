"""
Script to run the Telegram webhook server for the Financial Assistant bot.
"""

import uvicorn

from src.utils.logging_config import setup_logging

setup_logging()

from src.interfaces.telegram.telegram_app import app


if __name__ == "__main__":
    print("Starting Telegram Financial Assistant Bot...")
    print("\nServer will run on http://0.0.0.0:8002")
    print("Webhook endpoint: http://your-domain.com:8002/telegram/webhook")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info",
    )
