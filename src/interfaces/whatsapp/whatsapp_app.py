"""
FastAPI application for WhatsApp webhook integration with the financial assistance bot.
This runs separately from the Chainlit app and handles WhatsApp messages.
"""

import logging
from fastapi import FastAPI
from src.interfaces.whatsapp.whatsapp_handler import whatsapp_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Financial Assistant WhatsApp Bot", version="1.0.0")

# Include WhatsApp router
app.include_router(whatsapp_router)

@app.get("/")
async def root():
    return {"message": "Financial Assistant WhatsApp Bot is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)