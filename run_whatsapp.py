"""
Script to run the WhatsApp webhook server for the Financial Assistant bot.
"""

import uvicorn
from src.interfaces.whatsapp.whatsapp_app import app

if __name__ == "__main__":
    print("Starting WhatsApp Financial Assistant Bot...")
    print("\nServer will run on http://0.0.0.0:8001")
    print("Webhook endpoint: http://your-domain.com:8001/whatsapp_response")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        log_level="info"
    )