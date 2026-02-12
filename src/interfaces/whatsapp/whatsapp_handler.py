import logging
import os
import tempfile
from typing import Dict

import httpx
from fastapi import APIRouter, Request, Response
from src.settings import settings
from src.config.containers import get_stt_provider, get_vision_provider
from src.domain.models import TranscriptionRequest, VisionRequest, AudioFormat, ImageFormat
from src.data_inserting_flow import process_user_input

logger = logging.getLogger(__name__)

# Router for WhatsApp responses
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = settings.WHATSAPP_TOKEN
WHATSAPP_PHONE_NUMBER_ID = settings.WHATSAPP_PHONE_NUMBER_ID
WHATSAPP_VERIFY_TOKEN = settings.WHATSAPP_VERIFY_TOKEN


@whatsapp_router.api_route("/whatsapp_response", methods=["GET", "POST"])
async def whatsapp_handler(request: Request) -> Response:
    """Handles incoming messages and status updates from the WhatsApp Cloud API."""

    if request.method == "GET":
        # Webhook verification
        params = request.query_params
        if params.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
            return Response(content=params.get("hub.challenge"), status_code=200)
        return Response(content="Verification token mismatch", status_code=403)

    try:
        data = await request.json()
        change_value = data["entry"][0]["changes"][0]["value"]
        
        if "messages" in change_value:
            message = change_value["messages"][0]
            from_number = message["from"]
            
            # Send processing message first
            await send_response(from_number, "Processing your request... ")
            
            # Process different message types
            content = ""
            if message["type"] == "audio":
                content = await process_audio_message(message)
            elif message["type"] == "image":
                content = await process_image_message(message)
            elif message["type"] == "text":
                content = message["text"]["body"]
            else:
                await send_response(from_number, "Sorry, I can only process text, audio, and image messages.")
                return Response(content="Unsupported message type", status_code=200)

            # Process the content 
            try:
                response_message = await process_user_input(content)
                success = await send_response(from_number, response_message)
                
                if not success:
                    return Response(content="Failed to send message", status_code=500)
                    
            except Exception as e:
                logger.error(f"Error processing user input: {e}", exc_info=True)
                await send_response(from_number, "Sorry, I encountered an error processing your request. Please try again.")

            return Response(content="Message processed", status_code=200)

        elif "statuses" in change_value:
            return Response(content="Status update received", status_code=200)
        else:
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)


async def process_audio_message(message: Dict) -> str:
    """Download and transcribe audio message using STT port."""
    try:
        audio_id = message["audio"]["id"]
        audio_bytes = await download_media(audio_id)
        
        # Get STT provider
        stt_provider = get_stt_provider()
        
        # Create transcription request
        request = TranscriptionRequest(
            audio_data=audio_bytes,
            format=AudioFormat.OGG
        )
        
        # Transcribe using port
        response = stt_provider.transcribe(request)
        return response.text
            
    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        return "Sorry, I couldn't process your audio message. Please try again or send a text message."


async def process_image_message(message: Dict) -> str:
    """Download and analyze image message using Vision port."""
    try:
        # Get image caption if any
        content = message.get("image", {}).get("caption", "")
        
        # Download image
        image_id = message["image"]["id"]
        image_bytes = await download_media(image_id)
        
        # Get vision provider
        vision_provider = get_vision_provider()
        
        # Create vision request
        request = VisionRequest(
            image_data=image_bytes,
            format=ImageFormat.JPEG,
            prompt="Extract receipt data from this image. List all items with their quantities, unit prices, and total prices."
        )
        
        # Analyze using port
        response = vision_provider.analyze_image(request)
        image_analysis = response.extracted_text
        
        # Combine caption and image analysis
        if content:
            return f"{content}\n\n[Image Analysis: {image_analysis}]"
        else:
            return f"[Image Analysis: {image_analysis}]"
            
    except Exception as e:
        logger.error(f"Error processing image message: {e}")
        return "Sorry, I couldn't process your image. Please try again or send a text message."


async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp."""
    media_metadata_url = f"https://graph.facebook.com/v21.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        # Get media metadata
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

        # Download the actual media
        media_response = await client.get(download_url, headers=headers)
        media_response.raise_for_status()
        return media_response.content


async def send_response(from_number: str, response_text: str) -> bool:
    """Send text response to user via WhatsApp API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    json_data = {
        "messaging_product": "whatsapp",
        "to": from_number,
        "type": "text",
        "text": {"body": response_text},
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
                headers=headers,
                json=json_data,
            )
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        return False