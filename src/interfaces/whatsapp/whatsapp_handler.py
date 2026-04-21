"""
WhatsApp Interface Handler

This is the INTERFACE layer — it handles:
  - WhatsApp webhook verification (GET)
  - Receiving incoming messages (POST)
  - Multimodal input conversion (audio → text, image → text)
  - Sending responses back via WhatsApp API

For all AGENT LOGIC (LLM orchestration, saving purchases, querying spending),
it delegates to `process_user_input()` from the Main Agent orchestrator
"""

import logging
import time
from typing import Dict

import httpx
from fastapi import APIRouter, Request, Response
from src.utils.logging_config import get_logger
from src.settings import settings
from src.config.containers import get_stt_provider, get_vision_provider
from src.domain.models import TranscriptionRequest, VisionRequest, AudioFormat, ImageFormat

# Import the Main Agent orchestrator — this is the ONLY agent entry point
from src.agents.main_agent import process_user_input

logger = get_logger(__name__)

# Router for WhatsApp responses
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = settings.WHATSAPP_TOKEN
WHATSAPP_PHONE_NUMBER_ID = settings.WHATSAPP_PHONE_NUMBER_ID
WHATSAPP_VERIFY_TOKEN = settings.WHATSAPP_VERIFY_TOKEN

@whatsapp_router.api_route("/whatsapp_response", methods=["GET", "POST"])
async def whatsapp_handler(request: Request) -> Response:
    """Handles incoming messages and status updates from the WhatsApp Cloud API."""
    start_time = time.time()

    if request.method == "GET":
        # Webhook verification
        params = request.query_params
        verify_token = params.get("hub.verify_token")

        if verify_token == WHATSAPP_VERIFY_TOKEN:
            logger.info("WhatsApp webhook verification successful")
            return Response(content=params.get("hub.challenge"), status_code=200)
        else:
            logger.warning(f"WhatsApp webhook verification failed | Expected: {WHATSAPP_VERIFY_TOKEN}, Got: {verify_token}")
            return Response(content="Verification token mismatch", status_code=403)

    try:
        data = await request.json()
        logger.debug(f"Received WhatsApp webhook payload: {str(data)[:200]}...")

        change_value = data["entry"][0]["changes"][0]["value"]

        if "messages" in change_value:
            message = change_value["messages"][0]
            from_number = message["from"]
            message_type = message["type"]

            logger.info(f"Received {message_type} message from: {from_number}")

            # Send processing indicator
            await send_response(from_number, "Processing your request... ⏳")

            content = ""
            if message_type == "audio":
                logger.debug("Processing audio message type")
                content = await process_audio_message(message)
            elif message_type == "image":
                logger.debug("Processing image message type")
                content = await process_image_message(message)
            elif message_type == "text":
                content = message["text"]["body"]
                logger.debug(f"Text message content: '{content[:100]}{'...' if len(content) > 100 else ''}'")
            else:
                logger.warning(f"Unsupported message type received: {message_type}")
                await send_response(
                    from_number,
                    "Sorry, I can only process text, audio, and image messages."
                )
                return Response(content="Unsupported message type", status_code=200)

            if not content:
                logger.warning("Message processing resulted in empty content")
                await send_response(
                    from_number,
                    "Sorry, I couldn't understand your message. Please try again."
                )
                return Response(content="Empty content", status_code=200)

            logger.info(f"Delegating WhatsApp message to Main Agent: '{content[:100]}{'...' if len(content) > 100 else ''}'")

            try:
                response_message = await process_user_input(content)
                elapsed = time.time() - start_time

                logger.info(f"Main Agent response received after {elapsed:.2f}s | Response preview: {response_message[:150]}...")

                success = await send_response(from_number, response_message)

                if not success:
                    logger.error(f"Failed to send response to {from_number}")
                    return Response(content="Failed to send message", status_code=500)

                logger.info(f"Successfully sent response to {from_number}")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Error from Main Agent after {elapsed:.2f}s: {e}", exc_info=True)
                await send_response(
                    from_number,
                    "Sorry, I encountered an error processing your request. Please try again."
                )

            return Response(content="Message processed", status_code=200)

        elif "statuses" in change_value:
            logger.debug("Received status update")
            return Response(content="Status update received", status_code=200)
        else:
            logger.warning(f"Unknown event type in webhook payload: {list(change_value.keys())}")
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing WhatsApp message after {elapsed:.2f}s: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)



async def process_audio_message(message: Dict) -> str:
    """Download and transcribe audio message using STT port."""
    start_time = time.time()

    try:
        audio_id = message["audio"]["id"]
        logger.debug(f"Downloading audio media: {audio_id}")

        audio_bytes = await download_media(audio_id)
        logger.debug(f"Audio media downloaded: {len(audio_bytes)} bytes")

        stt_provider = get_stt_provider()
        request = TranscriptionRequest(
            audio_data=audio_bytes,
            format=AudioFormat.OGG
        )

        logger.debug("Starting speech-to-text transcription")
        response = stt_provider.transcribe(request)

        elapsed = time.time() - start_time
        logger.info(
            f"Audio message transcribed in {elapsed:.2f}s | "
            f"Transcription: '{response.text[:100]}{'...' if len(response.text) > 100 else ''}'"
        )

        return response.text

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing audio message after {elapsed:.2f}s: {e}", exc_info=True)
        return "Sorry, I couldn't process your audio message. Please try again or send a text message."


async def process_image_message(message: Dict) -> str:
    """Download and analyze image message using Vision port."""
    start_time = time.time()

    try:
        # Get image caption if any
        content = message.get("image", {}).get("caption", "")
        if content:
            logger.debug(f"Image caption: '{content}'")

        # Download image
        image_id = message["image"]["id"]
        logger.debug(f"Downloading image media: {image_id}")

        image_bytes = await download_media(image_id)
        logger.debug(f"Image media downloaded: {len(image_bytes)} bytes")

        vision_provider = get_vision_provider()
        request = VisionRequest(
            image_data=image_bytes,
            format=ImageFormat.JPEG,
            prompt="Extract receipt data from this image. List all items with their quantities, unit prices, and total prices."
        )

        logger.debug("Starting image analysis")
        response = vision_provider.analyze_image(request)
        image_analysis = response.extracted_text

        elapsed = time.time() - start_time
        logger.info(
            f"Image message analyzed in {elapsed:.2f}s | "
            f"Extracted text preview: {image_analysis[:100]}{'...' if len(image_analysis) > 100 else ''}"
        )

        # Combine caption and image analysis
        if content:
            result = f"{content}\n\n[Image Analysis: {image_analysis}]"
        else:
            result = f"[Image Analysis: {image_analysis}]"

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing image message after {elapsed:.2f}s: {e}", exc_info=True)
        return "Sorry, I couldn't process your image. Please try again or send a text message."



async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp."""
    logger.debug(f"Fetching media metadata for: {media_id}")

    media_metadata_url = f"https://graph.facebook.com/v21.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        # Get media metadata
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

        logger.debug(f"Downloading media from URL: {download_url}")

        # Download the actual media
        media_response = await client.get(download_url, headers=headers)
        media_response.raise_for_status()

        logger.debug(f"Media downloaded successfully: {len(media_response.content)} bytes")
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
        logger.debug(f"Sending response to {from_number} | Message length: {len(response_text)} chars")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
                headers=headers,
                json=json_data,
            )

            if response.status_code == 200:
                logger.debug(f"Response sent successfully to {from_number}")
                return True
            else:
                logger.error(f"Failed to send response to {from_number} | Status: {response.status_code} | Body: {response.text[:200]}")
                return False

    except Exception as e:
        logger.error(f"Error sending WhatsApp message to {from_number}: {e}", exc_info=True)
        return False