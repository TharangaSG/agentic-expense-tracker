"""
Telegram Interface Handler

This is the INTERFACE layer — it handles:
  - Telegram webhook updates (POST)
  - Multimodal input conversion (voice/audio -> text, photo -> text)
  - Sending responses back via the Telegram Bot API

For all AGENT LOGIC (LLM orchestration, saving purchases, querying spending),
it delegates to `process_user_input()` from the Main Agent orchestrator.
"""

import mimetypes
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request, Response

from src.agents.main_agent import process_user_input
from src.config.containers import get_stt_provider, get_vision_provider
from src.domain.models import AudioFormat, ImageFormat, TranscriptionRequest, VisionRequest
from src.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

telegram_router = APIRouter()

TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_WEBHOOK_SECRET_TOKEN = settings.TELEGRAM_WEBHOOK_SECRET_TOKEN
TELEGRAM_API_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_BASE_URL = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"
SUPPORTED_IMAGE_MIME_PREFIX = "image/"


@telegram_router.post("/telegram/webhook")
async def telegram_handler(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> Response:
    """Handle incoming Telegram updates."""
    start_time = time.time()

    if TELEGRAM_WEBHOOK_SECRET_TOKEN:
        if x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET_TOKEN:
            logger.warning("Telegram webhook secret token verification failed")
            return Response(content="Unauthorized", status_code=401)

    try:
        data = await request.json()
        logger.debug(f"Received Telegram webhook payload: {str(data)[:200]}...")

        message = data.get("message") or data.get("edited_message")
        if not message:
            logger.debug("Ignoring Telegram update without a message payload")
            return Response(content="Ignored", status_code=200)

        chat = message.get("chat", {})
        chat_id = chat.get("id")
        if chat_id is None:
            logger.warning("Telegram update missing chat id")
            return Response(content="Missing chat id", status_code=400)

        await send_chat_action(chat_id, "typing")

        content = await extract_message_content(message)
        if not content:
            logger.warning("Telegram message processing resulted in empty content")
            await send_response(
                chat_id,
                "Sorry, I couldn't understand your message. Please try again.",
            )
            return Response(content="Empty content", status_code=200)

        logger.info(
            f"Delegating Telegram message to Main Agent: "
            f"'{content[:100]}{'...' if len(content) > 100 else ''}'"
        )

        try:
            response_message = await process_user_input(
                content,
                session_id=f"telegram-chat-{chat_id}",
                user_id=str(message.get("from", {}).get("id", chat_id)),
                source="telegram",
            )
            elapsed = time.time() - start_time
            logger.info(
                f"Main Agent response received after {elapsed:.2f}s | "
                f"Response preview: {response_message[:150]}..."
            )

            success = await send_response(chat_id, response_message)
            if not success:
                logger.error(f"Failed to send Telegram response to chat {chat_id}")
                return Response(content="Failed to send message", status_code=500)

            logger.info(f"Successfully sent response to Telegram chat {chat_id}")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error from Main Agent after {elapsed:.2f}s: {e}", exc_info=True)
            await send_response(
                chat_id,
                "Sorry, I encountered an error processing your request. Please try again.",
            )

        return Response(content="Message processed", status_code=200)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing Telegram message after {elapsed:.2f}s: {e}", exc_info=True)
        return Response(content="Internal server error", status_code=500)


async def extract_message_content(message: dict[str, Any]) -> str:
    """Normalize Telegram text, photo, and voice/audio messages into agent input."""
    if message.get("text"):
        content = message["text"].strip()
        logger.debug(f"Telegram text content: '{content[:100]}{'...' if len(content) > 100 else ''}'")
        return content

    if message.get("voice"):
        logger.info("Received Telegram voice message")
        return await process_audio_file(message["voice"], caption=message.get("caption", ""))

    if message.get("audio"):
        logger.info("Received Telegram audio message")
        return await process_audio_file(message["audio"], caption=message.get("caption", ""))

    if message.get("photo"):
        logger.info("Received Telegram photo message")
        largest_photo = message["photo"][-1]
        return await process_image_file(largest_photo["file_id"], caption=message.get("caption", ""))

    if message.get("document"):
        document = message["document"]
        mime_type = document.get("mime_type", "")
        if mime_type.startswith(SUPPORTED_IMAGE_MIME_PREFIX):
            logger.info("Received Telegram image document")
            return await process_image_file(document["file_id"], caption=message.get("caption", ""))

        logger.warning(f"Unsupported Telegram document MIME type: {mime_type}")
        return ""

    logger.warning(f"Unsupported Telegram message keys: {list(message.keys())}")
    return ""


async def process_audio_file(audio_payload: dict[str, Any], caption: str = "") -> str:
    """Download and transcribe Telegram voice/audio messages."""
    start_time = time.time()

    try:
        file_id = audio_payload["file_id"]
        file_path = await get_file_path(file_id)
        audio_bytes = await download_file(file_path)
        audio_format = infer_audio_format(file_path)

        stt_provider = get_stt_provider()
        request = TranscriptionRequest(audio_data=audio_bytes, format=audio_format)

        logger.debug(f"Starting Telegram speech-to-text transcription | Format: {audio_format.value}")
        response = stt_provider.transcribe(request)

        transcript = response.text.strip()
        elapsed = time.time() - start_time
        logger.info(
            f"Telegram audio transcribed in {elapsed:.2f}s | "
            f"Transcription: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'"
        )

        if caption.strip():
            return f"{caption.strip()}\n\n[Voice Transcript: {transcript}]"
        return transcript

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing Telegram audio after {elapsed:.2f}s: {e}", exc_info=True)
        return "Sorry, I couldn't process your audio message. Please try again or send a text message."


async def process_image_file(file_id: str, caption: str = "") -> str:
    """Download and analyze Telegram images."""
    start_time = time.time()

    try:
        file_path = await get_file_path(file_id)
        image_bytes = await download_file(file_path)
        image_format = infer_image_format(file_path)

        vision_provider = get_vision_provider()
        request = VisionRequest(
            image_data=image_bytes,
            format=image_format,
            prompt="Extract receipt data from this image. List all items with their quantities, unit prices, and total prices.",
        )

        logger.debug(f"Starting Telegram image analysis | Format: {image_format.value}")
        response = vision_provider.analyze_image(request)
        image_analysis = response.extracted_text

        elapsed = time.time() - start_time
        logger.info(
            f"Telegram image analyzed in {elapsed:.2f}s | "
            f"Extracted text preview: {image_analysis[:100]}{'...' if len(image_analysis) > 100 else ''}"
        )

        if caption.strip():
            return f"{caption.strip()}\n\n[Image Analysis: {image_analysis}]"
        return f"[Image Analysis: {image_analysis}]"

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing Telegram image after {elapsed:.2f}s: {e}", exc_info=True)
        return "Sorry, I couldn't process your image. Please try again or send a text message."


async def get_file_path(file_id: str) -> str:
    """Resolve a Telegram file id into a downloadable file path."""
    response = await telegram_api_request("getFile", {"file_id": file_id})
    file_path = response.get("result", {}).get("file_path")
    if not file_path:
        raise ValueError("Telegram file path missing from getFile response")
    return file_path


async def download_file(file_path: str) -> bytes:
    """Download a Telegram file using the bot token."""
    file_url = f"{TELEGRAM_FILE_BASE_URL}/{file_path}"
    logger.debug(f"Downloading Telegram media from: {file_url}")

    async with httpx.AsyncClient() as client:
        response = await client.get(file_url)
        response.raise_for_status()
        logger.debug(f"Telegram media downloaded successfully: {len(response.content)} bytes")
        return response.content


async def send_chat_action(chat_id: int, action: str) -> bool:
    """Show Telegram chat activity while processing a request."""
    try:
        await telegram_api_request("sendChatAction", {"chat_id": chat_id, "action": action})
        return True
    except Exception as e:
        logger.warning(f"Failed to send Telegram chat action '{action}' to {chat_id}: {e}")
        return False


async def send_response(chat_id: int, response_text: str) -> bool:
    """Send a Telegram text response, splitting long messages if needed."""
    try:
        for chunk in split_text(response_text):
            await telegram_api_request("sendMessage", {"chat_id": chat_id, "text": chunk})
        return True
    except Exception as e:
        logger.error(f"Error sending Telegram message to {chat_id}: {e}", exc_info=True)
        return False


async def telegram_api_request(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the Telegram Bot API and raise on unsuccessful responses."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not configured")

    url = f"{TELEGRAM_API_BASE_URL}/{method}"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    if not data.get("ok"):
        raise ValueError(f"Telegram API error on {method}: {data}")

    return data


def split_text(text: str, max_length: int = 4096) -> list[str]:
    """Split long Telegram messages into chunks that preserve line boundaries when possible."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, max_length)
        if split_at <= 0:
            split_at = max_length

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return [chunk for chunk in chunks if chunk]


def infer_audio_format(file_path: str) -> AudioFormat:
    """Infer the audio format from the Telegram file path."""
    extension = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".ogg": AudioFormat.OGG,
        ".oga": AudioFormat.OGG,
        ".mp3": AudioFormat.MP3,
        ".wav": AudioFormat.WAV,
        ".m4a": AudioFormat.M4A,
    }
    return mapping.get(extension, AudioFormat.OGG)


def infer_image_format(file_path: str) -> ImageFormat:
    """Infer the image format from the Telegram file path or MIME type."""
    extension = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".jpg": ImageFormat.JPEG,
        ".jpeg": ImageFormat.JPEG,
        ".png": ImageFormat.PNG,
        ".gif": ImageFormat.GIF,
        ".webp": ImageFormat.WEBP,
    }
    if extension in mapping:
        return mapping[extension]

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == "image/png":
        return ImageFormat.PNG
    if mime_type == "image/gif":
        return ImageFormat.GIF
    if mime_type == "image/webp":
        return ImageFormat.WEBP
    return ImageFormat.JPEG
