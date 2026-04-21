"""
Chainlit Interface

This is the INTERFACE layer for the Chainlit web UI — it handles:
  - Audio recording & silence detection
  - File uploads (images, audio files)
  - Speech-to-text conversion (via STT port)
  - Text-to-speech responses (via TTS port)
  - Image analysis (via Vision port)
  - Chainlit UI messages and elements

For all AGENT LOGIC (LLM orchestration, saving purchases, querying spending),
it delegates to `process_user_input()` from the Main Agent orchestrator
"""

import io
import wave
import time
import numpy as np
import audioop


# Initialize logging first thing
from src.utils.logging_config import setup_logging
setup_logging()

from src.utils.logging_config import get_logger
from src.settings import settings
import chainlit as cl
from src.config.containers import get_stt_provider, get_tts_provider, get_vision_provider, get_async_database
from src.domain.models import TranscriptionRequest, VisionRequest, TTSRequest, AudioFormat, ImageFormat

# Import the Main Agent orchestrator — this is the ONLY agent entry point
from src.agents.main_agent import process_user_input

logger = get_logger(__name__)

if not settings.ELEVENLABS_API_KEY or not settings.ELEVENLABS_VOICE_ID:
    raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")



@cl.step(type="tool")
async def speech_to_text_chainlit(audio_file: tuple) -> str:
    """
    Transcribes an audio file using STT port for Chainlit.
    audio_file: ("audio.wav", audio_bytes, "audio/wav")
    """
    start_time = time.time()
    file_name, audio_bytes, mime_type = audio_file
    ext = file_name.split(".")[-1]

    logger.debug(f"Transcribing audio: {file_name} ({len(audio_bytes)} bytes, {mime_type})")

    stt_provider = get_stt_provider()
    request = TranscriptionRequest(
        audio_data=audio_bytes,
        format=AudioFormat(ext)
    )
    response = stt_provider.transcribe(request)

    elapsed = time.time() - start_time
    logger.info(f"Speech-to-text completed in {elapsed:.2f}s | Transcription: '{response.text[:100]}{'...' if len(response.text) > 100 else ''}'")

    return response.text


@cl.step(type="tool")
async def analyze_image_chainlit(image_data: bytes) -> str:
    """
    Analyze a receipt image using the Vision port.
    Returns extracted text from the image.
    """
    start_time = time.time()
    logger.debug(f"Analyzing receipt image ({len(image_data)} bytes)")

    vision_provider = get_vision_provider()
    request = VisionRequest(
        image_data=image_data,
        format=ImageFormat.JPEG,
        prompt="Extract receipt data from this image. List all items with their quantities, unit prices, and total prices."
    )
    response = vision_provider.analyze_image(request)

    elapsed = time.time() - start_time
    logger.info(
        f"Image analysis completed in {elapsed:.2f}s | "
        f"Extracted text preview: {response.extracted_text[:100]}{'...' if len(response.extracted_text) > 100 else ''}"
    )

    return response.extracted_text


@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    """Generate speech from text using TTS port."""
    start_time = time.time()
    logger.debug(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")

    tts_provider = get_tts_provider()
    request = TTSRequest(text=text)
    response = tts_provider.synthesize(request)

    elapsed = time.time() - start_time
    logger.debug(f"Text-to-speech completed in {elapsed:.2f}s | Audio size: {len(response.audio_data)} bytes")

    buffer = io.BytesIO(response.audio_data)
    buffer.name = f"output_audio.{response.format.value}"
    buffer.seek(0)

    return buffer.name, buffer.read()



@cl.on_chat_start
async def start():
    """Initialize the chat session and connect the async database."""
    logger.info("Chainlit chat session started")
    cl.user_session.set("message_history", [])

    # Ensure the async PostgreSQL pool is connected
    db = get_async_database()
    if db.pool is None:
        try:
            await db.connect()
            logger.info("✅ PostgreSQL connected successfully on Chainlit chat start.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL on Chainlit chat start: {e}")

    welcome_message = """
    🛒 Purchase Recording & Spending Assistant

    Welcome! I can help you:
    • **Record purchases** — via voice, text, or receipt images
    • **Query spending** — ask "How much did I spend on milk?"

    How to use:
    • Press the record button to dictate your purchase
    • Type your purchase details or spending question
    • Upload a receipt image for automatic processing

    Example inputs:
    • "I bought 2 bottles of milk for $5 each"
    • "How much have I spent on groceries?"

    Ready to help! 💰
    """
    await cl.Message(content=welcome_message).send()
    logger.debug("Welcome message sent to user")



@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    audio_energy = audioop.rms(chunk.data, 2)

    if audio_energy < settings.SILENCE_THRESHOLD:
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= settings.SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)


async def process_audio():
    """Process recorded audio: transcribe → delegate to Main Agent."""
    start_time = time.time()

    if audio_chunks := cl.user_session.get("audio_chunks"):
        concatenated = np.concatenate(list(audio_chunks))

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(concatenated.tobytes())

        wav_buffer.seek(0)
        cl.user_session.set("audio_chunks", [])
    else:
        logger.warning("No audio chunks captured")
        return

    with wave.open(wav_buffer, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    logger.debug(f"Audio recording duration: {duration:.2f}s")

    if duration <= 1.71:
        logger.info("Audio too short, asking user to try again")
        await cl.Message(content="The audio is too short, please try again.").send()
        return

    audio_buffer = wav_buffer.getvalue()
    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")
    logger.info(f"Processing audio recording ({len(audio_buffer)} bytes, {duration:.2f}s)")

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription_start = time.time()
    transcription = await speech_to_text_chainlit(whisper_input)
    transcription_elapsed = time.time() - transcription_start

    await cl.Message(
        author="You",
        type="user_message",
        content=f"Recorded: {transcription}",
        elements=[input_audio_el],
    ).send()

    logger.info(f"Delegating transcribed audio to Main Agent: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
    await cl.Message(content="Processing your request...").send()

    agent_start = time.time()
    result = await process_user_input(transcription)
    agent_elapsed = time.time() - agent_start

    await cl.Message(content=result).send()
    logger.info(f"Audio processing pipeline completed in {time.time() - start_time:.2f}s (transcription: {transcription_elapsed:.2f}s, agent: {agent_elapsed:.2f}s)")

    audio_response = "Your request has been processed successfully!"
    output_name, output_audio = await text_to_speech(audio_response, "audio/wav")
    output_audio_el = cl.Audio(auto_play=True, mime="audio/wav", content=output_audio)

    await cl.Message(
        content="Audio Confirmation:",
        elements=[output_audio_el]
    ).send()



@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages and file uploads."""
    logger.debug(f"Received message: content='{message.content[:100]}{'...' if len(message.content) > 100 else ''}', elements={len(message.elements) if message.elements else 0}")

    # Check if there are any uploaded files
    if message.elements:
        for element in message.elements:
            logger.info(f"Processing uploaded file: type={type(element).__name__}")
            if isinstance(element, cl.Image):
                await _handle_image_upload(element)
            elif isinstance(element, cl.Audio):
                await _handle_audio_upload(element)
        return

    # Handle text input — delegate entirely to Main Agent
    if message.content.strip():
        logger.info(f"Processing text message: '{message.content[:100]}{'...' if len(message.content) > 100 else ''}'")
        await cl.Message(content="Processing your request... ⏳").send()

        # Delegate to Main Agent (handles both saves AND spending queries)
        result = await process_user_input(message.content)
        await cl.Message(content=result).send()
    else:
        logger.debug("Received empty message, sending help tip")
        await cl.Message(
            content="Tip: You can:\n• Press P to record your purchase\n• Type your purchase details or spending questions\n• Upload a receipt image or audio file (.wav, .mp3, etc.)"
        ).send()



async def _handle_image_upload(element: cl.Image):
    """Process an uploaded receipt image: extract text → delegate to Main Agent."""
    start_time = time.time()
    logger.info(f"Handling image upload: name={getattr(element, 'name', 'N/A')}, size={getattr(element, 'size', 'N/A')} bytes")

    await cl.Message(content="Processing receipt image... 📷").send()

    try:
        # Get image data from various sources
        image_content = None
        if hasattr(element, 'content') and element.content:
            image_content = element.content
            logger.debug("Got image content from element.content")
        elif hasattr(element, 'path') and element.path:
            with open(element.path, 'rb') as f:
                image_content = f.read()
            logger.debug(f"Read image from path: {element.path}")
        elif hasattr(element, 'url') and element.url:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(element.url)
                image_content = resp.content
            logger.debug(f"Downloaded image from URL: {element.url}")

        if not image_content:
            logger.error("Could not access image data from any source")
            await cl.Message(content="Error: Could not access image data. Please try uploading again.").send()
            return

        logger.debug(f"Image data acquired ({len(image_content)} bytes), proceeding with analysis")

        # Step 1: Extract text from image (interface responsibility)
        extracted_text = await analyze_image_chainlit(image_content)

        # Step 2: Delegate extracted text to Main Agent (agent responsibility)
        logger.info(f"Delegating image analysis to Main Agent: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'")
        result = await process_user_input(f"[Receipt Image Analysis: {extracted_text}]")

        elapsed = time.time() - start_time
        await cl.Message(content=f"Receipt Processed!\n\n{result}").send()
        logger.info(f"Image upload handling completed in {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing image upload after {elapsed:.2f}s: {e}", exc_info=True)
        await cl.Message(content=f"Error processing image: {str(e)}").send()


async def _handle_audio_upload(element: cl.Audio):
    """Process an uploaded audio file: transcribe → delegate to Main Agent."""
    start_time = time.time()
    logger.info(f"Handling audio upload: name={getattr(element, 'name', 'N/A')}, size={getattr(element, 'size', 'N/A')} bytes")

    await cl.Message(content="Processing audio file... 🎙️").send()

    try:
        audio_content = None
        if hasattr(element, 'content') and element.content:
            audio_content = element.content
            logger.debug("Got audio content from element.content")
        elif hasattr(element, 'path') and element.path:
            with open(element.path, 'rb') as f:
                audio_content = f.read()
            logger.debug(f"Read audio from path: {element.path}")

        if not audio_content:
            logger.error("Could not access audio data from any source")
            await cl.Message(content="Error: Could not access audio data. Please try uploading again.").send()
            return

        whisper_input = ("audio.wav", audio_content, "audio/wav")
        transcription = await speech_to_text_chainlit(whisper_input)

        if transcription:
            logger.info(f"Audio transcribed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
            await cl.Message(
                author="You",
                type="user_message",
                content=f"Transcribed: {transcription}",
                elements=[element],
            ).send()

            logger.info(f"Delegating transcribed audio to Main Agent")
            result = await process_user_input(transcription)
            await cl.Message(content=result).send()
        else:
            logger.warning("Transcription returned empty result")
            await cl.Message(content="Failed to transcribe audio. Please try again.").send()

        elapsed = time.time() - start_time
        logger.info(f"Audio upload handling completed in {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing audio upload after {elapsed:.2f}s: {e}", exc_info=True)
        await cl.Message(content=f"Error processing audio: {str(e)}").send()