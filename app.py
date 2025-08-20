import io
import wave
import httpx
import numpy as np
import audioop
import base64
import json
import tempfile
import os
from typing import List
from src.settings import settings
import chainlit as cl
from src.utils.clients import gemini_client
from src.tools.save_data_to_db import save_data_to_db, ReceiptData
from src.tools.read_image import extract_data_from_image
from src.tools.speech_to_text import transcribe_audio


if not settings.ELEVENLABS_API_KEY or not settings.ELEVENLABS_VOICE_ID:
    raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")


tools = [
    {
        "type": "function",
        "function": {
            "name": "save_data_to_db",
            "description": "Takes structured JSON data of receipt items and saves it to the database.",
            "parameters": ReceiptData.model_json_schema(),
        },
    },
]

available_functions = {
    "save_data_to_db": save_data_to_db,
}


@cl.step(type="tool")
async def speech_to_text_chainlit(audio_file: tuple) -> str:
    """
    Transcribes an audio file using Gemini API for Chainlit.
    audio_file: ("audio.wav", audio_bytes, "audio/wav")
    """
    file_name, audio_bytes, mime_type = audio_file
    ext = file_name.split(".")[-1]

    # Convert to base64
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    response = gemini_client.chat.completions.create(
        model=settings.MAIN_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": ext,
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


@cl.step(type="tool")
async def process_purchase_data(transcription: str) -> str:
    """
    Process transcribed purchase data and save to database.
    """
    # Create a system message for processing purchase data
    messages = [
        {
            "role": "system", 
            "content": """You are a helpful assistant that processes purchase information. 
            When given purchase descriptions in natural language, extract the item details and structure them appropriately. 
            For text purchases, try to infer reasonable unit prices from the total if not explicitly stated.
            Always generate a unique receipt_id (you can use a timestamp-based approach or increment from 1).
            Make sure to call the save_data_to_db function with the extracted data."""
        },
        {
            "role": "user", 
            "content": f"I made a purchase: {transcription}. Please parse this and save to my database."
        }
    ]
    
    try:
        completion = gemini_client.chat.completions.create(
            model=settings.MAIN_MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Calling function: {function_name}")
                print(f"Arguments: {function_args}")
                
                function_response = function_to_call(**function_args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
            
            # Get final response
            final_completion = gemini_client.chat.completions.create(
                model=settings.MAIN_MODEL_NAME,
                messages=messages,
            )
            return final_completion.choices[0].message.content
        else:
            return response_message.content
            
    except Exception as e:
        return f"Error processing purchase data: {str(e)}"


@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    """Generate speech from text using ElevenLabs API."""
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.ELEVENLABS_VOICE_ID}"

    headers = {
        "Accept": mime_type,
        "Content-Type": "application/json",
        "xi-api-key": settings.ELEVENLABS_API_KEY,
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()

        buffer = io.BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"

        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)

        buffer.seek(0)
        return buffer.name, buffer.read()


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    welcome_message = """
    Purchase Recording Assistant
    
    Welcome! I can help you record and save your purchases to the database.
    
    How to use:
    • Press record button to record your purchase details via voice
    • Or type your purchase information directly
    • Upload receipt images for automatic processing
    
    Example voice input: 
    "I bought 2 bottles of milk for $5 each and 3 apples for $1 each"
    
    Ready to start recording your purchases!
    """
    await cl.Message(content=welcome_message).send()


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
    """Process recorded audio for purchase data extraction."""
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

    with wave.open(wav_buffer, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    if duration <= 1.71:
        await cl.Message(content="The audio is too short, please try again.").send()
        return

    audio_buffer = wav_buffer.getvalue()
    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    # Transcribe the audio
    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription = await speech_to_text_chainlit(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=f"Recorded: {transcription}",
        elements=[input_audio_el],
    ).send()

    # Process the purchase data
    processing_msg = await cl.Message(content="Processing your purchase data...").send()
    
    result = await process_purchase_data(transcription)
    
    # Send a new message with the result
    await cl.Message(content=f"Purchase Saved!\n\n{result}").send()
    
    # Generate audio response
    audio_response = "Your purchase has been successfully recorded and saved to the database!"
    output_name, output_audio = await text_to_speech(audio_response, "audio/wav")
    output_audio_el = cl.Audio(auto_play=True, mime="audio/wav", content=output_audio)
    
    await cl.Message(
        content="Audio Confirmation:", 
        elements=[output_audio_el]
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages and file uploads."""
    
    # Check if there are any uploaded files
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.Image):
                # Process uploaded image
                processing_msg = await cl.Message(content="Processing receipt image...").send()
                
                try:
                    #check different ways to access the image data
                    image_content = None
                    if hasattr(element, 'content') and element.content:
                        image_content = element.content
                    elif hasattr(element, 'path') and element.path:
                        # If content is None, try reading from path
                        with open(element.path, 'rb') as f:
                            image_content = f.read()
                    elif hasattr(element, 'url') and element.url:
                        # If it's a URL, process directly
                        try:
                            extracted_text = extract_data_from_image(element.url)
                            result = await process_purchase_data(extracted_text)
                            await cl.Message(content=f"Receipt Processed!\n\n{result}").send()
                            return
                        except Exception as e:
                            await cl.Message(content=f"Error processing image URL: {str(e)}").send()
                            return
                    
                    if not image_content:
                        await cl.Message(content="Error: Could not access image data. Please try uploading the image again.").send()
                        return
                    
                    # Save the image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(image_content)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        extracted_text = extract_data_from_image(tmp_file_path)
                        
                        result = await process_purchase_data(extracted_text)
                        await cl.Message(content=f"Receipt Processed!\n\n{result}").send()
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                    
                except Exception as e:
                    await cl.Message(content=f"Error processing image: {str(e)}").send()
                
            elif isinstance(element, cl.Audio):
                processing_msg = await cl.Message(content="Processing audio file...").send()
                
                try:
                    audio_content = None
                    if hasattr(element, 'content') and element.content:
                        audio_content = element.content
                    elif hasattr(element, 'path') and element.path:
                        with open(element.path, 'rb') as f:
                            audio_content = f.read()
                            
                    if not audio_content:
                        await cl.Message(content="Error: Could not access audio data. Please try uploading again.").send()
                        return
                        
                    # Save audio temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_content)
                        tmp_file_path = tmp_file.name
                        
                    try:
                        # Transcribe the audio using the file path
                        transcription = await transcribe_audio(tmp_file_path)
                        
                        if transcription:
                            await cl.Message(
                                author="You",
                                type="user_message",
                                content=f"Transcribed: {transcription}",
                                elements=[element],
                            ).send()
                            
                            result = await process_purchase_data(transcription)
                            await cl.Message(content=f"Purchase Saved!\n\n{result}").send()
                        else:
                            await cl.Message(content="Failed to transcribe audio. Please try again.").send()
                            
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                            
                except Exception as e:
                    await cl.Message(content=f"Error processing audio: {str(e)}").send()
        
        return
    
    # Handle text input as purchase description
    if message.content.strip():
        processing_msg = await cl.Message(content="Processing your purchase description...").send()
        result = await process_purchase_data(message.content)
        await cl.Message(content=f"Purchase Saved!\n\n{result}").send()
    else:
        await cl.Message(
            content="Tip: You can:\n• Press P to record your purchase\n• Type your purchase details\n• Upload a receipt image or audio file (.wav, .mp3, etc.)"
        ).send()