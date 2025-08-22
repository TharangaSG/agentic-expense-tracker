import base64
import tempfile
import os
from typing import Union
from src.utils.clients import gemini_client
from src.settings import settings


def transcribe_audio(file_input: Union[str, bytes]) -> str:
    """
    Transcribes an audio file using Gemini API and returns text.
    
    Args:
        file_input: Either a file path (str) or audio bytes (bytes)
    
    Returns:
        str: Transcribed text
    """
    if isinstance(file_input, str):
        # File path provided
        with open(file_input, "rb") as audio_file:
            audio_data = audio_file.read()
        file_extension = file_input.split(".")[-1]
    else:
        # Bytes provided (from WhatsApp)
        audio_data = file_input
        file_extension = "ogg"  # WhatsApp typically sends OGG format
    
    # Encode audio data to base64
    base64_audio = base64.b64encode(audio_data).decode("utf-8")

    client = gemini_client
    # Send request
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": file_extension
                        }
                    }
                ],
            }
        ],
    )

    return response.choices[0].message.content


import os
import google.genai as genai
from google.genai import types
from src.settings import settings

async def transcribe_ogg_file(file_path: str) -> str:
    """
    Transcribes an OGG audio file using the Gemini Flash 2.5 model.

    Args:
        file_path: The full path to the .ogg file.

    Returns:
        The transcribed text from the audio.
    """
    if not os.path.exists(file_path):
        return f"Error: The file at '{file_path}' was not found."

    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    response = client.models.generate_content(
        model=settings.STT_MODEL_NAME,
        contents=[
            types.Part.from_text(text='Transcribe this audio clip'),
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type='audio/ogg',
            )
        ]
    )

    return response.text


if __name__ == "__main__":
    ogg_file = 'src/tools/buy.ogg'
    transcription = transcribe_ogg_file(ogg_file)
    print("Transcription:", transcription)

    # transcription = transcribe_audio("src/tools/buy.mp3")
    # print("Transcription:", transcription)