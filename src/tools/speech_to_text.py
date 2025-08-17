import base64
import os
from openai import OpenAI
from src.utils.clients import gemini_client



def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Gemini API and returns text.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        str: Transcribed text
    """
    # Read and encode audio file
    with open(file_path, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

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
                            "format": file_path.split(".")[-1] 
                        }
                    }
                ],
            }
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    transcription = transcribe_audio("Ed Sheeran - Thinking out Loud (Lyrics) - 7clouds.mp3")
    print("Transcription:", transcription)
