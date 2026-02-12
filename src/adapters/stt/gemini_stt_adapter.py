"""
Gemini STT Adapter

Implements STTPort interface using Google Gemini for speech-to-text.
"""

import base64
from openai import OpenAI
from src.ports.stt_port import STTPort
from src.domain.models import TranscriptionRequest, TranscriptionResponse, AudioFormat
from src.settings import settings
from typing import List


class GeminiSTTAdapter(STTPort):
    """Gemini Speech-to-Text provider implementation"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini STT adapter.
        
        Args:
            api_key: Gemini API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.STT_MODEL_NAME
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio to text using Gemini.
        
        Args:
            request: TranscriptionRequest with audio data and format
            
        Returns:
            TranscriptionResponse with transcribed text
        """
        # Encode audio data to base64
        base64_audio = base64.b64encode(request.audio_data).decode("utf-8")
        
        # Prepare message content
        content = [
            {"type": "text", "text": "Transcribe this audio"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_audio,
                    "format": request.format.value
                }
            }
        ]
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        
        # Extract transcription
        transcribed_text = response.choices[0].message.content
        
        return TranscriptionResponse(
            text=transcribed_text,
            language=request.language,
            confidence=None  # Gemini doesn't provide confidence scores
        )
    
    def supported_formats(self) -> List[AudioFormat]:
        """List of supported audio formats"""
        return [
            AudioFormat.WAV,
            AudioFormat.MP3,
            AudioFormat.OGG,
            AudioFormat.M4A
        ]
