"""
ElevenLabs TTS Adapter

Implements TTSPort interface using ElevenLabs for text-to-speech.
"""

import httpx
from src.ports.tts_port import TTSPort
from src.domain.models import TTSRequest, TTSResponse, AudioFormat
from src.settings import settings
from typing import List, Dict


class ElevenLabsTTSAdapter(TTSPort):
    """ElevenLabs Text-to-Speech provider implementation"""
    
    def __init__(self, api_key: str = None, voice_id: str = None, model: str = None):
        """
        Initialize ElevenLabs TTS adapter.
        
        Args:
            api_key: ElevenLabs API key (defaults to settings)
            voice_id: Default voice ID (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.default_voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
        self.model = model or settings.TTS_MODEL_NAME
    
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech from text using ElevenLabs.
        
        Args:
            request: TTSRequest with text and voice configuration
            
        Returns:
            TTSResponse with generated audio data
        """
        voice_id = request.voice_id or self.default_voice_id
        model = request.model or self.model
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
        
        data = {
            "text": request.text,
            "model_id": model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            },
        }
        
        # Make synchronous request
        with httpx.Client(timeout=25.0) as client:
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            audio_data = response.content
        
        return TTSResponse(
            audio_data=audio_data,
            format=AudioFormat.MP3  # ElevenLabs returns MP3 by default
        )
    
    def list_voices(self) -> List[Dict[str, str]]:
        """List available voices with their IDs and names"""
        url = "https://api.elevenlabs.io/v1/voices"
        
        headers = {
            "xi-api-key": self.api_key,
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            voices_data = response.json()
            
            return [
                {
                    "voice_id": voice["voice_id"],
                    "name": voice["name"]
                }
                for voice in voices_data.get("voices", [])
            ]
