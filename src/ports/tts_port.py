"""
Text-to-Speech Port Interface

Defines the contract for TTS providers using Pydantic models.
"""

from abc import ABC, abstractmethod
from src.domain.models import TTSRequest, TTSResponse
from typing import List, Dict


class TTSPort(ABC):
    """Port interface for Text-to-Speech providers"""
    
    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech from text.
        
        Args:
            request: TTSRequest with text and voice configuration
            
        Returns:
            TTSResponse with generated audio data
        """
        pass
    
    @abstractmethod
    def list_voices(self) -> List[Dict[str, str]]:
        """List available voices with their IDs and names"""
        pass
