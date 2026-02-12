"""
Speech-to-Text Port Interface

Defines the contract for STT providers using Pydantic models.
"""

from abc import ABC, abstractmethod
from src.domain.models import TranscriptionRequest, TranscriptionResponse, AudioFormat
from typing import List


class STTPort(ABC):
    """Port interface for Speech-to-Text providers"""
    
    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio to text.
        
        Args:
            request: TranscriptionRequest with audio data and format
            
        Returns:
            TranscriptionResponse with transcribed text
        """
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[AudioFormat]:
        """List of supported audio formats"""
        pass
