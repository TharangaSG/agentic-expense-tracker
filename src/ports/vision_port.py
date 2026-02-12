"""
Vision Port Interface

Defines the contract for Vision/Image processing providers using Pydantic models.
"""

from abc import ABC, abstractmethod
from src.domain.models import VisionRequest, VisionResponse, ImageFormat
from typing import List


class VisionPort(ABC):
    """Port interface for Vision/Image processing providers"""
    
    @abstractmethod
    def analyze_image(self, request: VisionRequest) -> VisionResponse:
        """
        Analyze image and extract data.
        
        Args:
            request: VisionRequest with image data and prompt
            
        Returns:
            VisionResponse with extracted text/data
        """
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[ImageFormat]:
        """List of supported image formats"""
        pass
