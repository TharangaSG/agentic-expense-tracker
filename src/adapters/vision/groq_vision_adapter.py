"""
Groq Vision Adapter

Implements VisionPort interface using Groq for image analysis.
"""

import base64
from openai import OpenAI
from src.ports.vision_port import VisionPort
from src.domain.models import VisionRequest, VisionResponse, ImageFormat
from src.settings import settings
from typing import List


class GroqVisionAdapter(VisionPort):
    """Groq Vision provider implementation"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq Vision adapter.
        
        Args:
            api_key: Groq API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.VISION_MODEL_NAME
        
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
    
    def analyze_image(self, request: VisionRequest) -> VisionResponse:
        """
        Analyze image and extract data using Groq.
        
        Args:
            request: VisionRequest with image data and prompt
            
        Returns:
            VisionResponse with extracted text/data
        """
        # Encode image to base64
        base64_image = base64.b64encode(request.image_data).decode('utf-8')
        
        # Determine MIME type from format
        mime_type_map = {
            ImageFormat.JPEG: "image/jpeg",
            ImageFormat.PNG: "image/png",
            ImageFormat.GIF: "image/gif",
            ImageFormat.WEBP: "image/webp"
        }
        mime_type = mime_type_map.get(request.format, "image/jpeg")
        
        # Make API call
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=self.model,
        )
        
        # Extract analysis result
        extracted_text = response.choices[0].message.content
        
        return VisionResponse(
            extracted_text=extracted_text,
            confidence=None  # Groq doesn't provide confidence scores
        )
    
    def supported_formats(self) -> List[ImageFormat]:
        """List of supported image formats"""
        return [
            ImageFormat.JPEG,
            ImageFormat.PNG,
            ImageFormat.GIF,
            ImageFormat.WEBP
        ]
