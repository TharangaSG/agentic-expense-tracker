"""
Groq Vision Adapter

Implements VisionPort interface using Groq for image analysis.
"""

import base64
import logging
import time
from typing import List

from openai import OpenAI
from src.utils.logging_config import get_logger
from src.ports.vision_port import VisionPort
from src.domain.models import VisionRequest, VisionResponse, ImageFormat
from src.settings import settings

logger = get_logger(__name__)


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

        logger.info(f"Groq Vision adapter initialized | Model: {self.model}")
    
    def analyze_image(self, request: VisionRequest) -> VisionResponse:
        """
        Analyze image and extract data using Groq.

        Args:
            request: VisionRequest with image data and prompt

        Returns:
            VisionResponse with extracted text/data
        """
        start_time = time.time()

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
        logger.debug(
            f"Sending image to Groq Vision for analysis | "
            f"Format: {request.format.value} | Size: {len(request.image_data)} bytes | "
            f"Prompt: '{request.prompt[:50]}{'...' if len(request.prompt) > 50 else ''}'"
        )

        try:
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

            elapsed = time.time() - start_time
            logger.info(
                f"Vision analysis completed in {elapsed:.2f}s | "
                f"Extracted text length: {len(extracted_text) if extracted_text else 0} chars | "
                f"Preview: '{extracted_text[:100] if extracted_text else 'empty'}...'"
            )

            return VisionResponse(
                extracted_text=extracted_text,
                confidence=None  # Groq doesn't provide confidence scores
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Vision analysis failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise
    
    def supported_formats(self) -> List[ImageFormat]:
        """List of supported image formats"""
        return [
            ImageFormat.JPEG,
            ImageFormat.PNG,
            ImageFormat.GIF,
            ImageFormat.WEBP
        ]
