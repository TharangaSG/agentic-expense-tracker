"""
Domain models for the expense tracker application.

These Pydantic models represent the core business entities and are independent
of any external services or infrastructure concerns.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Item(BaseModel):
    """Purchase item domain model"""
    item_name: str = Field(description="The name of the purchased item")
    quantity: float = Field(description="The quantity of the item purchased")
    unit_price: float = Field(description="The price of a single unit")
    total_price: float = Field(description="The total price for the item")


class Receipt(BaseModel):
    """Receipt containing multiple items"""
    receipt_id: int = Field(description="Unique identifier for the receipt")
    items: List[Item] = Field(description="List of all items from the receipt")


class Message(BaseModel):
    """Chat message for LLM interactions"""
    role: str = Field(description="Message role: system, user, assistant, tool")
    content: str = Field(description="Message content")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    """LLM chat completion request"""
    messages: List[Message]
    model: str
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """LLM chat completion response"""
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str
    model: str
    usage: Optional[Dict[str, int]] = None


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    M4A = "m4a"


class TranscriptionRequest(BaseModel):
    """Speech-to-text transcription request"""
    audio_data: bytes = Field(description="Audio file bytes")
    format: AudioFormat = Field(description="Audio format")
    language: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Speech-to-text transcription response"""
    text: str = Field(description="Transcribed text")
    language: Optional[str] = None
    confidence: Optional[float] = None


class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"


class VisionRequest(BaseModel):
    """Vision/image analysis request"""
    image_data: bytes = Field(description="Image file bytes")
    format: ImageFormat = Field(description="Image format")
    prompt: str = Field(default="Extract data from this image")


class VisionResponse(BaseModel):
    """Vision/image analysis response"""
    extracted_text: str = Field(description="Extracted text or analysis")
    confidence: Optional[float] = None


class TTSRequest(BaseModel):
    """Text-to-speech synthesis request"""
    text: str = Field(description="Text to synthesize")
    voice_id: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None


class TTSResponse(BaseModel):
    """Text-to-speech synthesis response"""
    audio_data: bytes = Field(description="Generated audio bytes")
    format: AudioFormat = Field(description="Audio format")
