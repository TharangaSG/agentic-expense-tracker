"""
LLM Port Interface

Defines the contract for LLM providers using Pydantic models.
"""

from abc import ABC, abstractmethod
from src.domain.models import ChatRequest, ChatResponse


class LLMPort(ABC):
    """Port interface for LLM providers"""
    
    @abstractmethod
    def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat completion request and get response.
        
        Args:
            request: ChatRequest with messages and configuration
            
        Returns:
            ChatResponse with generated content
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current model name"""
        pass
