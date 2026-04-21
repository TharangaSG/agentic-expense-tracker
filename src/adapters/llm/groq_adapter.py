"""
Groq LLM Adapter

Implements LLMPort interface using Groq via OpenAI SDK.
"""

import logging
import time

from openai import OpenAI
from src.utils.logging_config import get_logger
from src.ports.llm_port import LLMPort
from src.domain.models import ChatRequest, ChatResponse
from src.settings import settings

logger = get_logger(__name__)


class GroqLLMAdapter(LLMPort):
    """Groq LLM provider implementation"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq LLM adapter.

        Args:
            api_key: Groq API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.MAIN_MODEL_NAME

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

        logger.info(f"Groq LLM adapter initialized | Model: {self.model}")
    
    def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Send chat completion request to Groq.

        Args:
            request: ChatRequest with messages and configuration

        Returns:
            ChatResponse with generated content
        """
        start_time = time.time()

        # Convert Pydantic messages to dict format for OpenAI SDK
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        # Prepare API call parameters
        api_params = {
            "model": request.model or self.model,
            "messages": messages,
        }

        # Add optional parameters
        if request.tools:
            api_params["tools"] = request.tools
            api_params["parallel_tool_calls"] = False
        if request.tool_choice:
            api_params["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            api_params["temperature"] = request.temperature
        if request.max_tokens:
            api_params["max_tokens"] = request.max_tokens

        # Make API call
        logger.debug(f"Sending chat completion request to Groq | Model: {api_params['model']} | Messages: {len(messages)}")

        try:
            response = self.client.chat.completions.create(**api_params)

            elapsed = time.time() - start_time

            # Extract response data
            choice = response.choices[0]
            message = choice.message

            has_tool_calls = bool(message.tool_calls)
            content_length = len(message.content) if message.content else 0

            logger.info(
                f"Groq chat completion completed in {elapsed:.2f}s | "
                f"Finish reason: {choice.finish_reason} | "
                f"Content length: {content_length} chars | "
                f"Tool calls: {len(message.tool_calls) if has_tool_calls else 0}"
            )

            if response.usage:
                logger.debug(
                    f"Token usage | Prompt: {response.usage.prompt_tokens} | "
                    f"Completion: {response.usage.completion_tokens} | "
                    f"Total: {response.usage.total_tokens}"
                )

            # Convert to ChatResponse
            return ChatResponse(
                content=message.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in (message.tool_calls or [])
                ] if message.tool_calls else None,
                finish_reason=choice.finish_reason,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Groq chat completion failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses"""
        return True
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model
