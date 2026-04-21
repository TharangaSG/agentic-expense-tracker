"""
Main Agent - Orchestrator

The Main Agent acts as a router and communicator:
- If the user provides a receipt/purchase → extracts and saves it (existing behavior)
- If the user asks a question about past spending → delegates to the Database Analyst Agent
- Formats all responses in a friendly, conversational tone
"""

import json
import time
from src.utils.logging_config import get_logger
from src.config.containers import get_llm_provider, get_async_database
from src.domain.models import Receipt, Message, ChatRequest
from src.agents.database_analyst_agent import ask_analyst

logger = get_logger(__name__)


def parse_text_purchase(text: str) -> str:
    """
    Parse natural language purchase description into structured format.
    
    Args:
        text: Natural language description of purchase
        
    Returns:
        Formatted text for the AI model to process
    """
    return f"Parse this purchase description and extract item details: {text}"


# ─── Tool definitions for the Main Agent ─────────────────────────────────────
save_tool = {
    "type": "function",
    "function": {
        "name": "save_data_to_db",
        "description": "Takes structured JSON data of receipt items and saves it to the database.",
        "parameters": Receipt.model_json_schema(),
    },
}

ask_analyst_tool = {
    "type": "function",
    "function": {
        "name": "ask_database_analyst",
        "description": (
            "Ask the Database Analyst Agent a question about the user's past spending "
            "or purchase history. The analyst will search the database and return "
            "factual findings. Use this tool when the user asks questions like "
            "'How much did I spend on X?', 'What did I buy last week?', "
            "'Show me my top expenses', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question about their spending, rephrased if needed for clarity.",
                },
            },
            "required": ["question"],
        },
    },
}

tools = [save_tool, ask_analyst_tool]

# ─── System prompt for the Main Agent (Orchestrator) ────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """You are a friendly financial assistant that helps users track their purchases and answer spending questions.

You have TWO main responsibilities:

## 1. Saving Purchases
When a user describes a purchase or provides receipt data:
- Extract item details: item_name, quantity, unit_price, total_price
- Call the `save_data_to_db` function with the structured data
- Always generate a unique receipt_id (increment from 1 or use a reasonable number)

## 2. Answering Spending Questions
When a user asks about their spending history (e.g., "How much did I spend on sugar?", 
"What were my expenses last week?", "Show me purchases of milk"):
- Use the `ask_database_analyst` tool to delegate the question to your colleague, 
  the Database Analyst Agent
- The analyst will search the database and return raw factual data
- YOUR job is to take that raw data and format it into a friendly, conversational response

## How to Decide
- If the input contains purchase data, items, prices, or receipts → SAVE it
- If the input is a question about past spending or history → ASK the analyst
- If the user just greets you → respond politely and explain what you can do

## Response Style
- Be warm, conversational, and helpful
- Use emojis sparingly (🛒, 💰, ✅) to make responses engaging
- Format monetary values as $X.XX
- When presenting analyst results, add context and friendly commentary
"""


async def process_user_input(user_input: str) -> str:
    """
    Process user input and return formatted response.
    
    Acts as the orchestrator: routes to save or analyst.
    
    Args:
        user_input: User's input (text, image analysis, or audio transcription)
        
    Returns:
        Formatted response message
    """
    start_time = time.time()
    logger.info(f"Processing user input: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")

    try:
        # Get providers
        llm_provider = get_llm_provider()
        db = get_async_database()

        logger.debug(f"LLM provider: {llm_provider.get_model_name()}")

        # Ensure DB connection is active
        if db.pool is None:
            logger.info("Database pool not initialized, connecting...")
            await db.connect()
        
        # Create chat request
        chat_request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content=ORCHESTRATOR_SYSTEM_PROMPT,
                ),
                Message(
                    role="user",
                    content=user_input,
                ),
            ],
            model=llm_provider.get_model_name(),
            tools=tools,
            tool_choice="auto",
        )

        logger.debug("Sending request to LLM...")
        response = llm_provider.chat_completion(chat_request)

        # Log LLM response metadata
        if response.usage:
            logger.info(
                f"LLM response received | Tokens: {response.usage.get('total_tokens', 'N/A')} "
                f"(prompt: {response.usage.get('prompt_tokens', 'N/A')}, "
                f"completion: {response.usage.get('completion_tokens', 'N/A')})"
            )

        # Check if the model wants to call a function
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                logger.info(f"LLM decided to call tool: {function_name}")

                if function_name == "save_data_to_db":
                    result = await _handle_save_receipt(function_args)
                    elapsed = time.time() - start_time
                    logger.info(f"Save receipt completed in {elapsed:.2f}s")
                    return result

                elif function_name == "ask_database_analyst":
                    result = await _handle_ask_analyst(
                        function_args, tool_call, chat_request
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"Ask analyst completed in {elapsed:.2f}s")
                    return result

        elapsed = time.time() - start_time
        logger.info(f"Direct response (no tool calls) completed in {elapsed:.2f}s")
        return response.content

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing user input after {elapsed:.2f}s: {e}", exc_info=True)
        return f"Error processing your message: {str(e)}"


async def _handle_save_receipt(function_args: dict) -> str:
    """Handle the save_data_to_db tool call using the async PostgreSQL adapter."""
    db = get_async_database()

    try:
        receipt = Receipt(**function_args)
        logger.info(
            f"Saving receipt {receipt.receipt_id} with {len(receipt.items)} items: "
            f"{', '.join([item.item_name for item in receipt.items])}"
        )

        success = await db.save_receipt(receipt)
        
        if success:
            output = "✅ Purchase Successfully Saved!\n\n"
            output += f"Items Saved: {len(receipt.items)}\n\n"
            output += "Saved Items:\n"
            
            total_amount = 0
            for i, item in enumerate(receipt.items, 1):
                total_amount += item.total_price
                
                output += f"{i}. {item.item_name}\n"
                output += f"   - Quantity: {item.quantity}\n"
                output += f"   - Unit Price: ${item.unit_price:.2f}\n"
                output += f"   - Total: ${item.total_price:.2f}\n\n"
            
            output += f"💰 Grand Total: ${total_amount:.2f}"

            logger.info(f"Receipt {receipt.receipt_id} saved successfully | Total: ${total_amount:.2f}")
            return output
        else:
            logger.error(f"Failed to save receipt {receipt.receipt_id}")
            return "❌ Error saving purchase to database"
            
    except Exception as e:
        logger.error(f"Error saving receipt: {e}", exc_info=True)
        return f"Error saving purchase: {str(e)}"


async def _handle_ask_analyst(
    function_args: dict,
    tool_call: dict,
    chat_request: ChatRequest,
) -> str:
    """
    Handle the ask_database_analyst tool call.
    
    Delegates to the Database Analyst Agent, then passes the raw results
    back to the Main Agent to format a friendly response.
    """

    llm_provider = get_llm_provider()
    question = function_args.get("question", "")

    logger.info(f"[Main Agent] Delegating to Analyst: '{question}'")
    analyst_start = time.time()

    # Call the Database Analyst Agent
    analyst_response = await ask_analyst(question)

    analyst_elapsed = time.time() - analyst_start
    logger.info(
        f"[Main Agent] Analyst response received after {analyst_elapsed:.2f}s | "
        f"Response preview: {analyst_response[:200]}..."
    )

    # Feed the analyst's response back to the Main Agent 
    chat_request.messages.append(
        Message(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        )
    )
    chat_request.messages.append(
        Message(
            tool_call_id=tool_call["id"],
            role="tool",
            name="ask_database_analyst",
            content=f"Database Analyst findings:\n{analyst_response}",
        )
    )
    
    # Remove tools so the Main Agent just formats the response
    final_request = ChatRequest(
        messages=chat_request.messages,
        model=llm_provider.get_model_name(),
    )

    logger.debug("Requesting LLM to format analyst response...")
    format_start = time.time()

    final_response = llm_provider.chat_completion(final_request)

    format_elapsed = time.time() - format_start
    if final_response.usage:
        logger.info(
            f"Response formatting completed in {format_elapsed:.2f}s | "
            f"Tokens: {final_response.usage.get('total_tokens', 'N/A')}"
        )

    return final_response.content or analyst_response