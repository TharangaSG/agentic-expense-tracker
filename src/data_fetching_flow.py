import json
from src.config.containers import get_llm_provider, get_database
from src.domain.models import Message, ChatRequest
from pydantic import BaseModel, Field


class SpendingQuery(BaseModel):
    item_name: str = Field(description="The name of the item to query for spending, e.g., 'soda', 'milk'.")


def run_conversation(user_prompt: str):
    """
    Runs the main conversation loop with the AI model using port-adapter architecture.
    """
    # Get providers from factory
    llm_provider = get_llm_provider()
    database = get_database()
    
    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_spending_for_item_last_week",
                "description": "Get the total amount of money spent on a specific item within the last 7 days.",
                "parameters": SpendingQuery.model_json_schema()
            }
        }
    ]
    
    print(f"\nUser: {user_prompt}")
    
    # Create chat request using Pydantic models
    chat_request = ChatRequest(
        messages=[
            Message(role="user", content=user_prompt)
        ],
        model=llm_provider.get_model_name(),
        tools=tools,
        tool_choice="auto"
    )
    
    # Send the prompt and tools to the model
    response = llm_provider.chat_completion(chat_request)
    
    # Check if the model wants to call the tool
    if response.tool_calls:
        print("\nModel wants to call a tool...")
        
        # Add assistant message to conversation
        chat_request.messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            )
        )
        
        for tool_call in response.tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            print(f"Calling function '{function_name}' with arguments: {function_args}")
            
            # Execute the tool using database port
            if function_name == "get_spending_for_item_last_week":
                item_name = function_args.get("item_name")
                total_spent = database.query_spending(item_name, days=7)
                function_response = f"Tool execution result: Total all-time spending on {item_name} is ${total_spent:.2f}"
            else:
                function_response = "Unknown function"
            
            print(f"Tool response: {function_response}")
            
            # Add tool response to conversation
            chat_request.messages.append(
                Message(
                    tool_call_id=tool_call["id"],
                    role="tool",
                    name=function_name,
                    content=function_response
                )
            )
        
        print("\nSending tool response back to the model...")
        
        # Get final response
        final_response = llm_provider.chat_completion(chat_request)
        
        print("\nLLM Final Answer:")
        print(final_response.content)
    else:
        print("\nLLM Final Answer:")
        print(response.content)


if __name__ == "__main__":
    user_question = "How much money have I spend to buy Cheese Crackers ?"
    run_conversation(user_question)

