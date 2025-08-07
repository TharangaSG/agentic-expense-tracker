import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from database_query_tool import get_spending_for_item_last_week


class SpendingQuery(BaseModel):
    item_name: str = Field(description="The name of the item to query for spending, e.g., 'soda', 'milk'.")

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL = "gemini-2.5-flash"


available_tools = {
    "get_spending_for_item_last_week": get_spending_for_item_last_week,
}

def run_conversation(user_prompt: str):
    """
    Runs the main conversation loop with the AI model.
    """
    if not client:
        print("Client not initialized. Exiting.")
        return

    messages = [{"role": "user", "content": user_prompt}]
    
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

    # Send the prompt and tools to the model
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    
    # Check if the model wants to call the tool
    if response_message.tool_calls:
        print("\nModel wants to call a tool...")
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Calling function '{function_name}' with arguments: {function_args}")
            
            function_response = function_to_call(**function_args)
            
            print(f"Tool response: {function_response}")
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        print("\nSending tool response back to the model...")
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        print("\nLLM Final Answer:")
        print(final_response.choices[0].message.content)
    else:
        print("\nLLM Final Answer:")
        print(response_message.content)

if __name__ == "__main__":

    user_question = "How much money have I spend to buy Cheese Crackers ?"
    run_conversation(user_question)
