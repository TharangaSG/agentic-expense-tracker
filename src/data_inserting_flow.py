import os
import json
from typing import List
from pydantic import BaseModel, Field
from src.settings import settings
from src.utils.clients import gemini_client
from src.tools.read_image import extract_data_from_image
from src.tools.save_data_to_db import save_data_to_db
from src.tools.speech_to_text import transcribe_audio

class Item(BaseModel):
    item_name: str = Field(description="The name of the purchased item")
    quantity: float = Field(description="The quantity of the item purchased")
    unit_price: float = Field(description="The price of a single unit of the item")
    total_price: float = Field(description="The total price for the item")

class ReceiptData(BaseModel):
    """This Pydantic model defines the parameters for the 'save_data_to_db' tool."""
    receipt_id: int = Field(description="A unique identifier for the receipt, e.g., 1.")
    items: List[Item] = Field(description="A list of all items from the receipt.")


def parse_text_purchase(text: str) -> str:
    """
    Parse natural language purchase description into structured format.
    
    Args:
        text: Natural language description of purchase
        
    Returns:
        Formatted text for the AI model to process
    """
    return f"Parse this purchase description and extract item details: {text}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_data_from_image",
            "description": "When the user provides an image URL, use this tool to extract raw text from it.",
            "parameters": {
                "type": "object",
                "properties": {"image_source": {"type": "string", "description": "The URL of the receipt image to process."}},
                "required": ["image_source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "When the user provides an audio URL, use this tool to extract raw text from it.",
            "parameters": {
                "type": "object",
                "properties": {"file_input": {"type": "string", "description": "The URL of the audio file to transcribe."}},
                "required": ["file_input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_data_to_db",
            "description": "Takes structured JSON data of receipt items and saves it to the database.",
            "parameters": ReceiptData.model_json_schema(),
        },
    },
]

available_functions = {
    "extract_data_from_image": extract_data_from_image,
    "save_data_to_db": save_data_to_db,
    "transcribe_audio": transcribe_audio,
    
}


def main():
    """
    Main function to run the data insertion flow.
    """
    print("Welcome to the Financial Assistant Data Insertion Tool!")
    print("\nYou can:")
    print("1. Upload a receipt image URL")
    print("2. Upload an audio file URL")
    print("3. Type your purchase description (e.g., 'I bought two milk packets for 10 dollars')")
    print("\nType 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter your input: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter some input.")
            continue
        
        result = process_user_input(user_input)
        print(f"\nResult: {result}")


async def process_user_input(user_input: str) -> str:
    """
    Process user input and return formatted response.
    
    Args:
        user_input: User's input (text, image analysis, or audio transcription)
        
    Returns:
        Formatted response message
    """
    try:
        # Determine input type and process accordingly
        if user_input.startswith("http"):
            if any(ext in user_input.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                # It's an image URL
                user_input = f"Image URL: {user_input}"
            elif any(ext in user_input.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
                # It's an audio URL
                user_input = f"Audio URL: {user_input}"
        
        # It's a text description of a purchase
        client = gemini_client
        
        # Generate a unique receipt ID (you might want to implement a better ID generation)
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a financial assistant that helps users track their purchases. 
                
                When given purchase descriptions in natural language, extract the item details and structure them appropriately. 
                
                For each item, extract:
                - item_name: The name of the item
                - quantity: How many units were purchased
                - unit_price: Price per single unit
                - total_price: Total cost for that item (quantity * unit_price)
                
                Call the save_data_to_db function with the structured data.
                
                If the user just says hello or greets you, respond politely and ask them to describe their purchase."""
            },
            {
                "role": "user", 
                "content": user_input
            }
        ]

        response = client.chat.completions.create(
            model=settings.MAIN_MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Check if the model wants to call a function
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in available_functions:
                    result = available_functions[function_name](**function_args)
                    
                    # Parse the result to create a better formatted output
                    if function_name == "save_data_to_db":
                        try:
                            result_data = json.loads(result)
                            if result_data.get("status") == "success":
                                # Extract items from function_args to show what was saved
                                items = function_args.get("items", [])
                                receipt_id = function_args.get("receipt_id", "N/A")
                                
                                output = "Purchase Successfully Saved!\n\n"
                                output += f"Items Saved: {len(items)}\n\n"
                                output += "Saved Items:\n"
                                
                                total_amount = 0
                                for i, item in enumerate(items, 1):
                                    item_name = item.get("item_name", "Unknown")
                                    quantity = item.get("quantity", 0)
                                    unit_price = item.get("unit_price", 0)
                                    total_price = item.get("total_price", 0)
                                    total_amount += total_price
                                    
                                    output += f"{i}. {item_name}\n"
                                    output += f"   - Quantity: {quantity}\n"
                                    output += f"   - Unit Price: ${unit_price:.2f}\n"
                                    output += f"   - Total: ${total_price:.2f}\n\n"
                                
                                output += f"Grand Total: ${total_amount:.2f}"
                                return output
                            else:
                                return f"Error saving purchase: {result_data.get('message', 'Unknown error')}"
                        except json.JSONDecodeError:
                            return f"Purchase saved! {result}"
                    
                    return f"Great! I've processed your request. {result}"
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error processing your message: {str(e)}"

if __name__ == "__main__":
    main()