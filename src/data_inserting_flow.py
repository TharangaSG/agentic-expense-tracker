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
                "required": ["image_source"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "When the user provides an audio URL, use this tool to extract raw text from it.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
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

try:
    client = gemini_client
except KeyError:
    print("ERROR: Please set the GEMINI_API_KEY environment variable.")
    exit()

MODEL = settings.MAIN_MODEL_NAME

available_functions = {
    "extract_data_from_image": extract_data_from_image,
    "save_data_to_db": save_data_to_db,
    "transcribe_audio": transcribe_audio,
}

def detect_input_type(user_input: str) -> str:
    """
    Detect the type of user input (image, audio, or text).
    
    Args:
        user_input: The user's input string
        
    Returns:
        Input type: 'image', 'audio', or 'text'
    """
    # Check for common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    # Check for common audio file extensions
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']
    
    user_input_lower = user_input.lower()
    
    if any(ext in user_input_lower for ext in image_extensions):
        return 'image'
    elif any(ext in user_input_lower for ext in audio_extensions):
        return 'audio'
    else:
        return 'text'

def process_user_input(user_input: str) -> str:
    """
    Process different types of user input and return appropriate content for the AI model.
    
    Args:
        user_input: Raw user input (could be image path, audio path, or text)
        
    Returns:
        Processed content ready for the AI model
    """
    input_type = detect_input_type(user_input)
    
    if input_type == 'image':
        return f"process the receipt at '{user_input}' and save its contents to my database."
    
    elif input_type == 'audio':
        return f"Transcribing audio file: '{user_input}' and save its contents to my database."
    
    else:
        # It's a text description of a purchase
        return f"I made a purchase: {user_input}. Please parse this and save to my database."

def main():
    """
    Main function to handle different types of user input.
    """
    # Get user input
    print("=== Receipt Processing System ===")
    print("You can:")
    print("1. Provide an image file path (e.g., 'bill4.jpg')")
    print("2. Provide an audio file path (e.g., 'purchase.wav')")
    print("3. Type your purchase description (e.g., 'I bought two milk packets for 10 dollars')")
    print()
    
    user_input = input("Enter your input: ").strip()
    
    if not user_input:
        print("No input provided. Exiting.")
        return
    
    # Process the user input
    processed_content = process_user_input(user_input)
    
    messages = [
        {"role": "system", "content": """You are a helpful assistant that processes purchase information. 
        When given purchase descriptions in natural language, extract the item details and structure them appropriately. 
        For text purchases, try to infer reasonable unit prices from the total if not explicitly stated.
        Always generate a unique receipt_id (you can use a timestamp-based approach or increment from 1).
        """},
        {"role": "user", "content": processed_content}
    ]
    
    # Process with AI model
    for attempt in range(3):
        print(f"\n--- Attempt {attempt + 1}: Sending request to model ---")
        print(f"  > Current message history length: {len(messages)}")
        print(f"  > Processing: {processed_content[:100]}...")
        
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = completion.choices[0].message
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                messages.append(response_message)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  > Calling function: {function_name}")
                    print(f"  > Arguments: {function_args}")
                    
                    function_response = function_to_call(**function_args)
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
                continue
            else:
                print("\n--- Final response from model ---")
                print(response_message.content)
                break
                
        except Exception as e:
            print(f"Error during processing: {e}")
            break

if __name__ == "__main__":
    main()