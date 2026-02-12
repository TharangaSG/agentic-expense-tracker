import json
from src.config.containers import get_llm_provider, get_database
from src.domain.models import Receipt, Message, ChatRequest


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
            "name": "save_data_to_db",
            "description": "Takes structured JSON data of receipt items and saves it to the database.",
            "parameters": Receipt.model_json_schema(),
        },
    },
]


async def process_user_input(user_input: str) -> str:
    """
    Process user input and return formatted response using port-adapter architecture.
    
    Args:
        user_input: User's input (text, image analysis, or audio transcription)
        
    Returns:
        Formatted response message
    """
    try:
        # Get providers
        llm_provider = get_llm_provider()
        database = get_database()
        
        # Create chat request
        chat_request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content="""You are a financial assistant that helps users track their purchases. 
                    
                    When given purchase descriptions in natural language, extract the item details and structure them appropriately. 
                    
                    For each item, extract:
                    - item_name: The name of the item
                    - quantity: How many units were purchased
                    - unit_price: Price per single unit
                    - total_price: Total cost for that item (quantity * unit_price)
                    
                    Call the save_data_to_db function with the structured data.
                    
                    If the user just says hello or greets you, respond politely and ask them to describe their purchase."""
                ),
                Message(
                    role="user",
                    content=user_input
                )
            ],
            model=llm_provider.get_model_name(),
            tools=tools,
            tool_choice="auto"
        )

        response = llm_provider.chat_completion(chat_request)
        
        # Check if the model wants to call a function
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                if function_name == "save_data_to_db":
                    # Create Receipt domain model and save using database port
                    receipt = Receipt(**function_args)
                    success = database.save_receipt(receipt)
                    
                    if success:
                        # Create formatted output
                        output = "Purchase Successfully Saved!\n\n"
                        output += f"Items Saved: {len(receipt.items)}\n\n"
                        output += "Saved Items:\n"
                        
                        total_amount = 0
                        for i, item in enumerate(receipt.items, 1):
                            total_amount += item.total_price
                            
                            output += f"{i}. {item.item_name}\n"
                            output += f"   - Quantity: {item.quantity}\n"
                            output += f"   - Unit Price: ${item.unit_price:.2f}\n"
                            output += f"   - Total: ${item.total_price:.2f}\n\n"
                        
                        output += f"Grand Total: ${total_amount:.2f}"
                        return output
                    else:
                        return "Error saving purchase to database"
        
        return response.content
        
    except Exception as e:
        return f"Error processing your message: {str(e)}"


# Commented out main function - not needed for WhatsApp bot
# def main():
#     """
#     Main function to run the data insertion flow.
#     """
#     print("Welcome to the Financial Assistant Data Insertion Tool!")
#     print("\nYou can:")
#     print("1. Upload a receipt image URL")
#     print("2. Upload an audio file URL")
#     print("3. Type your purchase description (e.g., 'I bought two milk packets for 10 dollars')")
#     print("\nType 'quit' to exit.")
    
#     while True:
#         user_input = input("\nEnter your input: ").strip()
        
#         if user_input.lower() == 'quit':
#             print("Goodbye!")
#             break
        
#         if not user_input:
#             print("Please enter some input.")
#             continue
        
#         result = process_user_input(user_input)
#         print(f"\nResult: {result}")

# if __name__ == "__main__":
#     main()