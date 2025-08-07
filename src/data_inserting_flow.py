import os
import json
import sqlite3
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field
from read_image import extract_data_from_image


class Item(BaseModel):
    item_name: str = Field(description="The name of the purchased item")
    quantity: float = Field(description="The quantity of the item purchased")
    unit_price: float = Field(description="The price of a single unit of the item")
    total_price: float = Field(description="The total price for the item")

class ReceiptData(BaseModel):
    """This Pydantic model defines the parameters for the 'save_data_to_db' tool."""
    receipt_id: int = Field(description="A unique identifier for the receipt, e.g., 1.")
    items: List[Item] = Field(description="A list of all items from the receipt.")


def save_data_to_db(receipt_id: int, items: List[dict]):
    """Saves the final, structured receipt data into a SQLite database."""
    print("\n[Tool Call] Running 'save_data_to_db'...")
    try:
        db_conn = sqlite3.connect('receipts_final.db')
        cursor = db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT, receipt_id INTEGER, item_name TEXT,
                quantity REAL, unit_price REAL, total_price REAL
            )''')

        for item_dict in items:
            item = Item.model_validate(item_dict)
            print(f"  > Inserting into DB: {item.item_name}")
            cursor.execute(
                "INSERT INTO items (receipt_id, item_name, quantity, unit_price, total_price) VALUES (?, ?, ?, ?, ?)",
                (receipt_id, item.item_name, item.quantity, item.unit_price, item.total_price)
            )
        
        db_conn.commit()
        db_conn.close()
        return json.dumps({"status": "success", "message": f"Saved {len(items)} items."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

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
            "name": "save_data_to_db",
            "description": "Takes structured JSON data of receipt items and saves it to the database.",
            "parameters": ReceiptData.model_json_schema(),
        },
    },
]

try:
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/"
    )
except KeyError:
    print("ERROR: Please set the GEMINI_API_KEY environment variable.")
    exit()

MODEL = "gemini-2.5-flash"

available_functions = {
    "extract_data_from_image": extract_data_from_image,
    "save_data_to_db": save_data_to_db,
}


messages = [
    {"role": "user", "content": "process the receipt at 'bill4.jpg' and save its contents to my database."}
]

for _ in range(3): 
    print("\n--- Sending request to model ---")
    print(f"  > Current message history length: {len(messages)}")
    
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
            
            function_response = function_to_call(**function_args)
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        continue
    else:
        print("\n--- Final response from model ---")
        print(response_message.content)
        break