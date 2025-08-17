import sqlite3
from typing import List
import json
from pydantic import BaseModel, Field

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