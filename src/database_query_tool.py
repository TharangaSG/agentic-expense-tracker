import sqlite3
from datetime import datetime, timedelta

def get_spending_for_item_last_week(item_name: str) -> str:
    """
    Queries the 'receipts_final.db' database to find the total all-time amount spent 
    on a specific item.
    
    NOTE: This function no longer filters by date due to the absence of a 
    'transaction_date' column. It sums all matching entries.
    
    Args:
        item_name: The name of the item to query (e.g., 'soda').
        
    Returns:
        A string with the result or an error message.
    """

    db_path = 'receipts_final.db'
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        sql_query = """
            SELECT SUM(total_price)
            FROM items
            WHERE item_name LIKE ?;
        """
        search_term = f'%{item_name}%'
        
        cur.execute(sql_query, (search_term,))
        result = cur.fetchone()[0]
        
        # The result will be None if no records are found
        total_spent = result if result is not None else 0.0
        
        return f"Tool execution result: Total all-time spending on {item_name} is ${total_spent:.2f}"

    except sqlite3.Error as e:
        if "no such table" in str(e):
             return f"Tool error: The database '{db_path}' exists, but the required 'items' table is missing. Details: {e}"
        return f"Tool error: Database error - {e}"
    except Exception as e:
        return f"Tool error: An unexpected error occurred - {e}"
    finally:
        if conn:
            conn.close()



