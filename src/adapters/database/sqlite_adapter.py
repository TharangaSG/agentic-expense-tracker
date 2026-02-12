"""
SQLite Database Adapter

Implements DatabasePort interface using SQLite.
"""

import sqlite3
import json
from typing import List, Optional
from src.ports.database_port import DatabasePort
from src.domain.models import Receipt, Item


class SQLiteDatabaseAdapter(DatabasePort):
    """SQLite database provider implementation"""
    
    def __init__(self, db_path: str = "receipts_final.db"):
        """
        Initialize SQLite database adapter.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER,
                item_name TEXT,
                quantity REAL,
                unit_price REAL,
                total_price REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_receipt(self, receipt: Receipt) -> bool:
        """
        Save receipt to database.
        
        Args:
            receipt: Receipt domain model with items
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for item in receipt.items:
                cursor.execute(
                    """INSERT INTO items 
                       (receipt_id, item_name, quantity, unit_price, total_price) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        receipt.receipt_id,
                        item.item_name,
                        item.quantity,
                        item.unit_price,
                        item.total_price
                    )
                )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving receipt: {e}")
            return False
    
    def query_spending(
        self, 
        item_name: str, 
        days: int = 7
    ) -> float:
        """
        Query total spending for an item within specified days.
        
        Args:
            item_name: Name of the item to query
            days: Number of days to look back (currently queries all-time)
            
        Returns:
            Total amount spent
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql_query = """
                SELECT SUM(total_price)
                FROM items
                WHERE item_name LIKE ?;
            """
            search_term = f'%{item_name}%'
            
            cursor.execute(sql_query, (search_term,))
            result = cursor.fetchone()[0]
            
            conn.close()
            
            return result if result is not None else 0.0
            
        except Exception as e:
            print(f"Error querying spending: {e}")
            return 0.0
    
    def get_items(
        self, 
        receipt_id: Optional[int] = None
    ) -> List[Item]:
        """
        Get items, optionally filtered by receipt_id.
        
        Args:
            receipt_id: Optional receipt ID to filter by
            
        Returns:
            List of Item domain models
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if receipt_id is not None:
                cursor.execute(
                    """SELECT item_name, quantity, unit_price, total_price 
                       FROM items WHERE receipt_id = ?""",
                    (receipt_id,)
                )
            else:
                cursor.execute(
                    """SELECT item_name, quantity, unit_price, total_price 
                       FROM items"""
                )
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                Item(
                    item_name=row[0],
                    quantity=row[1],
                    unit_price=row[2],
                    total_price=row[3]
                )
                for row in rows
            ]
            
        except Exception as e:
            print(f"Error getting items: {e}")
            return []
