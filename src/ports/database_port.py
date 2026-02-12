"""
Database Port Interface

Defines the contract for database operations using Pydantic models.
"""

from abc import ABC, abstractmethod
from src.domain.models import Receipt, Item
from typing import List, Optional


class DatabasePort(ABC):
    """Port interface for database operations"""
    
    @abstractmethod
    def save_receipt(self, receipt: Receipt) -> bool:
        """
        Save receipt to database.
        
        Args:
            receipt: Receipt domain model with items
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def query_spending(
        self, 
        item_name: str, 
        days: int = 7
    ) -> float:
        """
        Query total spending for an item within specified days.
        
        Args:
            item_name: Name of the item to query
            days: Number of days to look back
            
        Returns:
            Total amount spent
        """
        pass
    
    @abstractmethod
    def get_items(
        self, 
        receipt_id: Optional[int] = None
    ) -> List[Item]:
        """Get items, optionally filtered by receipt_id"""
        pass
