"""
Database Port Interface

Defines the contract for database operations using Pydantic models.
Supports both sync (legacy SQLite) and async (PostgreSQL) adapters.
"""

from abc import ABC, abstractmethod
from src.domain.models import Receipt, Item
from typing import List, Optional, Any, Dict


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


class AsyncDatabasePort(ABC):
    """
    Async port interface for database operations.
    
    Used by the PostgreSQL adapter and the multi-agent system.
    Extends capabilities with raw SQL execution and vector search
    for the Database Analyst Agent.
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """Initialize database connection pool and create tables if needed."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection pool."""
        pass
    
    @abstractmethod
    async def save_receipt(self, receipt: Receipt) -> bool:
        """
        Save receipt to database, including embedding generation.
        
        Args:
            receipt: Receipt domain model with items
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def query_spending(
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
    async def get_items(
        self, 
        receipt_id: Optional[int] = None
    ) -> List[Item]:
        """Get items, optionally filtered by receipt_id"""
        pass
    
    @abstractmethod
    async def execute_read_query(
        self,
        sql: str,
        params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query and return results.
        
        Used by the Database Analyst Agent for NL2SQL queries.
        Must only allow SELECT statements for safety.
        
        Args:
            sql: SQL SELECT query string  
            params: Optional query parameters
            
        Returns:
            List of dicts, each representing a row
        """
        pass
    
    @abstractmethod
    async def search_similar_items(
        self,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for items with similar name embeddings using pgvector.
        
        Used by the Database Analyst Agent to find exact DB spellings
        before generating SQL queries.
        
        Args:
            query_embedding: 768-dim embedding vector for the search query
            limit: Maximum number of results to return
            
        Returns:
            List of dicts with item_name, similarity_score, and other fields
        """
        pass
