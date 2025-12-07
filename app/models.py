"""
SQLAlchemy models for RAGify MVP.

Document model stores metadata about uploaded files.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.database import Base
import logging

logger = logging.getLogger(__name__)


class Document(Base):
    """
    Represents an uploaded document with metadata.
    
    The actual document embeddings are stored in ChromaDB,
    while this table stores metadata for tracking and status.
    """
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    
    # Status: "indexing", "indexed", "failed"
    status = Column(String(50), nullable=False, default="indexing", index=True)
    
    # Optional error message if status is "failed"
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Document(id={self.id}, tenant={self.tenant_id}, filename={self.filename}, status={self.status})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
