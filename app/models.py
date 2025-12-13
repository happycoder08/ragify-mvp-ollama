"""
SQLAlchemy models for RAGify MVP.

Document model stores metadata about uploaded files.
Conversation and Message models store chat history.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
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
    
    # Status: "pending" (uploaded, waiting to index), "indexing" (being processed), "indexed" (ready), "failed" (error)
    status = Column(String(50), nullable=False, default="pending", index=True)
    
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


class Conversation(Base):
    """
    Represents a conversation/chat session.
    
    Conversations belong to a tenant and contain multiple messages.
    """
    
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    title = Column(String(500), nullable=True)  # Optional conversation title
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, tenant={self.tenant_id}, title={self.title})>"
    
    def to_dict(self, include_messages=False):
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages else 0,
        }
        if include_messages:
            result["messages"] = [msg.to_dict() for msg in self.messages]
        return result


class Message(Base):
    """
    Represents a single message in a conversation.
    
    Messages have a role (user/assistant) and content.
    """
    
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    
    # Optional: sources cited in this message (for assistant messages)
    sources = Column(Text, nullable=True)  # JSON array of source filenames
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        import json
        sources_list = None
        if self.sources:
            try:
                sources_list = json.loads(self.sources)
            except:
                sources_list = []
        
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "sources": sources_list,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
