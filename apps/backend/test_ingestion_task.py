"""
Test ingestion task with database session injection.

Verifies that process_document_background:
- Accepts db_session_factory parameter
- Creates its own session (no global state)
- Updates document status in database
- Works with test session factory
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Document
from app.services import clients, rag_service
from main import process_document_background
import os


@pytest.fixture(scope="module", autouse=True)
def init_clients():
    """Initialize ChromaDB client and use mock LLM provider for all tests."""
    # Set mock LLM provider to avoid HTTP client dependency
    original_provider = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = "mock"
    
    # Initialize clients
    clients.initialize_chroma_client()
    
    # Force re-initialization of LLM provider in rag_service
    rag_service._llm_provider = None
    rag_service._embedding_provider = None
    
    yield
    
    # Cleanup
    clients.chroma_client = None
    rag_service._llm_provider = None
    rag_service._embedding_provider = None
    
    # Restore original provider
    if original_provider:
        os.environ["LLM_PROVIDER"] = original_provider
    elif "LLM_PROVIDER" in os.environ:
        del os.environ["LLM_PROVIDER"]


@pytest.fixture
def test_db_engine():
    """Create a temporary SQLite database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    engine.dispose()
    import os
    os.close(db_fd)
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


@pytest.fixture
def test_session_factory(test_db_engine):
    """Create a session factory for testing."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    return TestingSessionLocal


@pytest.fixture
def sample_document_file(tmp_path):
    """Create a sample document file for testing."""
    doc_path = tmp_path / "test_document.txt"
    doc_path.write_text("""
Employee Onboarding Guide

Welcome to the company!

Vacation Policy:
- Employees receive 15 days of vacation per year
- Vacation must be requested 2 weeks in advance

First Day Instructions:
- Arrive at 8:00 AM on your first day
- Report to the 3rd floor reception
""")
    return str(doc_path)


def test_process_document_background_with_session_factory(
    test_session_factory,
    sample_document_file
):
    """
    Test that process_document_background uses injected session factory.
    
    Verifies:
    - Task accepts db_session_factory parameter
    - Task creates its own session (no global state)
    - Task updates document status (to "indexed" on success or "failed" on error)
    - Database changes are persisted
    
    Note: This test may fail during indexing due to missing runtime dependencies,
    but it should still update the DB status to "failed" with error message.
    """
    # Create a document record in test database
    db = test_session_factory()
    doc = Document(
        tenant_id="test-tenant",
        filename="test_document.txt",
        file_path=sample_document_file,
        status="pending"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id
    db.close()
    
    # Verify initial status
    db = test_session_factory()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    assert doc.status == "pending", "Initial status should be pending"
    db.close()
    
    # Run background task with session factory
    asyncio.run(process_document_background(
        doc_id=doc_id,
        tenant_id="test-tenant",
        file_path=sample_document_file,
        filename="test_document.txt",
        db_session_factory=test_session_factory
    ))
    
    # Verify status was updated (either "indexed" or "failed")
    db = test_session_factory()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    assert doc is not None, "Document should exist"
    assert doc.status != "pending", "Status should be updated from pending"
    assert doc.status in ["indexed", "failed"], f"Status should be 'indexed' or 'failed', got: {doc.status}"
    
    # The key test: verify that DB session factory was used (status changed)
    print(f"✓ Task successfully used injected session factory (status: {doc.status})")
    db.close()


def test_process_document_background_handles_errors(
    test_session_factory,
    tmp_path
):
    """
    Test that process_document_background handles errors and updates status to 'failed'.
    """
    # Create a document record
    db = test_session_factory()
    doc = Document(
        tenant_id="test-tenant",
        filename="nonexistent.txt",
        file_path="/nonexistent/path/to/file.txt",  # Invalid path
        status="pending"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id
    db.close()
    
    # Run background task (should fail due to invalid file path)
    asyncio.run(process_document_background(
        doc_id=doc_id,
        tenant_id="test-tenant",
        file_path="/nonexistent/path/to/file.txt",
        filename="nonexistent.txt",
        db_session_factory=test_session_factory
    ))
    
    # Verify status was updated to "failed"
    db = test_session_factory()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    assert doc is not None, "Document should exist"
    assert doc.status == "failed", "Status should be updated to failed"
    assert doc.error_message is not None, "Error message should be set"
    assert len(doc.error_message) > 0, "Error message should not be empty"
    db.close()
    
    print("✓ Task successfully handled error and updated status to 'failed'")


def test_process_document_background_without_db(sample_document_file):
    """
    Test that process_document_background works without database.
    
    When db_session_factory is None, task should still process the document
    but skip database updates.
    """
    # Run task without database
    asyncio.run(process_document_background(
        doc_id=-1,  # -1 indicates no DB record
        tenant_id="test-tenant",
        file_path=sample_document_file,
        filename="test_document.txt",
        db_session_factory=None  # No database
    ))
    
    # Should complete without error
    print("✓ Task successfully processed document without database")


def test_process_document_background_explicit_parameters(
    test_session_factory,
    sample_document_file
):
    """
    Test that all parameters are passed explicitly (no globals).
    
    Verifies:
    - tenant_id is passed explicitly
    - doc_id is passed explicitly
    - db_session_factory is passed explicitly
    """
    # Create document
    db = test_session_factory()
    doc = Document(
        tenant_id="explicit-tenant",
        filename="explicit.txt",
        file_path=sample_document_file,
        status="pending"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id
    db.close()
    
    # Call with explicit keyword arguments
    asyncio.run(process_document_background(
        doc_id=doc_id,
        tenant_id="explicit-tenant",
        file_path=sample_document_file,
        filename="explicit.txt",
        db_session_factory=test_session_factory
    ))
    
    # Verify tenant isolation
    db = test_session_factory()
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.tenant_id == "explicit-tenant"
    ).first()
    assert doc is not None, "Document should be found with explicit tenant_id"
    assert doc.status in ["indexed", "failed"], "Status should be updated"
    db.close()
    
    print("✓ All parameters passed explicitly (tenant_id, doc_id, db_session_factory)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
