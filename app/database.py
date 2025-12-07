"""
Database connection and session management for RAGify MVP.

Uses SQLAlchemy with Postgres for document metadata storage.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

logger = logging.getLogger(__name__)

# Get database URL from environment or use default for local development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragify:ragify@localhost:5432/ragify_db"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    echo=False,  # Set to True to log all SQL statements
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    Yields a session and ensures it's closed after use.
    Returns None if database is not available.
    
    Usage in FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # use db here - check if db is None first!
    """
    db = None
    try:
        db = SessionLocal()
    except Exception as e:
        logger.warning("Database session creation failed: %s", e)
        yield None
        return
    
    try:
        yield db
    finally:
        if db is not None:
            try:
                db.close()
            except Exception as e:
                logger.warning("Database session close failed: %s", e)


def init_db() -> None:
    """
    Initialize database by creating all tables.
    Should be called on application startup.
    """
    logger.info("Initializing database...")
    try:
        # Import models to register them with Base
        from app.models import Document  # noqa: F401
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize database: %s", e)
        raise RuntimeError(f"Database initialization failed: {e}")


def test_connection() -> bool:
    """
    Test database connection.
    Returns True if connection is successful, False otherwise.
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error("Database connection test failed: %s", e)
        return False
