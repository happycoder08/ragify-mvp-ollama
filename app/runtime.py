"""
AppRuntime - Dependency injection container for RAGify.

Provides a clean way to manage dependencies (database, HTTP client, LLM providers)
that can be easily mocked or replaced in tests.
"""

import os
import logging
from typing import Optional, AsyncGenerator, Callable, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    """
    Runtime dependency container for RAGify application.
    
    Holds all major dependencies needed for upload, indexing, and query operations.
    Can be configured differently for production vs testing.
    
    Attributes:
        db_enabled: Whether database operations are enabled
        get_db_session: Factory for database sessions (context manager or async generator)
        http_client: Optional HTTP client for external API calls (None allowed for testing)
        llm_provider: LLM provider instance for chat/generation
        embedding_provider: Embedding provider instance for vectorization
        task_runner: Task execution strategy
            - Production: Factory function (BackgroundTasks -> TaskRunner)
            - Testing: InlineTaskRunner instance (immediate execution)
    """
    
    db_enabled: bool
    get_db_session: Callable
    http_client: Optional[Any]
    llm_provider: Any
    embedding_provider: Any
    task_runner: Any  # Callable[[BackgroundTasks], TaskRunner] or InlineTaskRunner


def build_runtime_from_env() -> AppRuntime:
    """
    Build AppRuntime from environment variables and application config.
    
    Used by main.py at startup to configure production runtime.
    
    Supports CI mode: If CI=true or APP_MODE=ci, automatically configures:
    - LLM_PROVIDER=mock
    - EMBEDDING_PROVIDER=mock
    - TASK_RUNNER=inline
    - No HTTP client required
    
    Returns:
        AppRuntime: Configured runtime with production or CI dependencies
    """
    from app.database import get_db
    from app.services import clients
    from app.services.llm_providers import create_llm_provider
    from app.services.embeddings import create_embedder
    from app.services.task_runner import create_background_task_runner, create_inline_task_runner
    
    # Detect CI mode from environment
    is_ci_mode = (
        os.getenv("CI", "").lower() in ("true", "1", "yes") or
        os.getenv("APP_MODE", "").lower() == "ci"
    )
    
    if is_ci_mode:
        # Override providers for CI mode
        logger.info("CI mode detected - configuring mock providers and inline task runner")
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
    
    # Database is always enabled in production
    db_enabled = True
    
    # Get database session factory
    get_db_session = get_db
    
    # Initialize HTTP client only if not in CI mode
    http_client = None
    if not is_ci_mode:
        clients.initialize_http_client()
        http_client = clients.get_http_client()
    
    # Create LLM providers based on environment
    llm_provider = create_llm_provider()
    
    # Create embedder based on environment
    provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
    if is_ci_mode:
        embedding_provider = create_embedder(provider_type="mock")
    else:
        embedding_provider = create_embedder(
            provider_type=provider_type,
            http_client=http_client
        )
    
    # Task runner: inline for CI, background for production
    if is_ci_mode:
        task_runner = create_inline_task_runner()
        task_runner_type = "inline"
    else:
        task_runner = create_background_task_runner
        task_runner_type = "background"
    
    logger.info(
        "AppRuntime initialized [CI_MODE=%s]: db_enabled=%s, http_client=%s, llm_provider=%s, embedding_provider=%s, task_runner=%s",
        is_ci_mode,
        db_enabled,
        http_client is not None,
        type(llm_provider).__name__,
        type(embedding_provider).__name__,
        task_runner_type
    )
    
    return AppRuntime(
        db_enabled=db_enabled,
        get_db_session=get_db_session,
        http_client=http_client,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        task_runner=task_runner
    )


def build_test_runtime(
    db_session_factory: Optional[Callable] = None,
    llm_provider: Optional[Any] = None,
    embedding_provider: Optional[Any] = None
) -> AppRuntime:
    """
    Build AppRuntime for testing with minimal dependencies.
    
    Args:
        db_session_factory: Optional database session factory (default: dummy generator)
        llm_provider: Optional LLM provider (default: creates MockLLMProvider)
        embedding_provider: Optional embedding provider (default: creates MockEmbeddingProvider)
    
    Returns:
        AppRuntime: Test-configured runtime with db_enabled=False and no HTTP client
    """
    from app.services.llm_providers import create_llm_provider
    from app.services.embeddings import create_embedder
    from app.services.task_runner import create_inline_task_runner
    
    # Database disabled by default in tests
    db_enabled = False
    
    # Dummy database session factory if none provided
    if db_session_factory is None:
        @asynccontextmanager
        async def dummy_db_session():
            """Dummy session that yields None."""
            yield None
        db_session_factory = dummy_db_session
    
    # No HTTP client needed for tests (mock providers don't use it)
    http_client = None
    
    # Use provided providers or create mocks
    if llm_provider is None:
        # Create mock provider directly without setting environment variable
        # (Setting LLM_PROVIDER=mock would cause is_mock_mode() to bypass ChromaDB queries)
        from app.services.llm_providers import MockLLMProvider
        llm_provider = MockLLMProvider()
    
    if embedding_provider is None:
        # Mock embedding provider (no HTTP client needed)
        embedding_provider = create_embedder(provider_type="mock")
    
    # Inline task runner for tests (immediate execution, no BackgroundTasks needed)
    task_runner = create_inline_task_runner()
    
    logger.info(
        "Test AppRuntime initialized: db_enabled=%s, http_client=%s, llm_provider=%s, embedding_provider=%s",
        db_enabled,
        http_client,
        type(llm_provider).__name__,
        type(embedding_provider).__name__
    )
    
    return AppRuntime(
        db_enabled=db_enabled,
        get_db_session=db_session_factory,
        http_client=http_client,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        task_runner=task_runner
    )
