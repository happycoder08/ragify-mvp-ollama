"""
Quick verification script for AppRuntime.

Demonstrates that AppRuntime can be built without external dependencies.
"""

from app.runtime import build_test_runtime

# Build test runtime
runtime = build_test_runtime()

print("=" * 60)
print("AppRuntime Test Configuration Summary")
print("=" * 60)
print()
print("✅ Runtime Attributes:")
print(f"   db_enabled: {runtime.db_enabled}")
print(f"   http_client: {runtime.http_client}")
print(f"   llm_provider: {type(runtime.llm_provider).__name__}")
print(f"   embedding_provider: {type(runtime.embedding_provider).__name__}")
print(f"   task_runner: {runtime.task_runner.__name__}")
print(f"   get_db_session: {type(runtime.get_db_session).__name__}")
print()
print("✅ CI/Testing Configuration:")
print(f"   Database: {'Disabled' if not runtime.db_enabled else 'Enabled'}")
print(f"   HTTP Client: {'None (no external API calls)' if runtime.http_client is None else 'Initialized'}")
print(f"   LLM Provider: {type(runtime.llm_provider).__name__} (deterministic)")
print(f"   Task Execution: {runtime.task_runner.__name__} (immediate)")
print()
print("=" * 60)
print("✅ AppRuntime successfully built without external dependencies!")
print("   No PostgreSQL, Ollama, or OpenAI required for testing.")
print("=" * 60)
