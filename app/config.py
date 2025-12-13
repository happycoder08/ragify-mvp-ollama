import os
from typing import Dict, Any

# ===========================
# RAGIFY MODE CONFIGURATION
# ===========================
# Set via env var: RAGIFY_MODE=dev|demo|prod
# - dev: Full features, verbose logging, no limits
# - demo: Fast responses, safe defaults, limited context
# - prod: Balanced performance, security, monitoring

RAGIFY_MODE = os.getenv("RAGIFY_MODE", "demo").lower()

# ===========================
# MODE-SPECIFIC SETTINGS
# ===========================

class ModeConfig:
    """Configuration presets for different RAGIFY modes."""
    
    DEV = {
        # LLM Settings
        "default_mode": "full",  # Query mode: fast or full
        "max_tokens_fast": None,  # Unlimited for dev testing
        "max_tokens_full": None,
        "top_k_fast": 3,
        "top_k_full": 6,
        
        # Retrieval Settings
        "similarity_threshold": 400,  # More lenient for testing
        "chunk_size": 800,
        "chunk_overlap": 200,
        
        # Performance Settings
        "request_timeout": 600,  # 10 min for slow models
        "embedding_batch_size": 10,
        
        # Provider Settings
        "llm_provider": "ollama",
        "llm_model": "llama3.2:1b",
        "embedding_model": "nomic-embed-text",
        
        # Reranker Settings
        "reranker_provider": "none",  # none, jina, cohere
        "reranker_top_n": None,  # None = use all retrieved docs
        "enable_reranking": False,  # Disable in dev for speed
        
        # Logging
        "log_level": "DEBUG",
        "enable_timing_logs": True,
    }
    
    DEMO = {
        # LLM Settings - optimized for speed
        "default_mode": "fast",
        "max_tokens_fast": 50,  # Brief answers only
        "max_tokens_full": 150,
        "top_k_fast": 2,  # Minimal context
        "top_k_full": 4,
        
        # Retrieval Settings - strict relevance
        "similarity_threshold": 350,
        "chunk_size": 800,
        "chunk_overlap": 200,
        
        # Performance Settings - fast responses
        "request_timeout": 300,  # 5 min max
        "embedding_batch_size": 10,
        
        # Provider Settings - fastest local model
        "llm_provider": "ollama",
        "llm_model": "llama3.2:1b",
        "embedding_model": "nomic-embed-text",
        
        # Reranker Settings
        "reranker_provider": "none",  # none, jina, cohere
        "reranker_top_n": 2,  # Keep top 2 after reranking
        "enable_reranking": False,  # Disable for speed in demo
        
        # Logging
        "log_level": "INFO",
        "enable_timing_logs": True,
    }
    
    PROD = {
        # LLM Settings - balanced quality/speed
        "default_mode": "full",
        "max_tokens_fast": 100,
        "max_tokens_full": 500,
        "top_k_fast": 3,
        "top_k_full": 5,
        
        # Retrieval Settings - balanced relevance
        "similarity_threshold": 350,
        "chunk_size": 800,
        "chunk_overlap": 200,
        
        # Performance Settings
        "request_timeout": 300,
        "embedding_batch_size": 10,
        
        # Provider Settings - production ready
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL", "llama3.2:1b"),
        "embedding_model": "nomic-embed-text",
        
        # Reranker Settings
        "reranker_provider": os.getenv("RERANKER_PROVIDER", "none"),
        "reranker_top_n": 3,  # Keep top 3 after reranking
        "enable_reranking": True,  # Enable for better accuracy in prod
        
        # Logging
        "log_level": "INFO",
        "enable_timing_logs": True,
    }
    
    @classmethod
    def get_config(cls, mode: str = None) -> Dict[str, Any]:
        """Get configuration for the specified mode."""
        mode = (mode or RAGIFY_MODE).lower()
        configs = {
            "dev": cls.DEV,
            "demo": cls.DEMO,
            "prod": cls.PROD,
        }
        return configs.get(mode, cls.DEMO)  # Default to demo if invalid mode


# ===========================
# ACTIVE CONFIGURATION
# ===========================

CONFIG = ModeConfig.get_config(RAGIFY_MODE)

# Export commonly used settings
DEFAULT_MODE = CONFIG["default_mode"]
MAX_TOKENS_FAST = CONFIG["max_tokens_fast"]
MAX_TOKENS_FULL = CONFIG["max_tokens_full"]
TOP_K_FAST = CONFIG["top_k_fast"]
TOP_K_FULL = CONFIG["top_k_full"]
SIMILARITY_THRESHOLD = CONFIG["similarity_threshold"]
CHUNK_SIZE = CONFIG["chunk_size"]
CHUNK_OVERLAP = CONFIG["chunk_overlap"]
REQUEST_TIMEOUT = CONFIG["request_timeout"]
EMBEDDING_BATCH_SIZE = CONFIG["embedding_batch_size"]
LLM_PROVIDER = CONFIG["llm_provider"]
LLM_MODEL = CONFIG["llm_model"]
EMBEDDING_MODEL = CONFIG["embedding_model"]
RERANKER_PROVIDER = CONFIG["reranker_provider"]
RERANKER_TOP_N = CONFIG["reranker_top_n"]
ENABLE_RERANKING = CONFIG["enable_reranking"]
LOG_LEVEL = CONFIG["log_level"]
ENABLE_TIMING_LOGS = CONFIG["enable_timing_logs"]

# ===========================
# PATHS AND DIRECTORIES
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===========================
# RUNTIME INFO
# ===========================

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of active configuration for logging/debugging."""
    return {
        "ragify_mode": RAGIFY_MODE,
        "default_mode": DEFAULT_MODE,
        "max_tokens_fast": MAX_TOKENS_FAST,
        "max_tokens_full": MAX_TOKENS_FULL,
        "top_k_fast": TOP_K_FAST,
        "top_k_full": TOP_K_FULL,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "request_timeout": REQUEST_TIMEOUT,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "reranker_provider": RERANKER_PROVIDER,
        "reranker_top_n": RERANKER_TOP_N,
        "enable_reranking": ENABLE_RERANKING,
        "log_level": LOG_LEVEL,
    }
