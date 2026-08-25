import os
from typing import Dict, Any

# ===========================
# RAGIFY MODE CONFIGURATION
# ===========================
# Set via env var: RAGIFY_MODE=dev|demo|pilot|prod
RAGIFY_MODE = os.getenv("RAGIFY_MODE", "demo").lower()

# ===========================
# MODE-SPECIFIC SETTINGS
# ===========================

class ModeConfig:
    """Configuration presets for different RAGIFY modes."""
    
    DEV = {
        "mode": "dev",
        "default_mode": "full",
        "debug": True,
        
        # LLM & Embedding
        "llm_provider": "ollama",
        "llm_model": "llama3.2:1b",
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        
        # Performance
        "request_timeout": 600,
        "embedding_batch_size": 10,
        "max_tokens_fast": None,
        "max_tokens_full": None,
        
        # Retrieval
        "top_k_fast": 3,
        "top_k_full": 3,
        "similarity_threshold": 400,
        "chunk_size": 800,
        "chunk_overlap": 200,
        "context_budget_chars": 3000,
        
        # Reranking
        "reranker_provider": "none",
        "reranker_top_n": None,
        "enable_reranking": False,
        
        # App Logic
        "max_conversation_turns": 10,
        "log_level": "DEBUG",
        "enable_timing_logs": True,
        "grounding_threshold": 0.65,
        "token_overlap_threshold": 1,
    }
    
    DEMO = {
        "mode": "demo",
        "default_mode": "demo",  # Critical fix for KeyError
        "debug": True,
        
        # LLM & Embedding
        "llm_provider": "ollama",
        "llm_model": "llama3.2:1b",
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        
        # Performance & Data Diet (Optimized for Local 8B Model)
        "request_timeout": 120,
        "embedding_batch_size": 1,
        "max_tokens_fast": 150,
        "max_tokens_full": 500,
        
        # Retrieval (Lightweight)
        "top_k_fast": 2,
        "top_k_full": 2,
        "similarity_threshold": 0.3,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "context_budget_chars": 1500,  # Strict budget to fix 90s latency
        
        # Reranking
        "reranker_provider": "none",
        "reranker_top_n": 2,
        "enable_reranking": False,
        
        # App Logic
        "max_conversation_turns": 5,
        "log_level": "INFO",
        "enable_timing_logs": True,
        "grounding_threshold": 0.65,
        "token_overlap_threshold": 1,
    }

    PILOT = {
        "mode": "pilot",
        "default_mode": "full",
        "debug": False,
        
        # LLM & Embedding
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL", "llama3.1:8b"),
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        
        # Performance
        "request_timeout": 300,
        "embedding_batch_size": 10,
        "max_tokens_fast": 100,
        "max_tokens_full": 500,
        
        # Retrieval
        "top_k_fast": 3,
        "top_k_full": 5,
        "similarity_threshold": 350,
        "chunk_size": 800,
        "chunk_overlap": 200,
        "context_budget_chars": 4000,
        
        # Reranking
        "reranker_provider": os.getenv("RERANKER_PROVIDER", "none"),
        "reranker_top_n": 3,
        "enable_reranking": True,
        
        # App Logic
        "max_conversation_turns": 8,
        "log_level": "INFO",
        "enable_timing_logs": True,
        "grounding_threshold": 0.65,
        "token_overlap_threshold": 1,
    }
    
    PROD = {
        "mode": "prod",
        "default_mode": "full",
        "debug": False,
        
        # LLM & Embedding
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL", "llama3.1:8b"),
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        
        # Performance
        "request_timeout": 300,
        "embedding_batch_size": 10,
        "max_tokens_fast": 100,
        "max_tokens_full": 500,
        
        # Retrieval
        "top_k_fast": 3,
        "top_k_full": 5,
        "similarity_threshold": 350,
        "chunk_size": 800,
        "chunk_overlap": 200,
        "context_budget_chars": 4000,
        
        # Reranking
        "reranker_provider": os.getenv("RERANKER_PROVIDER", "none"),
        "reranker_top_n": 3,
        "enable_reranking": True,
        
        # App Logic
        "max_conversation_turns": 8,
        "grounding_threshold": 0.85,
        "token_overlap_threshold": 2,
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
            "pilot": cls.PILOT,
            "prod": cls.PROD,
        }
        return configs.get(mode, cls.DEMO)


# ===========================
# SETTINGS (PILOT MODE)
# ===========================

class Settings:
    @property
    def GROUNDING_THRESHOLD(self) -> float:
        mode = (RAGIFY_MODE or "").lower()
        if mode in {"demo", "pilot"}:
            return 0.45
        return 0.85

    @property
    def TOKEN_OVERLAP_THRESHOLD(self) -> int:
        mode = (RAGIFY_MODE or "").lower()
        if mode in {"demo", "pilot"}:
            return 1
        return 2

# ===========================
# ACTIVE CONFIGURATION
# ===========================

CONFIG = ModeConfig.get_config(RAGIFY_MODE)
settings = Settings()

# Export commonly used settings
DEFAULT_MODE = CONFIG["default_mode"]
MAX_TOKENS_FAST = CONFIG["max_tokens_fast"]
MAX_TOKENS_FULL = CONFIG["max_tokens_full"]
TOP_K_FAST = CONFIG["top_k_fast"]
TOP_K_FULL = CONFIG["top_k_full"]
TOP_N_FAST = CONFIG.get("top_n_fast", None)
TOP_N_FULL = CONFIG.get("top_n_full", None)
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
CONTEXT_BUDGET_CHARS = CONFIG.get("context_budget_chars", None)
MAX_CONVERSATION_TURNS = CONFIG["max_conversation_turns"]
GROUNDING_THRESHOLD = settings.GROUNDING_THRESHOLD
TOKEN_OVERLAP_THRESHOLD = settings.TOKEN_OVERLAP_THRESHOLD
LOG_LEVEL = CONFIG["log_level"]
ENABLE_TIMING_LOGS = CONFIG["enable_timing_logs"]

# ===========================
# PATHS AND DIRECTORIES
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.getenv("VECTOR_DIR", os.path.join(BASE_DIR, "vectorstore"))
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
        "context_budget_chars": CONTEXT_BUDGET_CHARS,
        "request_timeout": REQUEST_TIMEOUT,
        "llm_provider": LLM_PROVIDER,
        "log_level": LOG_LEVEL,
    }
