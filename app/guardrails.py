"""
Guardrails and rate limiting for RAGify MVP.

Implements:
- File upload size limits
- Allowed file extensions validation
- Per-tenant rate limiting (in-memory)
- Request timeout enforcement
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# Guardrail Configuration per Tenant
# ============================================================================

@dataclass
class GuardrailConfig:
    """Guardrail limits for a specific tenant."""
    
    # File upload limits
    max_file_size_mb: int  # Maximum file size in MB
    max_files_per_request: int  # Maximum files per upload request
    allowed_extensions: set[str]  # Allowed file extensions (e.g., {'.pdf', '.txt'})
    
    # Rate limiting
    max_requests_per_minute: int  # Max API requests per minute
    max_requests_per_hour: int  # Max API requests per hour
    max_upload_mb_per_hour: int  # Max total upload MB per hour
    
    # Request timeouts
    llm_timeout_seconds: int  # LLM request timeout
    upload_timeout_seconds: int  # Upload processing timeout
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_file_size_mb": self.max_file_size_mb,
            "max_files_per_request": self.max_files_per_request,
            "allowed_extensions": list(self.allowed_extensions),
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_upload_mb_per_hour": self.max_upload_mb_per_hour,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "upload_timeout_seconds": self.upload_timeout_seconds,
        }


# ============================================================================
# Per-Tenant Guardrail Configurations
# ============================================================================

GUARDRAIL_CONFIGS: Dict[str, GuardrailConfig] = {
    "default": GuardrailConfig(
        max_file_size_mb=10,
        max_files_per_request=5,
        allowed_extensions={'.pdf', '.txt', '.docx', '.md'},
        max_requests_per_minute=20,
        max_requests_per_hour=500,
        max_upload_mb_per_hour=100,
        llm_timeout_seconds=300,  # 5 minutes
        upload_timeout_seconds=120,  # 2 minutes
    ),
    
    "acme": GuardrailConfig(
        max_file_size_mb=25,  # Higher limits for enterprise
        max_files_per_request=10,
        allowed_extensions={'.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx'},
        max_requests_per_minute=50,
        max_requests_per_hour=2000,
        max_upload_mb_per_hour=500,
        llm_timeout_seconds=600,  # 10 minutes
        upload_timeout_seconds=300,  # 5 minutes
    ),
    
    "finance": GuardrailConfig(
        max_file_size_mb=15,
        max_files_per_request=8,
        allowed_extensions={'.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx'},
        max_requests_per_minute=30,
        max_requests_per_hour=1000,
        max_upload_mb_per_hour=200,
        llm_timeout_seconds=300,
        upload_timeout_seconds=180,
    ),
}


def get_guardrail_config(tenant_id: str) -> GuardrailConfig:
    """
    Get guardrail configuration for a tenant.
    Falls back to 'default' if tenant not found.
    """
    config = GUARDRAIL_CONFIGS.get(tenant_id)
    if not config:
        logger.warning(f"No guardrail config for tenant {tenant_id}, using default")
        config = GUARDRAIL_CONFIGS["default"]
    return config


# ============================================================================
# In-Memory Rate Limiter
# ============================================================================

@dataclass
class RateLimitBucket:
    """Track requests and uploads for rate limiting."""
    
    requests_per_minute: deque  # Timestamps of requests in last minute
    requests_per_hour: deque  # Timestamps of requests in last hour
    uploads_mb_per_hour: deque  # (timestamp, size_mb) tuples in last hour
    
    def __init__(self):
        self.requests_per_minute = deque()
        self.requests_per_hour = deque()
        self.uploads_mb_per_hour = deque()


class RateLimiter:
    """In-memory rate limiter for per-tenant request tracking."""
    
    def __init__(self):
        self.buckets: Dict[str, RateLimitBucket] = defaultdict(RateLimitBucket)
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        
        for tenant_id, bucket in self.buckets.items():
            # Clean minute-window requests
            while bucket.requests_per_minute and bucket.requests_per_minute[0] < one_minute_ago:
                bucket.requests_per_minute.popleft()
            
            # Clean hour-window requests
            while bucket.requests_per_hour and bucket.requests_per_hour[0] < one_hour_ago:
                bucket.requests_per_hour.popleft()
            
            # Clean hour-window uploads
            while bucket.uploads_mb_per_hour and bucket.uploads_mb_per_hour[0][0] < one_hour_ago:
                bucket.uploads_mb_per_hour.popleft()
        
        self._last_cleanup = now
        logger.debug("Rate limiter cleanup completed")
    
    def check_rate_limit(self, tenant_id: str, upload_size_mb: float = 0) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.
        
        Args:
            tenant_id: Tenant identifier
            upload_size_mb: Size of upload in MB (0 for non-upload requests)
        
        Returns:
            (allowed, error_message) tuple
        """
        self._cleanup_old_entries()
        
        config = get_guardrail_config(tenant_id)
        bucket = self.buckets[tenant_id]
        now = time.time()
        
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        
        # Count requests in last minute
        recent_requests_minute = sum(1 for ts in bucket.requests_per_minute if ts > one_minute_ago)
        if recent_requests_minute >= config.max_requests_per_minute:
            return False, f"Rate limit exceeded: {config.max_requests_per_minute} requests per minute"
        
        # Count requests in last hour
        recent_requests_hour = sum(1 for ts in bucket.requests_per_hour if ts > one_hour_ago)
        if recent_requests_hour >= config.max_requests_per_hour:
            return False, f"Rate limit exceeded: {config.max_requests_per_hour} requests per hour"
        
        # Check upload bandwidth in last hour
        if upload_size_mb > 0:
            recent_uploads_mb = sum(size for ts, size in bucket.uploads_mb_per_hour if ts > one_hour_ago)
            if recent_uploads_mb + upload_size_mb > config.max_upload_mb_per_hour:
                return False, f"Upload bandwidth limit exceeded: {config.max_upload_mb_per_hour} MB per hour"
        
        return True, None
    
    def record_request(self, tenant_id: str, upload_size_mb: float = 0):
        """
        Record a request for rate limiting tracking.
        
        Args:
            tenant_id: Tenant identifier
            upload_size_mb: Size of upload in MB (0 for non-upload requests)
        """
        bucket = self.buckets[tenant_id]
        now = time.time()
        
        bucket.requests_per_minute.append(now)
        bucket.requests_per_hour.append(now)
        
        if upload_size_mb > 0:
            bucket.uploads_mb_per_hour.append((now, upload_size_mb))
        
        logger.debug(f"Recorded request for tenant {tenant_id}, upload_size={upload_size_mb}MB")
    
    def get_current_usage(self, tenant_id: str) -> dict:
        """
        Get current usage statistics for a tenant.
        
        Returns:
            Dictionary with current counts
        """
        bucket = self.buckets[tenant_id]
        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        
        requests_last_minute = sum(1 for ts in bucket.requests_per_minute if ts > one_minute_ago)
        requests_last_hour = sum(1 for ts in bucket.requests_per_hour if ts > one_hour_ago)
        uploads_mb_last_hour = sum(size for ts, size in bucket.uploads_mb_per_hour if ts > one_hour_ago)
        
        config = get_guardrail_config(tenant_id)
        
        return {
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "uploads_mb_last_hour": round(uploads_mb_last_hour, 2),
            "limits": {
                "max_requests_per_minute": config.max_requests_per_minute,
                "max_requests_per_hour": config.max_requests_per_hour,
                "max_upload_mb_per_hour": config.max_upload_mb_per_hour,
            }
        }


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


# ============================================================================
# Validation Functions
# ============================================================================

def validate_file_extension(filename: str, tenant_id: str) -> tuple[bool, Optional[str]]:
    """
    Validate file extension against allowed list.
    
    Returns:
        (valid, error_message) tuple
    """
    config = get_guardrail_config(tenant_id)
    
    # Extract extension (case-insensitive)
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if ext not in config.allowed_extensions:
        allowed_list = ', '.join(sorted(config.allowed_extensions))
        return False, f"File type '{ext}' not allowed. Allowed types: {allowed_list}"
    
    return True, None


def validate_file_size(file_size_bytes: int, tenant_id: str) -> tuple[bool, Optional[str]]:
    """
    Validate file size against limit.
    
    Returns:
        (valid, error_message) tuple
    """
    config = get_guardrail_config(tenant_id)
    max_bytes = config.max_file_size_mb * 1024 * 1024
    
    if file_size_bytes > max_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, f"File size ({size_mb:.2f} MB) exceeds limit of {config.max_file_size_mb} MB"
    
    return True, None


def validate_file_count(file_count: int, tenant_id: str) -> tuple[bool, Optional[str]]:
    """
    Validate number of files in upload request.
    
    Returns:
        (valid, error_message) tuple
    """
    config = get_guardrail_config(tenant_id)
    
    if file_count > config.max_files_per_request:
        return False, f"Too many files ({file_count}). Maximum {config.max_files_per_request} files per request"
    
    return True, None
