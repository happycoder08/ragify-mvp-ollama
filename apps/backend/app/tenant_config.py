"""
Tenant configuration management for multi-tenant RAGify MVP.

Each tenant has customizable branding and UI settings.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Configuration for a single tenant with branding and customization."""
    
    tenant_id: str
    title: str
    primary_color: str  # Hex color code for primary branding
    logo_url: Optional[str] = None  # URL to logo image
    disclaimer: Optional[str] = None  # Custom disclaimer text for this tenant
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tenant_id": self.tenant_id,
            "title": self.title,
            "primary_color": self.primary_color,
            "logo_url": self.logo_url,
            "disclaimer": self.disclaimer,
        }


# ============================================================================
# Hardcoded tenant configurations for MVP
# ============================================================================

TENANT_CONFIGS: Dict[str, TenantConfig] = {
    "default": TenantConfig(
        tenant_id="default",
        title="RAGify AI - Demo",
        primary_color="#3b82f6",  # Blue
        logo_url=None,
        disclaimer="This is a demonstration of RAGify AI. Answers are generated from uploaded documents.",
    ),
    
    "acme": TenantConfig(
        tenant_id="acme",
        title="ACME Corp Knowledge Base",
        primary_color="#ef4444",  # Red
        logo_url=None,
        disclaimer="ACME Corp internal knowledge base. For employee use only. Answers are based on official company policies.",
    ),
    
    "finance": TenantConfig(
        tenant_id="finance",
        title="Financial Services Assistant",
        primary_color="#10b981",  # Green
        logo_url=None,
        disclaimer="Financial information assistant. This is not financial advice. Always consult with a licensed professional.",
    ),
}


def get_tenant_config(tenant_id: str) -> Optional[TenantConfig]:
    """
    Retrieve configuration for a specific tenant.
    
    Args:
        tenant_id: Unique identifier for the tenant
        
    Returns:
        TenantConfig if found, None otherwise
    """
    config = TENANT_CONFIGS.get(tenant_id)
    if config:
        logger.info("Retrieved config for tenant: %s", tenant_id)
    else:
        logger.warning("No config found for tenant: %s", tenant_id)
    return config


def list_tenant_ids() -> list[str]:
    """Get list of all configured tenant IDs."""
    return list(TENANT_CONFIGS.keys())
