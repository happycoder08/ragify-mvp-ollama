"""
Test script to verify RAGIFY_MODE configuration.
Run with different modes to see configuration changes:
  python test_config.py
  RAGIFY_MODE=dev python test_config.py
  RAGIFY_MODE=demo python test_config.py
  RAGIFY_MODE=prod python test_config.py
"""

import os
import json

# Test with different modes
test_modes = ["dev", "demo", "prod"]

for mode in test_modes:
    os.environ["RAGIFY_MODE"] = mode
    
    # Reimport config to get fresh settings
    import importlib
    if 'app.config' in __import__('sys').modules:
        importlib.reload(__import__('sys').modules['app.config'])
    
    from app.config import get_config_summary
    
    print(f"\n{'='*60}")
    print(f"RAGIFY_MODE = {mode.upper()}")
    print(f"{'='*60}")
    
    summary = get_config_summary()
    print(json.dumps(summary, indent=2))
    
    print("\nKey differences:")
    print(f"  - Default query mode: {summary['default_mode']}")
    print(f"  - Fast mode: {summary['top_k_fast']} chunks, {summary['max_tokens_fast']} tokens")
    print(f"  - Full mode: {summary['top_k_full']} chunks, {summary['max_tokens_full']} tokens")
    print(f"  - Similarity threshold: {summary['similarity_threshold']}")
    print(f"  - Request timeout: {summary['request_timeout']}s")
