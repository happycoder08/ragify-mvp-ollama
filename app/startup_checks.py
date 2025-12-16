"""
Startup assertions for demo mode.
Logs readiness state and fails fast if preconditions are not met.
"""

import logging
from app.database import SessionLocal
from app.models import Document
from app.services import clients

logger = logging.getLogger(__name__)


def demo_startup_check(tenant_id: str = "default"):
    """
    Run demo mode startup checks.
    Logs document counts and vector store readiness.
    Fails fast if more than one document is indexed (demo expects single doc).
    
    In demo mode, gracefully handles Postgres unavailability (not critical for RAG).
    
    Args:
        tenant_id: Tenant to check
        
    Raises:
        RuntimeError: If checks fail critically
    """
    logger.info("=" * 70)
    logger.info("DEMO MODE STARTUP CHECKS")
    logger.info("=" * 70)
    
    # Check Postgres documents (graceful failure in demo mode)
    db = SessionLocal()
    db_count = 0
    try:
        db_docs = db.query(Document).filter(Document.tenant_id == tenant_id).all()
        db_count = len(db_docs)
        logger.info(f"✓ Postgres documents (tenant={tenant_id}): {db_count}")
        for doc in db_docs:
            logger.info(f"  - ID={doc.id}, filename={doc.filename}, status={doc.status}")
    except Exception as e:
        logger.warning(f"⚠ Postgres unavailable (not critical for RAG): {e}")
        logger.info("  Application will run with ChromaDB only.")
        db_count = 0
    finally:
        db.close()
    
    # Check Chroma collection
    try:
        chroma_client = clients.get_chroma_client()
        collection = chroma_client.get_or_create_collection(f"documents_{tenant_id}")
        chroma_count = collection.count()
        logger.info(f"✓ Chroma vectors (collection=documents_{tenant_id}): {chroma_count}")
        
        # Get unique filenames from metadata
        if chroma_count > 0:
            res = collection.get()
            filenames = set()
            for meta in res.get("metadatas", []):
                fn = meta.get("filename") or meta.get("source_file", "unknown")
                filenames.add(fn)
            
            logger.info(f"✓ Unique filenames in Chroma: {len(filenames)}")
            for fn in sorted(filenames):
                logger.info(f"  - {fn}")
            
            # Demo constraint: only one document
            if len(filenames) > 1:
                msg = f"DEMO CONSTRAINT VIOLATED: Expected 1 document, found {len(filenames)}: {filenames}"
                logger.error(f"✗ {msg}")
                raise RuntimeError(msg)
        else:
            logger.warning("⚠️  Chroma collection is empty (no vectors indexed yet)")
    
    except Exception as e:
        logger.error(f"✗ Failed to check Chroma: {e}")
        raise RuntimeError(f"Chroma check failed: {e}")
    
    logger.info("=" * 70)
    logger.info("✓ DEMO STARTUP CHECKS PASSED")
    logger.info("=" * 70)
