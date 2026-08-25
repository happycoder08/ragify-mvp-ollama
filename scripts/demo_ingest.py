import sys
from pathlib import Path

# Fix python path to allow imports from root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import asyncio
import inspect
import os

from app.config import UPLOAD_DIR
from app.database import get_db, init_db, test_connection
from app.models import Document
from app.services import clients, ingestion, rag_service
from main import process_document_background


def _get_db_session():
    db_gen = get_db()
    try:
        db = next(db_gen)
    except StopIteration:
        return None, None
    return db, db_gen


def _close_db_session(db_gen):
    if db_gen:
        db_gen.close()


def _is_safe_upload_path(path: str) -> bool:
    if not path:
        return False
    try:
        upload_root = os.path.abspath(UPLOAD_DIR)
        target = os.path.abspath(path)
        return os.path.commonpath([upload_root, target]) == upload_root
    except Exception:
        return False


async def _init_clients():
    clients.initialize_chroma_client()
    # Check if mock mode via simple check or env var to avoid circular imports if specific service logic needed
    is_mock = False
    if hasattr(rag_service, "is_mock_mode"):
        is_mock = rag_service.is_mock_mode()
    
    if not is_mock and os.getenv("EMBEDDING_PROVIDER", "").lower() != "mock":
        await clients.initialize_http_client()


async def _clean_existing(tenant_id: str, filenames):
    try:
        collection = await rag_service.get_collection_async(tenant_id)
        for name in filenames:
            try:
                collection.delete(where={"filename": {"$eq": name}})
            except Exception as e:
                print(f"warn: failed to delete vectors for {name}: {e}")
    except Exception as e:
        print(f"warn: failed to access collection for cleanup: {e}")

    db, db_gen = _get_db_session()
    try:
        if not db:
            print("warn: database unavailable; skipping document cleanup")
            return
        docs = (
            db.query(Document)
            .filter(Document.tenant_id == tenant_id)
            .filter(Document.filename.in_(filenames))
            .all()
        )
        for doc in docs:
            # Only delete file if it looks like it is in our upload dir (safety)
            if _is_safe_upload_path(doc.file_path) and os.path.exists(doc.file_path):
                try:
                    os.remove(doc.file_path)
                except Exception as e:
                    print(f"warn: failed to delete file {doc.file_path}: {e}")
            db.delete(doc)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"warn: document cleanup failed: {e}")
    finally:
        _close_db_session(db_gen)


async def _ingest_files(tenant_id: str, files: list[Path], clean: bool):
    if not files:
        print("No files to ingest.")
        return

    await _init_clients()

    if clean:
        await _clean_existing(tenant_id, [p.name for p in files])

    for path in files:
        raw_bytes = path.read_bytes()
        saved_path = ingestion.save_upload(raw_bytes, path.name)

        doc_id = -1
        db, db_gen = _get_db_session()
        try:
            if db:
                doc_record = Document(
                    tenant_id=tenant_id,
                    filename=path.name,
                    file_path=saved_path,
                    status="pending",
                )
                db.add(doc_record)
                db.commit()
                db.refresh(doc_record)
                doc_id = doc_record.id
            else:
                print(f"warn: no DB session; proceeding without record for {path.name}")
        except Exception as e:
            if db:
                db.rollback()
            print(f"warn: failed to create DB record for {path.name}: {e}")
        finally:
            _close_db_session(db_gen)

        # Call the worker directly
        num_chunks = await process_document_background(
            doc_id=doc_id,
            tenant_id=tenant_id,
            file_path=saved_path,
            filename=path.name,
            db_session_factory=get_db,
        )
        print(f"✅ {path.name} | doc_id={doc_id} | chunks={num_chunks}")


def _run_async(coro):
    return asyncio.run(coro)


def _ingest_files_sync(tenant_id: str, files: list[Path], clean: bool):
    """Sync wrapper for non-async environments, though rarely used now."""
    if not files:
        print("No files to ingest.")
        return

    _run_async(_init_clients())

    if clean:
        _run_async(_clean_existing(tenant_id, [p.name for p in files]))

    for path in files:
        raw_bytes = path.read_bytes()
        saved_path = ingestion.save_upload(raw_bytes, path.name)

        doc_id = -1
        db, db_gen = _get_db_session()
        try:
            if db:
                doc_record = Document(
                    tenant_id=tenant_id,
                    filename=path.name,
                    file_path=saved_path,
                    status="pending",
                )
                db.add(doc_record)
                db.commit()
                db.refresh(doc_record)
                doc_id = doc_record.id
        except Exception as e:
            if db: db.rollback()
            print(f"warn: DB error {e}")
        finally:
            _close_db_session(db_gen)

        num_chunks = process_document_background(
            doc_id=doc_id,
            tenant_id=tenant_id,
            file_path=saved_path,
            filename=path.name,
            db_session_factory=get_db,
        )
        print(f"✅ {path.name} | doc_id={doc_id} | chunks={num_chunks}")


def main():
    parser = argparse.ArgumentParser(description="Ingest demo documents.")
    # Positional argument 'path' that accepts a file or directory
    parser.add_argument("path", nargs="?", default="demo/policy_pack", help="File or Directory to ingest")
    parser.add_argument("--tenant", default="default", help="Tenant ID to ingest into.")
    parser.add_argument("--clean", action="store_true", help="Delete prior demo docs for this tenant.")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"❌ Path not found: {target}")

    # Determine file list based on whether input is file or dir
    if target.is_file():
        files = [target]
    else:
        files = [p for p in sorted(target.iterdir()) if p.is_file()]

    try:
        init_db()
        test_connection()
    except Exception as e:
        print(f"warn: database init failed: {e}")

    # Detect if the worker is async (it usually is)
    if inspect.iscoroutinefunction(process_document_background):
        asyncio.run(_ingest_files(args.tenant, files, args.clean))
    else:
        _ingest_files_sync(args.tenant, files, args.clean)


if __name__ == "__main__":
    main()