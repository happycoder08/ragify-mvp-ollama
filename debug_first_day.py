import asyncio
import sys
sys.path.append('.')
from app.runtime import build_test_runtime
from app.services import clients
from app.services.rag_service import query_collection

async def debug_query():
    try:
        # Initialize clients for testing
        clients.initialize_chroma_client()
        chroma_client = clients.get_chroma_client()
        collections = chroma_client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        for col in collections:
            try:
                count = col.count()
                print(f"Collection {col.name}: {count} documents")
            except Exception as e:
                print(f"Collection {col.name}: error getting count - {e}")
        await clients.initialize_http_client()
        print("Clients initialized successfully")

        # Initialize test runtime
        runtime = build_test_runtime()
        print("Test runtime initialized successfully")

        gen, sources, evidence, context, debug_info = await query_collection(
            tenant_id='default',
            question='What do I do on my first day?',
            top_k=4,
            mode='full',
            debug=1,
            request_id='debug-test-123'
        )
        print('=== DEBUG INFO ===')
        print(f'Full debug_info: {debug_info}')
        retrieved = debug_info.get('retrieved_chunks_top20', [])
        print(f'Retrieved chunks: {len(retrieved)}')
        print(f'Total retrieved: {debug_info.get("total_retrieved", "N/A")}')
        print(f'K final: {debug_info.get("k_final", "N/A")}')
        print(f'Is broad: {debug_info.get("is_broad", "N/A")}')
        print(f'Selected chunk IDs: {debug_info.get("selected_chunk_ids", [])}')
        print(f'Selected headings: {debug_info.get("selected_headings", [])}')
        for i, chunk in enumerate(retrieved[:10]):
            chunk_id = chunk.get('chunk_id', 'unknown')
            dist = chunk.get('dist', 0)
            anchor = chunk.get('anchor_type', 'None')
            header = chunk.get('header_first_line', '')[:100]
            lexical = chunk.get('lexical_score', 0)
            final = chunk.get('final_score', 0)
            print(f'{i+1}. {chunk_id} - dist: {dist:.4f} - anchor: {anchor}')
            print(f'   Header: {header}...')
            print(f'   Lexical: {lexical:.2f} - Final: {final:.2f}')
            print()

        print('=== SELECTED CHUNKS ===')
        selected = debug_info.get('selected_chunks', [])
        for i, chunk in enumerate(selected):
            chunk_id = chunk.get('chunk_id', 'unknown')
            dist = chunk.get('dist', 0)
            source = chunk.get('source_file', 'unknown')
            header = chunk.get('header_first_line', '')[:100]
            anchor = chunk.get('anchor_type', 'None')
            print(f'{i+1}. {chunk_id} - dist: {dist:.4f}')
            print(f'   Source: {source}')
            print(f'   Header: {header}...')
            print(f'   Anchor: {anchor}')
            print()

        # Get the answer
        answer = ''
        async for chunk in gen:
            answer += chunk
        print('=== ANSWER ===')
        print(answer[:500])

    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(debug_query())