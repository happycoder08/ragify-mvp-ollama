import asyncio
import sys
sys.path.append('.')
from app.runtime import AppRuntime

async def test_debug():
    # Initialize the app runtime
    runtime = AppRuntime()
    await runtime.initialize()

    from app.services.rag_service import query_collection
    result = await query_collection(
        tenant_id='default',
        question='What do I do on my first day?',
        top_k=4,
        mode='full',
        debug=1,
        request_id='debug-test-123'
    )

    gen, sources, evidence, context, debug_info = result

    # Collect the answer
    answer_parts = []
    async for chunk in gen:
        answer_parts.append(chunk)
    answer = ''.join(answer_parts)

    print('=== DEBUG INFORMATION ===')
    print(f'Question: What do I do on my first day?')
    print(f'Answer: {answer[:100]}...')
    print(f'Debug keys: {list(debug_info.keys()) if debug_info else "None"}')

    if debug_info:
        retrieved = debug_info.get('retrieved_chunks_top20', [])
        print(f'\n=== RETRIEVED CHUNKS (top {len(retrieved)}) ===')
        for i, chunk in enumerate(retrieved[:10]):
            header = chunk.get('header_first_line', '')
            print(f'{i+1:2d}. chunk_id={chunk.get("chunk_id")}, dist={chunk.get("dist", 0):.4f}, final_score={chunk.get("final_score", 0):.4f}')
            print(f'    header: "{header}"')
            print(f'    anchor_type: {chunk.get("anchor_type")}, anchor_detected: {chunk.get("anchor_detected")}')
            print()

        selected = debug_info.get('selected_chunks', [])
        print(f'=== SELECTED CHUNKS ({len(selected)}) ===')
        for i, chunk in enumerate(selected):
            header = chunk.get('header_first_line', '')
            print(f'{i+1}. chunk_id={chunk.get("chunk_id")}, dist={chunk.get("dist", 0):.4f}, final_score={chunk.get("final_score", "N/A")}')
            print(f'   header: "{header}"')
            print(f'   anchor_type: {chunk.get("anchor_type")}, anchor_detected: {chunk.get("anchor_detected")}')
            print()

        print(f'=== EVIDENCE ITEMS ({len(evidence)}) ===')
        for i, ev in enumerate(evidence):
            print(f'{i+1}. chunk_id={ev.chunk_id}, heading="{ev.heading}", anchor_type={ev.anchor_type}, anchor_detected={ev.anchor_detected}')

        print(f'\n=== PIPELINE INFO ===')
        print(f'Pipeline marker: {debug_info.get("pipeline_marker")}')
        print(f'Hits count: {debug_info.get("hits_count")}')
        print(f'Selected count: {debug_info.get("selected_count")}')

        # Check chunk metadata consistency
        print(f'\n=== CHUNK METADATA ANALYSIS ===')
        all_chunks = retrieved + selected
        metadata_fields = {}
        for chunk in all_chunks:
            for key in chunk.keys():
                if key not in ['chunk_id', 'dist', 'final_score', 'header_first_line', 'anchor_type', 'anchor_detected']:
                    metadata_fields[key] = metadata_fields.get(key, 0) + 1

        print('Metadata fields found across chunks:')
        for field, count in metadata_fields.items():
            print(f'  {field}: present in {count} chunks')

asyncio.run(test_debug())