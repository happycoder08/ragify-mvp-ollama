import pytest
from app.services.rag_service import query_collection
import asyncio

@pytest.mark.asyncio
async def test_arrival_time_force_include(monkeypatch):
    # Simulate a collection with schedule and arrival+clock time chunks
    class DummyCollection:
        def __init__(self):
            self.metadata = {'embed_dim': 3}
            self.name = 'dummy'
        def query(self, query_embeddings, n_results, include):
            # Return 3 hits: 1 schedule, 1 arrival+clock, 1 generic
            return {
                'documents': [[
                    'MORNING (8:00 AM - 12:00 PM)\nWelcome to the company!',
                    'Please ARRIVE at 9:00 AM at the main reception.\nCheck in with HR.',
                    'Lunch (12:00 PM - 1:00 PM)\nEnjoy your meal.'
                ]],
                'metadatas': [[
                    {'source_file': 'onboarding.txt', 'header': 'MORNING'},
                    {'source_file': 'onboarding.txt', 'header': 'ARRIVAL'},
                    {'source_file': 'onboarding.txt', 'header': 'LUNCH'}
                ]],
                'distances': [[0.2, 0.1, 0.3]],
                'ids': [[
                    'onboarding.txt_1',
                    'onboarding.txt_2',
                    'onboarding.txt_3'
                ]]
            }
    # Patch get_collection_async and embed_texts
    async def dummy_get_collection_async(tenant_id):
        return DummyCollection()
    async def dummy_embed_texts(texts, tenant_id=None):
        return [[0.1, 0.2, 0.3] for _ in texts]
    monkeypatch.setattr('app.services.rag_service.get_collection_async', dummy_get_collection_async)
    monkeypatch.setattr('app.services.rag_service.embed_texts', dummy_embed_texts)

    # Run query_collection for arrival intent
    gen, sources, evidence, context, debug_info = await query_collection(
        tenant_id='default',
        question='What time should I arrive on my first day?',
        top_k=2,
        mode='full',
        conversation_history=None,
        doc_ids=None,
        debug=1,
        request_id='test-req-1'
    )
    # Validate that the arrival+clock time chunk is force-included
    selected_chunks = debug_info['selected_chunks'] if 'selected_chunks' in debug_info else debug_info.get('selected', [])
    chunk_texts = [c['header_first_line'].lower() for c in selected_chunks]
    assert any('arrive' in t or 'reception' in t for t in chunk_texts), 'Arrival chunk not included!'
    assert any(c['contains_clock_time'] for c in selected_chunks), 'No clock time in selected chunks!'
    # Validate that selected size == top_k
    assert len(selected_chunks) == 2
    # Validate that the best (lowest dist) arrival+clock chunk is first
    assert selected_chunks[0]['header_first_line'].lower().startswith('please arrive') or 'arrive' in selected_chunks[0]['header_first_line'].lower()

@pytest.mark.asyncio
async def test_arrival_time_refusal(monkeypatch):
    # Simulate a collection with only schedule/generic chunks (no arrival+clock)
    class DummyCollection:
        def __init__(self):
            self.metadata = {'embed_dim': 3}
            self.name = 'dummy'
        def query(self, query_embeddings, n_results, include):
            return {
                'documents': [[
                    'MORNING (8:00 AM - 12:00 PM)\nWelcome!',
                    'Lunch (12:00 PM - 1:00 PM)\nEnjoy.',
                ]],
                'metadatas': [[
                    {'source_file': 'onboarding.txt', 'header': 'MORNING'},
                    {'source_file': 'onboarding.txt', 'header': 'LUNCH'}
                ]],
                'distances': [[0.2, 0.3]],
                'ids': [[
                    'onboarding.txt_1',
                    'onboarding.txt_2'
                ]]
            }
    async def dummy_get_collection_async(tenant_id):
        return DummyCollection()
    async def dummy_embed_texts(texts, tenant_id=None):
        return [[0.1, 0.2, 0.3] for _ in texts]
    monkeypatch.setattr('app.services.rag_service.get_collection_async', dummy_get_collection_async)
    monkeypatch.setattr('app.services.rag_service.embed_texts', dummy_embed_texts)

    # Run query_collection for arrival intent
    gen, sources, evidence, context, debug_info = await query_collection(
        tenant_id='default',
        question='What time should I arrive on my first day?',
        top_k=2,
        mode='full',
        conversation_history=None,
        doc_ids=None,
        debug=1,
        request_id='test-req-2'
    )
    # Should refuse (no clock time in any selected chunk)
    assert debug_info.get('refused') is True
    assert debug_info.get('refusal_reason') == 'NO_CLOCK_TIME_FOR_ARRIVAL'
