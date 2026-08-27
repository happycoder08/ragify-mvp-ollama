import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import * as sse from '../../sse'
import * as api from '../../api'
import { MemoryRouter } from 'react-router-dom'

// Mock import.meta.env before importing Query
vi.mock('import.meta.env', () => ({
  VITE_DEMO_MODE: 'true',
  VITE_SHOW_DEVTOOLS: 'true'
}))

// Mock listDocuments to avoid network calls
vi.mock('../../api')

// Enable dev tools for tests
;(globalThis as any).__TEST_SHOW_DEVTOOLS__ = true

import Query, { buildDisambiguatedQuestion } from '../Query'

describe('Query SSE behavior', () => {
  beforeEach(() => {
    // @ts-ignore
    api.listDocuments = vi.fn().mockResolvedValue({ documents: [] })
  })

  test('buildDisambiguatedQuestion appends option in parens', () => {
    expect(buildDisambiguatedQuestion('Original Q', '2026')).toBe('Original Q (2026)')
    expect(buildDisambiguatedQuestion('Who?', 'Me')).toBe('Who? (Me)')
  })

  test('includes conversation_id and capped history in /api/query request', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      // Immediately complete with a long answer to populate history
      handlers.onFinal?.({
        answer: 'A'.repeat(1200),
        refused: false,
        evidence: [],
        sources: [],
      })
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })

    // Use paste for instant update of long text to avoid timeout
    const longQuestion = 'Q'.repeat(1200)
    await userEvent.clear(textarea)
    await userEvent.paste(longQuestion)
    
    await userEvent.click(ask)

    await waitFor(() => {
      expect(mockQuery).toHaveBeenCalledTimes(1)
    })

    const firstRequest = mockQuery.mock.calls[0][0] as any
    expect(firstRequest.conversation_id).toBeDefined()
    expect(typeof firstRequest.conversation_id).toBe('number')
    expect(firstRequest.question.length).toBeLessThanOrEqual(800)
    expect(firstRequest.conversation_history ?? []).toHaveLength(0)

    // Second query should carry forward capped history with same conversation_id
    await userEvent.clear(textarea)
    await userEvent.type(textarea, 'Second turn question')
    await userEvent.click(ask)

    await waitFor(() => {
      expect(mockQuery).toHaveBeenCalledTimes(2)
    })

    const secondRequest = mockQuery.mock.calls[1][0] as any
    expect(secondRequest.conversation_id).toBe(firstRequest.conversation_id)
    const history = secondRequest.conversation_history as Array<{ role: string; content: string }>
    expect(Array.isArray(history)).toBe(true)
    // One prior turn -> up to 2 messages (user + assistant)
    expect(history.length).toBeGreaterThanOrEqual(1)
    expect(history.length).toBeLessThanOrEqual(2)
    for (const msg of history) {
      expect(msg.content.length).toBeLessThanOrEqual(800)
    }
  })

  test('final overwrites streamed tokens and refusal clears evidence', async () => {
    // Prepare a mock implementation of queryWithSSE that calls handlers
    const mockQuery = vi.fn((_request, handlers) => {
      // stream tokens
      setTimeout(() => handlers.onToken?.('ARRIVE '), 10)
      setTimeout(() => handlers.onToken?.('AT '), 20)
      setTimeout(() => handlers.onToken?.('8:00 '), 30)
      // then final (refused=true)
      setTimeout(() => handlers.onFinal?.({
        answer: 'The document does not specify this.',
        refused: true,
        refusal_reason: 'No matching info',
        evidence: [],
        sources: []
      }), 50)

      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    // Enter a question and submit
    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'What time do we arrive?')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    // After final arrives, the answer must be the canonical refusal message
    await waitFor(() => {
      expect(screen.getByText('The document does not specify this.')).toBeInTheDocument()
    })

    // Evidence panel should NOT be present when refused
    expect(screen.queryByText(/Employee_Onboarding_Guide_1.txt/)).not.toBeInTheDocument()
  })

  test('dev invariant shows when final missing evidence/sources', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      setTimeout(() => handlers.onFinal?.({
        answer: 'Some answer',
        refused: false,
        evidence: [],
        sources: [],
      }), 10)
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'Check invariant')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText(/DEV ERROR: final payload missing evidence or sources/)).toBeInTheDocument()
    })
  })

  test('clarification response skips dev invariant and follow-up renders answer', async () => {
    let callCount = 0
    const mockQuery = vi.fn((_request, handlers) => {
      callCount += 1
      if (callCount === 1) {
        setTimeout(() => handlers.onFinal?.({
          answer: '',
          refused: false,
          evidence: [],
          sources: [],
          needs_clarification: true,
          pipeline_marker: 'CLARIFICATION_REQUIRED',
          clarification: {
            type: 'policy_year',
            question: 'Which policy year?',
            options: ['2025', '2026']
          }
        }), 10)
      } else {
        setTimeout(() => handlers.onFinal?.({
          answer: 'Final answer for 2025',
          refused: false,
          evidence: [{ chunk_id: 'c1', snippet: 'Policy for 2025', heading: 'Policy' }],
          sources: [{ filename: 'policy.pdf' }]
        }), 10)
      }
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'What is the policy year?')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText('Which policy year?')).toBeInTheDocument()
    })

    expect(screen.queryByText(/DEV ERROR: final payload missing evidence or sources/)).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: '2025' }))

    await waitFor(() => {
      expect(screen.getByText('Final answer for 2025')).toBeInTheDocument()
    })

    expect(mockQuery).toHaveBeenCalledTimes(2)
    const followupRequest = mockQuery.mock.calls[1][0] as any
    expect(followupRequest.question).toBe('What is the policy year? (policy year: 2025)')
  })

  test('dev mismatch detected when answer time not in evidence', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      setTimeout(() => handlers.onFinal?.({
        answer: 'Please arrive at 8:00 AM',
        refused: false,
        evidence: [{ chunk_id: 'c1', snippet: 'No time here' }],
        sources: [{ filename: 'doc.txt' }],
      }), 10)
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'Check mismatch')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText('Answer/Evidence mismatch')).toBeInTheDocument()
    })
  })

  test('dev warning when LLM_VALIDATED has numeric evidence but no numeric answer', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      setTimeout(() => handlers.onFinal?.({
        answer: 'No numbers here',
        refused: false,
        pipeline_marker: 'LLM_VALIDATED',
        evidence: [{ chunk_id: 'c1', snippet: 'The total is 42 items.' }],
        sources: [{ filename: 'doc.txt' }],
      }), 10)
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'Check numeric invariant')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText('Validated answer missing numeric evidence')).toBeInTheDocument()
    })
  })

  test('dev success banner when EXTRACTOR_FALLBACK is used', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      setTimeout(() => handlers.onFinal?.({
        answer: 'Fallback extracted answer',
        refused: false,
        pipeline_marker: 'EXTRACTOR_FALLBACK',
        evidence: [{ chunk_id: 'c1', snippet: 'Strict extraction used' }],
        sources: [{ filename: 'doc.txt' }],
      }), 10)
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'Check fallback')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText('🛡️ Hallucination prevented: Fallback to strict extraction.')).toBeInTheDocument()
    })
  })

  test('dev flags canonical refusal text when refused=false', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      setTimeout(() => handlers.onFinal?.({
        answer: 'The document does not specify this.',
        refused: false,
        evidence: [{ chunk_id: 'c1', snippet: 'some snippet' }],
        sources: [{ filename: 'doc.txt' }],
      }), 10)
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'Check canonical refusal')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText(/DEV ERROR: final.answer equals canonical refusal while refused=false/)).toBeInTheDocument()
    })
  })

  test('copy answer + citations includes question, mode, sources, and all evidence', async () => {
    const mockQuery = vi.fn((_request, handlers) => {
      setTimeout(() => handlers.onFinal?.({
        answer: 'Final answer body',
        refused: false,
        evidence: [
          { chunk_id: 'c1', snippet: 'snippet one', heading: 'H1' },
          { chunk_id: 'c2', snippet: 'snippet two' },
        ],
        sources: [
          { filename: 'doc-one.txt' },
          { filename: 'doc-two.txt' },
        ],
      }), 10)
      return { abort: () => {}, done: Promise.resolve() }
    })

    // @ts-ignore
    vi.spyOn(sse, 'queryWithSSE').mockImplementation(mockQuery)

    const writeText = vi.fn().mockResolvedValue(undefined)
    // @ts-ignore
    navigator.clipboard = { writeText }

    render(
      <MemoryRouter>
        <Query />
      </MemoryRouter>
    )

    const textarea = screen.getByPlaceholderText(/What is the company vacation policy/i)
    await userEvent.type(textarea, 'What is the company vacation policy?')
    const ask = screen.getByRole('button', { name: /Ask|Asking/i })
    await userEvent.click(ask)

    await waitFor(() => {
      expect(screen.getByText('Final answer body')).toBeInTheDocument()
    })

    const copyButton = screen.getByRole('button', { name: 'Copy answer + citations' })
    await userEvent.click(copyButton)

    expect(writeText).toHaveBeenCalledTimes(1)
    const copied = writeText.mock.calls[0][0] as string

    expect(copied).toContain('Question: What is the company vacation policy?')
    expect(copied).toMatch(/Answer Mode: (EXTRACTED|CITED|NOT FOUND|CLARIFY)/)
    expect(copied).toContain('Answer: Final answer body')
    expect(copied).toContain('Sources:')
    expect(copied).toContain('doc-one.txt')
    expect(copied).toContain('doc-two.txt')
    expect(copied).toContain('[1] H1 (chunk_id=c1)')
    expect(copied).toContain('snippet one')
    expect(copied).toContain('[2] Evidence 2 (chunk_id=c2)')
    expect(copied).toContain('snippet two')
  })
})
