import { describe, it, expect } from 'vitest';
import { computeAnswerMode, labelForMode, tooltipForMode, tooltipForModeWithContext } from '../../utils/computeAnswerMode';

describe('computeAnswerMode', () => {
  it('returns NOT_FOUND when refused is true', () => {
    const mode = computeAnswerMode({ refused: true });
    expect(mode).toBe('NOT_FOUND');
    expect(labelForMode(mode)).toBe('NOT FOUND');
    expect(tooltipForMode(mode)).toBe('Model refused or no supported evidence.');
  });

  it('returns EXTRACTED when pipeline_marker startsWith EXTRACTOR_', () => {
    const mode = computeAnswerMode({ refused: false, pipeline_marker: 'EXTRACTOR_XYZ' });
    expect(mode).toBe('EXTRACTED');
    expect(labelForMode(mode)).toBe('EXTRACTED');
    expect(tooltipForMode(mode)).toBe('Answer synthesized from extracted evidence.');
  });

  it('returns EXTRACTED for specific extractor overrides', () => {
    expect(computeAnswerMode({ refused: false, pipeline_marker: 'EXTRACTOR_FACT_SINGLE' })).toBe('EXTRACTED');
    expect(computeAnswerMode({ refused: false, pipeline_marker: 'EXTRACTOR_EVIDENCE_FALLBACK' })).toBe('EXTRACTED');
    expect(computeAnswerMode({ refused: false, pipeline_marker: 'EXTRACTOR_FALLBACK' })).toBe('EXTRACTED');
  });

  it('returns EXTRACTED when debug_info.pipeline_marker indicates extractor', () => {
    const mode = computeAnswerMode({ refused: false, debug_info: { pipeline_marker: 'EXTRACTOR_ABC' } });
    expect(mode).toBe('EXTRACTED');
  });

  it('does not treat debug_info.extractor_used as authoritative (CITED without pipeline marker)', () => {
    const mode = computeAnswerMode({ refused: false, debug_info: { extractor_used: true } });
    expect(mode).toBe('CITED');
  });

  it('returns CITED by default', () => {
    const mode = computeAnswerMode({ refused: false });
    expect(mode).toBe('CITED');
    expect(labelForMode(mode)).toBe('CITED');
    expect(tooltipForMode(mode)).toBe('Answer supported by citations/evidence.');
  });

  it('returns CLARIFICATION when pipeline_marker is CLARIFICATION_REQUIRED', () => {
    const mode = computeAnswerMode({ refused: false, pipeline_marker: 'CLARIFICATION_REQUIRED' });
    expect(mode).toBe('CLARIFICATION');
    expect(labelForMode(mode)).toBe('CLARIFY');
    expect(tooltipForMode(mode)).toBe('Needs clarification to answer accurately.');
  });

  it('returns CLARIFICATION when debug_info.pipeline_marker is CLARIFICATION_REQUIRED', () => {
    const mode = computeAnswerMode({ refused: false, debug_info: { pipeline_marker: 'CLARIFICATION_REQUIRED' } });
    expect(mode).toBe('CLARIFICATION');
  });

  it('returns CLARIFICATION when needs_clarification is true', () => {
    const mode = computeAnswerMode({ refused: false, needs_clarification: true });
    expect(mode).toBe('CLARIFICATION');
  });

  it('tooltipForModeWithContext returns clarification message when needed', () => {
    const msg = tooltipForModeWithContext('CITED', { pipeline_marker: 'CLARIFICATION_REQUIRED' });
    expect(msg).toBe('Needs clarification to answer accurately.');

    const msg2 = tooltipForModeWithContext('CITED', { needs_clarification: true });
    expect(msg2).toBe('Needs clarification to answer accurately.');

    const msg3 = tooltipForModeWithContext('CITED', {
      debug_info: { pipeline_marker: 'CLARIFICATION_REQUIRED' },
    });
    expect(msg3).toBe('Needs clarification to answer accurately.');
  });

  it('tooltipForModeWithContext returns standard message when no clarification needed', () => {
    const msg = tooltipForModeWithContext('NOT_FOUND', { pipeline_marker: 'OTHER' });
    expect(msg).toBe('Model refused or no supported evidence.');
  });
});
