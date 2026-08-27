export type AnswerMode = 'NOT_FOUND' | 'EXTRACTED' | 'CITED' | 'CLARIFICATION';

export function computeAnswerMode(input: {
  refused: boolean;
  needs_clarification?: boolean;
  pipeline_marker?: string | null;
  debug_info?: any;
}): AnswerMode {
  if (input.refused === true) return 'NOT_FOUND';

  const pm = input.pipeline_marker;
  const debugPm = input.debug_info?.pipeline_marker;
  const needsClarification =
    input.needs_clarification === true ||
    pm === 'CLARIFICATION_REQUIRED' ||
    debugPm === 'CLARIFICATION_REQUIRED' ||
    input.debug_info?.needs_clarification === true;
  if (needsClarification) return 'CLARIFICATION';

  if (typeof pm === 'string' && pm.startsWith('EXTRACTOR_')) return 'EXTRACTED';

  const fallbackPm = debugPm;
  if (typeof fallbackPm === 'string' && fallbackPm.startsWith('EXTRACTOR_')) return 'EXTRACTED';

  return 'CITED';
}

export function labelForMode(mode: AnswerMode): 'NOT FOUND' | 'EXTRACTED' | 'CITED' | 'CLARIFY' {
  switch (mode) {
    case 'NOT_FOUND':
      return 'NOT FOUND';
    case 'EXTRACTED':
      return 'EXTRACTED';
    case 'CLARIFICATION':
      return 'CLARIFY';
    case 'CITED':
    default:
      return 'CITED';
  }
}

export function tooltipForMode(mode: AnswerMode): string {
  switch (mode) {
    case 'NOT_FOUND':
      return 'Model refused or no supported evidence.';
    case 'EXTRACTED':
      return 'Answer synthesized from extracted evidence.';
    case 'CLARIFICATION':
      return 'Needs clarification to answer accurately.';
    case 'CITED':
    default:
      return 'Answer supported by citations/evidence.';
  }
}

export function tooltipForModeWithContext(
  mode: AnswerMode,
  ctx: { needs_clarification?: boolean; pipeline_marker?: string | null; debug_info?: any }
): string {
  const debugMarker = ctx.debug_info?.pipeline_marker;
  const debugNeedsClarification = ctx.debug_info?.needs_clarification === true;
  if (mode === 'CLARIFICATION') {
    return 'Needs clarification to answer accurately.';
  }
  if (
    ctx.needs_clarification === true ||
    ctx.pipeline_marker === 'CLARIFICATION_REQUIRED' ||
    debugMarker === 'CLARIFICATION_REQUIRED' ||
    debugNeedsClarification
  ) {
    return 'Needs clarification to answer accurately.';
  }
  return tooltipForMode(mode);
}

export default { computeAnswerMode, labelForMode, tooltipForMode, tooltipForModeWithContext };
