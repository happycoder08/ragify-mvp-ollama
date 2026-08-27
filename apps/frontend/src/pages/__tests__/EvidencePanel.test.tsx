import { render, fireEvent, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import EvidencePanel from '../EvidencePanel';

describe('EvidencePanel collapse/expand', () => {
  const makeItem = (i: number) => ({ snippet: `snippet ${i}`, chunk_id: `chunk-${i}`, heading: `H${i}`, doc_id: i });

  it('renders 2 items by default when evidence length >= 3', () => {
    const items = [makeItem(1), makeItem(2), makeItem(3)];
    const { container } = render(<EvidencePanel evidence={items} query="" refused={false} sources={[]} />);
    const itemsRendered = container.querySelectorAll('.evidence-item');
    expect(itemsRendered.length).toBe(2);
  });

  it('toggle shows all N items and back to 2', () => {
    const items = [makeItem(1), makeItem(2), makeItem(3)];
    render(<EvidencePanel evidence={items} query="" refused={false} sources={[]} />);

    const showAllBtn = screen.getByText('Show all (3)');
    fireEvent.click(showAllBtn);
    expect(document.querySelectorAll('.evidence-item').length).toBe(3);

    const showLessBtn = screen.getByText('Show less');
    fireEvent.click(showLessBtn);
    expect(document.querySelectorAll('.evidence-item').length).toBe(2);
  });

  it('renders "No evidence returned." when evidence is empty and not refused', () => {
    render(<EvidencePanel evidence={[]} query="" refused={false} sources={[]} />);
    expect(screen.getByText('No evidence returned.')).toBeTruthy();
  });

  it('transitions between refusal and evidence states without hook-order errors', () => {
    const evidence = [{ snippet: 'Policy evidence', chunk_id: 'chunk-1' }];
    const view = render(<EvidencePanel evidence={[]} query="" refused={true} sources={[]} />);

    expect(screen.getByText('No evidence available')).toBeInTheDocument();

    view.rerender(<EvidencePanel evidence={evidence} query="policy" refused={false} sources={[]} />);
    expect(screen.getByText('Evidence (1)')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(view.container.querySelector('.evidence-snippet')).toHaveTextContent('Policy evidence');

    view.rerender(<EvidencePanel evidence={[]} query="policy" refused={true} sources={[]} />);
    expect(screen.getByText('No evidence available')).toBeInTheDocument();
  });
});
