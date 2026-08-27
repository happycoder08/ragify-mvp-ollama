/**
 * Document selection state management for active document filtering
 * Stores selected doc IDs in localStorage
 */

const SELECTED_DOCS_KEY = 'ragify_selected_docs';

/**
 * Get selected document IDs from localStorage
 * Returns empty array if none selected (means "All docs")
 */
export function getSelectedDocIds(): number[] {
  try {
    const stored = localStorage.getItem(SELECTED_DOCS_KEY);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('Failed to get selected docs:', error);
    return [];
  }
}

/**
 * Set selected document IDs in localStorage
 */
export function setSelectedDocIds(docIds: number[]): void {
  localStorage.setItem(SELECTED_DOCS_KEY, JSON.stringify(docIds));
}

/**
 * Toggle a document ID in the selection
 */
export function toggleDocId(docId: number): number[] {
  const selected = getSelectedDocIds();
  const index = selected.indexOf(docId);
  
  if (index > -1) {
    // Remove it
    selected.splice(index, 1);
  } else {
    // Add it
    selected.push(docId);
  }
  
  setSelectedDocIds(selected);
  return selected;
}

/**
 * Clear all selections (default to "All docs")
 */
export function clearSelection(): void {
  localStorage.removeItem(SELECTED_DOCS_KEY);
}

/**
 * Check if a document ID is selected
 */
export function isDocSelected(docId: number): boolean {
  const selected = getSelectedDocIds();
  return selected.includes(docId);
}
