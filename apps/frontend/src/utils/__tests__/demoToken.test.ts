import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getDemoToken, consumeDemoTokenFromUrl } from '../demoToken';

describe('demoToken helpers', () => {
  const originalLocation = window.location;
  const originalReplace = history.replaceState;

  beforeEach(() => {
    // Clear sessionStorage
    sessionStorage.clear();
  });

  afterEach(() => {
    // Restore mocks
    Object.defineProperty(window, 'location', { value: originalLocation, configurable: true });
    history.replaceState = originalReplace;
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it('consumes token from URL and clears query string', () => {
    // Mock location
    const fakeLocation: any = {
      pathname: '/docs',
      search: '?token=abc123',
      hash: '#x'
    };
    Object.defineProperty(window, 'location', { value: fakeLocation, configurable: true });

    const replaceSpy = vi.spyOn(history, 'replaceState');

    const token = consumeDemoTokenFromUrl();
    expect(token).toBe('abc123');
    expect(sessionStorage.getItem('ragify_demo_token')).toBe('abc123');
    expect(replaceSpy).toHaveBeenCalledWith(null, '', '/docs#x');
  });

  it('returns token from sessionStorage when URL has no token', () => {
    const fakeLocation: any = { pathname: '/docs', search: '', hash: '' };
    Object.defineProperty(window, 'location', { value: fakeLocation, configurable: true });
    sessionStorage.setItem('ragify_demo_token', 'stored-token');

    const replaceSpy = vi.spyOn(history, 'replaceState');

    const token = consumeDemoTokenFromUrl();
    expect(token).toBe('stored-token');
    expect(replaceSpy).not.toHaveBeenCalled();
  });

  it('getDemoToken reads sessionStorage', () => {
    sessionStorage.setItem('ragify_demo_token', 's2');
    expect(getDemoToken()).toBe('s2');
  });

  it('returns null when no token anywhere', () => {
    const fakeLocation: any = { pathname: '/docs', search: '', hash: '' };
    Object.defineProperty(window, 'location', { value: fakeLocation, configurable: true });
    expect(consumeDemoTokenFromUrl()).toBeNull();
    expect(getDemoToken()).toBeNull();
  });
});
