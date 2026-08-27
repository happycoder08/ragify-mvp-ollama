// Helpers for consuming a demo token from the URL and sessionStorage
const STORAGE_KEY = 'ragify_demo_token';

export function getDemoToken(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

// Reads `token` from window.location.search if present, stores it into sessionStorage
// and removes the query string from the address bar via history.replaceState.
// If no token in URL, returns value read from sessionStorage.
export function consumeDemoTokenFromUrl(): string | null {
  try {
    const search = window.location.search || '';
    const params = new URLSearchParams(search.replace(/^\?/, ''));
    const token = params.get('token');
    if (token) {
      try {
        sessionStorage.setItem(STORAGE_KEY, token);
      } catch {}

      // Remove query string while preserving pathname and hash
      const pathname = window.location.pathname || '';
      const hash = window.location.hash || '';
      try {
        history.replaceState(null, '', pathname + hash);
      } catch {}

      return token;
    }

    return getDemoToken();
  } catch {
    return null;
  }
}

export default { getDemoToken, consumeDemoTokenFromUrl };
