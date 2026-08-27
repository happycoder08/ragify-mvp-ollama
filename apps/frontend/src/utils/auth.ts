/**
 * Authentication utilities for RAGify frontend.
 * Manages JWT token in localStorage.
 */

const TOKEN_KEY = 'ragify_jwt';

export interface JWTPayload {
  sub: string; // username
  tenant_id: string;
  exp: number; // expiration timestamp
}

/**
 * Decode JWT token (without verification - server validates).
 * Returns null if token is invalid or expired.
 */
export function decodeToken(token: string): JWTPayload | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    
    const payload = JSON.parse(atob(parts[1]));
    
    // Check expiration
    const now = Date.now();
    const expMs = (payload.exp || 0) * 1000;
    const isExpired = !!payload.exp && expMs < now;

    if (import.meta.env.DEV) {
      console.log('[Auth Check]', {
        tokenPresent: true,
        source: TOKEN_KEY,
        exp: payload.exp,
        now: Math.floor(now / 1000),
        expired: isExpired
      });
    }

    if (isExpired) {
      return null; // Token expired
    }
    
    return payload as JWTPayload;
  } catch (error) {
    console.error('Failed to decode token:', error);
    return null;
  }
}

/**
 * Get stored JWT token from localStorage.
 */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Store JWT token in localStorage.
 */
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

/**
 * Remove JWT token from localStorage.
 */
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/**
 * Check if user is logged in (has valid token).
 */
export function isLoggedIn(): boolean {
  const token = getToken();
  if (!token) {
    if (import.meta.env.DEV) console.log('[Auth Check] No token found in', TOKEN_KEY);
    return false;
  }
  
  const payload = decodeToken(token);
  if (payload === null) {
    clearToken(); // Auto-clear if expired/invalid
    return false;
  }
  return true;
}

/**
 * Get current user info from token.
 */
export function getCurrentUser(): JWTPayload | null {
  const token = getToken();
  if (!token) return null;
  return decodeToken(token);
}
