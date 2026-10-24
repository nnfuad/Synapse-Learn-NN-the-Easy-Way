import { useState, useEffect, useCallback } from 'react';

/**
 * Normalizes any module ID input (e.g. 0, '0', '00', 'module/0', 1, '01')
 * into canonical 2-digit format ('00', '01', ... '09').
 */
export function normalizeModuleId(rawId) {
  if (rawId === undefined || rawId === null) return '00';
  const clean = String(rawId).trim().replace(/^module\/?/i, '').replace(/^#\/?/i, '');
  const num = parseInt(clean, 10);
  if (!isNaN(num) && num >= 0 && num <= 9) {
    return String(num).padStart(2, '0');
  }
  return '00';
}

/**
 * Parses either the URL hash (e.g. #/module/00, #module/0, #00)
 * OR the URL pathname (e.g. /module/00, /module/0, /roadmap)
 * guaranteeing seamless navigation whether accessed with or without hash.
 */
export function parseCurrentRoute() {
  if (typeof window === 'undefined') {
    return { raw: '', page: 'home', id: null };
  }

  const hash = window.location.hash || '';
  const pathname = window.location.pathname || '';

  // 1. Process hash if present
  const cleanHash = hash.replace(/^#\/?/, '').trim();

  // 2. Process pathname (stripping leading slash and index.html)
  const cleanPath = pathname
    .replace(/^\//, '')
    .replace(/index\.html$/i, '')
    .replace(/\/$/, '')
    .trim();

  // Candidate path: prefer hash if present and non-empty, otherwise fallback to pathname
  const candidate = (cleanHash && cleanHash !== '/' && cleanHash !== 'index.html')
    ? cleanHash
    : cleanPath;

  if (!candidate || candidate === 'home' || candidate === 'dashboard' || candidate === '/') {
    return { raw: candidate, page: 'home', id: null };
  }

  if (candidate === 'roadmap') {
    return { raw: candidate, page: 'roadmap', id: null };
  }

  // Matches "module/0", "module/00", "module/01", "00", "0", etc.
  const moduleMatch = candidate.match(/^module\/([0-9]{1,2})$/i) || candidate.match(/^([0-9]{1,2})$/);
  if (moduleMatch) {
    const id = normalizeModuleId(moduleMatch[1]);
    return { raw: candidate, page: 'module', id };
  }

  if (candidate.startsWith('module/')) {
    const parts = candidate.split('/');
    const id = normalizeModuleId(parts[1]);
    return { raw: candidate, page: 'module', id };
  }

  // Fallback to Home
  return { raw: candidate, page: 'home', id: null };
}

export function useHashRoute() {
  const [route, setRoute] = useState(parseCurrentRoute);

  useEffect(() => {
    const handleRouteSync = () => {
      setRoute(parseCurrentRoute());
    };

    window.addEventListener('hashchange', handleRouteSync);
    window.addEventListener('popstate', handleRouteSync);

    return () => {
      window.removeEventListener('hashchange', handleRouteSync);
      window.removeEventListener('popstate', handleRouteSync);
    };
  }, []);

  const navigate = useCallback((path) => {
    const target = (path || '').trim();

    if (!target || target === 'home' || target === 'dashboard' || target === '/') {
      window.location.hash = '#/';
      if (window.location.pathname !== '/' && window.location.pathname !== '/index.html') {
        window.history.pushState(null, '', '/#/');
      }
    } else {
      const formatted = target.startsWith('#') ? target : `#/${target.replace(/^\//, '')}`;
      window.location.hash = formatted;
      if (window.location.pathname !== '/' && window.location.pathname !== '/index.html') {
        window.history.pushState(null, '', `/${formatted}`);
      }
    }

    // Immediately trigger route update in React state
    setRoute(parseCurrentRoute());

    // Scroll to top of the new view smoothly
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

  return { route, navigate };
}
