import { useState, useEffect } from 'react';

/**
 * Lightweight, robust URL Hash Router for Synapse.
 * Defaults to Home Page ('#/' or empty) upon opening the application.
 *
 * Supported Routes:
 *   - '' or '#/' or '#/home' or '#/dashboard' -> Home Page
 *   - '#/module/00' ... '#/module/09'        -> Dedicated Module Pages
 *   - '#/roadmap'                            -> 80-Year Timeline Explorer
 */
export function useHashRoute() {
  const parseHash = () => {
    const rawHash = window.location.hash.replace(/^#\/?/, '').trim();

    if (!rawHash || rawHash === 'home' || rawHash === 'dashboard') {
      return { raw: rawHash, page: 'home', id: null };
    }

    if (rawHash === 'roadmap') {
      return { raw: rawHash, page: 'roadmap', id: null };
    }

    if (rawHash.startsWith('module/')) {
      const parts = rawHash.split('/');
      const id = parts[1] || '00';
      return { raw: rawHash, page: 'module', id };
    }

    // Default fallback to Home
    return { raw: rawHash, page: 'home', id: null };
  };

  const [route, setRoute] = useState(parseHash);

  useEffect(() => {
    const handleHashChange = () => {
      setRoute(parseHash());
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigate = (path) => {
    if (!path || path === 'home' || path === 'dashboard' || path === '/') {
      window.location.hash = '#/';
    } else {
      window.location.hash = path.startsWith('#') ? path : `#/${path.replace(/^\//, '')}`;
    }
  };

  return { route, navigate };
}
