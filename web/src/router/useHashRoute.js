import { useState, useEffect } from 'react';

/**
 * Lightweight, robust URL Hash Router for Synapse.
 * Supports:
 *   - '#/' or '' -> Defaults to '#/module/00' (or dashboard)
 *   - '#/module/00' ... '#/module/09'
 *   - '#/roadmap'
 *   - '#/dashboard'
 */
export function useHashRoute() {
  const parseHash = () => {
    const rawHash = window.location.hash.replace(/^#\/?/, '') || 'module/00';
    const parts = rawHash.split('/');
    const page = parts[0] || 'module';
    const id = parts[1] || '00';
    return { raw: rawHash, page, id };
  };

  const [route, setRoute] = useState(parseHash);

  useEffect(() => {
    // If user opens root http://localhost:5173/ without hash, redirect to #/module/00
    if (!window.location.hash || window.location.hash === '#' || window.location.hash === '#/') {
      window.location.hash = '#/module/00';
    }

    const handleHashChange = () => {
      setRoute(parseHash());
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const navigate = (path) => {
    window.location.hash = path.startsWith('#') ? path : `#/${path.replace(/^\//, '')}`;
  };

  return { route, navigate };
}
