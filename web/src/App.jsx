import React from 'react';
import { useHashRoute } from './router/useHashRoute.js';
import Layout from './components/layout/Layout.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';

// Dedicated Page Views
import ModulePage from './pages/ModulePage.jsx';
import DashboardPage from './pages/DashboardPage.jsx';
import RoadmapPage from './pages/RoadmapPage.jsx';

export default function App() {
  const { route, navigate } = useHashRoute();

  // Active module ID for layout navigation
  const currentModuleId = route.page === 'module' ? route.id || '00' : '00';

  return (
    <Layout
      currentModuleId={currentModuleId}
      currentPage={route.page}
      onNavigate={navigate}
    >
      <ErrorBoundary key={route.page + (route.id || '')}>
        {(route.page === 'home' || route.page === 'dashboard') && <DashboardPage onNavigate={navigate} />}
        {route.page === 'roadmap' && <RoadmapPage onNavigate={navigate} />}
        {route.page === 'module' && (
          <ModulePage
            key={route.id || '00'}
            moduleId={route.id || '00'}
            onNavigate={navigate}
          />
        )}
      </ErrorBoundary>
    </Layout>
  );
}
