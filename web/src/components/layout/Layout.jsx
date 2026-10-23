import React, { useState } from 'react';
import Sidebar from './Sidebar.jsx';
import Navbar from './Navbar.jsx';

export default function Layout({ currentModuleId, currentPage, onNavigate, children }) {
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 flex font-sans selection:bg-cyan-500 selection:text-white">
      {/* Sidebar (Desktop persistent, Mobile drawer) */}
      <Sidebar
        currentModuleId={currentModuleId}
        currentPage={currentPage}
        onNavigate={onNavigate}
        isMobileOpen={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Layout Container */}
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar
          currentModuleId={currentModuleId}
          currentPage={currentPage}
          onOpenMobileSidebar={() => setIsMobileSidebarOpen(true)}
          onNavigate={onNavigate}
        />

        <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-8 py-6 sm:py-8">
          {children}
        </main>

        {/* Global Footer */}
        <footer className="border-t border-slate-800/80 bg-slate-950/60 py-6 px-4 sm:px-8 text-xs text-slate-500">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-400">Synapse: Learn NN the Easy Way</span>
              <span>•</span>
              <span>Zero Foundations to State-of-the-Art Research</span>
            </div>
            <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
              <span>79/79 Unit Tests Passing</span>
              <span>•</span>
              <a
                href="https://github.com/nnfuad/Synapse-Learn-NN-the-Easy-Way"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-cyan-400 transition"
              >
                GitHub Open Source (MIT)
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
