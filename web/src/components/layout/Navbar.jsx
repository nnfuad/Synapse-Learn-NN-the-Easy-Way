import React from 'react';
import { Menu, ChevronRight, Github, ExternalLink, Compass, BookOpen, Terminal } from 'lucide-react';
import { MODULES_MAP, MODULES_LIST } from '../../data/curriculumData.js';

export default function Navbar({ currentModuleId, currentPage, onOpenMobileSidebar, onNavigate }) {
  const currentMod = MODULES_MAP[currentModuleId] || MODULES_MAP['00'];

  return (
    <header className="sticky top-0 z-30 bg-[#090d16]/90 backdrop-blur-md border-b border-slate-800/80 px-4 sm:px-8 py-3.5 flex items-center justify-between">
      {/* Left: Mobile Menu Button & Breadcrumbs */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobileSidebar}
          className="lg:hidden p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-white"
          aria-label="Open navigation sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Dynamic Breadcrumb */}
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span
            onClick={() => onNavigate('dashboard')}
            className="cursor-pointer hover:text-cyan-400 transition hidden sm:inline"
          >
            Synapse
          </span>
          <ChevronRight className="w-3.5 h-3.5 text-slate-600 hidden sm:inline" />
          {currentPage === 'module' ? (
            <>
              <span className="text-slate-400 hidden md:inline">{currentMod.part}</span>
              <ChevronRight className="w-3.5 h-3.5 text-slate-600 hidden md:inline" />
              <span className="text-white font-bold truncate max-w-[200px] sm:max-w-none">
                Stage {currentMod.number}: {currentMod.title}
              </span>
            </>
          ) : currentPage === 'roadmap' ? (
            <span className="text-white font-bold">Historical Roadmap (1943 - 2026)</span>
          ) : (
            <span className="text-white font-bold">Course Dashboard</span>
          )}
        </div>
      </div>

      {/* Right: Quick Module Jumper & GitHub Link */}
      <div className="flex items-center gap-3">
        {/* Module Fast Selector Dropdown */}
        <div className="relative">
          <select
            value={currentModuleId}
            onChange={(e) => onNavigate(`module/${e.target.value}`)}
            className="bg-slate-900 border border-slate-700/80 hover:border-cyan-500/60 text-xs font-bold text-white px-3 py-1.5 rounded-xl appearance-none pr-7 cursor-pointer transition shadow-sm"
          >
            {MODULES_LIST.map((m) => (
              <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200">
                Stage {m.number}: {m.title}
              </option>
            ))}
          </select>
          <span className="pointer-events-none absolute right-2.5 top-2.5 text-[10px] text-slate-400">▾</span>
        </div>

        <a
          href="https://github.com/nnfuad/Synapse-Learn-NN-the-Easy-Way"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/80 text-xs font-semibold text-slate-200 transition"
        >
          <Github className="w-3.5 h-3.5 text-white" />
          <span>Star on GitHub</span>
        </a>
      </div>
    </header>
  );
}
