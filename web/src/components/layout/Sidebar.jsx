import React from 'react';
import {
  Brain,
  Compass,
  CheckCircle2,
  Clock,
  Sparkles,
  ExternalLink,
  ChevronRight,
  BookOpen,
  Code2,
  Layers,
  Cpu,
  Zap,
  Activity,
  Eye,
  Microscope,
} from 'lucide-react';
import { CURRICULUM_SECTIONS, MODULES_MAP } from '../../data/curriculumData.js';

const MODULE_ICONS = {
  '00': Brain,
  '01': Activity,
  '02': Activity,
  '03': Activity,
  '04': Eye,
  '05': Clock,
  '06': Sparkles,
  '07': Zap,
  '08': Cpu,
  '09': Microscope,
};

export default function Sidebar({ currentModuleId, currentPage, onNavigate, isMobileOpen, onCloseMobile }) {
  const handleNav = (target) => {
    onNavigate(target);
    if (onCloseMobile) onCloseMobile();
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          onClick={onCloseMobile}
          className="fixed inset-0 z-40 bg-black/80 backdrop-blur-sm lg:hidden transition-opacity"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed lg:sticky top-0 left-0 z-50 h-screen w-72 bg-[#090d16] border-r border-slate-800/80 flex flex-col transition-transform duration-300 ease-in-out ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-800/80 flex items-center justify-between">
          <div
            onClick={() => handleNav('dashboard')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-tight text-white text-base">SYNAPSE</span>
                <span className="text-[9px] uppercase font-mono font-bold px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                  Zero→SOTA
                </span>
              </div>
              <p className="text-[10px] text-slate-400 font-medium">Neural Networks Explained</p>
            </div>
          </div>
        </div>

        {/* Navigation Quick Links */}
        <div className="px-3 pt-4 pb-2 space-y-1">
          <button
            onClick={() => handleNav('dashboard')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-semibold transition ${
              currentPage === 'home' || currentPage === 'dashboard'
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Compass className="w-4 h-4 text-cyan-400" />
            <span>Course Dashboard & Overview</span>
          </button>

          <button
            onClick={() => handleNav('roadmap')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-semibold transition ${
              currentPage === 'roadmap'
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Clock className="w-4 h-4 text-blue-400" />
            <span>80-Year Historical Timeline</span>
          </button>
        </div>

        {/* Modules Section List (Scrollable) */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-5 scrollbar-thin">
          {CURRICULUM_SECTIONS.map((section, sIdx) => (
            <div key={sIdx} className="space-y-1.5">
              <div className="px-3 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500">
                {section.title}
              </div>

              <div className="space-y-0.5">
                {section.modules.map((modId) => {
                  const mod = MODULES_MAP[modId];
                  const Icon = MODULE_ICONS[modId] || Brain;
                  const isActive = currentPage === 'module' && currentModuleId === modId;

                  return (
                    <button
                      key={modId}
                      onClick={() => handleNav(`module/${modId}`)}
                      className={`w-full text-left px-3 py-2 rounded-xl text-xs transition flex items-center justify-between group ${
                        isActive
                          ? 'bg-cyan-500/15 border border-cyan-500/50 text-cyan-200 font-bold shadow-md shadow-cyan-500/5'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-2.5 truncate">
                        <span
                          className={`w-5 h-5 rounded-lg flex items-center justify-center text-[10px] font-mono font-bold flex-shrink-0 ${
                            isActive
                              ? 'bg-cyan-500 text-slate-950 font-black'
                              : 'bg-slate-800 text-slate-400 group-hover:text-slate-200'
                          }`}
                        >
                          {mod.number}
                        </span>
                        <span className="truncate">{mod.title.split(': ')[0].replace(/^[0-9]{2}\.\s*/, '')}</span>
                      </div>
                      <ChevronRight
                        className={`w-3.5 h-3.5 flex-shrink-0 transition-transform ${
                          isActive ? 'text-cyan-400 translate-x-0.5' : 'text-slate-600 group-hover:text-slate-400'
                        }`}
                      />
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Footer Status */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-slate-400">Automated Tests:</span>
            <span className="text-emerald-400 font-bold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              79/79 Passing
            </span>
          </div>

          <a
            href="https://github.com/nnfuad/Synapse-Learn-NN-the-Easy-Way"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-slate-900 hover:bg-slate-850 border border-slate-800 text-xs text-slate-300 font-medium transition"
          >
            <span>GitHub Repository</span>
            <ExternalLink className="w-3.5 h-3.5 text-slate-500" />
          </a>
        </div>
      </aside>
    </>
  );
}
