import React from 'react';
import {
  Brain,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  BookOpen,
  Terminal,
  Clock,
  Code2,
  Activity,
  Layers,
  Cpu,
  Zap,
  Eye,
  Microscope,
} from 'lucide-react';
import { CURRICULUM_SECTIONS, MODULES_MAP } from '../data/curriculumData.js';

export default function DashboardPage({ onNavigate }) {
  return (
    <div className="space-y-10 animate-fadeIn">
      {/* Hero Welcome Banner */}
      <section className="bg-gradient-to-r from-blue-950/50 via-slate-900 to-purple-950/40 border border-slate-800 rounded-3xl p-6 sm:p-10 relative overflow-hidden shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-3xl space-y-3 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            Zero to State-of-the-Art Research (1943 → 2026)
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black text-white tracking-tight leading-tight">
            Master Neural Networks <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">The Visual Way</span>
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed pt-1">
            Demystifying artificial intelligence from the ground up. Every module pairs real physical analogies and 60 FPS interactive visualizers with zero-black-box pure NumPy scratch code and production PyTorch 2.x implementations.
          </p>

          <div className="pt-4 flex flex-wrap items-center gap-3">
            <button
              onClick={() => onNavigate('module/00')}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm transition shadow-lg shadow-cyan-500/20 flex items-center gap-2"
            >
              <span>Begin Stage 00: The Genesis (1943)</span>
              <ArrowRight className="w-4 h-4 text-slate-950" />
            </button>

            <button
              onClick={() => onNavigate('roadmap')}
              className="px-5 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm transition border border-slate-700 flex items-center gap-2"
            >
              <Clock className="w-4 h-4 text-blue-400" />
              <span>Explore 80-Year Timeline</span>
            </button>
          </div>
        </div>
      </section>

      {/* The 5-Step Learning Loop */}
      <section className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 sm:p-8 space-y-5">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">The 5-Step Learning Loop</h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            How to learn from this tool for deep, permanent conceptual mastery:
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 pt-2">
          {[
            { step: '01', title: 'The Crisis & Story', desc: 'Read why previous methods broke down and what historical crisis forced researchers to invent this.' },
            { step: '02', title: 'Interactive Visualizer', desc: 'Play with sliders, drag coordinates, and watch vectors and hyperplanes move before reading math.' },
            { step: '03', title: 'Scratch Python/NumPy', desc: 'Inspect every arithmetic operation from raw mathematical equations without framework magic.' },
            { step: '04', title: 'Production PyTorch', desc: 'See how frontier research labs write hardware-accelerated, modular production models.' },
            { step: '05', title: 'Notebooks & Quiz', desc: 'Experiment on custom datasets in Jupyter and validate your mental models with the concept quiz.' },
          ].map((s, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-mono font-bold text-cyan-400">Step {s.step}</span>
                <h3 className="text-sm font-bold text-white mt-1">{s.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed mt-1">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Curriculum Stages Grid */}
      <section className="space-y-6">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">Curriculum Stages (10 Chapters)</h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Click any stage to launch its dedicated interactive learning environment:
          </p>
        </div>

        <div className="space-y-8">
          {CURRICULUM_SECTIONS.map((section, sIdx) => (
            <div key={sIdx} className="space-y-3">
              <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-cyan-400">
                <span>{section.title}</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {section.modules.map((mId) => {
                  const m = MODULES_MAP[mId];
                  return (
                    <div
                      key={mId}
                      onClick={() => onNavigate(`module/${mId}`)}
                      className="p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 hover:border-cyan-500/50 cursor-pointer transition-all duration-200 group flex flex-col justify-between shadow-lg"
                    >
                      <div className="space-y-2.5">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                            STAGE {m.number}
                          </span>
                          <span className="text-[11px] font-mono text-slate-500">{m.era}</span>
                        </div>

                        <h3 className="text-base font-bold text-white group-hover:text-cyan-300 transition-colors">
                          {m.title}
                        </h3>

                        <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                          {m.subtitle}
                        </p>
                      </div>

                      <div className="pt-4 mt-4 border-t border-slate-800/80 flex items-center justify-between text-xs">
                        <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          {m.tests}
                        </span>

                        <span className="text-cyan-400 font-semibold group-hover:translate-x-1 transition-transform flex items-center gap-1">
                          Launch Stage →
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
