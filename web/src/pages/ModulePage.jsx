import React, { useState, useEffect } from 'react';
import {
  Brain,
  Clock,
  Sparkles,
  Code2,
  HelpCircle,
  Cpu,
  Layers,
  ArrowRight,
  ArrowLeft,
  BookOpen,
  Terminal,
  Activity,
  Eye,
  Zap,
  Microscope,
} from 'lucide-react';
import { MODULES_MAP, MODULES_LIST } from '../data/curriculumData.js';

// Visualizers
import HistoryTimeline from '../components/HistoryTimeline.jsx';
import McCullochPittsSim from '../components/McCullochPittsSim.jsx';
import PerceptronPlayground from '../components/PerceptronPlayground.jsx';
import XorCrisisVisualizer from '../components/XorCrisisVisualizer.jsx';
import CodeViewer from '../components/CodeViewer.jsx';
import StudentQuiz from '../components/StudentQuiz.jsx';

import Module01Visualizer from '../components/modules/Module01Visualizer.jsx';
import Module02Visualizer from '../components/modules/Module02Visualizer.jsx';
import Module03Visualizer from '../components/modules/Module03Visualizer.jsx';
import Module04Visualizer from '../components/modules/Module04Visualizer.jsx';
import Module05Visualizer from '../components/modules/Module05Visualizer.jsx';
import Module06Visualizer from '../components/modules/Module06Visualizer.jsx';
import Module07Visualizer from '../components/modules/Module07Visualizer.jsx';
import Module08Visualizer from '../components/modules/Module08Visualizer.jsx';
import Module09Visualizer from '../components/modules/Module09Visualizer.jsx';

export default function ModulePage({ moduleId, onNavigate }) {
  const mod = MODULES_MAP[moduleId] || MODULES_MAP['00'];

  // Default tab based on module
  const getDefaultTab = (mId) => (mId === '00' ? 'perceptron' : 'visualizer');
  const [activeTab, setActiveTab] = useState(() => getDefaultTab(moduleId));

  // Reset tab when moduleId changes
  useEffect(() => {
    setActiveTab(getDefaultTab(moduleId));
  }, [moduleId]);

  // Determine available tabs
  const tabs =
    moduleId === '00'
      ? [
          { id: 'perceptron', label: '1957 Perceptron 2D', icon: Brain, color: 'text-emerald-400' },
          { id: 'mcculloch', label: '1943 McCulloch-Pitts', icon: Cpu, color: 'text-cyan-400' },
          { id: 'xor_warp', label: '1986 XOR Space Warp', icon: Layers, color: 'text-purple-400' },
          { id: 'timeline', label: 'Historical Timeline', icon: Clock, color: 'text-blue-400' },
          { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
          { id: 'quiz', label: 'Mastery Quiz', icon: HelpCircle, color: 'text-rose-400' },
        ]
      : [
          { id: 'visualizer', label: 'Interactive Visualizer', icon: Activity, color: 'text-cyan-400' },
          { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
          { id: 'quiz', label: 'Mastery Quiz', icon: HelpCircle, color: 'text-rose-400' },
        ];

  // Prev / Next Module calculations
  const currentIndex = MODULES_LIST.findIndex((m) => m.id === moduleId);
  const prevModule = currentIndex > 0 ? MODULES_LIST[currentIndex - 1] : null;
  const nextModule = currentIndex < MODULES_LIST.length - 1 ? MODULES_LIST[currentIndex + 1] : null;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Module Hero Header */}
      <section className="bg-gradient-to-b from-[#0c1322] to-[#070b12] border border-slate-800/80 rounded-2xl p-6 sm:p-8 relative overflow-hidden shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2 max-w-3xl">
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono font-semibold text-cyan-400">
              <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                STAGE {mod.number} OF 10
              </span>
              <span>•</span>
              <span className="text-slate-400">ERA: {mod.era}</span>
              <span>•</span>
              <span className="text-purple-400">{mod.part}</span>
            </div>

            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-black text-white tracking-tight">
              {mod.title}
            </h1>

            <p className="text-slate-300 text-xs sm:text-sm leading-relaxed pt-1">
              {mod.subtitle}
            </p>

            {/* Sensory Physical Analogy Pill */}
            <div className="pt-2 flex items-start gap-2 text-xs text-amber-300/90 font-medium">
              <Sparkles className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <span><strong>Physical Analogy:</strong> {mod.analogy}</span>
            </div>
          </div>

          {/* Quick Specs HUD */}
          <div className="flex-shrink-0 grid grid-cols-3 gap-2 bg-slate-900/90 border border-slate-800 p-3 rounded-xl text-center font-mono">
            <div className="px-2">
              <div className="text-[10px] text-slate-500 uppercase">Scratch</div>
              <div className="text-xs font-bold text-cyan-400 mt-0.5">{mod.scratch}</div>
            </div>
            <div className="px-2 border-l border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase">SOTA</div>
              <div className="text-xs font-bold text-purple-400 mt-0.5">{mod.sota}</div>
            </div>
            <div className="px-2 border-l border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase">Tests</div>
              <div className="text-xs font-bold text-emerald-400 mt-0.5">{mod.tests}</div>
            </div>
          </div>
        </div>

        {/* Interactive Tabs Bar */}
        <div className="flex items-center gap-2 mt-6 overflow-x-auto pb-1 scrollbar-none border-t border-slate-800/80 pt-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${
                  isActive
                    ? 'bg-slate-800 text-white border border-slate-700 shadow-md shadow-cyan-500/5'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? tab.color : 'text-slate-500'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Main Tab View Content */}
      <section className="space-y-6">
        {/* Module 00 Views */}
        {moduleId === '00' && (
          <>
            {activeTab === 'perceptron' && <PerceptronPlayground />}
            {activeTab === 'mcculloch' && <McCullochPittsSim />}
            {activeTab === 'xor_warp' && <XorCrisisVisualizer />}
            {activeTab === 'timeline' && <HistoryTimeline />}
            {activeTab === 'code' && <CodeViewer currentModule="00" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="00" />}
          </>
        )}

        {/* Module 01 Views */}
        {moduleId === '01' && (
          <>
            {activeTab === 'visualizer' && <Module01Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="01" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="01" />}
          </>
        )}

        {/* Module 02 Views */}
        {moduleId === '02' && (
          <>
            {activeTab === 'visualizer' && <Module02Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="02" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="02" />}
          </>
        )}

        {/* Module 03 Views */}
        {moduleId === '03' && (
          <>
            {activeTab === 'visualizer' && <Module03Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="03" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="03" />}
          </>
        )}

        {/* Module 04 Views */}
        {moduleId === '04' && (
          <>
            {activeTab === 'visualizer' && <Module04Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="04" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="04" />}
          </>
        )}

        {/* Module 05 Views */}
        {moduleId === '05' && (
          <>
            {activeTab === 'visualizer' && <Module05Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="05" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="05" />}
          </>
        )}

        {/* Module 06 Views */}
        {moduleId === '06' && (
          <>
            {activeTab === 'visualizer' && <Module06Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="06" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="06" />}
          </>
        )}

        {/* Module 07 Views */}
        {moduleId === '07' && (
          <>
            {activeTab === 'visualizer' && <Module07Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="07" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="07" />}
          </>
        )}

        {/* Module 08 Views */}
        {moduleId === '08' && (
          <>
            {activeTab === 'visualizer' && <Module08Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="08" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="08" />}
          </>
        )}

        {/* Module 09 Views */}
        {moduleId === '09' && (
          <>
            {activeTab === 'visualizer' && <Module09Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="09" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="09" />}
          </>
        )}
      </section>

      {/* Chapter Bottom Navigation & Local Terminal Command Helper */}
      <section className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        {prevModule ? (
          <button
            onClick={() => onNavigate(`module/${prevModule.id}`)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-750 text-xs font-semibold text-slate-200 transition"
          >
            <ArrowLeft className="w-4 h-4 text-slate-400" />
            <div className="text-left">
              <div className="text-[10px] text-slate-500 uppercase font-mono">Previous Stage</div>
              <div>{prevModule.number}. {prevModule.title.split(':')[0]}</div>
            </div>
          </button>
        ) : (
          <div />
        )}

        {/* Terminal Run Helper */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-400">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>python3 {mod.workspace}</span>
        </div>

        {nextModule ? (
          <button
            onClick={() => onNavigate(`module/${nextModule.id}`)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white transition shadow-lg shadow-cyan-600/20"
          >
            <div className="text-right">
              <div className="text-[10px] text-cyan-200 uppercase font-mono">Next Stage</div>
              <div>{nextModule.number}. {nextModule.title.split(':')[0]}</div>
            </div>
            <ArrowRight className="w-4 h-4 text-white" />
          </button>
        ) : (
          <div />
        )}
      </section>
    </div>
  );
}
