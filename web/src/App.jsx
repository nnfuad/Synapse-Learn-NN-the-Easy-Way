import React, { useState } from 'react';
import {
  Brain,
  Layers,
  Cpu,
  Clock,
  Sparkles,
  Code2,
  HelpCircle,
  FolderGit2,
  BookOpen,
  ChevronDown,
  ExternalLink,
  Zap,
  CheckCircle2,
  Compass,
  Eye,
  Activity,
  Sliders,
  Scale,
  Microscope,
  GitBranch,
} from 'lucide-react';

import HistoryTimeline from './components/HistoryTimeline.jsx';
import McCullochPittsSim from './components/McCullochPittsSim.jsx';
import PerceptronPlayground from './components/PerceptronPlayground.jsx';
import XorCrisisVisualizer from './components/XorCrisisVisualizer.jsx';
import CodeViewer from './components/CodeViewer.jsx';
import StudentQuiz from './components/StudentQuiz.jsx';

import Module01Visualizer from './components/modules/Module01Visualizer.jsx';
import Module02Visualizer from './components/modules/Module02Visualizer.jsx';
import Module03Visualizer from './components/modules/Module03Visualizer.jsx';
import Module04Visualizer from './components/modules/Module04Visualizer.jsx';
import Module05Visualizer from './components/modules/Module05Visualizer.jsx';
import Module06Visualizer from './components/modules/Module06Visualizer.jsx';
import Module07Visualizer from './components/modules/Module07Visualizer.jsx';
import Module08Visualizer from './components/modules/Module08Visualizer.jsx';
import Module09Visualizer from './components/modules/Module09Visualizer.jsx';

const ALL_MODULES = [
  { id: '00', title: '00. Historical Genesis & The Perceptron', status: 'Active (Implemented)', era: '1943 - 1969' },
  { id: '01', title: '01. Computational Graphs, Autograd & Reverse Chain Rule', status: 'Active (Implemented)', era: '1970 - 1986' },
  { id: '02', title: '02. Loss Geometry, Optimization & SOTA Optimizers', status: 'Active (Implemented)', era: '1950 - 2017' },
  { id: '03', title: '03. Generalization, Regularization & Normalization', status: 'Active (Implemented)', era: '2014 - 2019' },
  { id: '04', title: '04. Spatial Vision & Convolutions (LeNet to ConvNeXt)', status: 'Active (Implemented)', era: '1998 - 2022' },
  { id: '05', title: '05. Sequential Dynamics & Attention Spark (RNN/LSTM)', status: 'Active (Implemented)', era: '1997 - 2014' },
  { id: '06', title: '06. Modern Transformers & Attention Matrix (RoPE/KV)', status: 'Active (Implemented)', era: '2017 - 2024' },
  { id: '07', title: '07. Generative Paradigms (Diffusion & Flow Matching)', status: 'Active (Implemented)', era: '2020 - 2024' },
  { id: '08', title: '08. LLMs, Alignment & Reasoning (RLHF/DPO/GRPO/MoE)', status: 'Active (Implemented)', era: '2022 - 2026' },
  { id: '09', title: '09. Mechanistic Interpretability & Circuit Scans (SAE)', status: 'Active (Implemented)', era: '2024 - 2026' },
];

const MODULE_META = {
  '00': {
    badge: 'STAGE 1 OF 10',
    era: '1943 - 1969',
    title: 'The Genesis & The Atomic Neuron',
    subtitle: "From biological logic gates to Frank Rosenblatt's 1957 Perceptron and the devastating 1969 XOR crisis that triggered the First AI Winter.",
    tests: '6/6 Passing',
    scratch: 'NumPy 100%',
    sota: 'PyTorch 2.x',
    workspace: 'python/module00_history_and_perceptron/',
    notebook: 'notebooks/00_genesis_and_perceptrons.ipynb',
    tabs: [
      { id: 'timeline', label: 'Historical Odyssey', icon: Clock, color: 'text-blue-400' },
      { id: 'mcculloch', label: '1943 McCulloch-Pitts', icon: Cpu, color: 'text-cyan-400' },
      { id: 'perceptron', label: '1957 Perceptron 2D', icon: Brain, color: 'text-emerald-400' },
      { id: 'xor_warp', label: '1986 XOR Space Warp', icon: Layers, color: 'text-purple-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
      { id: 'quiz', label: 'Mastery Quiz', icon: HelpCircle, color: 'text-rose-400' },
    ],
  },
  '01': {
    badge: 'STAGE 2 OF 10',
    era: '1970 - 1986',
    title: 'Computational Graphs, Autograd & Reverse Chain Rule',
    subtitle: 'Linnainmaa 1970 reverse mode automatic differentiation, Rumelhart/Hinton/Williams 1986 backprop, and hydraulic DAG backpressure waves.',
    tests: '9/9 Passing',
    scratch: 'Python Value Engine',
    sota: 'torch.autograd.Function',
    workspace: 'python/module01_computational_graphs_autograd/',
    notebook: 'notebooks/01_computational_graphs_and_autograd.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive DAG Visualizer', icon: Activity, color: 'text-cyan-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
    ],
  },
  '02': {
    badge: 'STAGE 3 OF 10',
    era: '1950 - 2017',
    title: 'Loss Geometry, Optimization & SOTA Optimizers',
    subtitle: 'From Robbins-Monro 1951 SGD to Polyak 1964 Momentum, Kingma & Ba 2014 Adam, and Loshchilov & Hutter 2017 AdamW.',
    tests: '11/11 Passing',
    scratch: 'NumPy Optimizers',
    sota: 'PyTorch AdamW + Cosine',
    workspace: 'python/module02_optimization_and_loss_landscapes/',
    notebook: 'notebooks/02_optimization_and_loss_landscapes.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Loss Landscape', icon: Activity, color: 'text-amber-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
    ],
  },
  '03': {
    badge: 'STAGE 4 OF 10',
    era: '2014 - 2019',
    title: 'Generalization, Regularization & Normalization',
    subtitle: 'Dropout co-adaptation breaking, Santurkar loss landscape smoothing, modern LLaMA RMSNorm, and Decoupled AdamW.',
    tests: '6/6 Passing',
    scratch: 'NumPy 100%',
    sota: 'PyTorch Pre-LN',
    workspace: 'python/module03_generalization_and_normalization/',
    notebook: 'notebooks/03_generalization_and_normalization.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Visualizer', icon: Activity, color: 'text-cyan-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
    ],
  },
  '04': {
    badge: 'STAGE 5 OF 10',
    era: '1998 - 2022',
    title: 'Spatial Vision & Convolutions (CNNs)',
    subtitle: 'From 1998 LeNet-5 spatial translation invariance to the 2015 ResNet identity highway solving deep degradation, and 2022 ConvNeXt.',
    tests: '7/7 Passing',
    scratch: 'NumPy im2col',
    sota: 'ResNet / ConvNeXt',
    workspace: 'python/module04_spatial_vision_and_cnns/',
    notebook: 'notebooks/04_spatial_vision_and_cnns.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Visualizer', icon: Eye, color: 'text-emerald-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
    ],
  },
  '05': {
    badge: 'STAGE 6 OF 10',
    era: '1997 - 2014',
    title: 'Sequential Dynamics & Recurrent Memory',
    subtitle: 'Vanishing gradient exponential decay in vanilla RNNs vs the Hochreiter & Schmidhuber 1997 LSTM Constant Error Carousel highway and Bahdanau Attention.',
    tests: '8/8 Passing',
    scratch: 'NumPy LSTM Cell',
    sota: 'PyTorch Bahdanau Attention',
    workspace: 'python/module05_sequential_processing_and_attention/',
    notebook: 'notebooks/05_sequential_dynamics_and_attention.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Memory Tape & Spotlight', icon: Activity, color: 'text-cyan-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
    ],
  },
  '06': {
    badge: 'STAGE 7 OF 10',
    era: '2017 - 2024',
    title: 'The Modern Transformer Revolution & Attention Mechanics',
    subtitle: 'Vaswani 2017 Scaled Dot-Product Multi-Head Attention, Su 2021 RoPE 2D rotation geometry, and autoregressive KV-Cache acceleration.',
    tests: '7/7 Passing',
    scratch: 'NumPy Multi-Head Attention',
    sota: 'FlashAttention / PyTorch 2.x',
    workspace: 'python/module06_transformers_and_attention/',
    notebook: 'notebooks/06_transformers_and_attention_mechanics.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive QKV & KV-Cache Sandbox', icon: Sparkles, color: 'text-purple-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
    ],
  },
  '07': {
    badge: 'STAGE 8 OF 10',
    era: '2020 - 2024',
    title: 'Generative Paradigms (Diffusion & Flow Matching)',
    subtitle: 'From thermodynamic non-equilibrium & DDPM (2020) to Optimal Transport Continuous Flow Matching (2023).',
    tests: '10/10 Passing',
    scratch: 'NumPy & PyTorch',
    sota: 'OT-CFM / Flux',
    workspace: 'python/module07_generative_diffusion_flow/',
    notebook: 'notebooks/07_diffusion_and_flow_matching.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Visualizer', icon: Zap, color: 'text-cyan-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
      { id: 'quiz', label: 'Mastery Quiz', icon: HelpCircle, color: 'text-rose-400' },
    ],
  },
  '08': {
    badge: 'STAGE 9 OF 10',
    era: '2022 - 2026',
    title: 'Modern LLMs, Alignment & Reasoning',
    subtitle: 'Sparse Mixture of Experts (MoE), Direct Preference Optimization (DPO), and Critic-Free GRPO Test-Time Reasoning.',
    tests: '9/9 Passing',
    scratch: 'PyTorch Scratch',
    sota: 'DeepSeek-V3 / R1',
    workspace: 'python/module08_llm_reasoning_and_alignment/',
    notebook: 'notebooks/08_reasoning_and_alignment.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Visualizer', icon: Cpu, color: 'text-purple-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
      { id: 'quiz', label: 'Mastery Quiz', icon: HelpCircle, color: 'text-rose-400' },
    ],
  },
  '09': {
    badge: 'STAGE 10 OF 10',
    era: '2024 - 2026',
    title: 'Mechanistic Interpretability & Circuit Scans',
    subtitle: 'Anthropic Transformer Circuits, 2-layer induction heads, and Sparse Autoencoder (SAE) monosemantic feature dictionary extraction.',
    tests: '6/6 Passing',
    scratch: 'PyTorch Scratch',
    sota: 'Anthropic SAE',
    workspace: 'python/module09_mechanistic_interpretability/',
    notebook: 'notebooks/09_mechanistic_interpretability.ipynb',
    tabs: [
      { id: 'visualizer', label: 'Interactive Visualizer', icon: Microscope, color: 'text-rose-400' },
      { id: 'code', label: 'Code Laboratory', icon: Code2, color: 'text-amber-400' },
      { id: 'quiz', label: 'Mastery Quiz', icon: HelpCircle, color: 'text-rose-400' },
    ],
  },
};

export default function App() {
  const [currentModule, setCurrentModule] = useState('03');
  const [activeTab, setActiveTab] = useState('visualizer');
  const [showRoadmapModal, setShowRoadmapModal] = useState(false);

  const meta = MODULE_META[currentModule] || MODULE_META['00'];

  const handleModuleChange = (modId) => {
    setCurrentModule(modId);
    setShowRoadmapModal(false);
    if (modId === '00') {
      setActiveTab('perceptron');
    } else {
      setActiveTab('visualizer');
    }
  };

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-[#090d16]/90 backdrop-blur-md border-b border-slate-800/80 px-4 sm:px-8 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-tight text-white text-base sm:text-lg">SYNAPSE</span>
                <span className="text-[10px] uppercase font-mono font-bold px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                  Zero → SOTA
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                The High-Class Graphical Neural Network Explainer
              </p>
            </div>
          </div>

          {/* Module Switcher */}
          <div className="flex items-center gap-2">
            <div className="relative">
              <select
                value={currentModule}
                onChange={(e) => handleModuleChange(e.target.value)}
                className="bg-slate-900 border border-slate-700 hover:border-cyan-500/60 text-xs font-bold text-white px-3 py-1.5 rounded-xl appearance-none pr-8 cursor-pointer shadow-sm transition"
              >
                {ALL_MODULES.map((m) => (
                  <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200 py-1">
                    {m.title}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-2.5 pointer-events-none" />
            </div>

            <button
              onClick={() => setShowRoadmapModal(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-600/20 border border-cyan-500/40 hover:bg-cyan-600/30 text-xs font-semibold text-cyan-300 transition"
            >
              <Compass className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Roadmap</span>
              <span>(10/10)</span>
            </button>
          </div>
        </div>
      </header>

      {/* Hero Sub-header with Module Metadata */}
      <section className="bg-gradient-to-b from-[#0c1322] to-[#070b12] border-b border-slate-800/60 px-4 sm:px-8 py-8 sm:py-10">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-2 max-w-3xl">
              <div className="flex items-center gap-2 text-xs font-mono font-semibold text-cyan-400">
                <span>MODULE {currentModule}</span>
                <span>•</span>
                <span>ERA: {meta.era}</span>
                <span>•</span>
                <span className="text-emerald-400">{meta.badge}</span>
              </div>
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black text-white tracking-tight">
                {meta.title}
              </h1>
              <p className="text-slate-300 text-sm sm:text-base leading-relaxed pt-1">
                {meta.subtitle}
              </p>
            </div>

            {/* Workspace Stats Badge */}
            <div className="flex-shrink-0 grid grid-cols-2 sm:grid-cols-3 gap-3 bg-slate-900/80 border border-slate-800 p-3.5 rounded-2xl">
              <div className="text-center px-2">
                <div className="text-xs text-slate-400 font-mono">Scratch Tier</div>
                <div className="text-sm font-bold text-cyan-400 mt-0.5">{meta.scratch}</div>
              </div>
              <div className="text-center px-2 border-l border-slate-800">
                <div className="text-xs text-slate-400 font-mono">SOTA Tier</div>
                <div className="text-sm font-bold text-purple-400 mt-0.5">{meta.sota}</div>
              </div>
              <div className="text-center px-2 border-l border-slate-800 col-span-2 sm:col-span-1">
                <div className="text-xs text-slate-400 font-mono">Test Suite</div>
                <div className="text-sm font-bold text-emerald-400 mt-0.5">{meta.tests}</div>
              </div>
            </div>
          </div>

          {/* Interactive Navigation Tabs */}
          <div className="flex items-center gap-1.5 sm:gap-2 mt-8 overflow-x-auto pb-1 scrollbar-none">
            {meta.tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 sm:px-4 py-2.5 rounded-xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${
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
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-8 py-8">
        {/* Module 00 Views */}
        {currentModule === '00' && (
          <>
            {activeTab === 'timeline' && <HistoryTimeline />}
            {activeTab === 'mcculloch' && <McCullochPittsSim />}
            {activeTab === 'perceptron' && <PerceptronPlayground />}
            {activeTab === 'xor_warp' && <XorCrisisVisualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="00" />}
            {activeTab === 'quiz' && <StudentQuiz />}
          </>
        )}

        {/* Module 01 Views */}
        {currentModule === '01' && (
          <>
            {activeTab === 'visualizer' && <Module01Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="01" />}
          </>
        )}

        {/* Module 02 Views */}
        {currentModule === '02' && (
          <>
            {activeTab === 'visualizer' && <Module02Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="02" />}
          </>
        )}

        {/* Module 03 Views */}
        {currentModule === '03' && (
          <>
            {activeTab === 'visualizer' && <Module03Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="03" />}
          </>
        )}

        {/* Module 04 Views */}
        {currentModule === '04' && (
          <>
            {activeTab === 'visualizer' && <Module04Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="04" />}
          </>
        )}

        {/* Module 05 Views */}
        {currentModule === '05' && (
          <>
            {activeTab === 'visualizer' && <Module05Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="05" />}
          </>
        )}

        {/* Module 06 Views */}
        {currentModule === '06' && (
          <>
            {activeTab === 'visualizer' && <Module06Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="06" />}
          </>
        )}

        {/* Module 07 Views */}
        {currentModule === '07' && (
          <>
            {activeTab === 'visualizer' && <Module07Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="07" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="07" />}
          </>
        )}

        {/* Module 08 Views */}
        {currentModule === '08' && (
          <>
            {activeTab === 'visualizer' && <Module08Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="08" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="08" />}
          </>
        )}

        {/* Module 09 Views */}
        {currentModule === '09' && (
          <>
            {activeTab === 'visualizer' && <Module09Visualizer />}
            {activeTab === 'code' && <CodeViewer currentModule="09" />}
            {activeTab === 'quiz' && <StudentQuiz currentModule="09" />}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/60 py-6 px-4 sm:px-8 mt-12 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="font-bold text-slate-400">Project Synapse</span>
            <span>•</span>
            <span>Zero to State-of-the-Art Research</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400 font-mono">
            <span>Workspace: <code>{meta.workspace}</code></span>
            <span>•</span>
            <span>Notebook: <code>{meta.notebook}</code></span>
          </div>
        </div>
      </footer>

      {/* Curriculum Roadmap Modal */}
      {showRoadmapModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-white font-bold text-lg">
                <Compass className="w-5 h-5 text-cyan-400" />
                <span>Zero-to-SOTA Curriculum Roadmap</span>
              </div>
              <button
                onClick={() => setShowRoadmapModal(false)}
                className="text-slate-400 hover:text-white text-xs px-2.5 py-1 rounded-lg bg-slate-800"
              >
                Close
              </button>
            </div>

            <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2">
              {ALL_MODULES.map((m) => {
                const isImplemented = true;
                const isCurrent = m.id === currentModule;
                return (
                  <div
                    key={m.id}
                    onClick={() => handleModuleChange(m.id)}
                    className={`p-3 rounded-xl border flex items-center justify-between text-xs transition cursor-pointer hover:border-cyan-500/80 ${
                      isCurrent
                        ? 'bg-cyan-950/40 border-cyan-500/60 text-cyan-200'
                        : 'bg-slate-950/40 border-slate-800 text-slate-400'
                    }`}
                  >
                    <div>
                      <div className="font-bold text-slate-200">{m.title}</div>
                      <div className="text-[11px] text-slate-500 font-mono mt-0.5">Historical Era: {m.era}</div>
                    </div>
                    <span className="px-2 py-0.5 rounded font-mono text-[10px] font-bold bg-cyan-500/20 text-cyan-400">
                      {m.status}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
