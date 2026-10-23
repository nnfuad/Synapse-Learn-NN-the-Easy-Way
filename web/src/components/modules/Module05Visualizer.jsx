import React, { useState, useEffect } from 'react';
import {
  Brain,
  Layers,
  Sparkles,
  ArrowRight,
  ArrowLeft,
  Play,
  Pause,
  RotateCcw,
  Sliders,
  Eye,
  Activity,
  Zap,
  Info,
  CheckCircle2,
  Lock,
  Unlock,
} from 'lucide-react';

export default function Module05Visualizer() {
  const [activeSubTab, setActiveSubTab] = useState('tape');

  // --- SubTab 1: Recurrent Tape & Vanishing Gradient ---
  const [isLstmMode, setIsLstmMode] = useState(false); // false = Vanilla RNN, true = LSTM CEC
  const [currentStep, setCurrentStep] = useState(7); // current time step (0 to 7)
  const [isPlaying, setIsPlaying] = useState(false);
  const totalSteps = 8;
  const decayRate = 0.72; // Vanilla RNN spectral decay factor

  // Auto-play backward gradient pulse
  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => (prev > 0 ? prev - 1 : 7));
      }, 700);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  // Compute gradient signal strength at step t (backpropagated from step 7)
  const getGradientSignal = (step) => {
    if (isLstmMode) {
      // In LSTM CEC, signal is maintained along cell state highway
      return 1.0 - (7 - step) * 0.035; // minimal loss (~0.76 at t=0)
    } else {
      // In Vanilla RNN, exponential decay (decayRate ^ (7 - step))
      return Math.pow(decayRate, 7 - step);
    }
  };

  // --- SubTab 2: LSTM Gate Controls ---
  const [forgetGate, setForgetGate] = useState(0.92);
  const [inputGate, setInputGate] = useState(0.45);
  const [candidateCell, setCandidateCell] = useState(0.8);
  const [outputGate, setOutputGate] = useState(0.75);
  const [prevCellState, setPrevCellState] = useState(1.2);

  // Derived calculations
  const cellState = forgetGate * prevCellState + inputGate * candidateCell;
  const tanhCell = Math.tanh(cellState);
  const hiddenState = outputGate * tanhCell;

  // Preset setter
  const setPreset = (type) => {
    if (type === 'preserve') {
      setForgetGate(1.0);
      setInputGate(0.0);
      setOutputGate(0.8);
    } else if (type === 'flush') {
      setForgetGate(0.0);
      setInputGate(1.0);
      setCandidateCell(0.9);
      setOutputGate(0.85);
    } else if (type === 'mute') {
      setOutputGate(0.0);
    } else if (type === 'balanced') {
      setForgetGate(0.85);
      setInputGate(0.5);
      setCandidateCell(0.6);
      setOutputGate(0.7);
    }
  };

  // --- SubTab 3: Bahdanau Attention Matrix ---
  const sourceTokens = ['Le', 'chat', 'noir', 'dort', 'paisiblement'];
  const targetTokens = ['The', 'black', 'cat', 'sleeps', 'peacefully'];
  const [selectedTargetIdx, setSelectedTargetIdx] = useState(1); // Default: "black" -> focuses on "noir"

  // Pre-calculated attention alignment matrix (Target rows x Source columns)
  const alignmentMatrix = [
    [0.86, 0.05, 0.04, 0.03, 0.02], // The -> Le
    [0.04, 0.14, 0.76, 0.04, 0.02], // black -> noir (Adjective inversion!)
    [0.02, 0.81, 0.11, 0.04, 0.02], // cat -> chat
    [0.02, 0.03, 0.03, 0.88, 0.04], // sleeps -> dort
    [0.01, 0.02, 0.02, 0.05, 0.90], // peacefully -> paisiblement
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Historical Context */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 relative overflow-hidden backdrop-blur-sm">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono font-semibold text-cyan-400 mb-2">
              <span>MODULE 05</span>
              <span>•</span>
              <span>ERA: 1997 - 2014</span>
              <span>•</span>
              <span className="text-emerald-400">HOCHREITER & SCHMIDHUBER → BAHDANAU</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
              Sequential Processing & Recurrent Dynamics
            </h2>
            <p className="text-slate-300 text-sm max-w-3xl mt-1.5 leading-relaxed">
              Experience the 1990s <span className="text-rose-400 font-semibold">Vanishing Gradient Crisis</span> in Vanilla RNNs,
              uncover how the <span className="text-cyan-400 font-semibold">LSTM Constant Error Carousel (CEC)</span> created an
              unattenuated memory highway, and see how <span className="text-purple-400 font-semibold">Bahdanau Attention</span> shattered
              the fixed-length bottleneck by turning sequential memory into an interactive spotlight.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <span className="px-3 py-1 rounded-xl bg-slate-800/80 border border-slate-700/60 text-xs font-mono text-slate-300">
              PyPure NumPy 100%
            </span>
            <span className="px-3 py-1 rounded-xl bg-purple-950/40 border border-purple-500/40 text-xs font-mono text-purple-300">
              PyTorch Seq2Seq
            </span>
          </div>
        </div>

        {/* Sub-Tabs Navigation */}
        <div className="flex items-center gap-2 mt-6 border-t border-slate-800/80 pt-4 overflow-x-auto pb-1 scrollbar-none">
          {[
            { id: 'tape', label: '1. Vanishing Gradient Tape', icon: Activity, color: 'text-rose-400' },
            { id: 'highway', label: '2. LSTM Highway & Gate Valves', icon: Sliders, color: 'text-cyan-400' },
            { id: 'attention', label: '3. Bahdanau Attention Spotlight', icon: Eye, color: 'text-purple-400' },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeSubTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${
                  active
                    ? 'bg-slate-800 text-white border border-slate-700 shadow-md shadow-cyan-500/5'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? tab.color : 'text-slate-500'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* --- SUBTAB 1: UNROLLED RECURRENT TAPE & VANISHING GRADIENT SIMULATOR --- */}
      {activeSubTab === 'tape' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-rose-400" />
                    Unrolled Temporal Memory Tape (t = 0 ... 7)
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Inject a loss gradient dL/dh₇ = 1.0 at the final step and watch it backpropagate to step 0.
                  </p>
                </div>

                {/* Model Architecture Toggle */}
                <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800 self-start sm:self-auto">
                  <button
                    onClick={() => setIsLstmMode(false)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      !isLstmMode
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Vanilla RNN
                  </button>
                  <button
                    onClick={() => setIsLstmMode(true)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      isLstmMode
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    LSTM (Constant Error Carousel)
                  </button>
                </div>
              </div>

              {/* The Unrolled Chain Diagram */}
              <div className="relative py-8 px-2 overflow-x-auto">
                {/* Upper Cell State Highway (for LSTM) */}
                {isLstmMode && (
                  <div className="relative mb-6 pb-2">
                    <div className="absolute top-1/2 left-4 right-4 h-2 bg-gradient-to-r from-cyan-500/40 via-cyan-400/80 to-cyan-500 -translate-y-1/2 rounded-full shadow-[0_0_12px_rgba(6,182,212,0.4)]" />
                    <div className="flex justify-between relative z-10 px-6 text-[10px] font-mono text-cyan-300 uppercase tracking-wider font-bold">
                      <span>Highway Start (c₀)</span>
                      <span>Constant Error Carousel: ∂cₜ / ∂cₜ₋₁ = fₜ ≈ 1.0</span>
                      <span>Highway End (c₇)</span>
                    </div>
                  </div>
                )}

                {/* Time Step Nodes */}
                <div className="grid grid-cols-8 gap-2 min-w-[620px]">
                  {Array.from({ length: totalSteps }).map((_, step) => {
                    const signal = getGradientSignal(step);
                    const isActive = step === currentStep;
                    const isTarget = step === 7;

                    // Compute dynamic color and opacity based on gradient signal
                    let bgStyle, borderStyle, textColor, glowStyle;
                    if (isLstmMode) {
                      bgStyle = `rgba(6, 182, 212, ${Math.max(0.15, signal * 0.4)})`;
                      borderStyle = `rgba(6, 182, 212, ${Math.max(0.3, signal)})`;
                      textColor = 'text-cyan-300';
                      glowStyle = `0 0 ${signal * 18}px rgba(6, 182, 212, ${signal * 0.6})`;
                    } else {
                      bgStyle = `rgba(244, 63, 94, ${Math.max(0.05, signal * 0.8)})`;
                      borderStyle = `rgba(244, 63, 94, ${Math.max(0.15, signal)})`;
                      textColor = signal > 0.1 ? 'text-rose-300' : 'text-slate-600';
                      glowStyle = `0 0 ${signal * 22}px rgba(244, 63, 94, ${signal * 0.7})`;
                    }

                    return (
                      <div
                        key={step}
                        onClick={() => setCurrentStep(step)}
                        className={`cursor-pointer transition-all duration-300 flex flex-col items-center p-3 rounded-2xl border text-center ${
                          isActive ? 'ring-2 ring-white scale-105 z-20' : ''
                        }`}
                        style={{
                          backgroundColor: bgStyle,
                          borderColor: borderStyle,
                          boxShadow: glowStyle,
                        }}
                      >
                        <div className="text-[10px] font-mono text-slate-400 font-bold">
                          Step t = {step}
                        </div>

                        {/* Cell Icon / State */}
                        <div className="my-3 w-10 h-10 rounded-xl bg-slate-900/90 border border-slate-700/80 flex items-center justify-center font-bold font-mono text-sm shadow-inner">
                          {isLstmMode ? (
                            <span className="text-cyan-400">c_{step}</span>
                          ) : (
                            <span className="text-rose-400">h_{step}</span>
                          )}
                        </div>

                        {/* Signal Magnitude */}
                        <div className={`text-xs font-mono font-bold ${textColor}`}>
                          {signal < 0.001 ? signal.toExponential(1) : signal.toFixed(3)}
                        </div>
                        <div className="text-[9px] text-slate-500 font-mono mt-0.5">
                          {isTarget ? 'Target' : 'Signal'}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Flow Direction Indicator */}
                <div className="flex items-center justify-between text-xs text-slate-500 font-mono mt-6 px-4">
                  <div className="flex items-center gap-1.5 text-cyan-400">
                    <ArrowLeft className="w-3.5 h-3.5" />
                    <span>Backpropagation Through Time (BPTT Flow ←)</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-slate-400">
                    <span>Forward Sequence Processing (t=0 → t=7)</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                </div>
              </div>

              {/* Playback & Step Controls */}
              <div className="flex flex-wrap items-center justify-between gap-4 mt-6 pt-6 border-t border-slate-800">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition shadow-lg shadow-cyan-500/20"
                  >
                    {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    <span>{isPlaying ? 'Pause Pulse' : 'Play Gradient Pulse'}</span>
                  </button>

                  <button
                    onClick={() => {
                      setIsPlaying(false);
                      setCurrentStep(7);
                    }}
                    className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition text-xs"
                    title="Reset to Step 7"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <span className="text-xs text-slate-400 font-mono whitespace-nowrap">
                    Active Step: <strong className="text-white">t = {currentStep}</strong>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="7"
                    value={currentStep}
                    onChange={(e) => {
                      setIsPlaying(false);
                      setCurrentStep(parseInt(e.target.value));
                    }}
                    className="w-full sm:w-36 accent-cyan-400 cursor-pointer"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Mathematical Card & Physical Metaphor */}
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-base font-bold text-white flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-cyan-400" />
                The Telephone Game Metaphor
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                {isLstmMode ? (
                  <>
                    <strong className="text-cyan-400">The Conveyor Belt Highway:</strong> The LSTM builds an explicit parallel conveyor belt (<code className="text-cyan-300">c_t</code>).
                    The cell does not force gradients through repeated weight matrix squashing. Instead, additive gates act as mechanical valves.
                    When the forget gate is open (<code className="text-cyan-300">f_t ≈ 1</code>), the gradient slides down the belt across hundreds of steps completely unharmed!
                  </>
                ) : (
                  <>
                    <strong className="text-rose-400">Diminishing Notepad:</strong> In a vanilla RNN, 50 people whisper a secret down a line.
                    Each person summarizes what they heard on a tiny paper note (<code className="text-rose-300">W_hh</code>).
                    By person 30, the original message is completely lost.
                    Mathematically, the gradient is repeatedly multiplied by <code className="text-rose-300">W_hh^T</code>, decaying to ~0.00003!
                  </>
                )}
              </p>

              <div className="mt-4 p-4 rounded-xl bg-slate-950/80 border border-slate-800 font-mono text-xs space-y-2">
                <div className="text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
                  Jacobian Product Across Time:
                </div>
                {isLstmMode ? (
                  <div className="text-cyan-300 text-sm font-bold">
                    ∂c_t / ∂c_{'{t-1}'} = f_t ≈ 1.0 (No Matrix!)
                  </div>
                ) : (
                  <div className="text-rose-300 text-sm font-bold">
                    ∂h_T / ∂h_t = ∏_{'{k=t+1}'}^T (diag(1-h_k²) W_hh^T)
                  </div>
                )}
                <div className="text-slate-400 text-[11px]">
                  Current signal at step {currentStep}:{' '}
                  <strong className={isLstmMode ? 'text-cyan-400' : 'text-rose-400'}>
                    {(getGradientSignal(currentStep) * 100).toFixed(2)}%
                  </strong>{' '}
                  of original gradient.
                </div>
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-3">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <Info className="w-4 h-4 text-amber-400" />
                Historical Milestone (1997)
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Published by <strong>Sepp Hochreiter & Jürgen Schmidhuber</strong> in <em>Neural Computation</em>.
                In 2000, Felix Gers added the <strong>Forget Gate</strong>, allowing the network to continually reset its own memory buffer
                when sequences switch contexts.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* --- SUBTAB 2: LSTM HIGHWAY & GATE CONTROLS --- */}
      {activeSubTab === 'highway' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Interactive Gate Valves & Sliders */}
          <div className="lg:col-span-1 space-y-4 bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                Gate Valves & Potentiometers
              </h3>
            </div>
            <p className="text-xs text-slate-400">
              Adjust each gate valve to see how information flows onto the conveyor belt highway.
            </p>

            {/* Quick Presets */}
            <div className="flex flex-wrap gap-1.5 pt-1 pb-2">
              <button
                onClick={() => setPreset('preserve')}
                className="px-2.5 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-[11px] font-mono text-cyan-300 hover:bg-cyan-500/20 transition"
              >
                Keep Memory (f=1)
              </button>
              <button
                onClick={() => setPreset('flush')}
                className="px-2.5 py-1 rounded-lg bg-rose-500/10 border border-rose-500/30 text-[11px] font-mono text-rose-300 hover:bg-rose-500/20 transition"
              >
                Flush & Overwrite
              </button>
              <button
                onClick={() => setPreset('mute')}
                className="px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-[11px] font-mono text-slate-300 hover:bg-slate-700 transition"
              >
                Mute Output (o=0)
              </button>
            </div>

            {/* Forget Gate Slider */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-purple-400">Forget Gate (f_t)</span>
                <span className="font-mono text-purple-300">{forgetGate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={forgetGate}
                onChange={(e) => setForgetGate(parseFloat(e.target.value))}
                className="w-full accent-purple-400 cursor-pointer"
              />
              <div className="text-[10px] text-slate-500">
                Valve controlling past memory retention: <code className="text-slate-400">f_t · c_{'{t-1}'}</code>
              </div>
            </div>

            {/* Input Gate Slider */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-emerald-400">Input Gate (i_t)</span>
                <span className="font-mono text-emerald-300">{inputGate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={inputGate}
                onChange={(e) => setInputGate(parseFloat(e.target.value))}
                className="w-full accent-emerald-400 cursor-pointer"
              />
              <div className="text-[10px] text-slate-500">
                Valve controlling fresh candidate write: <code className="text-slate-400">i_t · ~c_t</code>
              </div>
            </div>

            {/* Candidate Memory Slider */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-amber-400">Candidate Memory (~c_t)</span>
                <span className="font-mono text-amber-300">{candidateCell.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-1.0"
                max="1.0"
                step="0.05"
                value={candidateCell}
                onChange={(e) => setCandidateCell(parseFloat(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer"
              />
              <div className="text-[10px] text-slate-500">
                Tanh proposal of incoming cargo: <code className="text-slate-400">tanh(W_c x_t + U_c h_{'{t-1}'})</code>
              </div>
            </div>

            {/* Output Gate Slider */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-cyan-400">Output Gate (o_t)</span>
                <span className="font-mono text-cyan-300">{outputGate.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={outputGate}
                onChange={(e) => setOutputGate(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="text-[10px] text-slate-500">
                Valve revealing state to outside world: <code className="text-slate-400">h_t = o_t · tanh(c_t)</code>
              </div>
            </div>

            {/* Previous Cell State Slider */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-slate-300">Prior Cell State (c_{'{t-1}'})</span>
                <span className="font-mono text-slate-200">{prevCellState.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.1"
                value={prevCellState}
                onChange={(e) => setPrevCellState(parseFloat(e.target.value))}
                className="w-full accent-slate-300 cursor-pointer"
              />
            </div>
          </div>

          {/* Interactive Schematic Diagram & Energy Meters */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                <Activity className="w-5 h-5 text-cyan-400" />
                The Constant Error Carousel Highway Schematic
              </h3>

              {/* Graphic Flow Representation */}
              <div className="relative p-6 bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden">
                {/* Upper Highway (c_t) */}
                <div className="p-4 rounded-xl bg-slate-900 border border-cyan-500/40 relative shadow-lg shadow-cyan-500/10">
                  <div className="flex items-center justify-between text-xs font-mono font-bold text-cyan-300 mb-2">
                    <span className="flex items-center gap-1.5">
                      <Zap className="w-4 h-4 text-cyan-400" />
                      CELL STATE HIGHWAY: c_t = f_t · c_{'{t-1}'} + i_t · ~c_t
                    </span>
                    <span className="text-base text-white font-mono bg-cyan-950 px-3 py-1 rounded-lg border border-cyan-500/50">
                      c_t = {cellState.toFixed(3)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs font-mono text-slate-400 mt-3 pt-3 border-t border-slate-800">
                    <div>
                      Retained from Past: <strong className="text-purple-400">{(forgetGate * prevCellState).toFixed(3)}</strong>
                    </div>
                    <div>
                      Added from New Input: <strong className="text-emerald-400">{(inputGate * candidateCell).toFixed(3)}</strong>
                    </div>
                  </div>
                </div>

                {/* Vertical Bridge down to Hidden State */}
                <div className="flex justify-center my-4">
                  <div className="w-0.5 h-8 bg-gradient-to-b from-cyan-500 to-amber-500" />
                </div>

                {/* Lower Hidden State (h_t) */}
                <div className="p-4 rounded-xl bg-slate-900 border border-amber-500/40 relative shadow-lg shadow-amber-500/10">
                  <div className="flex items-center justify-between text-xs font-mono font-bold text-amber-300 mb-2">
                    <span className="flex items-center gap-1.5">
                      <Activity className="w-4 h-4 text-amber-400" />
                      HIDDEN STATE OUTPUT: h_t = o_t · tanh(c_t)
                    </span>
                    <span className="text-base text-white font-mono bg-amber-950 px-3 py-1 rounded-lg border border-amber-500/50">
                      h_t = {hiddenState.toFixed(3)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs font-mono text-slate-400 mt-3 pt-3 border-t border-slate-800">
                    <div>
                      Output Gate Valve (o_t): <strong className="text-cyan-400">{outputGate.toFixed(2)}</strong>
                    </div>
                    <div>
                      Squashed Cell tanh(c_t): <strong className="text-amber-400">{tanhCell.toFixed(3)}</strong>
                    </div>
                  </div>
                </div>
              </div>

              {/* Real-time Dynamic Energy Bar Meters */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-purple-400 font-bold mb-1">Forget Valve</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-500 transition-all duration-200"
                      style={{ width: `${forgetGate * 100}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono font-bold text-white mt-1.5">{(forgetGate * 100).toFixed(0)}%</div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-emerald-400 font-bold mb-1">Input Valve</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 transition-all duration-200"
                      style={{ width: `${inputGate * 100}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono font-bold text-white mt-1.5">{(inputGate * 100).toFixed(0)}%</div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-cyan-400 font-bold mb-1">Output Valve</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-500 transition-all duration-200"
                      style={{ width: `${outputGate * 100}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono font-bold text-white mt-1.5">{(outputGate * 100).toFixed(0)}%</div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-amber-400 font-bold mb-1">Hidden Charge</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 transition-all duration-200"
                      style={{ width: `${Math.abs(hiddenState) * 100}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono font-bold text-white mt-1.5">{hiddenState.toFixed(2)}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* --- SUBTAB 3: BAHDANAU ATTENTION SPOTLIGHT --- */}
      {activeSubTab === 'attention' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Eye className="w-5 h-5 text-purple-400" />
                    Bahdanau Additive Attention Spotlight
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Click any target word below to see the dynamic attention beam shine across source encoder representations.
                  </p>
                </div>

                <span className="px-3 py-1 rounded-xl bg-purple-500/10 border border-purple-500/30 text-xs font-mono text-purple-300">
                  ICLR 2015 Landmark
                </span>
              </div>

              {/* Target Words Selector Pills */}
              <div className="space-y-2 mb-6">
                <div className="text-xs font-mono text-slate-400 uppercase tracking-wider font-semibold">
                  Decoder Output Step (Click Target Word):
                </div>
                <div className="flex flex-wrap gap-2">
                  {targetTokens.map((token, idx) => {
                    const isSelected = selectedTargetIdx === idx;
                    return (
                      <button
                        key={idx}
                        onClick={() => setSelectedTargetIdx(idx)}
                        className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-mono font-bold transition ${
                          isSelected
                            ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/30 ring-2 ring-purple-400'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        <span>Step {idx}:</span>
                        <span>"{token}"</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Dynamic Attention Spotlight Beams */}
              <div className="p-6 bg-slate-950 rounded-2xl border border-slate-800 space-y-6">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <span>Decoder Query State: <strong className="text-purple-400">"{targetTokens[selectedTargetIdx]}"</strong></span>
                  <span>∑ α_ij = 1.000 (Softmax Normalized)</span>
                </div>

                {/* Source Tokens Beams */}
                <div className="grid grid-cols-5 gap-3">
                  {sourceTokens.map((token, srcIdx) => {
                    const weight = alignmentMatrix[selectedTargetIdx][srcIdx];
                    const isDominant = weight > 0.5;

                    return (
                      <div
                        key={srcIdx}
                        className={`p-3 rounded-xl border flex flex-col items-center justify-between text-center transition-all duration-300 ${
                          isDominant
                            ? 'bg-purple-950/40 border-purple-500 text-white shadow-lg shadow-purple-500/20 scale-105'
                            : 'bg-slate-900 border-slate-800 text-slate-400'
                        }`}
                      >
                        <span className="text-[10px] font-mono text-slate-500">Key h_{srcIdx}</span>
                        <div className="text-sm font-bold my-2 font-mono text-white">"{token}"</div>

                        {/* Weight Pill */}
                        <div
                          className={`text-xs font-mono font-bold px-2 py-0.5 rounded-full ${
                            isDominant
                              ? 'bg-purple-500 text-white'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {(weight * 100).toFixed(1)}%
                        </div>

                        {/* Bar representation */}
                        <div className="w-full h-1.5 bg-slate-800 rounded-full mt-2.5 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-purple-500 to-cyan-400 transition-all duration-300"
                            style={{ width: `${weight * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* The 2D Alignment Matrix Heatmap View */}
              <div className="mt-6 pt-6 border-t border-slate-800">
                <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400 font-semibold mb-3">
                  Full 2D Alignment Matrix Heatmap:
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 text-left text-slate-500">Target ↓ \ Source →</th>
                        {sourceTokens.map((s, i) => (
                          <th key={i} className="p-2 text-center text-slate-300 font-bold">
                            {s}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {targetTokens.map((t, rowIdx) => (
                        <tr
                          key={rowIdx}
                          onClick={() => setSelectedTargetIdx(rowIdx)}
                          className={`cursor-pointer transition ${
                            selectedTargetIdx === rowIdx ? 'bg-purple-950/30' : 'hover:bg-slate-850'
                          }`}
                        >
                          <td className="p-2 font-bold text-white border-t border-slate-800/60">
                            {t}
                          </td>
                          {alignmentMatrix[rowIdx].map((val, colIdx) => {
                            const opacity = Math.max(0.08, val);
                            const isCellSelected = selectedTargetIdx === rowIdx;
                            return (
                              <td
                                key={colIdx}
                                className="p-2 text-center border-t border-slate-800/60"
                                style={{
                                  backgroundColor: isCellSelected
                                    ? `rgba(168, 85, 247, ${val * 0.7})`
                                    : `rgba(6, 182, 212, ${val * 0.4})`,
                                }}
                              >
                                <span className={val > 0.4 ? 'text-white font-bold' : 'text-slate-400'}>
                                  {val.toFixed(2)}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Contextual & Mathematical Card */}
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-base font-bold text-white flex items-center gap-2 mb-3">
                <Zap className="w-4 h-4 text-purple-400" />
                The Adjective Inversion Phenomenon
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Observe step 1 where the decoder produces the English word <strong className="text-purple-300">"black"</strong>.
                In French, the input was <em>"Le chat noir"</em> (noun before adjective).
                Notice how the attention spotlight jumps forward to <strong className="text-white">"noir" (76%)</strong>,
                bypassing <em>"chat"</em>!
                Then on step 2 (<strong className="text-purple-300">"cat"</strong>), it looks backward to <strong className="text-white">"chat" (81%)</strong>.
              </p>

              <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs space-y-2">
                <div className="text-purple-400 text-[11px] uppercase tracking-wider font-semibold">
                  Additive Alignment Score:
                </div>
                <div className="text-white font-bold">
                  e_ij = v_aᵀ tanh(W_a s_{'{i-1}'} + U_a h_j)
                </div>
                <div className="text-slate-400 text-[11px] pt-1">
                  Softmax across source length produces probabilities α_ij. Dynamic context vector:{' '}
                  <code className="text-cyan-400">c_i = ∑ α_ij h_j</code>.
                </div>
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-3">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Why This Changed AI History
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Before Bahdanau Attention, machine translation systems evaluated on sentences longer than 20 words experienced catastrophic
                performance collapse. Attention completely decoupled sequence length from vector capacity, directly paving the way for the
                2017 Transformer revolution.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
