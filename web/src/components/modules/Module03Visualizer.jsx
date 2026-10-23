import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Sliders,
  Play,
  RotateCcw,
  Zap,
  Activity,
  Layers,
  Sparkles,
  BookOpen,
  Scale,
  Cpu,
  BarChart3,
  CheckCircle2,
  TrendingDown,
  Info,
  Maximize2,
} from 'lucide-react';

export default function Module03Visualizer() {
  const [activeSubTab, setActiveSubTab] = useState('distributions');

  // Distribution Engine State
  const [inputMean, setInputMean] = useState(4.5);
  const [inputStd, setInputStd] = useState(3.2);
  const [normType, setNormType] = useState('rms'); // 'raw', 'bn', 'ln', 'rms'
  const [gamma, setGamma] = useState(1.0);
  const [beta, setBeta] = useState(0.0);

  // Dropout Strobe Engine State
  const [dropoutP, setDropoutP] = useState(0.4);
  const [isTraining, setIsTraining] = useState(true);
  const [isStrobing, setIsStrobing] = useState(false);
  const [strobeSeed, setStrobeSeed] = useState(0);

  // Auto strobe timer
  useEffect(() => {
    let interval;
    if (isStrobing) {
      interval = setInterval(() => {
        setStrobeSeed((s) => s + 1);
      }, 350);
    }
    return () => clearInterval(interval);
  }, [isStrobing]);

  // Generate synthetic sample vector for distributions
  const { rawData, normData, stats } = useMemo(() => {
    const N = 400;
    // Box-Muller transform for deterministic Gaussian pseudo-randoms
    const raw = [];
    for (let i = 0; i < N; i++) {
      const u1 = Math.max(1e-6, ((i * 1234567 + 98765) % 100000) / 100000);
      const u2 = ((i * 7654321 + 54321) % 100000) / 100000;
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      raw.push(inputMean + z * inputStd);
    }

    const mean = raw.reduce((a, b) => a + b, 0) / N;
    const variance = raw.reduce((a, b) => a + (b - mean) ** 2, 0) / N;
    const std = Math.sqrt(variance + 1e-6);
    const rms = Math.sqrt(raw.reduce((a, b) => a + b * b, 0) / N + 1e-6);

    let normalized = [];
    if (normType === 'raw') {
      normalized = raw;
    } else if (normType === 'bn' || normType === 'ln') {
      // (x - mean) / std * gamma + beta
      normalized = raw.map((x) => ((x - mean) / std) * gamma + beta);
    } else if (normType === 'rms') {
      // x / rms * gamma (No bias/centering)
      normalized = raw.map((x) => (x / rms) * gamma);
    }

    const normMean = normalized.reduce((a, b) => a + b, 0) / N;
    const normVariance = normalized.reduce((a, b) => a + (b - normMean) ** 2, 0) / N;
    const normStd = Math.sqrt(normVariance + 1e-6);
    const normRms = Math.sqrt(normalized.reduce((a, b) => a + b * b, 0) / N + 1e-6);

    return {
      rawData: raw,
      normData: normalized,
      stats: {
        rawMean: mean,
        rawStd: std,
        rawRms: rms,
        normMean,
        normStd,
        normRms,
      },
    };
  }, [inputMean, inputStd, normType, gamma, beta]);

  // Compute histogram bins for normalized data
  const bins = useMemo(() => {
    const numBins = 28;
    const minVal = -4.0;
    const maxVal = 4.0;
    const binWidth = (maxVal - minVal) / numBins;
    const counts = new Array(numBins).fill(0);

    normData.forEach((val) => {
      const idx = Math.floor((val - minVal) / binWidth);
      if (idx >= 0 && idx < numBins) {
        counts[idx]++;
      }
    });

    const maxCount = Math.max(...counts, 1);
    return counts.map((count, i) => ({
      x: minVal + (i + 0.5) * binWidth,
      height: (count / maxCount) * 100,
      count,
    }));
  }, [normData]);

  // Generate 2-layer Dropout Neural Circuit state
  const circuit = useMemo(() => {
    const numInput = 6;
    const numHidden1 = 8;
    const numHidden2 = 8;
    const numOutput = 4;

    const getMask = (count, layerId) => {
      return Array.from({ length: count }, (_, i) => {
        if (!isTraining) return true; // Inference mode: all active
        // Pseudo-random based on strobeSeed
        const seedVal = Math.sin(strobeSeed * 37.1 + layerId * 19.3 + i * 7.7);
        const rand = (seedVal - Math.floor(seedVal));
        return rand >= dropoutP;
      });
    };

    const maskH1 = getMask(numHidden1, 1);
    const maskH2 = getMask(numHidden2, 2);

    return {
      numInput,
      numHidden1,
      numHidden2,
      numOutput,
      maskH1,
      maskH2,
      scalingFactor: isTraining && dropoutP < 1.0 ? (1.0 / (1.0 - dropoutP)).toFixed(2) : '1.00',
    };
  }, [dropoutP, isTraining, strobeSeed]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Sub-navigation pill menu */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-4 overflow-x-auto">
        {[
          { id: 'distributions', label: '1. Distribution Engine (RMSNorm vs LN/BN)', icon: BarChart3 },
          { id: 'dropout', label: '2. Dropout Strobe & Co-Adaptation', icon: Zap },
          { id: 'adamw', label: '3. AdamW Decoupled Decay Sandbox', icon: Scale },
          { id: 'research', label: '4. Research Hall of Fame (Citations)', icon: BookOpen },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${
                isActive
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* ========================================================================================= */}
      {/* TAB 1: DISTRIBUTION ENGINE */}
      {/* ========================================================================================= */}
      {activeSubTab === 'distributions' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Controls Panel */}
          <div className="lg:col-span-4 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 space-y-6">
            <div>
              <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono font-bold uppercase tracking-wider">
                <Sliders className="w-4 h-4" />
                <span>Input Activation Shift</span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">Upstream Drift Simulator</h3>
              <p className="text-xs text-slate-400 mt-1">
                Simulate how upstream unnormalized weight changes throw activations into arbitrary mean and variance shifts.
              </p>
            </div>

            {/* Normalization Mode Selector */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300">Normalization Mode:</label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'raw', name: 'Raw Unnormalized', tag: 'Shifted' },
                  { id: 'bn', name: 'BatchNorm (2015)', tag: 'Batch (B)' },
                  { id: 'ln', name: 'LayerNorm (2016)', tag: 'Channels (D)' },
                  { id: 'rms', name: 'RMSNorm (2019)', tag: 'LLaMA SOTA' },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => setNormType(mode.id)}
                    className={`p-2.5 rounded-xl border text-left transition ${
                      normType === mode.id
                        ? 'bg-cyan-500/15 border-cyan-500/60 text-white shadow-md shadow-cyan-500/10'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <div className="text-xs font-bold">{mode.name}</div>
                    <div className="text-[10px] text-cyan-400/80 font-mono">{mode.tag}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Mean Shift Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Upstream Mean Shift (μ):</span>
                <span className="font-mono font-bold text-cyan-400">{inputMean.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="-8.0"
                max="8.0"
                step="0.2"
                value={inputMean}
                onChange={(e) => setInputMean(parseFloat(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            {/* Standard Deviation Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Upstream Std Dev (σ):</span>
                <span className="font-mono font-bold text-cyan-400">{inputStd.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="6.0"
                step="0.1"
                value={inputStd}
                onChange={(e) => setInputStd(parseFloat(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            {/* Learnable Affine Scale (Gamma) */}
            <div className="space-y-1.5 border-t border-slate-800/80 pt-4">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Learnable Scale (γ):</span>
                <span className="font-mono font-bold text-purple-400">{gamma.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="2.5"
                step="0.05"
                value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                className="w-full accent-purple-500 cursor-pointer"
              />
            </div>

            {/* Learnable Shift (Beta) - only active if not RMSNorm */}
            {normType !== 'rms' && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Learnable Shift (β):</span>
                  <span className="font-mono font-bold text-emerald-400">{beta.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-2.0"
                  max="2.0"
                  step="0.1"
                  value={beta}
                  onChange={(e) => setBeta(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer"
                />
              </div>
            )}

            {normType === 'rms' && (
              <div className="p-3 bg-purple-950/20 border border-purple-800/40 rounded-xl text-[11px] text-purple-300">
                <span className="font-bold">⚡ RMSNorm Efficiency:</span> No learnable bias $\beta$ is allocated. By skipping mean centering and shift additions, modern LLMs save 10-50% computational wall-clock time!
              </div>
            )}
          </div>

          {/* Right SVG Histogram & Stats */}
          <div className="lg:col-span-8 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-cyan-400" />
                    <span>Real-Time Activation Distribution</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Live 400-sample activation profile under{' '}
                    <span className="font-mono text-cyan-400 uppercase font-bold">{normType}</span>
                  </p>
                </div>

                <button
                  onClick={() => {
                    setInputMean(0.0);
                    setInputStd(1.0);
                    setGamma(1.0);
                    setBeta(0.0);
                  }}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset Normal</span>
                </button>
              </div>

              {/* Live SVG Histogram */}
              <div className="relative mt-6 h-64 w-full bg-[#070b14] border border-slate-800/60 rounded-xl p-4 flex items-end">
                {/* Reference center line (0.0) */}
                <div className="absolute top-0 bottom-0 left-1/2 w-px border-l border-dashed border-slate-600/50 z-0"></div>

                {/* Histogram Bars */}
                <div className="w-full h-full flex items-end justify-between gap-1 z-10">
                  {bins.map((bin, idx) => (
                    <div key={idx} className="flex-1 flex flex-col items-center justify-end h-full group relative">
                      <div
                        style={{ height: `${Math.max(4, bin.height)}%` }}
                        className={`w-full rounded-t-sm transition-all duration-150 ${
                          normType === 'rms'
                            ? 'bg-gradient-to-t from-purple-600/40 to-cyan-400 shadow-sm shadow-cyan-500/20'
                            : normType === 'raw'
                            ? 'bg-gradient-to-t from-rose-600/40 to-rose-400'
                            : 'bg-gradient-to-t from-emerald-600/40 to-emerald-400'
                        }`}
                      ></div>
                      {/* Tooltip on hover */}
                      <div className="absolute bottom-full mb-1 hidden group-hover:block bg-slate-900 border border-slate-700 text-[10px] text-white px-2 py-0.5 rounded shadow-lg pointer-events-none whitespace-nowrap z-20">
                        x: {bin.x.toFixed(2)} | count: {bin.count}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* X Axis Labels */}
              <div className="flex justify-between text-[11px] font-mono text-slate-500 px-2 mt-2">
                <span>-4.0σ</span>
                <span>-2.0σ</span>
                <span className="text-cyan-400 font-bold">0.0 (Zero Centered)</span>
                <span>+2.0σ</span>
                <span>+4.0σ</span>
              </div>
            </div>

            {/* Live Metrics Comparison Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6 pt-4 border-t border-slate-800/80">
              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <div className="text-[11px] text-slate-400 font-mono">Sample Mean (μ)</div>
                <div className="text-base font-black text-cyan-400 mt-1">
                  {stats.normMean.toFixed(3)}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Raw: {stats.rawMean.toFixed(2)}</div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <div className="text-[11px] text-slate-400 font-mono">Std Dev (σ)</div>
                <div className="text-base font-black text-emerald-400 mt-1">
                  {stats.normStd.toFixed(3)}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Raw: {stats.rawStd.toFixed(2)}</div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <div className="text-[11px] text-slate-400 font-mono">Root Mean Square</div>
                <div className="text-base font-black text-purple-400 mt-1">
                  {stats.normRms.toFixed(3)}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Raw: {stats.rawRms.toFixed(2)}</div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
                <div className="text-[11px] text-slate-400 font-mono">Loss Smoothing</div>
                <div className="text-base font-black text-amber-400 mt-1">
                  {normType === 'raw' ? 'Poor (Rugged)' : 'Lipschitz (Smooth)'}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Santurkar et al. 2018</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================================= */}
      {/* TAB 2: DROPOUT STROBE & CO-ADAPTATION BREAKER */}
      {/* ========================================================================================= */}
      {activeSubTab === 'dropout' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls */}
          <div className="lg:col-span-4 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 space-y-6">
            <div>
              <div className="flex items-center gap-2 text-rose-400 text-xs font-mono font-bold uppercase tracking-wider">
                <Zap className="w-4 h-4" />
                <span>Srivastava et al. (2014)</span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">Dropout Co-Adaptation Breaker</h3>
              <p className="text-xs text-slate-400 mt-1">
                Randomly dropping hidden units prevents neurons from relying on "star" neighbors, forcing distributed representations.
              </p>
            </div>

            {/* Mode Switcher */}
            <div className="flex items-center p-1 bg-slate-900 border border-slate-800 rounded-xl">
              <button
                onClick={() => setIsTraining(true)}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition ${
                  isTraining ? 'bg-cyan-500 text-white shadow-md' : 'text-slate-400 hover:text-white'
                }`}
              >
                Training Phase (Sparsified)
              </button>
              <button
                onClick={() => {
                  setIsTraining(false);
                  setIsStrobing(false);
                }}
                className={`flex-1 py-2 text-xs font-bold rounded-lg transition ${
                  !isTraining ? 'bg-purple-600 text-white shadow-md' : 'text-slate-400 hover:text-white'
                }`}
              >
                Inference (Dense Ensemble)
              </button>
            </div>

            {/* Dropout Probability Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Dropout Probability (p):</span>
                <span className="font-mono font-bold text-rose-400">{(dropoutP * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={dropoutP}
                disabled={!isTraining}
                onChange={(e) => setDropoutP(parseFloat(e.target.value))}
                className="w-full accent-rose-500 cursor-pointer disabled:opacity-30"
              />
            </div>

            {/* Strobe Rehearsal Button */}
            {isTraining && (
              <div className="pt-2">
                <button
                  onClick={() => setIsStrobing(!isStrobing)}
                  className={`w-full py-3 rounded-xl font-bold text-xs flex items-center justify-center gap-2 transition shadow-lg ${
                    isStrobing
                      ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/30 animate-pulse'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                  }`}
                >
                  <Play className="w-4 h-4" />
                  <span>{isStrobing ? 'Pause Strobe Rehearsal' : 'Start Strobe Rehearsal (10 Hz)'}</span>
                </button>
                <p className="text-[11px] text-slate-500 text-center mt-2">
                  Rapidly flickers random sub-networks, breaking neuron co-adaptation live.
                </p>
              </div>
            )}

            {/* Inverted Dropout Math Callout */}
            <div className="p-3.5 bg-slate-900/80 border border-slate-800 rounded-xl space-y-2 text-xs">
              <div className="font-bold text-cyan-400 flex items-center gap-1.5">
                <Info className="w-4 h-4" />
                <span>Inverted Dropout Scaling Formula</span>
              </div>
              <div className="font-mono text-[11px] text-slate-300 bg-slate-950 p-2 rounded border border-slate-800/80">
                y = (x · Bernoulli(1 - p)) / (1 - p)
              </div>
              <div className="flex justify-between text-[11px] text-slate-400 pt-1">
                <span>Active Boost Multiplier:</span>
                <span className="font-mono font-bold text-emerald-400">×{circuit.scalingFactor}</span>
              </div>
            </div>
          </div>

          {/* Interactive Neural Network Graph Display */}
          <div className="lg:col-span-8 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-rose-400" />
                    <span>Neural Network Topology & Co-Adaptation</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {isTraining
                      ? `Training Mode: Neurons randomly muted with p=${dropoutP}; active units scaled by ×${circuit.scalingFactor}`
                      : 'Inference Mode: All neurons firing simultaneously as an ensemble average with weight scaling = 1.0'}
                  </p>
                </div>
              </div>

              {/* SVG Network Visualizer */}
              <div className="relative mt-6 h-80 w-full bg-[#070b14] border border-slate-800/60 rounded-xl p-4 flex items-center justify-center">
                <svg className="w-full h-full" viewBox="0 0 600 300">
                  {/* Synaptic Connections Layer 1 -> Layer 2 */}
                  {Array.from({ length: circuit.numInput }).map((_, i) =>
                    Array.from({ length: circuit.numHidden1 }).map((_, j) => {
                      const isActive = circuit.maskH1[j];
                      return (
                        <line
                          key={`in-h1-${i}-${j}`}
                          x1="70"
                          y1={30 + i * 45}
                          x2="220"
                          y2={25 + j * 35}
                          stroke={isActive ? '#06b6d4' : '#1e293b'}
                          strokeWidth={isActive ? '1.2' : '0.5'}
                          strokeOpacity={isActive ? '0.6' : '0.2'}
                        />
                      );
                    })
                  )}

                  {/* Synaptic Connections Layer 2 -> Layer 3 */}
                  {Array.from({ length: circuit.numHidden1 }).map((_, i) =>
                    Array.from({ length: circuit.numHidden2 }).map((_, j) => {
                      const isActive = circuit.maskH1[i] && circuit.maskH2[j];
                      return (
                        <line
                          key={`h1-h2-${i}-${j}`}
                          x1="220"
                          y1={25 + i * 35}
                          x2="380"
                          y2={25 + j * 35}
                          stroke={isActive ? '#a855f7' : '#1e293b'}
                          strokeWidth={isActive ? '1.2' : '0.5'}
                          strokeOpacity={isActive ? '0.6' : '0.2'}
                        />
                      );
                    })
                  )}

                  {/* Synaptic Connections Layer 3 -> Output */}
                  {Array.from({ length: circuit.numHidden2 }).map((_, i) =>
                    Array.from({ length: circuit.numOutput }).map((_, j) => {
                      const isActive = circuit.maskH2[i];
                      return (
                        <line
                          key={`h2-out-${i}-${j}`}
                          x1="380"
                          y1={25 + i * 35}
                          x2="530"
                          y2={50 + j * 65}
                          stroke={isActive ? '#10b981' : '#1e293b'}
                          strokeWidth={isActive ? '1.2' : '0.5'}
                          strokeOpacity={isActive ? '0.6' : '0.2'}
                        />
                      );
                    })
                  )}

                  {/* Input Nodes */}
                  {Array.from({ length: circuit.numInput }).map((_, i) => (
                    <g key={`in-${i}`}>
                      <circle cx="70" cy={30 + i * 45} r="12" fill="#0c4a6e" stroke="#0ea5e9" strokeWidth="2" />
                      <text x="70" y={34 + i * 45} textAnchor="middle" fill="#e0f2fe" fontSize="9" fontWeight="bold">
                        x{i + 1}
                      </text>
                    </g>
                  ))}

                  {/* Hidden Layer 1 Nodes */}
                  {Array.from({ length: circuit.numHidden1 }).map((_, i) => {
                    const active = circuit.maskH1[i];
                    return (
                      <g key={`h1-${i}`}>
                        <circle
                          cx="220"
                          cy={25 + i * 35}
                          r="12"
                          fill={active ? '#581c87' : '#0f172a'}
                          stroke={active ? '#c084fc' : '#334155'}
                          strokeWidth="2"
                          className="transition-all duration-150"
                        />
                        <text
                          x="220"
                          y={29 + i * 35}
                          textAnchor="middle"
                          fill={active ? '#ffffff' : '#475569'}
                          fontSize="9"
                          fontWeight="bold"
                        >
                          {active ? `h${i + 1}` : '✕'}
                        </text>
                      </g>
                    );
                  })}

                  {/* Hidden Layer 2 Nodes */}
                  {Array.from({ length: circuit.numHidden2 }).map((_, i) => {
                    const active = circuit.maskH2[i];
                    return (
                      <g key={`h2-${i}`}>
                        <circle
                          cx="380"
                          cy={25 + i * 35}
                          r="12"
                          fill={active ? '#701a75' : '#0f172a'}
                          stroke={active ? '#f472b6' : '#334155'}
                          strokeWidth="2"
                          className="transition-all duration-150"
                        />
                        <text
                          x="380"
                          y={29 + i * 35}
                          textAnchor="middle"
                          fill={active ? '#ffffff' : '#475569'}
                          fontSize="9"
                          fontWeight="bold"
                        >
                          {active ? `h${i + 1}` : '✕'}
                        </text>
                      </g>
                    );
                  })}

                  {/* Output Nodes */}
                  {Array.from({ length: circuit.numOutput }).map((_, i) => (
                    <g key={`out-${i}`}>
                      <circle cx="530" cy={50 + i * 65} r="14" fill="#064e3b" stroke="#10b981" strokeWidth="2" />
                      <text x="530" y={54 + i * 65} textAnchor="middle" fill="#d1fae5" fontSize="10" fontWeight="bold">
                        y{i + 1}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            </div>

            {/* Bottom Status Bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 mt-6 pt-4 border-t border-slate-800 text-xs">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5 text-cyan-400">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                  <span>Active Hidden Units: {circuit.maskH1.filter(Boolean).length + circuit.maskH2.filter(Boolean).length} / 16</span>
                </span>
                <span className="text-slate-500">•</span>
                <span className="text-rose-400 font-mono">Dropped Sub-networks: 2^{circuit.numHidden1 + circuit.numHidden2} combinations</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================================= */}
      {/* TAB 3: ADAMW DECOUPLED DECAY SANDBOX */}
      {/* ========================================================================================= */}
      {activeSubTab === 'adamw' && (
        <div className="bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2 text-amber-400 text-xs font-mono font-bold uppercase tracking-wider">
                <Scale className="w-4 h-4" />
                <span>Loshchilov & Hutter (ICLR 2019)</span>
              </div>
              <h3 className="text-xl font-bold text-white mt-1">Decoupled Weight Decay (AdamW) vs L2 Regularization</h3>
              <p className="text-xs text-slate-400 mt-1">
                Why standard L2 regularization breaks in adaptive moment optimizers and how AdamW rescued modern LLM pretraining.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Standard Adam with L2 Card */}
            <div className="p-5 rounded-xl border border-rose-900/40 bg-rose-950/10 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-rose-300">Coupled Adam with L2</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">
                  Flawed in Adaptive Settings
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                L2 regularization adds $\lambda w$ directly to the gradient vector $g_t$. The second moment estimate $v_t$ records this gradient magnitude squared.
              </p>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-[11px] font-mono text-rose-300 space-y-1">
                <div>g_t = ∇L(w_t) + λ · w_t</div>
                <div>v_t = β₂ v_{'{t-1}'} + (1 - β₂) (g_t)²</div>
                <div className="text-rose-400 font-bold">w_{'{t+1}'} = w_t - (η / √(v_t)) · m_t</div>
              </div>
              <div className="text-xs text-slate-400 bg-rose-950/30 p-2.5 rounded border border-rose-800/40">
                <span className="font-bold text-rose-200">The Failure Mode:</span> If a parameter has frequent large gradients, $v_t$ is huge. When $v_t$ divides the step, the weight decay penalty is virtually canceled!
              </div>
            </div>

            {/* Decoupled AdamW Card */}
            <div className="p-5 rounded-xl border border-emerald-900/40 bg-emerald-950/10 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-emerald-300">Decoupled AdamW</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  Modern Production Standard
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                AdamW keeps the gradient moments pure ($g_t = \nabla L$) and applies the decay step directly to the weights, fully decoupled from learning rate schedules.
              </p>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-[11px] font-mono text-emerald-300 space-y-1">
                <div>g_t = ∇L(w_t)   [No λ term in moment!]</div>
                <div>v_t = β₂ v_{'{t-1}'} + (1 - β₂) (g_t)²</div>
                <div className="text-emerald-400 font-bold">w_{'{t+1}'} = w_t - η · λ · w_t - (η / √(v_t)) · m_t</div>
              </div>
              <div className="text-xs text-slate-400 bg-emerald-950/30 p-2.5 rounded border border-emerald-800/40">
                <span className="font-bold text-emerald-200">The SOTA Fix:</span> Every weight decays at a constant proportional rate $\eta \lambda$, preserving weight norm regularization across all layers uniformly.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================================= */}
      {/* TAB 4: RESEARCH HALL OF FAME */}
      {/* ========================================================================================= */}
      {activeSubTab === 'research' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            {
              title: 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting',
              authors: 'Srivastava, Hinton, Krizhevsky, Sutskever, & Salakhutdinov (2014)',
              venue: 'JMLR 2014',
              badge: 'Regularization',
              color: 'text-rose-400 border-rose-500/30 bg-rose-500/10',
              insight:
                'Demonstrated that dropping hidden units breaks complex co-adaptations where neurons rely on specific peers. Provides exponential sub-network ensembling at test time for free.',
            },
            {
              title: 'Batch Normalization: Accelerating Deep Network Training',
              authors: 'Sergey Ioffe & Christian Szegedy (2015)',
              venue: 'ICML 2015',
              badge: 'Normalization',
              color: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
              insight:
                'Introduced mini-batch mean and variance standardization, allowing significantly higher learning rates and reducing dependency on hyper-delicate weight initialization.',
            },
            {
              title: 'How Does Batch Normalization Help Optimization?',
              authors: 'Santurkar, Tsipras, Tran, & Madry (2018)',
              venue: 'NeurIPS 2018',
              badge: 'Theoretical Breakthrough',
              color: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
              insight:
                'Debunked the Internal Covariate Shift hypothesis. Proved that BatchNorm works primarily by smoothing the loss landscape, making gradients Lipschitz continuous.',
            },
            {
              title: 'Root Mean Square Normalization (RMSNorm)',
              authors: 'Biao Zhang & Rico Sennrich (2019)',
              venue: 'NeurIPS 2019',
              badge: 'Modern LLM Standard',
              color: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
              insight:
                'Discovered that mean-centering is redundant in LayerNorm. Enforcing only root-mean-square scale invariance achieves identical convergence while slashing 10-50% computational overhead.',
            },
          ].map((paper, i) => (
            <div key={i} className="bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className={`text-[10px] uppercase font-mono font-bold px-2 py-0.5 rounded border ${paper.color}`}>
                  {paper.badge}
                </span>
                <span className="text-xs font-mono text-slate-500">{paper.venue}</span>
              </div>
              <h4 className="text-sm font-bold text-white leading-snug">{paper.title}</h4>
              <p className="text-xs text-slate-400 font-mono">{paper.authors}</p>
              <p className="text-xs text-slate-300 leading-relaxed pt-1 border-t border-slate-800/60">
                {paper.insight}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
