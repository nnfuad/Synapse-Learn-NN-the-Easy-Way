import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  RotateCcw,
  Sparkles,
  ArrowRight,
  Info,
  Droplets,
  Layers,
  Activity,
  CheckCircle2,
  TrendingDown,
  HelpCircle,
  Eye,
  FastForward,
} from 'lucide-react';

// Graph Presets
const GRAPH_PRESETS = {
  DIAMOND: {
    id: 'DIAMOND',
    name: 'Diamond DAG (Gradient Superposition)',
    description: 'A branching DAG where x feeds two downstream operations (x² and 3x). Demonstrates multivariable chain rule summation.',
    inputs: { x: 3.0 },
    buildGraph: (inputs) => {
      const x = inputs.x;
      const x2 = x * x;
      const x3 = 3.0 * x;
      const c = x2 + x3;

      // Backward derivatives
      const grad_c = 1.0;
      const grad_x2 = grad_c * 1.0;
      const grad_x3 = grad_c * 1.0;
      const grad_x_from_x2 = grad_x2 * (2.0 * x);
      const grad_x_from_x3 = grad_x3 * 3.0;
      const grad_x = grad_x_from_x2 + grad_x_from_x3;

      return {
        nodes: [
          { id: 'x', label: 'x', type: 'input', x: 80, y: 200, data: x, grad: grad_x, formula: 'Input variable' },
          { id: 'sq', label: 'x²', op: '**2', type: 'op', x: 280, y: 120, data: x2, grad: grad_x2, formula: 'd(x²)/dx = 2x', localGrad: 2 * x },
          { id: 'mul', label: '3 · x', op: '*', type: 'op', x: 280, y: 280, data: x3, grad: grad_x3, formula: 'd(3x)/dx = 3.0', localGrad: 3.0 },
          { id: 'c', label: 'c = x² + 3x', op: '+', type: 'output', x: 480, y: 200, data: c, grad: grad_c, formula: 'dc/dc = 1.0 (Loss Seed)', localGrad: 1.0 },
        ],
        edges: [
          { source: 'x', target: 'sq', label: 'dx²' },
          { source: 'x', target: 'mul', label: 'd(3x)' },
          { source: 'sq', target: 'c', label: '+1.0' },
          { source: 'mul', target: 'c', label: '+1.0' },
        ],
        topoOrder: ['c', 'sq', 'mul', 'x'],
      };
    },
  },

  NEURON: {
    id: 'NEURON',
    name: 'Single Sigmoid Neuron',
    description: 'A biological-artificial neuron: y = σ(w₁x₁ + w₂x₂ + b). Shows activation flattening & gradient vanishing.',
    inputs: { x1: 1.5, w1: 0.8, x2: -1.0, w2: 1.2, b: -0.5 },
    buildGraph: (inputs) => {
      const { x1, w1, x2, w2, b } = inputs;
      const prod1 = x1 * w1;
      const prod2 = x2 * w2;
      const sum_wx = prod1 + prod2;
      const z = sum_wx + b;
      const sig = 1.0 / (1.0 + Math.exp(-Math.max(-30, Math.min(30, z))));

      // Backward
      const grad_sig = 1.0;
      const dsig_dz = sig * (1.0 - sig);
      const grad_z = grad_sig * dsig_dz;
      const grad_b = grad_z * 1.0;
      const grad_sum_wx = grad_z * 1.0;
      const grad_prod1 = grad_sum_wx * 1.0;
      const grad_prod2 = grad_sum_wx * 1.0;
      const grad_x1 = grad_prod1 * w1;
      const grad_w1 = grad_prod1 * x1;
      const grad_x2 = grad_prod2 * w2;
      const grad_w2 = grad_prod2 * x2;

      return {
        nodes: [
          { id: 'x1', label: 'x₁', type: 'input', x: 60, y: 80, data: x1, grad: grad_x1, formula: 'Input Feature 1' },
          { id: 'w1', label: 'w₁', type: 'input', x: 60, y: 150, data: w1, grad: grad_w1, formula: 'Synaptic Weight 1' },
          { id: 'prod1', label: 'w₁x₁', op: '*', type: 'op', x: 220, y: 115, data: prod1, grad: grad_prod1, formula: 'd(wx)/dw = x, d(wx)/dx = w', localGrad: w1 },

          { id: 'x2', label: 'x₂', type: 'input', x: 60, y: 250, data: x2, grad: grad_x2, formula: 'Input Feature 2' },
          { id: 'w2', label: 'w₂', type: 'input', x: 60, y: 320, data: w2, grad: grad_w2, formula: 'Synaptic Weight 2' },
          { id: 'prod2', label: 'w₂x₂', op: '*', type: 'op', x: 220, y: 285, data: prod2, grad: grad_prod2, formula: 'd(wx)/dw = x, d(wx)/dx = w', localGrad: w2 },

          { id: 'b', label: 'b', type: 'input', x: 220, y: 200, data: b, grad: grad_b, formula: 'Neuron Bias' },

          { id: 'sum_wx', label: 'Σwx', op: '+', type: 'op', x: 360, y: 200, data: sum_wx, grad: grad_sum_wx, formula: 'Summation of dendrites', localGrad: 1.0 },
          { id: 'z', label: 'z = Σwx + b', op: '+', type: 'op', x: 470, y: 200, data: z, grad: grad_z, formula: 'Pre-activation potential', localGrad: 1.0 },
          { id: 'out', label: 'σ(z)', op: 'sigmoid', type: 'output', x: 610, y: 200, data: sig, grad: grad_sig, formula: 'σ\'(z) = σ(z)(1 - σ(z))', localGrad: dsig_dz },
        ],
        edges: [
          { source: 'x1', target: 'prod1' },
          { source: 'w1', target: 'prod1' },
          { source: 'x2', target: 'prod2' },
          { source: 'w2', target: 'prod2' },
          { source: 'prod1', target: 'sum_wx' },
          { source: 'prod2', target: 'sum_wx' },
          { source: 'sum_wx', target: 'z' },
          { source: 'b', target: 'z' },
          { source: 'z', target: 'out' },
        ],
        topoOrder: ['out', 'z', 'b', 'sum_wx', 'prod2', 'prod1', 'w2', 'x2', 'w1', 'x1'],
      };
    },
  },

  COMPOSITE_LOSS: {
    id: 'COMPOSITE_LOSS',
    name: 'MSE Loss & Squared Error DAG',
    description: 'Loss = (prediction - target)². Shows how error signals emanate from loss and propagate back to parameters.',
    inputs: { target: 1.0, w: 0.6, x: 2.0, b: 0.1 },
    buildGraph: (inputs) => {
      const { target, w, x, b } = inputs;
      const wx = w * x;
      const pred = wx + b;
      const diff = pred - target;
      const loss = diff * diff;

      // Backward
      const grad_loss = 1.0;
      const dloss_ddiff = 2.0 * diff;
      const grad_diff = grad_loss * dloss_ddiff;
      const grad_pred = grad_diff * 1.0;
      const grad_target = grad_diff * -1.0;
      const grad_wx = grad_pred * 1.0;
      const grad_b = grad_pred * 1.0;
      const grad_w = grad_wx * x;
      const grad_x = grad_wx * w;

      return {
        nodes: [
          { id: 'w', label: 'w', type: 'input', x: 80, y: 100, data: w, grad: grad_w, formula: 'Weight parameter' },
          { id: 'x', label: 'x', type: 'input', x: 80, y: 180, data: x, grad: grad_x, formula: 'Input sample' },
          { id: 'wx', label: 'w·x', op: '*', type: 'op', x: 230, y: 140, data: wx, grad: grad_wx, formula: 'Linear product' },
          { id: 'b', label: 'b', type: 'input', x: 230, y: 240, data: b, grad: grad_b, formula: 'Bias parameter' },
          { id: 'pred', label: 'ŷ (pred)', op: '+', type: 'op', x: 370, y: 190, data: pred, grad: grad_pred, formula: 'Model prediction' },
          { id: 'target', label: 'y (target)', type: 'input', x: 370, y: 300, data: target, grad: grad_target, formula: 'Ground truth label' },
          { id: 'diff', label: 'ŷ - y', op: '-', type: 'op', x: 490, y: 240, data: diff, grad: grad_diff, formula: 'Residual error' },
          { id: 'loss', label: 'Loss = (ŷ-y)²', op: '**2', type: 'output', x: 630, y: 240, data: loss, grad: grad_loss, formula: 'dLoss/dLoss = 1.0' },
        ],
        edges: [
          { source: 'w', target: 'wx' },
          { source: 'x', target: 'wx' },
          { source: 'wx', target: 'pred' },
          { source: 'b', target: 'pred' },
          { source: 'pred', target: 'diff' },
          { source: 'target', target: 'diff' },
          { source: 'diff', target: 'loss' },
        ],
        topoOrder: ['loss', 'diff', 'target', 'pred', 'b', 'wx', 'x', 'w'],
      };
    },
  },
};

export default function Module01Visualizer() {
  const [selectedPresetKey, setSelectedPresetKey] = useState('DIAMOND');
  const preset = GRAPH_PRESETS[selectedPresetKey];
  const [inputs, setInputs] = useState(preset.inputs);
  const [selectedNodeId, setSelectedNodeId] = useState('x');

  // Animation states: 'idle', 'forward_surge', 'backward_surge'
  const [flowMode, setFlowMode] = useState('forward_surge');
  const [activeStepIndex, setActiveStepIndex] = useState(-1);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const animFrameRef = useRef(null);
  const [pulseProgress, setPulseProgress] = useState(0);

  // Re-sync inputs when preset changes
  useEffect(() => {
    setInputs(GRAPH_PRESETS[selectedPresetKey].inputs);
    setSelectedNodeId(GRAPH_PRESETS[selectedPresetKey].topoOrder[GRAPH_PRESETS[selectedPresetKey].topoOrder.length - 1]);
    setActiveStepIndex(-1);
    setIsAutoPlaying(false);
  }, [selectedPresetKey]);

  // Compute live graph
  const graph = preset.buildGraph(inputs);
  const selectedNode = graph.nodes.find((n) => n.id === selectedNodeId) || graph.nodes[0];

  // 60 FPS Pulse Animation along edges
  useEffect(() => {
    let progress = 0;
    const animate = () => {
      progress = (progress + 0.02) % 1;
      setPulseProgress(progress);
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animFrameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []);

  // Handle auto-step play through reverse topological order
  useEffect(() => {
    if (!isAutoPlaying) return;
    const interval = setInterval(() => {
      setActiveStepIndex((prev) => {
        const next = prev + 1;
        if (next >= graph.topoOrder.length) {
          setIsAutoPlaying(false);
          return prev;
        }
        setSelectedNodeId(graph.topoOrder[next]);
        return next;
      });
    }, 1200);
    return () => clearInterval(interval);
  }, [isAutoPlaying, graph.topoOrder]);

  const handleInputChange = (key, val) => {
    setInputs((prev) => ({ ...prev, [key]: parseFloat(val) }));
  };

  const triggerBackwardWave = () => {
    setFlowMode('backward_surge');
    setActiveStepIndex(0);
    setSelectedNodeId(graph.topoOrder[0]);
    setIsAutoPlaying(true);
  };

  const triggerForwardFlow = () => {
    setFlowMode('forward_surge');
    setActiveStepIndex(-1);
    setIsAutoPlaying(false);
  };

  return (
    <div className="space-y-6">
      {/* Module 01 Pedagogical Banner */}
      <div className="bg-gradient-to-r from-blue-950/40 via-cyan-950/20 to-slate-900 border border-cyan-800/40 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Linnainmaa 1970 • Rumelhart 1986
              </span>
              <span className="text-xs text-slate-400 font-mono">The Reverse Chain Rule</span>
            </div>
            <h2 className="text-xl font-black text-white flex items-center gap-2">
              <Droplets className="w-5 h-5 text-cyan-400" />
              Dynamic Computational DAG & Reverse Pressure Wave
            </h2>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Watch intermediate values flow forward like water in pipes (<span className="text-cyan-400 font-semibold">Blue</span>),
              then trigger the hydraulic backpressure surge (<span className="text-rose-400 font-semibold">Red</span>) where unit loss pressure propagates backward, splitting and superimposing across branching junctions.
            </p>
          </div>

          {/* Preset Selector */}
          <div className="flex flex-wrap items-center gap-2">
            {Object.keys(GRAPH_PRESETS).map((k) => (
              <button
                key={k}
                onClick={() => setSelectedPresetKey(k)}
                className={`px-3 py-2 rounded-xl text-xs font-semibold transition ${
                  selectedPresetKey === k
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-md shadow-cyan-500/10'
                    : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-850 hover:text-white'
                }`}
              >
                {GRAPH_PRESETS[k].name.split(' (')[0]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Visualizer Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive DAG Canvas (8 cols) */}
        <div className="lg:col-span-8 bg-slate-900/90 border border-slate-800 rounded-2xl p-4 flex flex-col shadow-2xl relative overflow-hidden">
          {/* Canvas Header & Action Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80 mb-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold text-slate-300">FLOW MODE:</span>
              <button
                onClick={triggerForwardFlow}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                  flowMode === 'forward_surge'
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                    : 'bg-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                <ArrowRight className="w-3.5 h-3.5 text-cyan-300" />
                Forward Values (Blue)
              </button>
              <button
                onClick={triggerBackwardWave}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                  flowMode === 'backward_surge'
                    ? 'bg-rose-600 text-white shadow-lg shadow-rose-500/20'
                    : 'bg-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                <Activity className="w-3.5 h-3.5 text-rose-300" />
                Surge Backward Wave (Red)
              </button>
            </div>

            <div className="flex items-center gap-2">
              {flowMode === 'backward_surge' && (
                <button
                  onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 flex items-center gap-1.5 border border-slate-700 transition"
                >
                  {isAutoPlaying ? <RotateCcw className="w-3.5 h-3.5 text-amber-400" /> : <Play className="w-3.5 h-3.5 text-emerald-400" />}
                  <span>{isAutoPlaying ? 'Pause Step' : 'Auto Step Wave'}</span>
                </button>
              )}
            </div>
          </div>

          {/* SVG DAG Visualizer Surface */}
          <div className="relative w-full h-[440px] bg-[#070b13] rounded-xl border border-slate-800/80 overflow-hidden select-none">
            {/* Background Grid Pattern */}
            <svg className="absolute inset-0 w-full h-full opacity-20 pointer-events-none">
              <defs>
                <pattern id="dag-grid" width="30" height="30" patternUnits="userSpaceOnUse">
                  <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#334155" strokeWidth="0.8" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#dag-grid)" />
            </svg>

            {/* Edges & Water Pipes */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <defs>
                {/* Flow glow filters */}
                <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <filter id="glow-red" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {graph.edges.map((edge, idx) => {
                const s = graph.nodes.find((n) => n.id === edge.source);
                const t = graph.nodes.find((n) => n.id === edge.target);
                if (!s || !t) return null;

                const dx = t.x - s.x;
                const pathData = `M ${s.x + 35} ${s.y} C ${s.x + dx * 0.5} ${s.y}, ${t.x - dx * 0.5} ${t.y}, ${t.x - 35} ${t.y}`;

                // Pulse position
                const pulseT = flowMode === 'backward_surge' ? 1 - pulseProgress : pulseProgress;
                const pulseX = s.x + 35 + (t.x - 35 - (s.x + 35)) * pulseT;
                const pulseY = s.y + (t.y - s.y) * pulseT;

                return (
                  <g key={`${edge.source}-${edge.target}-${idx}`}>
                    {/* Background pipe */}
                    <path
                      d={pathData}
                      fill="none"
                      stroke="#1e293b"
                      strokeWidth="6"
                      strokeLinecap="round"
                    />
                    {/* Active flow stream */}
                    <path
                      d={pathData}
                      fill="none"
                      stroke={flowMode === 'backward_surge' ? '#f43f5e' : '#0ea5e9'}
                      strokeWidth="2.5"
                      strokeDasharray="6 4"
                      strokeOpacity="0.8"
                    />
                    {/* Travelling water pulse bead */}
                    <circle
                      cx={pulseX}
                      cy={pulseY}
                      r="4"
                      fill={flowMode === 'backward_surge' ? '#fda4af' : '#7dd3fc'}
                      filter={flowMode === 'backward_surge' ? 'url(#glow-red)' : 'url(#glow-blue)'}
                    />
                  </g>
                );
              })}
            </svg>

            {/* Graph Nodes */}
            <div className="absolute inset-0 pointer-events-auto">
              {graph.nodes.map((node) => {
                const isSelected = selectedNodeId === node.id;
                const isCurrentWaveNode =
                  flowMode === 'backward_surge' &&
                  activeStepIndex >= 0 &&
                  graph.topoOrder[activeStepIndex] === node.id;

                const isInput = node.type === 'input';
                const isOutput = node.type === 'output';

                return (
                  <div
                    key={node.id}
                    onClick={() => setSelectedNodeId(node.id)}
                    style={{ left: `${node.x - 45}px`, top: `${node.y - 32}px` }}
                    className={`absolute w-[92px] h-[64px] rounded-xl flex flex-col items-center justify-center cursor-pointer transition-all duration-200 select-none ${
                      isCurrentWaveNode
                        ? 'ring-4 ring-rose-500 scale-110 z-30 shadow-2xl shadow-rose-500/50 bg-rose-950/80 border-rose-400'
                        : isSelected
                        ? 'ring-2 ring-cyan-400 scale-105 z-20 shadow-xl bg-slate-800 border-cyan-500/80'
                        : 'bg-slate-900/90 border border-slate-700/80 hover:border-slate-500 hover:scale-102 z-10'
                    }`}
                  >
                    {/* Node Header Label & Op */}
                    <div className="flex items-center gap-1">
                      <span className="text-[11px] font-bold text-white font-mono">{node.label}</span>
                      {node.op && (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-purple-500/20 text-purple-300 font-mono">
                          {node.op}
                        </span>
                      )}
                    </div>

                    {/* Forward Data Value (Blue) */}
                    <div className="text-[10px] font-mono font-semibold text-cyan-300 mt-0.5">
                      v: {node.data >= 0 ? '+' : ''}{node.data.toFixed(2)}
                    </div>

                    {/* Backward Gradient Value (Red) */}
                    <div className={`text-[10px] font-mono font-bold mt-0.5 ${
                      Math.abs(node.grad) > 1e-4 ? 'text-rose-400' : 'text-slate-500'
                    }`}>
                      ∇: {node.grad >= 0 ? '+' : ''}{node.grad.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Bottom Live Formula Bar */}
          <div className="mt-3 bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2 text-slate-300">
              <span className="text-slate-500">Selected:</span>
              <span className="text-cyan-400 font-bold">{selectedNode.label}</span>
              <span className="text-slate-600">|</span>
              <span className="text-slate-400">Formula:</span>
              <span className="text-purple-300">{selectedNode.formula}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-cyan-300 font-semibold">Forward v = {selectedNode.data.toFixed(4)}</span>
              <span className="text-rose-400 font-bold">Gradient dL/dv = {selectedNode.grad.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* Right Column: Dynamic Sliders & Physical Analogy Gauge (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Interactive Input Variables Tuning Panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>Input Variables & Parameters</span>
              </div>
              <span className="text-[11px] font-mono text-cyan-400">Live Evaluation</span>
            </div>

            <div className="space-y-3 pt-1">
              {Object.keys(inputs).map((paramKey) => (
                <div key={paramKey} className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-300 font-bold">{paramKey}</span>
                    <span className="text-cyan-400 font-bold">{inputs[paramKey].toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-4.0"
                    max="4.0"
                    step="0.1"
                    value={inputs[paramKey]}
                    onChange={(e) => handleInputChange(paramKey, e.target.value)}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Water Pipe Pressure Gauge Metaphor */}
          <div className="bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center gap-2 text-white font-bold text-sm pb-2 border-b border-slate-800">
              <Droplets className="w-4 h-4 text-blue-400" />
              <span>Hydraulic Pipe & Pressure Metaphor</span>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <div className="flex items-center justify-between p-2 rounded-xl bg-slate-950/60 border border-slate-800/80 font-mono">
                <span className="text-slate-400">Throughput Value (Data):</span>
                <span className="text-cyan-400 font-bold">{selectedNode.data.toFixed(3)} L/s</span>
              </div>

              <div className="flex items-center justify-between p-2 rounded-xl bg-slate-950/60 border border-slate-800/80 font-mono">
                <span className="text-slate-400">Reverse Backpressure (Adjoint):</span>
                <span className="text-rose-400 font-bold">{selectedNode.grad.toFixed(3)} PSI</span>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed pt-1">
                Notice: In a Diamond DAG, variable <code className="text-cyan-300">x</code> feeds both <code className="text-purple-300">x²</code> and <code className="text-purple-300">3x</code>. When backpressure surges from the loss, it splits across both branches and recombines additively at <code className="text-cyan-300">x</code>.
              </p>
            </div>
          </div>

          {/* Historical Fact Card */}
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-3.5 text-xs text-slate-400 space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold text-slate-200">
              <Info className="w-4 h-4 text-cyan-400" />
              <span>Why Reverse Mode (1970)?</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              Forward-mode autodiff requires one forward sweep per parameter ($O(P)$). For modern 70B parameter LLMs, forward mode is impossible. Reverse mode computes all gradients in $O(1)$ loss evaluations!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
