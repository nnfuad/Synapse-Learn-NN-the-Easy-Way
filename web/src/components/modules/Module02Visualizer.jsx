import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  Mountain,
  Sliders,
  Activity,
  Layers,
  Compass,
  Zap,
  Info,
  CheckCircle2,
} from 'lucide-react';

// =============================================================================
// Mathematical Loss Landscapes & Analytical Gradients
// =============================================================================

const LANDSCAPES = {
  RAVINE: {
    id: 'RAVINE',
    name: 'Ill-Conditioned Ravine (Condition Number = 50)',
    description: 'f(x, y) = 0.5(x² + 50y²). Steep walls cause SGD to oscillate wildly while Momentum & Adam glide down the valley floor.',
    bounds: { xMin: -5, xMax: 5, yMin: -3, yMax: 3 },
    startPos: [-4.0, 2.0],
    minima: [0, 0],
    loss: (x, y) => 0.5 * (x * x + 50 * y * y),
    grad: (x, y) => [x, 50 * y],
  },

  ROSENBROCK: {
    id: 'ROSENBROCK',
    name: 'Rosenbrock Banana Valley',
    description: 'f(x, y) = (1 - x)² + 100(y - x²)². Steep parabolic canyon walls leading to a flat, gently curving valley floor toward (1, 1).',
    bounds: { xMin: -2.0, xMax: 2.0, yMin: -1.0, yMax: 3.0 },
    startPos: [-1.5, 2.0],
    minima: [1, 1],
    loss: (x, y) => (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2,
    grad: (x, y) => [
      -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
      200.0 * (y - x * x),
    ],
  },

  SADDLE: {
    id: 'SADDLE',
    name: 'Hyperbolic Saddle Point',
    description: 'f(x, y) = 0.5x² - 0.5y². Stationary point at (0, 0). Tests ability to escape plateaus along negative curvature.',
    bounds: { xMin: -3, xMax: 3, yMin: -3, yMax: 3 },
    startPos: [0.05, 0.02],
    minima: null,
    loss: (x, y) => 0.5 * x * x - 0.5 * y * y,
    grad: (x, y) => [x, -y],
  },

  RASTRIGIN: {
    id: 'RASTRIGIN',
    name: 'Rastrigin Multimodal Terrain',
    description: 'f(x, y) = 20 + x² - 10cos(2πx) + y² - 10cos(2πy). Dozens of deceptive local basins testing momentum breakout.',
    bounds: { xMin: -3, xMax: 3, yMin: -3, yMax: 3 },
    startPos: [-2.2, 1.8],
    minima: [0, 0],
    loss: (x, y) =>
      20.0 +
      (x * x - 10.0 * Math.cos(2.0 * Math.PI * x)) +
      (y * y - 10.0 * Math.cos(2.0 * Math.PI * y)),
    grad: (x, y) => [
      2.0 * x + 20.0 * Math.PI * Math.sin(2.0 * Math.PI * x),
      2.0 * y + 20.0 * Math.PI * Math.sin(2.0 * Math.PI * y),
    ],
  },
};

const OPTIMIZER_CONFIGS = [
  { id: 'sgd', name: 'SGD (1951)', color: '#ef4444', desc: 'Steepest descent; ping-pongs in ravines' },
  { id: 'momentum', name: 'Polyak Momentum (1964)', color: '#a855f7', desc: 'Inertia cancels oscillations' },
  { id: 'adam', name: 'Adam (2014)', color: '#0ea5e9', desc: 'Adaptive coordinate crampons' },
  { id: 'adamw', name: 'AdamW (2017)', color: '#10b981', desc: 'Decoupled weight decay SOTA' },
];

export default function Module02Visualizer() {
  const canvasRef = useRef(null);

  // Landscape & Simulation state
  const [selectedLandscapeKey, setSelectedLandscapeKey] = useState('RAVINE');
  const landscape = LANDSCAPES[selectedLandscapeKey];

  const [lr, setLr] = useState(0.025);
  const [momentumBeta, setMomentumBeta] = useState(0.9);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepCount, setStepCount] = useState(0);

  // Active optimizers toggle
  const [activeOpts, setActiveOpts] = useState({
    sgd: true,
    momentum: true,
    adam: true,
    adamw: true,
  });

  // Optimizer Marble States
  const [trajectories, setTrajectories] = useState({
    sgd: { pos: [...landscape.startPos], vel: [0, 0], path: [[...landscape.startPos]], lossHistory: [] },
    momentum: { pos: [...landscape.startPos], vel: [0, 0], path: [[...landscape.startPos]], lossHistory: [] },
    adam: { pos: [...landscape.startPos], m: [0, 0], v: [0, 0], path: [[...landscape.startPos]], lossHistory: [] },
    adamw: { pos: [...landscape.startPos], m: [0, 0], v: [0, 0], path: [[...landscape.startPos]], lossHistory: [] },
  });

  // Reset simulation when landscape changes
  const resetSimulation = (startPos = landscape.startPos) => {
    setIsPlaying(false);
    setStepCount(0);
    const initialLoss = landscape.loss(startPos[0], startPos[1]);
    setTrajectories({
      sgd: { pos: [...startPos], vel: [0, 0], path: [[...startPos]], lossHistory: [initialLoss] },
      momentum: { pos: [...startPos], vel: [0, 0], path: [[...startPos]], lossHistory: [initialLoss] },
      adam: { pos: [...startPos], m: [0, 0], v: [0, 0], path: [[...startPos]], lossHistory: [initialLoss] },
      adamw: { pos: [...startPos], m: [0, 0], v: [0, 0], path: [[...startPos]], lossHistory: [initialLoss] },
    });
  };

  useEffect(() => {
    resetSimulation(landscape.startPos);
  }, [selectedLandscapeKey]);

  // Coordinate transforms: world space <-> canvas pixels
  const worldToCanvas = (wx, wy, width, height) => {
    const { xMin, xMax, yMin, yMax } = landscape.bounds;
    const px = ((wx - xMin) / (xMax - xMin)) * width;
    const py = height - ((wy - yMin) / (yMax - yMin)) * height;
    return [px, py];
  };

  const canvasToWorld = (px, py, width, height) => {
    const { xMin, xMax, yMin, yMax } = landscape.bounds;
    const wx = xMin + (px / width) * (xMax - xMin);
    const wy = yMin + ((height - py) / height) * (yMax - yMin);
    return [wx, wy];
  };

  // Perform one step of each active optimizer
  const stepSimulation = () => {
    setTrajectories((prev) => {
      const next = { ...prev };
      const currentStep = stepCount + 1;

      // 1. SGD
      if (activeOpts.sgd) {
        let [x, y] = next.sgd.pos;
        let [gx, gy] = landscape.grad(x, y);
        // Clip gradients to prevent canvas blast
        const norm = Math.hypot(gx, gy);
        if (norm > 100) {
          gx = (gx / norm) * 100;
          gy = (gy / norm) * 100;
        }
        x -= lr * gx;
        y -= lr * gy;
        const lossVal = landscape.loss(x, y);
        next.sgd = {
          ...next.sgd,
          pos: [x, y],
          path: [...next.sgd.path, [x, y]],
          lossHistory: [...next.sgd.lossHistory, lossVal],
        };
      }

      // 2. Momentum
      if (activeOpts.momentum) {
        let [x, y] = next.momentum.pos;
        let [vx, vy] = next.momentum.vel;
        let [gx, gy] = landscape.grad(x, y);
        const norm = Math.hypot(gx, gy);
        if (norm > 100) {
          gx = (gx / norm) * 100;
          gy = (gy / norm) * 100;
        }
        vx = momentumBeta * vx + gx;
        vy = momentumBeta * vy + gy;
        x -= lr * vx;
        y -= lr * vy;
        const lossVal = landscape.loss(x, y);
        next.momentum = {
          ...next.momentum,
          pos: [x, y],
          vel: [vx, vy],
          path: [...next.momentum.path, [x, y]],
          lossHistory: [...next.momentum.lossHistory, lossVal],
        };
      }

      // 3. Adam
      if (activeOpts.adam) {
        let [x, y] = next.adam.pos;
        let [mx, my] = next.adam.m;
        let [vx, vy] = next.adam.v;
        let [gx, gy] = landscape.grad(x, y);

        // Standard Adam couples L2 penalty into gradient
        if (weightDecay > 0) {
          gx += weightDecay * x;
          gy += weightDecay * y;
        }

        const beta1 = 0.9;
        const beta2 = 0.999;
        const eps = 1e-8;

        mx = beta1 * mx + (1 - beta1) * gx;
        my = beta1 * my + (1 - beta1) * gy;
        vx = beta2 * vx + (1 - beta2) * (gx * gx);
        vy = beta2 * vy + (1 - beta2) * (gy * gy);

        const m_hat_x = mx / (1 - Math.pow(beta1, currentStep));
        const m_hat_y = my / (1 - Math.pow(beta1, currentStep));
        const v_hat_x = vx / (1 - Math.pow(beta2, currentStep));
        const v_hat_y = vy / (1 - Math.pow(beta2, currentStep));

        x -= (lr / (Math.sqrt(v_hat_x) + eps)) * m_hat_x;
        y -= (lr / (Math.sqrt(v_hat_y) + eps)) * m_hat_y;
        const lossVal = landscape.loss(x, y);
        next.adam = {
          ...next.adam,
          pos: [x, y],
          m: [mx, my],
          v: [vx, vy],
          path: [...next.adam.path, [x, y]],
          lossHistory: [...next.adam.lossHistory, lossVal],
        };
      }

      // 4. AdamW
      if (activeOpts.adamw) {
        let [x, y] = next.adamw.pos;
        let [mx, my] = next.adamw.m;
        let [vx, vy] = next.adamw.v;
        let [gx, gy] = landscape.grad(x, y);

        // Decoupled weight decay
        if (weightDecay > 0) {
          x -= lr * weightDecay * x;
          y -= lr * weightDecay * y;
        }

        const beta1 = 0.9;
        const beta2 = 0.999;
        const eps = 1e-8;

        mx = beta1 * mx + (1 - beta1) * gx;
        my = beta1 * my + (1 - beta1) * gy;
        vx = beta2 * vx + (1 - beta2) * (gx * gx);
        vy = beta2 * vy + (1 - beta2) * (gy * gy);

        const m_hat_x = mx / (1 - Math.pow(beta1, currentStep));
        const m_hat_y = my / (1 - Math.pow(beta1, currentStep));
        const v_hat_x = vx / (1 - Math.pow(beta2, currentStep));
        const v_hat_y = vy / (1 - Math.pow(beta2, currentStep));

        x -= (lr / (Math.sqrt(v_hat_x) + eps)) * m_hat_x;
        y -= (lr / (Math.sqrt(v_hat_y) + eps)) * m_hat_y;
        const lossVal = landscape.loss(x, y);
        next.adamw = {
          ...next.adamw,
          pos: [x, y],
          m: [mx, my],
          v: [vx, vy],
          path: [...next.adamw.path, [x, y]],
          lossHistory: [...next.adamw.lossHistory, lossVal],
        };
      }

      return next;
    });

    setStepCount((s) => s + 1);
  };

  // Simulation tick loop
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      stepSimulation();
    }, 45);
    return () => clearInterval(timer);
  }, [isPlaying, stepCount, lr, momentumBeta, weightDecay, activeOpts]);

  // Click on canvas to spawn new starting position
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const [wx, wy] = canvasToWorld(px, py, canvas.width, canvas.height);
    resetSimulation([wx, wy]);
  };

  // Render 60 FPS Canvas with Contour Landscape and Trajectory Trails
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw Loss Heatmap Background & Contour Lines
    const step = 8;
    for (let px = 0; px < width; px += step) {
      for (let py = 0; py < height; py += step) {
        const [wx, wy] = canvasToWorld(px, py, width, height);
        const l = landscape.loss(wx, wy);

        // Normalize loss to log color scale
        const logL = Math.log10(Math.max(0.01, l + 1));
        const intensity = Math.min(1.0, logL / 4.0);

        // Dark navy to deep cyan-purple heatmap palette
        const r = Math.floor(10 + intensity * 35);
        const g = Math.floor(15 + intensity * 45);
        const b = Math.floor(35 + intensity * 70);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(px, py, step, step);
      }
    }

    // 2. Draw Subtle Iso-contour Level Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    const numContours = 12;
    for (let c = 0; c < numContours; c++) {
      // Draw grid references
      ctx.beginPath();
      const [cx, cy] = worldToCanvas(0, 0, width, height);
      ctx.arc(cx, cy, (c + 1) * 35, 0, Math.PI * 2);
      ctx.stroke();
    }

    // 3. Mark Global Minima if present
    if (landscape.minima) {
      const [mx, my] = worldToCanvas(landscape.minima[0], landscape.minima[1], width, height);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(mx, my, 8, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(mx, my, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#fde68a';
      ctx.font = 'bold 10px monospace';
      ctx.fillText('GLOBAL MINIMUM', mx + 12, my + 4);
    }

    // 4. Draw Trajectory Trails & Rolling Marbles
    OPTIMIZER_CONFIGS.forEach((opt) => {
      if (!activeOpts[opt.id]) return;
      const data = trajectories[opt.id];
      const path = data.path;
      if (path.length === 0) return;

      // Draw Path Trail
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      path.forEach(([wx, wy], idx) => {
        const [px, py] = worldToCanvas(wx, wy, width, height);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      // Draw Dots along path
      path.forEach(([wx, wy], idx) => {
        if (idx % 2 === 0 || idx === path.length - 1) {
          const [px, py] = worldToCanvas(wx, wy, width, height);
          ctx.fillStyle = opt.color;
          ctx.beginPath();
          ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Draw Head Marble
      const [curWx, curWy] = data.pos;
      const [curPx, curPy] = worldToCanvas(curWx, curWy, width, height);

      // Glow halo
      ctx.fillStyle = `${opt.color}40`;
      ctx.beginPath();
      ctx.arc(curPx, curPy, 10, 0, Math.PI * 2);
      ctx.fill();

      // Solid Marble
      ctx.fillStyle = opt.color;
      ctx.beginPath();
      ctx.arc(curPx, curPy, 5.5, 0, Math.PI * 2);
      ctx.fill();

      // White center highlight
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(curPx - 1.5, curPy - 1.5, 1.8, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [trajectories, landscape, activeOpts]);

  return (
    <div className="space-y-6">
      {/* Module 02 Header Banner */}
      <div className="bg-gradient-to-r from-purple-950/40 via-indigo-950/20 to-slate-900 border border-purple-800/40 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                Robbins-Monro 1951 • Polyak 1964 • AdamW 2017
              </span>
              <span className="text-xs text-slate-400 font-mono">Loss Geometry & SOTA Optimization</span>
            </div>
            <h2 className="text-xl font-black text-white flex items-center gap-2">
              <Mountain className="w-5 h-5 text-purple-400" />
              Loss Landscape Simulator & Real-Time Marble Race
            </h2>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Watch marble trajectories navigate non-convex ravines and saddle points. Compare how <span className="text-red-400 font-semibold">SGD</span> bounces wildly across steep canyon walls while <span className="text-purple-400 font-semibold">Momentum</span> builds forward inertia and <span className="text-emerald-400 font-semibold">AdamW</span> glides effortlessly along the ravine floor.
            </p>
          </div>

          {/* Landscape Preset Selector */}
          <div className="flex flex-wrap items-center gap-2">
            {Object.keys(LANDSCAPES).map((k) => (
              <button
                key={k}
                onClick={() => setSelectedLandscapeKey(k)}
                className={`px-3 py-2 rounded-xl text-xs font-semibold transition ${
                  selectedLandscapeKey === k
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50 shadow-md shadow-purple-500/10'
                    : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-850 hover:text-white'
                }`}
              >
                {LANDSCAPES[k].name.split(' (')[0]}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: 2D Contour Canvas (8 cols) */}
        <div className="lg:col-span-8 bg-slate-900/90 border border-slate-800 rounded-2xl p-4 flex flex-col shadow-2xl relative">
          {/* Controls Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80 mb-2">
            {/* Play / Step / Reset */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition flex items-center gap-1.5 shadow-md ${
                  isPlaying
                    ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-500/20'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/20'
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isPlaying ? 'Pause Marbles' : 'Release Marbles'}</span>
              </button>

              <button
                onClick={() => stepSimulation()}
                disabled={isPlaying}
                className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-bold text-slate-200 transition border border-slate-700"
              >
                Step +1
              </button>

              <button
                onClick={() => resetSimulation()}
                className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-400 hover:text-white transition border border-slate-700 flex items-center gap-1"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            </div>

            {/* Step Counter Badge */}
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="text-slate-400">Step:</span>
              <span className="font-bold text-cyan-400 bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800">
                {stepCount}
              </span>
            </div>
          </div>

          {/* Canvas Wrapper */}
          <div className="relative w-full h-[450px] bg-[#070b13] rounded-xl border border-slate-800/80 overflow-hidden cursor-crosshair">
            <canvas
              ref={canvasRef}
              width={680}
              height={450}
              onClick={handleCanvasClick}
              className="w-full h-full block"
            />
            {/* Click to drop hint */}
            <div className="absolute bottom-2.5 left-3 px-2 py-1 rounded bg-black/60 backdrop-blur-sm border border-slate-800 text-[10px] font-mono text-slate-400 pointer-events-none">
              💡 Click anywhere on the terrain to set a new starting marble position!
            </div>
          </div>

          {/* Active Optimizers Pills Bar */}
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-800/80">
            <div className="flex flex-wrap items-center gap-2">
              {OPTIMIZER_CONFIGS.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() =>
                    setActiveOpts((prev) => ({ ...prev, [opt.id]: !prev[opt.id] }))
                  }
                  className={`px-3 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition border ${
                    activeOpts[opt.id]
                      ? 'bg-slate-800/90 text-white border-slate-700 shadow-sm'
                      : 'bg-slate-950/60 text-slate-500 border-slate-900 opacity-60'
                  }`}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: opt.color }}
                  />
                  <span>{opt.name}</span>
                </button>
              ))}
            </div>

            <span className="text-[11px] font-mono text-slate-400">
              Current Loss: SGD ({trajectories.sgd.lossHistory.slice(-1)[0]?.toFixed(3) || '0.00'}) | AdamW ({trajectories.adamw.lossHistory.slice(-1)[0]?.toFixed(3) || '0.00'})
            </span>
          </div>
        </div>

        {/* Right Column: Hyperparameters & Mountain Skiing Metaphor (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Hyperparameters Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3.5">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Sliders className="w-4 h-4 text-purple-400" />
                <span>Hyperparameter Cockpit</span>
              </div>
              <span className="text-[11px] font-mono text-purple-400">Live Reaction</span>
            </div>

            {/* Learning Rate Slider */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300 font-bold">Learning Rate (η)</span>
                <span className="text-cyan-400 font-bold">{lr.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.002"
                max="0.08"
                step="0.002"
                value={lr}
                onChange={(e) => setLr(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
              <p className="text-[10px] text-slate-500">
                Notice: Large η causes SGD to explode across ravine walls!
              </p>
            </div>

            {/* Momentum Beta Slider */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300 font-bold">Momentum Inertia (β)</span>
                <span className="text-purple-400 font-bold">{momentumBeta.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.98"
                step="0.02"
                value={momentumBeta}
                onChange={(e) => setMomentumBeta(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-400"
              />
            </div>

            {/* Decoupled Weight Decay Slider */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300 font-bold">Decoupled Weight Decay (λ)</span>
                <span className="text-emerald-400 font-bold">{weightDecay.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.05"
                step="0.005"
                value={weightDecay}
                onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
            </div>
          </div>

          {/* Mountain Skiing Metaphor Box */}
          <div className="bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center gap-2 text-white font-bold text-sm pb-2 border-b border-slate-800">
              <Mountain className="w-4 h-4 text-cyan-400" />
              <span>Skiing Down the Foggy Ravine</span>
            </div>

            <div className="space-y-2 text-xs text-slate-300 leading-relaxed">
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="text-red-400 font-bold">1. SGD Skier: </span>
                Blindly follows steepest local slope. Smashes back and forth across the ravine walls without forward progress.
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="text-purple-400 font-bold">2. Momentum Skier: </span>
                Acts like a heavy bowling ball. Lateral crashes cancel out, while forward speed carries through the gentle valley.
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="text-emerald-400 font-bold">3. AdamW Skier: </span>
                Wears smart crampons adjusting friction independently on both axes, plus a gentle bungee tether (weight decay) pulling back to safety.
              </div>
            </div>
          </div>

          {/* SOTA Insight */}
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-3.5 text-xs text-slate-400 space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold text-slate-200">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Why AdamW in 2026?</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              Loshchilov & Hutter (2017) discovered that standard Adam couples L2 penalty into the adaptive gradient moments, causing weights with large gradients to decay too slowly. AdamW decouples weight decay, enabling SOTA performance across Transformers!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
