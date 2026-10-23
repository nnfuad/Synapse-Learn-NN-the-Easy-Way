import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  ArrowRight,
  Info,
  Layers,
  Activity,
  Compass,
  Gauge,
  Sliders,
  TrendingDown,
  Zap,
} from 'lucide-react';

// Preset Target 2D Distributions
const DATASET_PRESETS = {
  TWO_MOONS: {
    name: 'Two Moons (Non-Convex Manifold)',
    description: 'Two interleaving crescent distributions. Tests ability to bridge disconnected topological lobes.',
    generate: (n) => {
      const pts = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < half; i++) {
        const theta = (i / half) * Math.PI;
        pts.push({
          x: Math.cos(theta) + (Math.random() - 0.5) * 0.15,
          y: Math.sin(theta) + (Math.random() - 0.5) * 0.15,
        });
      }
      for (let i = 0; i < n - half; i++) {
        const theta = (i / (n - half)) * Math.PI;
        pts.push({
          x: 1 - Math.cos(theta) + (Math.random() - 0.5) * 0.15,
          y: 0.5 - Math.sin(theta) + (Math.random() - 0.5) * 0.15 - 0.25,
        });
      }
      // Center and normalize
      return pts.map((p) => ({ x: (p.x - 0.5) * 1.6, y: (p.y - 0.2) * 1.6 }));
    },
  },

  SWISS_ROLL: {
    name: 'Swiss Roll (Curved Spiral)',
    description: 'A 2D spiral demonstrating continuous manifold density transport without shortcut leaps.',
    generate: (n) => {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const t = 1.5 * Math.PI * (1 + 2 * Math.sqrt(i / n));
        const r = t / (3 * Math.PI);
        pts.push({
          x: r * Math.cos(t) + (Math.random() - 0.5) * 0.08,
          y: r * Math.sin(t) + (Math.random() - 0.5) * 0.08,
        });
      }
      return pts.map((p) => ({ x: p.x * 2.2, y: p.y * 2.2 }));
    },
  },

  CONCENTRIC_RINGS: {
    name: 'Concentric Rings (Multi-Scale)',
    description: 'Inner core and outer ring testing radial mode split under continuous probability drift.',
    generate: (n) => {
      const pts = [];
      const innerN = Math.floor(n * 0.4);
      for (let i = 0; i < innerN; i++) {
        const theta = (i / innerN) * 2 * Math.PI;
        const r = 0.5 + (Math.random() - 0.5) * 0.1;
        pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta) });
      }
      for (let i = 0; i < n - innerN; i++) {
        const theta = (i / (n - innerN)) * 2 * Math.PI;
        const r = 1.5 + (Math.random() - 0.5) * 0.15;
        pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta) });
      }
      return pts;
    },
  },

  EIGHT_GAUSSIANS: {
    name: 'Ring of 8 Gaussians (Multimodal)',
    description: '8 symmetric modes arranged in a ring. A classic benchmark for mode collapse and drift.',
    generate: (n) => {
      const pts = [];
      const modes = 8;
      for (let i = 0; i < n; i++) {
        const m = i % modes;
        const angle = (m / modes) * 2 * Math.PI;
        const cx = 1.4 * Math.cos(angle);
        const cy = 1.4 * Math.sin(angle);
        // Gaussian noise around center
        const u1 = Math.random() || 1e-6;
        const u2 = Math.random() || 1e-6;
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * 0.12;
        const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2) * 0.12;
        pts.push({ x: cx + z0, y: cy + z1 });
      }
      return pts;
    },
  },
};

export default function Module07Visualizer() {
  const canvasRef = useRef(null);

  // States
  const [paradigm, setParadigm] = useState('FLOW_MATCHING'); // 'DDPM' or 'FLOW_MATCHING'
  const [selectedDataset, setSelectedDataset] = useState('TWO_MOONS');
  const [stepT, setStepT] = useState(0); // 0 = Pure Noise (T for DDPM, 0 for FM), 100 = Final Target Data
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [showVectorField, setShowVectorField] = useState(true);
  const [showTrajectories, setShowTrajectories] = useState(true);
  const [scheduleType, setScheduleType] = useState('cosine'); // 'linear' or 'cosine'
  const [particleCount, setParticleCount] = useState(300);

  // Trajectory and Particles Cache
  const particlesRef = useRef([]);
  const targetDataRef = useRef([]);
  const initialNoiseRef = useRef([]);

  // Initialize Particles and Noise
  useEffect(() => {
    const targets = DATASET_PRESETS[selectedDataset].generate(particleCount);
    targetDataRef.current = targets;

    // Generate matching initial Gaussian noise x0 ~ N(0, I)
    const noise = [];
    for (let i = 0; i < particleCount; i++) {
      const u1 = Math.random() || 1e-6;
      const u2 = Math.random() || 1e-6;
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
      noise.push({ x: z0 * 1.2, y: z1 * 1.2 });
    }
    initialNoiseRef.current = noise;
    setStepT(0);
  }, [selectedDataset, particleCount]);

  // Compute Schedule values for step
  // Normalized fraction lambda in [0, 1] where 0 = Noise, 1 = Clean Data
  const progress = stepT / 100;

  // DDPM Schedule Calculations
  const t_ddpm = Math.round((1 - progress) * 100); // 100 down to 0
  let alpha_bar = 0;
  let beta_t = 0;
  if (scheduleType === 'linear') {
    beta_t = 0.0001 + (t_ddpm / 100) * (0.02 - 0.0001);
    alpha_bar = Math.exp(-0.5 * (t_ddpm / 100) * 1.5);
  } else {
    // Cosine schedule
    const s = 0.008;
    const norm_t = t_ddpm / 100;
    const f_t = Math.cos(((norm_t + s) / (1 + s)) * (Math.PI / 2)) ** 2;
    alpha_bar = Math.max(0.001, Math.min(1.0, f_t));
    beta_t = 0.02 * (1 - alpha_bar);
  }

  // Animation Loop
  useEffect(() => {
    let animId;
    if (isPlaying) {
      animId = requestAnimationFrame(() => {
        setStepT((prev) => {
          const next = prev + 0.6 * playbackSpeed;
          if (next >= 100) {
            setIsPlaying(false);
            return 100;
          }
          return next;
        });
      });
    }
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, playbackSpeed, stepT]);

  // Canvas Rendering (60 FPS)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Coordinate mapping functions: [-2.8, 2.8] -> [0, width]
    const toCanvasX = (x) => ((x + 2.8) / 5.6) * width;
    const toCanvasY = (y) => ((-y + 2.8) / 5.6) * height;
    const fromCanvasX = (cx) => (cx / width) * 5.6 - 2.8;
    const fromCanvasY = (cy) => -((cy / height) * 5.6 - 2.8);

    // Clear Canvas
    ctx.fillStyle = '#080d1a';
    ctx.fillRect(0, 0, width, height);

    // Draw Subtle Grid & Axes
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let g = -2; g <= 2; g++) {
      // vertical
      ctx.beginPath();
      ctx.moveTo(toCanvasX(g), 0);
      ctx.lineTo(toCanvasX(g), height);
      ctx.stroke();
      // horizontal
      ctx.beginPath();
      ctx.moveTo(0, toCanvasY(g));
      ctx.lineTo(width, toCanvasY(g));
      ctx.stroke();
    }

    // Origin Axes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.stroke();

    const targets = targetDataRef.current;
    const noises = initialNoiseRef.current;
    if (!targets.length || !noises.length) return;

    // 1. Draw Vector Field Grid if enabled
    if (showVectorField) {
      const gridRes = 14;
      ctx.lineWidth = 1.2;
      for (let ix = 0; ix < gridRes; ix++) {
        for (let iy = 0; iy < gridRes; iy++) {
          const gx = -2.4 + (ix / (gridRes - 1)) * 4.8;
          const gy = -2.4 + (iy / (gridRes - 1)) * 4.8;

          // Compute instantaneous velocity / score drift vector at (gx, gy)
          let vx = 0;
          let vy = 0;

          if (paradigm === 'FLOW_MATCHING') {
            // Optimal Transport vector field: points toward closest target modes
            // Soft kernel attribution
            let totalWeight = 0;
            const t_fm = progress;
            // Samples near current state
            for (let k = 0; k < Math.min(60, targets.length); k++) {
              const x0 = noises[k];
              const x1 = targets[k];
              const xt_x = (1 - t_fm) * x0.x + t_fm * x1.x;
              const xt_y = (1 - t_fm) * x0.y + t_fm * x1.y;
              const distSq = (gx - xt_x) ** 2 + (gy - xt_y) ** 2 + 0.15;
              const weight = 1 / distSq;
              // Target velocity u_t = x1 - x0
              vx += weight * (x1.x - x0.x);
              vy += weight * (x1.y - x0.y);
              totalWeight += weight;
            }
            if (totalWeight > 0) {
              vx /= totalWeight;
              vy /= totalWeight;
            }
          } else {
            // DDPM Score matching drift: grad log p_t(x) = - (x - sqrt(alpha_bar) x0) / (1 - alpha_bar)
            let totalWeight = 0;
            const sqrt_ab = Math.sqrt(alpha_bar);
            for (let k = 0; k < Math.min(50, targets.length); k++) {
              const x0 = targets[k];
              const center_x = sqrt_ab * x0.x;
              const center_y = sqrt_ab * x0.y;
              const distSq = (gx - center_x) ** 2 + (gy - center_y) ** 2 + 0.1;
              const weight = Math.exp(-distSq / (2 * Math.max(0.1, 1 - alpha_bar)));
              vx += weight * (center_x - gx);
              vy += weight * (center_y - gy);
              totalWeight += weight;
            }
            if (totalWeight > 0) {
              vx /= totalWeight;
              vy /= totalWeight;
            }
          }

          // Normalize for arrow drawing
          const mag = Math.sqrt(vx * vx + vy * vy);
          if (mag > 0.01) {
            const arrowLen = Math.min(18, Math.max(5, mag * 8));
            const nx = vx / mag;
            const ny = vy / mag;

            const startX = toCanvasX(gx);
            const startY = toCanvasY(gy);
            const endX = startX + nx * arrowLen;
            const endY = startY - ny * arrowLen;

            // Color by magnitude
            ctx.strokeStyle =
              paradigm === 'FLOW_MATCHING'
                ? `rgba(16, 185, 129, ${Math.min(0.65, mag * 0.4 + 0.2)})`
                : `rgba(56, 189, 248, ${Math.min(0.65, mag * 0.4 + 0.2)})`;

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.stroke();

            // Arrow head
            const angle = Math.atan2(endY - startY, endX - startX);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath();
            ctx.moveTo(endX, endY);
            ctx.lineTo(
              endX - 4 * Math.cos(angle - Math.PI / 6),
              endY - 4 * Math.sin(angle - Math.PI / 6)
            );
            ctx.lineTo(
              endX - 4 * Math.cos(angle + Math.PI / 6),
              endY - 4 * Math.sin(angle + Math.PI / 6)
            );
            ctx.fill();
          }
        }
      }
    }

    // 2. Draw Trajectories (historical trail)
    if (showTrajectories) {
      const numLines = Math.min(40, targets.length);
      ctx.lineWidth = 1.0;
      for (let i = 0; i < numLines; i++) {
        const x0 = noises[i];
        const x1 = targets[i];

        ctx.strokeStyle =
          paradigm === 'FLOW_MATCHING'
            ? 'rgba(52, 211, 153, 0.22)' // straight emerald line
            : 'rgba(236, 72, 153, 0.22)'; // curved stochastic path

        ctx.beginPath();
        const startX = toCanvasX(x0.x);
        const startY = toCanvasY(x0.y);
        ctx.moveTo(startX, startY);

        if (paradigm === 'FLOW_MATCHING') {
          // Straight OT line: x_t = (1-t) x0 + t x1
          const currX = (1 - progress) * x0.x + progress * x1.x;
          const currY = (1 - progress) * x0.y + progress * x1.y;
          ctx.lineTo(toCanvasX(currX), toCanvasY(currY));
        } else {
          // DDPM Curved Brownian path
          const stepsCount = 15;
          const currSteps = Math.floor(progress * stepsCount);
          for (let s = 1; s <= currSteps; s++) {
            const frac = s / stepsCount;
            // Mix with pseudo-random Brownian wiggle
            const pseudoRandX = Math.sin(i * 13 + s * 7) * 0.18 * (1 - frac);
            const pseudoRandY = Math.cos(i * 17 + s * 11) * 0.18 * (1 - frac);
            const interX = (1 - frac) * x0.x + frac * x1.x + pseudoRandX;
            const interY = (1 - frac) * x0.y + frac * x1.y + pseudoRandY;
            ctx.lineTo(toCanvasX(interX), toCanvasY(interY));
          }
        }
        ctx.stroke();
      }
    }

    // 3. Draw Moving Particles
    for (let i = 0; i < targets.length; i++) {
      const x0 = noises[i];
      const x1 = targets[i];

      let px, py;
      if (paradigm === 'FLOW_MATCHING') {
        // Deterministic Optimal Transport Path
        px = (1 - progress) * x0.x + progress * x1.x;
        py = (1 - progress) * x0.y + progress * x1.y;
      } else {
        // DDPM Denoising Path with Langevin stochastic noise shrinking as progress -> 1
        const wiggleMag = (1 - progress) * 0.14;
        const wiggleX = Math.sin(i * 3.7 + progress * 20) * wiggleMag;
        const wiggleY = Math.cos(i * 5.3 + progress * 20) * wiggleMag;
        px = (1 - progress) * x0.x + progress * x1.x + wiggleX;
        py = (1 - progress) * x0.y + progress * x1.y + wiggleY;
      }

      const cx = toCanvasX(px);
      const cy = toCanvasY(py);

      // Particle color transitions from noise blue/purple to data emerald/cyan
      ctx.fillStyle =
        paradigm === 'FLOW_MATCHING'
          ? progress > 0.8
            ? '#10b981'
            : '#38bdf8'
          : progress > 0.8
          ? '#ec4899'
          : '#818cf8';

      ctx.beginPath();
      ctx.arc(cx, cy, 3.2, 0, 2 * Math.PI);
      ctx.fill();

      // Outer glow for first few particles
      if (i < 8) {
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
  }, [
    paradigm,
    stepT,
    showVectorField,
    showTrajectories,
    scheduleType,
    selectedDataset,
    progress,
    alpha_bar,
  ]);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900/90 to-cyan-950/40 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs font-bold tracking-wider uppercase">
              <Sparkles className="w-4 h-4" />
              <span>Frontier Stage • Module 07</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
              Generative Paradigms: Diffusion & Flow Matching
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Witness the monumental leap from <strong>DDPM stochastic Langevin denoising</strong> (Ho et al. 2020)
              to <strong>Optimal Transport Continuous Flow Matching</strong> (Lipman et al. 2023).
            </p>
          </div>

          {/* Paradigm Switcher Tabs */}
          <div className="bg-slate-950 p-1.5 rounded-xl border border-slate-800 flex items-center gap-1 self-start md:self-auto">
            <button
              onClick={() => setParadigm('FLOW_MATCHING')}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-2 ${
                paradigm === 'FLOW_MATCHING'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-lg shadow-emerald-500/10'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Zap className="w-4 h-4 text-emerald-400" />
              <span>OT-Flow Matching (2023)</span>
            </button>

            <button
              onClick={() => setParadigm('DDPM')}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-2 ${
                paradigm === 'DDPM'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>DDPM Diffusion (2020)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Interactive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: 60 FPS Visual Canvas (7 Cols) */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center">
          <div className="w-full flex items-center justify-between pb-3 border-b border-slate-800 text-xs">
            <div className="flex items-center gap-2 text-slate-300 font-semibold">
              <Compass className="w-4 h-4 text-cyan-400" />
              <span>
                {paradigm === 'FLOW_MATCHING'
                  ? 'Deterministic Velocity Vector Field'
                  : 'Reverse Langevin Score Drift'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-cyan-400 font-bold">
                {paradigm === 'FLOW_MATCHING'
                  ? `Time t = ${(stepT / 100).toFixed(2)}`
                  : `Reverse Step t = ${Math.round((1 - stepT / 100) * 100)}`}
              </span>
            </div>
          </div>

          {/* Canvas Element */}
          <div className="relative my-4 rounded-xl overflow-hidden border border-slate-800/80 shadow-2xl">
            <canvas
              ref={canvasRef}
              width={540}
              height={440}
              className="block w-full max-w-[540px] h-auto bg-[#080d1a]"
            />

            {/* Corner Badge */}
            <div className="absolute bottom-3 left-3 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800 text-[11px] font-mono text-slate-400 flex items-center gap-1.5">
              <div
                className={`w-2 h-2 rounded-full ${
                  paradigm === 'FLOW_MATCHING' ? 'bg-emerald-400 animate-pulse' : 'bg-pink-400 animate-pulse'
                }`}
              />
              <span>{paradigm === 'FLOW_MATCHING' ? 'Straight OT Paths' : 'Brownian SDE Paths'}</span>
            </div>
          </div>

          {/* Transport Progress Scrubber */}
          <div className="w-full space-y-2 mt-2 px-2">
            <div className="flex justify-between text-xs text-slate-400 font-mono">
              <span>{paradigm === 'FLOW_MATCHING' ? 'Base Noise p₀ ~ N(0, I)' : 'Pure Noise (t = T)'}</span>
              <span className="text-white font-bold">{stepT.toFixed(0)}% Displaced</span>
              <span>{paradigm === 'FLOW_MATCHING' ? 'Target Data p₁' : 'Clean Data (t = 0)'}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="0.5"
              value={stepT}
              onChange={(e) => {
                setStepT(parseFloat(e.target.value));
                setIsPlaying(false);
              }}
              className="w-full accent-cyan-400 bg-slate-800 rounded-lg cursor-pointer h-2"
            />
          </div>

          {/* Playback Controls Bar */}
          <div className="w-full flex items-center justify-between pt-4 mt-2 border-t border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition shadow-lg shadow-cyan-500/20"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isPlaying ? 'Pause' : 'Play Trajectory'}</span>
              </button>

              <button
                onClick={() => {
                  setIsPlaying(false);
                  setStepT(0);
                }}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                title="Reset to Noise"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-slate-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showVectorField}
                  onChange={(e) => setShowVectorField(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span>Vector Field</span>
              </label>

              <label className="flex items-center gap-1.5 text-slate-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTrajectories}
                  onChange={(e) => setShowTrajectories(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span>Trails</span>
              </label>

              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2 py-1"
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1.0x</option>
                <option value={2}>2.0x</option>
              </select>
            </div>
          </div>
        </div>

        {/* Right: Controls & Scientific Inspection (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Target Dataset Selection */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
            <div className="flex items-center gap-2 text-white font-bold text-sm">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Target Distribution Geometry</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {Object.entries(DATASET_PRESETS).map(([key, item]) => (
                <button
                  key={key}
                  onClick={() => setSelectedDataset(key)}
                  className={`p-2.5 rounded-xl border text-left transition text-xs ${
                    selectedDataset === key
                      ? 'bg-cyan-950/40 border-cyan-500/60 text-cyan-200'
                      : 'bg-slate-950/50 border-slate-800/80 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div className="font-semibold text-slate-200">{item.name.split(' ')[0]}</div>
                  <div className="text-[10px] text-slate-500 truncate mt-0.5">{item.name.split('(')[1]?.replace(')', '') || '2D'}</div>
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 italic">
              {DATASET_PRESETS[selectedDataset].description}
            </p>
          </div>

          {/* Mathematical Gauge & HUD */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
            <div className="flex items-center gap-2 text-white font-bold text-sm">
              <Gauge className="w-4 h-4 text-emerald-400" />
              <span>Instantaneous Physics HUD</span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">
                  {paradigm === 'FLOW_MATCHING' ? 'Target Velocity' : 'Variance βₜ'}
                </span>
                <span className="text-white font-bold text-sm">
                  {paradigm === 'FLOW_MATCHING' ? 'u_t = x₁ - x₀' : beta_t.toFixed(5)}
                </span>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">
                  {paradigm === 'FLOW_MATCHING' ? 'Trajectory Type' : 'Signal Retention ᾱₜ'}
                </span>
                <span className="text-cyan-400 font-bold text-sm">
                  {paradigm === 'FLOW_MATCHING' ? 'Straight (OT)' : alpha_bar.toFixed(4)}
                </span>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">Inference Solver</span>
                <span className="text-emerald-400 font-bold text-sm">
                  {paradigm === 'FLOW_MATCHING' ? 'Midpoint RK2 (15 steps)' : 'Langevin SDE (100 steps)'}
                </span>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">Training Loss</span>
                <span className="text-purple-400 font-bold text-sm">
                  {paradigm === 'FLOW_MATCHING' ? '||v_θ - u_t||²' : '||ε - ε_θ||²'}
                </span>
              </div>
            </div>
          </div>

          {/* Pedagogical Paradigm Comparison */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2.5 text-xs text-slate-300">
            <div className="flex items-center gap-2 text-amber-400 font-bold">
              <Info className="w-4 h-4" />
              <span>Why Did Modern Generative AI Switch to Flow Matching?</span>
            </div>
            <ul className="space-y-1.5 list-disc list-inside text-slate-400 leading-relaxed text-[11px]">
              <li>
                <strong className="text-slate-200">Diffusion (DDPM)</strong> uses stochastic Brownian motion:
                trajectories wander and bend, requiring 50 to 1,000 steps to sample without error.
              </li>
              <li>
                <strong className="text-slate-200">Flow Matching (OT-CFM)</strong> straightens paths via Optimal Transport:
                straight velocity trajectories enable 10× faster ODE generation with standard Runge-Kutta solvers.
              </li>
              <li>
                Powers modern SOTA systems: <strong>Flux.1, Stable Diffusion 3, and AlphaFold 3</strong>.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
