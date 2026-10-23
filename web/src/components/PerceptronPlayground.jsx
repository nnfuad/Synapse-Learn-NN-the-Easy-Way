import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, StepForward, RotateCcw, Plus, CheckCircle, AlertOctagon, HelpCircle, ArrowRight } from 'lucide-react';
import confetti from 'canvas-confetti';

const PRESETS = {
  OR: [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 1 },
  ],
  AND: [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 0 },
    { x: 1, y: 0, label: 0 },
    { x: 1, y: 1, label: 1 },
  ],
  SEPARABLE: [
    { x: -0.8, y: -0.5, label: 0 },
    { x: -0.5, y: -0.8, label: 0 },
    { x: -0.2, y: -0.3, label: 0 },
    { x: 0.1, y: -0.6, label: 0 },
    { x: 0.3, y: 0.6, label: 1 },
    { x: 0.6, y: 0.3, label: 1 },
    { x: 0.8, y: 0.8, label: 1 },
    { x: 0.4, y: 0.9, label: 1 },
  ],
  XOR: [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 0 },
  ],
};

export default function PerceptronPlayground() {
  const canvasRef = useRef(null);

  // State
  const [points, setPoints] = useState(PRESETS.OR);
  const [weights, setWeights] = useState([0.1, -0.2]); // [w1, w2]
  const [bias, setBias] = useState(0.0);
  const [lr, setLr] = useState(0.1);
  const [epoch, setEpoch] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [addModeLabel, setAddModeLabel] = useState(1); // 0 or 1
  const [lastUpdatedPoint, setLastUpdatedPoint] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState('OR');

  // Classification stats
  const predictions = points.map(pt => {
    const sum = weights[0] * pt.x + weights[1] * pt.y + bias;
    const pred = sum >= 0 ? 1 : 0;
    return { ...pt, sum, pred, correct: pred === pt.label };
  });

  const correctCount = predictions.filter(p => p.correct).length;
  const accuracy = points.length > 0 ? (correctCount / points.length) * 100 : 0;
  const isFullyConverged = accuracy === 100 && points.length > 0;

  // Trigger celebration on convergence
  const prevConvergedRef = useRef(false);
  useEffect(() => {
    if (isFullyConverged && !prevConvergedRef.current && epoch > 0) {
      try {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.6 }
        });
      } catch (e) {
        // Fallback gracefully if confetti fails
      }
      setIsPlaying(false);
    }
    prevConvergedRef.current = isFullyConverged;
  }, [isFullyConverged, epoch]);

  // Reset Model
  const resetWeights = () => {
    setWeights([0.05, -0.05]);
    setBias(0.0);
    setEpoch(0);
    setIsPlaying(false);
    setLastUpdatedPoint(null);
  };

  // Load Preset
  const loadPreset = (name) => {
    setSelectedPreset(name);
    setPoints(PRESETS[name]);
    resetWeights();
  };

  // Perform 1 Step (Perceptron Learning Rule on one misclassified sample)
  const stepOne = () => {
    if (isFullyConverged) return;

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      const sum = weights[0] * pt.x + weights[1] * pt.y + bias;
      const pred = sum >= 0 ? 1 : 0;
      const err = pt.label - pred;

      if (err !== 0) {
        // Perceptron Learning Rule:
        // w <- w + lr * err * x
        // b <- b + lr * err
        const newW1 = weights[0] + lr * err * pt.x;
        const newW2 = weights[1] + lr * err * pt.y;
        const newBias = bias + lr * err;

        setWeights([newW1, newW2]);
        setBias(newBias);
        setLastUpdatedPoint(pt);
        return;
      }
    }
  };

  // Perform 1 Epoch (Full sweep over all points)
  const stepEpoch = () => {
    let currentW1 = weights[0];
    let currentW2 = weights[1];
    let currentBias = bias;
    let anyError = false;

    for (const pt of points) {
      const sum = currentW1 * pt.x + currentW2 * pt.y + currentBias;
      const pred = sum >= 0 ? 1 : 0;
      const err = pt.label - pred;

      if (err !== 0) {
        anyError = true;
        currentW1 += lr * err * pt.x;
        currentW2 += lr * err * pt.y;
        currentBias += lr * err;
      }
    }

    setWeights([currentW1, currentW2]);
    setBias(currentBias);
    setEpoch(e => e + 1);

    if (!anyError) {
      setIsPlaying(false);
    }
  };

  // Auto-play loop
  useEffect(() => {
    let timer;
    if (isPlaying && !isFullyConverged) {
      timer = setInterval(() => {
        stepEpoch();
      }, 300);
    }
    return () => clearInterval(timer);
  }, [isPlaying, isFullyConverged, weights, bias, points, lr]);

  // Coordinate transformation: from [-1.5, 1.5] math space to Canvas pixel space
  const mathToCanvas = (mx, my, width, height) => {
    const range = 3.0; // from -1.5 to 1.5
    const cx = ((mx + 1.5) / range) * width;
    const cy = height - ((my + 1.5) / range) * height; // Invert Y
    return [cx, cy];
  };

  const canvasToMath = (cx, cy, width, height) => {
    const range = 3.0;
    const mx = (cx / width) * range - 1.5;
    const my = ((height - cy) / height) * range - 1.5;
    return [mx, my];
  };

  // Handle Canvas Click to add point
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const [mx, my] = canvasToMath(cx, cy, canvas.width, canvas.height);
    // Round to 2 decimal places
    const newPt = {
      x: Math.round(mx * 10) / 10,
      y: Math.round(my * 10) / 10,
      label: addModeLabel
    };

    setPoints([...points, newPt]);
    setEpoch(0);
  };

  // Canvas Rendering (60 FPS redraw on state change)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // 1. Draw Grid Lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;

    for (let x = -1.5; x <= 1.5; x += 0.5) {
      const [cx1, cy1] = mathToCanvas(x, -1.5, width, height);
      const [cx2, cy2] = mathToCanvas(x, 1.5, width, height);
      ctx.beginPath();
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
      ctx.stroke();
    }

    for (let y = -1.5; y <= 1.5; y += 0.5) {
      const [cx1, cy1] = mathToCanvas(-1.5, y, width, height);
      const [cx2, cy2] = mathToCanvas(1.5, y, width, height);
      ctx.beginPath();
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
      ctx.stroke();
    }

    // 2. Draw Main Axes (x=0, y=0)
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.5;

    const [origX, origY] = mathToCanvas(0, 0, width, height);
    // X Axis
    ctx.beginPath();
    ctx.moveTo(0, origY);
    ctx.lineTo(width, origY);
    ctx.stroke();
    // Y Axis
    ctx.beginPath();
    ctx.moveTo(origX, 0);
    ctx.lineTo(origX, height);
    ctx.stroke();

    // 3. Draw Decision Boundary Line & Shaded Halves
    // w1 * x + w2 * y + b = 0
    const [w1, w2] = weights;

    if (Math.abs(w1) > 1e-5 || Math.abs(w2) > 1e-5) {
      // Find two points on boundary at borders x = -1.5 and x = 1.5
      // y = -(w1*x + b) / w2
      let p1, p2;

      if (Math.abs(w2) > 0.001) {
        const y_left = -(w1 * (-1.5) + bias) / w2;
        const y_right = -(w1 * (1.5) + bias) / w2;
        p1 = mathToCanvas(-1.5, y_left, width, height);
        p2 = mathToCanvas(1.5, y_right, width, height);
      } else {
        // Vertical line x = -b / w1
        const x_vert = -bias / w1;
        p1 = mathToCanvas(x_vert, -1.5, width, height);
        p2 = mathToCanvas(x_vert, 1.5, width, height);
      }

      // Draw decision line
      ctx.save();
      ctx.strokeStyle = '#38bdf8'; // Cyan
      ctx.lineWidth = 3;
      ctx.shadowColor = 'rgba(56, 189, 248, 0.6)';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
      ctx.restore();

      // Draw Normal Vector (Weight Arrow)
      // Normal points towards (w1, w2) from boundary midpoint
      const midMathX = 0;
      const midMathY = Math.abs(w2) > 0.001 ? -(w1 * midMathX + bias) / w2 : 0;
      const [midCx, midCy] = mathToCanvas(midMathX, midMathY, width, height);

      // Scale weight vector for visibility
      const vecLen = Math.sqrt(w1 * w1 + w2 * w2) || 1;
      const normW1 = (w1 / vecLen) * 0.4;
      const normW2 = (w2 / vecLen) * 0.4;
      const [arrowCx, arrowCy] = mathToCanvas(midMathX + normW1, midMathY + normW2, width, height);

      ctx.save();
      ctx.strokeStyle = '#f59e0b'; // Amber weight normal
      ctx.fillStyle = '#f59e0b';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(midCx, midCy);
      ctx.lineTo(arrowCx, arrowCy);
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(arrowCy - midCy, arrowCx - midCx);
      ctx.beginPath();
      ctx.moveTo(arrowCx, arrowCy);
      ctx.lineTo(arrowCx - 10 * Math.cos(angle - Math.PI / 6), arrowCy - 10 * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(arrowCx - 10 * Math.cos(angle + Math.PI / 6), arrowCy - 10 * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // 4. Draw Data Points
    predictions.forEach(pt => {
      const [cx, cy] = mathToCanvas(pt.x, pt.y, width, height);
      const isClass1 = pt.label === 1;

      ctx.save();
      // Outer glow if misclassified
      if (!pt.correct) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Point Circle
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.fillStyle = isClass1 ? '#38bdf8' : '#f43f5e'; // Blue vs Rose
      ctx.shadowColor = isClass1 ? 'rgba(56, 189, 248, 0.5)' : 'rgba(244, 63, 94, 0.5)';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Coordinates text
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px monospace';
      ctx.fillText(`(${pt.x},${pt.y})`, cx + 12, cy + 4);
      ctx.restore();
    });
  }, [points, weights, bias, predictions]);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold mb-1">
            <CheckCircle className="w-3.5 h-3.5" />
            1957 Interactive Playground
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            The Perceptron 2D Decision Boundary
          </h3>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Watch Rosenblatt's Perceptron Learning Rule physically rotate and translate the decision hyperplane in real-time.
          </p>
        </div>

        {/* Presets Bar */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <span className="text-[10px] uppercase font-mono font-bold text-slate-500 px-2">Presets:</span>
          {['OR', 'AND', 'SEPARABLE', 'XOR'].map((name) => (
            <button
              key={name}
              onClick={() => loadPreset(name)}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition ${
                selectedPreset === name
                  ? name === 'XOR'
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                    : 'bg-cyan-500 text-slate-950'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: 2D Canvas */}
        <div className="lg:col-span-7 flex flex-col items-center">
          <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl">
            <canvas
              ref={canvasRef}
              width={480}
              height={440}
              onClick={handleCanvasClick}
              className="cursor-crosshair block"
            />
            {/* Overlay hint */}
            <div className="absolute bottom-2 left-3 text-[11px] font-mono text-slate-500 bg-slate-950/80 px-2 py-0.5 rounded border border-slate-800">
              Click canvas to add Class {addModeLabel} ({addModeLabel === 1 ? 'Blue' : 'Red'}) point
            </div>
            {/* Legend */}
            <div className="absolute top-2 right-3 flex items-center gap-3 text-[11px] font-mono bg-slate-950/80 px-2.5 py-1 rounded border border-slate-800">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
                <span className="text-slate-300">Class 1</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-400" />
                <span className="text-slate-300">Class 0</span>
              </div>
              <div className="flex items-center gap-1.5 text-amber-400">
                <span>→</span>
                <span>Normal w</span>
              </div>
            </div>
          </div>

          {/* Add Point Mode Toggle */}
          <div className="flex items-center gap-3 mt-3">
            <span className="text-xs font-mono text-slate-400">Next Click Adds:</span>
            <button
              onClick={() => setAddModeLabel(1)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-bold border transition ${
                addModeLabel === 1
                  ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                  : 'bg-slate-900 border-slate-800 text-slate-500'
              }`}
            >
              + Blue Point (Class 1)
            </button>
            <button
              onClick={() => setAddModeLabel(0)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-bold border transition ${
                addModeLabel === 0
                  ? 'bg-rose-500/20 border-rose-400 text-rose-300'
                  : 'bg-slate-900 border-slate-800 text-slate-500'
              }`}
            >
              + Red Point (Class 0)
            </button>
          </div>
        </div>

        {/* Right: Controls & Real-Time Parameter Meters */}
        <div className="lg:col-span-5 space-y-4">
          {/* Controls Bar */}
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
              Training Controls
            </div>

            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition ${
                  isPlaying
                    ? 'bg-amber-500 hover:bg-amber-600 text-slate-950'
                    : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isPlaying ? 'Pause' : 'Auto Train'}
              </button>

              <button
                onClick={stepEpoch}
                disabled={isPlaying || isFullyConverged}
                className="py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-semibold text-slate-200 flex items-center justify-center gap-1 transition"
              >
                <StepForward className="w-3.5 h-3.5" />
                1 Epoch
              </button>

              <button
                onClick={resetWeights}
                className="py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 flex items-center justify-center gap-1 transition"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            </div>

            {/* Learning Rate Slider */}
            <div className="pt-2">
              <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
                <span>Learning Rate (η)</span>
                <span className="text-cyan-400 font-bold">{lr.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={lr}
                onChange={(e) => setLr(parseFloat(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>
          </div>

          {/* Model Mathematical State */}
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3 font-mono text-xs">
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
              <span>Neuron State</span>
              <span className="text-slate-500">Epoch: {epoch}</span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <div className="text-slate-500 text-[10px]">Weight w₁</div>
                <div className="text-cyan-400 font-bold text-sm mt-0.5">{weights[0].toFixed(3)}</div>
              </div>
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <div className="text-slate-500 text-[10px]">Weight w₂</div>
                <div className="text-cyan-400 font-bold text-sm mt-0.5">{weights[1].toFixed(3)}</div>
              </div>
              <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                <div className="text-slate-500 text-[10px]">Bias b</div>
                <div className="text-cyan-400 font-bold text-sm mt-0.5">{bias.toFixed(3)}</div>
              </div>
            </div>

            {/* Hyperplane Formula */}
            <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-center text-slate-300">
              <span className="text-slate-500">Decision Line: </span>
              <span className="text-cyan-300">
                {weights[0].toFixed(2)}x₁ + {weights[1].toFixed(2)}x₂ + ({bias.toFixed(2)}) = 0
              </span>
            </div>

            {/* Accuracy & Convergence Meter */}
            <div className="pt-1">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Classification Accuracy:</span>
                <span className={`font-bold ${isFullyConverged ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {accuracy.toFixed(0)}% ({correctCount}/{points.length} correct)
                </span>
              </div>
              <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${isFullyConverged ? 'bg-emerald-400' : 'bg-amber-400'}`}
                  style={{ width: `${accuracy}%` }}
                />
              </div>
            </div>

            {selectedPreset === 'XOR' && (
              <div className="mt-3 p-3 rounded-lg bg-rose-950/40 border border-rose-500/40 text-[11px] text-rose-300 flex items-start gap-2 leading-relaxed">
                <AlertOctagon className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <div>
                  <strong>The 1969 XOR Barrier in Action!</strong> No matter how many epochs you run, a single line CANNOT achieve 100% accuracy on XOR. Notice how it endlessly oscillates between 50% and 75%.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
