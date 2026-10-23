import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, ArrowRight, ShieldCheck, HelpCircle, Layers, Lightbulb } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function XorCrisisVisualizer() {
  const [warpProgress, setWarpProgress] = useState(0.0); // 0.0 (Input Space) to 1.0 (Hidden Feature Space)
  const [isWarped, setIsWarped] = useState(false);

  // XOR 4 Points in original (x1, x2) space
  const initialPoints = [
    { name: "(0,0)", x: 0, y: 0, label: 0 },
    { name: "(0,1)", x: 0, y: 1, label: 1 },
    { name: "(1,0)", x: 1, y: 0, label: 1 },
    { name: "(1,1)", x: 1, y: 1, label: 0 },
  ];

  // Hidden Layer Transformation:
  // Let hidden neuron 1: h1 = ReLU(x1 + x2 - 0.5)
  // Let hidden neuron 2: h2 = ReLU(x1 + x2 - 1.5)
  // For (0,0): h1 = 0,    h2 = 0
  // For (0,1): h1 = 0.5,  h2 = 0
  // For (1,0): h1 = 0.5,  h2 = 0
  // For (1,1): h1 = 1.5,  h2 = 0.5
  // Now Class 1 is at (0.5, 0), and Class 0 is at (0,0) and (1.5, 0.5) - completely linearly separable!
  const getTransformedCoordinates = (p, t) => {
    // Hidden target coords
    const h1 = Math.max(0, p.x + p.y - 0.5);
    const h2 = Math.max(0, p.x + p.y - 1.5);

    // Interpolate smoothly with progress t
    const curX = p.x * (1 - t) + h1 * t;
    const curY = p.y * (1 - t) + h2 * t;

    return { curX, curY, h1, h2 };
  };

  const canvasRef = useRef(null);

  // Trigger celebration when fully warped
  useEffect(() => {
    if (warpProgress >= 0.99 && !isWarped) {
      setIsWarped(true);
      try {
        confetti({
          particleCount: 60,
          spread: 70,
          origin: { y: 0.7 }
        });
      } catch (e) {}
    } else if (warpProgress < 0.99) {
      setIsWarped(false);
    }
  }, [warpProgress, isWarped]);

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Map math [-0.5, 2.0] to canvas pixels
    const toPx = (mx, my) => {
      const minX = -0.4, maxX = 1.8;
      const minY = -0.4, maxY = 1.8;
      const px = ((mx - minX) / (maxX - minX)) * width;
      const py = height - ((my - minY) / (maxY - minY)) * height;
      return [px, py];
    };

    // Draw grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 1.5; g += 0.5) {
      const [x1, y1] = toPx(g, -0.4);
      const [x2, y2] = toPx(g, 1.8);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      const [hx1, hy1] = toPx(-0.4, g);
      const [hx2, hy2] = toPx(1.8, g);
      ctx.beginPath();
      ctx.moveTo(hx1, hy1);
      ctx.lineTo(hx2, hy2);
      ctx.stroke();
    }

    // Draw Axes
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.5;
    const [ax0, ay0] = toPx(0, 0);
    // X axis
    ctx.beginPath();
    ctx.moveTo(0, ay0);
    ctx.lineTo(width, ay0);
    ctx.stroke();
    // Y axis
    ctx.beginPath();
    ctx.moveTo(ax0, 0);
    ctx.lineTo(ax0, height);
    ctx.stroke();

    // If fully warped, draw the triumphant separating line!
    if (warpProgress > 0.8) {
      const alpha = (warpProgress - 0.8) / 0.2;
      // In hidden space, Class 1 is at (0.5, 0).
      // Class 0 is at (0,0) and (1.5, 0.5).
      // Separating line 1: h1 = 0.25
      const [sp1, sp2] = [toPx(0.25, -0.3), toPx(0.25, 1.6)];
      ctx.save();
      ctx.strokeStyle = `rgba(16, 185, 129, ${alpha})`; // Emerald
      ctx.lineWidth = 3;
      ctx.shadowColor = 'rgba(16, 185, 129, 0.8)';
      ctx.shadowBlur = 15;
      ctx.beginPath();
      ctx.moveTo(sp1[0], sp1[1]);
      ctx.lineTo(sp2[0], sp2[1]);
      ctx.stroke();

      // Separating line 2: h1 - h2 = 0.8
      const [lp1, lp2] = [toPx(0.8, 0.0), toPx(1.8, 1.0)];
      ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`; // Cyan
      ctx.beginPath();
      ctx.moveTo(lp1[0], lp1[1]);
      ctx.lineTo(lp2[0], lp2[1]);
      ctx.stroke();
      ctx.restore();
    }

    // Draw Points and motion trails
    initialPoints.forEach(p => {
      const { curX, curY } = getTransformedCoordinates(p, warpProgress);
      const [cx, cy] = toPx(curX, curY);
      const isClass1 = p.label === 1;

      // Draw dashed motion vector trail from initial pos
      if (warpProgress > 0.05) {
        const [initPx, initPy] = toPx(p.x, p.y);
        ctx.save();
        ctx.strokeStyle = '#334155';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(initPx, initPy);
        ctx.lineTo(cx, cy);
        ctx.stroke();
        ctx.restore();
      }

      // Point circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fillStyle = isClass1 ? '#38bdf8' : '#f43f5e';
      ctx.shadowColor = isClass1 ? 'rgba(56, 189, 248, 0.6)' : 'rgba(244, 63, 94, 0.6)';
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '11px monospace';
      ctx.fillText(`${p.name} [Class ${p.label}]`, cx + 14, cy + 4);
      ctx.restore();
    });
  }, [warpProgress]);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 text-xs font-semibold mb-1">
            <Layers className="w-3.5 h-3.5" />
            The Architectural Breakthrough (1986)
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            How Hidden Layers Resolve XOR via "Space Warping"
          </h3>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Drag the slider to watch non-linear hidden neurons twist the 2D coordinate space, transforming an impossible problem into trivial linear separation.
          </p>
        </div>

        <button
          onClick={() => setWarpProgress(warpProgress === 1 ? 0 : 1)}
          className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs flex items-center gap-2 transition shadow-lg shadow-purple-600/20"
        >
          <Sparkles className="w-3.5 h-3.5" />
          {warpProgress > 0.5 ? 'Reset to Input Space' : 'Snap to Warped Hidden Space'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        {/* Left: Dynamic 2D Canvas */}
        <div className="lg:col-span-7 flex flex-col items-center">
          <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl">
            <canvas
              ref={canvasRef}
              width={460}
              height={420}
              className="block"
            />
            {/* Axis Title Overlay */}
            <div className="absolute top-3 left-4 text-xs font-mono font-bold text-slate-400 bg-slate-950/80 px-2.5 py-1 rounded border border-slate-800">
              {warpProgress < 0.3
                ? 'Input Space: (x₁, x₂)'
                : warpProgress > 0.7
                ? 'Hidden Feature Space: (h₁, h₂)'
                : 'Intermediate Nonlinear Interpolation'}
            </div>
            {warpProgress > 0.8 && (
              <div className="absolute bottom-3 right-4 text-xs font-mono font-bold text-emerald-400 bg-emerald-950/80 px-3 py-1 rounded border border-emerald-500/40 animate-pulse">
                ✨ Linearly Separable!
              </div>
            )}
          </div>

          {/* Slider */}
          <div className="w-full max-w-md mt-4 space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-slate-400">
              <span>Input Space (t = 0.0)</span>
              <span className="text-purple-400 font-bold">Warp Factor: {(warpProgress * 100).toFixed(0)}%</span>
              <span>Hidden Space (t = 1.0)</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={warpProgress}
              onChange={(e) => setWarpProgress(parseFloat(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
          </div>
        </div>

        {/* Right: The Intuition & Mathematical Proof */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-amber-400 text-xs font-bold uppercase tracking-wider">
              <Lightbulb className="w-4 h-4" />
              The Geometric Intuition
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              In raw input space $(x_1, x_2)$, Class 1 $(0,1)$ and $(1,0)$ are trapped diagonally between Class 0 $(0,0)$ and $(1,1)$. No straight line can cut between them without cutting the wrong pairs.
            </p>
            <p className="text-xs text-slate-300 leading-relaxed">
              When we add a hidden layer with non-linear activation (like ReLU), the network bends the coordinate sheet:
            </p>
            <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 font-mono text-[11px] text-cyan-300 space-y-1">
              <div>h₁ = ReLU(x₁ + x₂ - 0.5)</div>
              <div>h₂ = ReLU(x₁ + x₂ - 1.5)</div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Points $(0,1)$ and $(1,0)$ collapse onto the EXACT same hidden position $(0.5, 0.0)$, untangling the diagonal trap and allowing the final neuron to slice cleanly!
            </p>
          </div>

          {/* Transformation Table */}
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 font-mono text-xs">
            <div className="text-[10px] font-bold uppercase text-slate-500 mb-2">
              Coordinate Mapping Matrix
            </div>
            <div className="grid grid-cols-4 pb-1 border-b border-slate-800 text-slate-400 text-center text-[11px]">
              <span>Point</span>
              <span>Class</span>
              <span>Input (x₁,x₂)</span>
              <span>Hidden (h₁,h₂)</span>
            </div>
            <div className="divide-y divide-slate-800/40 text-[11px]">
              <div className="grid grid-cols-4 py-1.5 text-center text-slate-300">
                <span>(0,0)</span>
                <span className="text-rose-400 font-bold">0</span>
                <span>[0, 0]</span>
                <span className="text-cyan-400 font-bold">[0.0, 0.0]</span>
              </div>
              <div className="grid grid-cols-4 py-1.5 text-center text-slate-300">
                <span>(0,1)</span>
                <span className="text-cyan-400 font-bold">1</span>
                <span>[0, 1]</span>
                <span className="text-cyan-400 font-bold">[0.5, 0.0]</span>
              </div>
              <div className="grid grid-cols-4 py-1.5 text-center text-slate-300">
                <span>(1,0)</span>
                <span className="text-cyan-400 font-bold">1</span>
                <span>[1, 0]</span>
                <span className="text-cyan-400 font-bold">[0.5, 0.0]</span>
              </div>
              <div className="grid grid-cols-4 py-1.5 text-center text-slate-300">
                <span>(1,1)</span>
                <span className="text-rose-400 font-bold">0</span>
                <span>[1, 1]</span>
                <span className="text-cyan-400 font-bold">[1.5, 0.5]</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
