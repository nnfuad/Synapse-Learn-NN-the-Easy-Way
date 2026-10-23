import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  Microscope,
  Layers,
  Sparkles,
  ArrowRight,
  Eye,
  Activity,
  Sliders,
  CheckCircle2,
  Info,
  Compass,
  Cpu,
} from 'lucide-react';

// Preset Induction Sequences
const INDUCTION_PRESETS = [
  {
    id: 'ALPHABET',
    name: 'Letter Sequence [A, B, C, D, A] -> B',
    tokens: ['A', 'B', 'C', 'D', 'A'],
    target: 'B',
  },
  {
    id: 'WORDS',
    name: 'Language Copy [The, Cat, Sat, On, The] -> Cat',
    tokens: ['The', 'Cat', 'Sat', 'On', 'The'],
    target: 'Cat',
  },
  {
    id: 'CODE',
    name: 'Variable Syntax [def, foo, int, x, def] -> foo',
    tokens: ['def', 'foo', 'int', 'x', 'def'],
    target: 'foo',
  },
];

// SAE Interpretable Monosemantic Dictionary Features
const SAE_DICTIONARY_FEATURES = [
  { id: 0, label: 'Feature 0: Golden Gate Bridge', domain: 'Geography & Landmarks', angle: 0, color: '#f43f5e' },
  { id: 1, label: 'Feature 1: Differential Equations', domain: 'Mathematics & Calculus', angle: (2 * Math.PI) / 5, color: '#06b6d4' },
  { id: 2, label: 'Feature 2: Python Code Syntax', domain: 'Computer Programming', angle: (4 * Math.PI) / 5, color: '#10b981' },
  { id: 3, label: 'Feature 3: Shakespearean Sonnets', domain: 'Literature & Verse', angle: (6 * Math.PI) / 5, color: '#a855f7' },
  { id: 4, label: 'Feature 4: Quantum Entanglement', domain: 'Theoretical Physics', angle: (8 * Math.PI) / 5, color: '#f59e0b' },
];

export default function Module09Visualizer() {
  const [activeTab, setActiveTab] = useState('INDUCTION_CIRCUITS'); // 'INDUCTION_CIRCUITS' or 'SAE_SUPERPOSITION'
  const canvasRef = useRef(null);

  // Induction Circuit States
  const [selectedInductionPreset, setSelectedInductionPreset] = useState(0);
  const [hoveredCell, setHoveredCell] = useState(null); // { layer: 0|1, row: i, col: j }
  const [activeStep, setActiveStep] = useState(2); // 0 = Input, 1 = L0, 2 = L1

  // SAE Superposition States
  const [lambdaSparsity, setLambdaSparsity] = useState(0.04);
  const [selectedSaeFeature, setSelectedSaeFeature] = useState(null);

  // Active Induction Sequence
  const activeSeq = INDUCTION_PRESETS[selectedInductionPreset];
  const seqTokens = activeSeq.tokens;
  const seqLen = seqTokens.length;

  // Build Layer 0 (Previous-Token Head) Attention Matrix
  // Strictly attends to token at i - 1
  const layer0Attn = [];
  for (let i = 0; i < seqLen; i++) {
    const row = [];
    for (let j = 0; j < seqLen; j++) {
      if (j > i) {
        row.push(0); // causal mask
      } else if (j === i - 1) {
        row.push(0.85); // strong previous token attention
      } else if (j === i) {
        row.push(0.12); // self attention
      } else {
        row.push(0.03 / Math.max(1, i - 1));
      }
    }
    // Normalize row
    const sum = row.reduce((a, b) => a + b, 0);
    layer0Attn.push(row.map((v) => v / sum));
  }

  // Build Layer 1 (Induction Head) Attention Matrix
  // At the query token (position 4: repeated token A), spikes attention to position 1 (token B)!
  const layer1Attn = [];
  for (let i = 0; i < seqLen; i++) {
    const row = [];
    for (let j = 0; j < seqLen; j++) {
      if (j > i) {
        row.push(0); // causal mask
      } else if (i === seqLen - 1 && j === 1) {
        // Induction spike! Query A matches predecessor tag at B
        row.push(0.94);
      } else if (j === i) {
        row.push(0.04);
      } else {
        row.push(0.02);
      }
    }
    // Normalize row
    const sum = row.reduce((a, b) => a + b, 0);
    layer1Attn.push(row.map((v) => v / sum));
  }

  // Canvas Rendering for SAE Superposition Disentanglement
  useEffect(() => {
    if (activeTab !== 'SAE_SUPERPOSITION') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const toCanvasX = (x) => ((x + 2.4) / 4.8) * width;
    const toCanvasY = (y) => ((-y + 2.4) / 4.8) * height;

    // Clear Canvas
    ctx.fillStyle = '#080d1a';
    ctx.fillRect(0, 0, width, height);

    // Axes
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.stroke();

    // Unit Circle
    ctx.strokeStyle = '#334155';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(0), (1.0 / 4.8) * width, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // 1. Draw Polysemantic Activation Points in Superposition
    // Generate synthetic cloud of activations along combinations of the 5 features
    ctx.fillStyle = 'rgba(100, 116, 139, 0.45)';
    const numPoints = 180;
    for (let i = 0; i < numPoints; i++) {
      // Pick 1 or 2 active features
      const f1 = SAE_DICTIONARY_FEATURES[i % 5];
      const f2 = SAE_DICTIONARY_FEATURES[(i * 3 + 1) % 5];
      const c1 = Math.random() * 1.4;
      // Shrink secondary feature if lambda is high (sparse collapse)
      const c2 = Math.random() < 0.35 && lambdaSparsity < 0.05 ? Math.random() * 0.8 : 0;

      const px = c1 * Math.cos(f1.angle) + c2 * Math.cos(f2.angle);
      const py = c1 * Math.sin(f1.angle) + c2 * Math.sin(f2.angle);

      ctx.beginPath();
      ctx.arc(toCanvasX(px), toCanvasY(py), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // 2. Draw Ground Truth / Monosemantic Dictionary Feature Axes
    SAE_DICTIONARY_FEATURES.forEach((f, idx) => {
      const isSelected = selectedSaeFeature === idx;
      const vx = Math.cos(f.angle) * 1.5;
      const vy = Math.sin(f.angle) * 1.5;

      const startX = toCanvasX(0);
      const startY = toCanvasY(0);
      const endX = toCanvasX(vx);
      const endY = toCanvasY(vy);

      // Arrow stem
      ctx.strokeStyle = isSelected ? '#ffffff' : f.color;
      ctx.lineWidth = isSelected ? 3.5 : 2.2;
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Arrow head
      const angle = Math.atan2(endY - startY, endX - startX);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - 7 * Math.cos(angle - Math.PI / 6), endY - 7 * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(endX - 7 * Math.cos(angle + Math.PI / 6), endY - 7 * Math.sin(angle + Math.PI / 6));
      ctx.fill();

      // Feature Label
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = isSelected ? '#ffffff' : f.color;
      const textX = toCanvasX(vx * 1.15);
      const textY = toCanvasY(vy * 1.15);
      ctx.fillText(`F${f.id}`, textX - 6, textY + 4);
    });
  }, [activeTab, lambdaSparsity, selectedSaeFeature]);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900/90 to-rose-950/40 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-rose-400 font-mono text-xs font-bold tracking-wider uppercase">
              <Microscope className="w-4 h-4" />
              <span>Frontier Stage • Module 09</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
              Mechanistic Interpretability & Circuit Scans
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Peel open the neural network black box: examine the microscopic <strong>2-Layer Induction Head Circuit</strong> (Anthropic 2021)
              and untangle polysemantic neurons via <strong>Sparse Autoencoders (SAEs)</strong> (Anthropic 2023/2024).
            </p>
          </div>

          {/* Sub-Tab Selector */}
          <div className="bg-slate-950 p-1.5 rounded-xl border border-slate-800 flex items-center gap-1 self-start md:self-auto">
            <button
              onClick={() => setActiveTab('INDUCTION_CIRCUITS')}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-2 ${
                activeTab === 'INDUCTION_CIRCUITS'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-lg shadow-rose-500/10'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Cpu className="w-4 h-4 text-rose-400" />
              <span>Induction Head Circuit</span>
            </button>

            <button
              onClick={() => setActiveTab('SAE_SUPERPOSITION')}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-2 ${
                activeTab === 'SAE_SUPERPOSITION'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Brain className="w-4 h-4 text-cyan-400" />
              <span>SAE Superposition Scan</span>
            </button>
          </div>
        </div>
      </div>

      {/* VIEW 1: INDUCTION HEAD CIRCUITS */}
      {activeTab === 'INDUCTION_CIRCUITS' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Dual Layer Attention Heatmap Scanners (7 Cols) */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Eye className="w-4 h-4 text-rose-400" />
                <span>Microscopic Attention Circuit Heatmaps</span>
              </div>

              {/* Preset Selector */}
              <div className="flex items-center gap-2">
                <select
                  value={selectedInductionPreset}
                  onChange={(e) => setSelectedInductionPreset(parseInt(e.target.value))}
                  className="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-1.5 font-mono"
                >
                  {INDUCTION_PRESETS.map((p, idx) => (
                    <option key={p.id} value={idx}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Sequence Token Indicator Bar */}
            <div className="bg-[#080d1a] border border-slate-800 rounded-xl p-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500 font-mono">Prompt:</span>
                <div className="flex items-center gap-1.5 font-mono text-xs">
                  {seqTokens.map((tok, i) => (
                    <span
                      key={i}
                      className={`px-2.5 py-1 rounded-lg border font-bold ${
                        i === seqLen - 1
                          ? 'bg-rose-500/20 border-rose-500/50 text-rose-300'
                          : i === 1
                          ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                          : 'bg-slate-900 border-slate-800 text-slate-300'
                      }`}
                    >
                      {tok}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs font-mono">
                <ArrowRight className="w-4 h-4 text-emerald-400" />
                <span className="text-slate-400">Target Next:</span>
                <span className="px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-bold">
                  {activeSeq.target}
                </span>
              </div>
            </div>

            {/* Side-by-Side Heatmaps: Layer 0 vs Layer 1 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Layer 0 Heatmap */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between text-xs font-bold text-blue-400">
                  <span>Layer 0: Previous-Token Head</span>
                  <span className="text-[10px] text-slate-500 font-mono">Attends to i-1</span>
                </div>

                <div className="relative pt-2">
                  <div className="grid grid-cols-5 gap-1.5">
                    {layer0Attn.map((row, rIdx) =>
                      row.map((val, cIdx) => (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          onMouseEnter={() => setHoveredCell({ layer: 0, row: rIdx, col: cIdx, val })}
                          onMouseLeave={() => setHoveredCell(null)}
                          className="aspect-square rounded flex items-center justify-center font-mono text-[10px] transition cursor-pointer border border-transparent hover:border-white"
                          style={{
                            backgroundColor:
                              val > 0
                                ? `rgba(59, 130, 246, ${Math.max(0.1, val)})`
                                : 'rgba(15, 23, 42, 0.4)',
                            color: val > 0.4 ? '#ffffff' : '#94a3b8',
                          }}
                        >
                          {(val * 100).toFixed(0)}%
                        </div>
                      ))
                    )}
                  </div>
                </div>
                <div className="text-[10px] text-slate-500 font-mono text-center pt-1">
                  Subdiagonal line shows each token attending to its predecessor
                </div>
              </div>

              {/* Layer 1 Induction Head Heatmap */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between text-xs font-bold text-rose-400">
                  <span>Layer 1: Induction Head</span>
                  <span className="text-[10px] text-rose-400 font-mono font-bold">Induction Spike!</span>
                </div>

                <div className="relative pt-2">
                  <div className="grid grid-cols-5 gap-1.5">
                    {layer1Attn.map((row, rIdx) =>
                      row.map((val, cIdx) => {
                        const isInductionSpike = rIdx === seqLen - 1 && cIdx === 1;
                        return (
                          <div
                            key={`${rIdx}-${cIdx}`}
                            onMouseEnter={() => setHoveredCell({ layer: 1, row: rIdx, col: cIdx, val })}
                            onMouseLeave={() => setHoveredCell(null)}
                            className={`aspect-square rounded flex items-center justify-center font-mono text-[10px] transition cursor-pointer border ${
                              isInductionSpike
                                ? 'border-rose-400 animate-pulse font-extrabold text-white'
                                : 'border-transparent hover:border-white'
                            }`}
                            style={{
                              backgroundColor:
                                val > 0
                                  ? `rgba(244, 63, 94, ${Math.max(0.1, val)})`
                                  : 'rgba(15, 23, 42, 0.4)',
                              color: val > 0.4 ? '#ffffff' : '#94a3b8',
                            }}
                          >
                            {(val * 100).toFixed(0)}%
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
                <div className="text-[10px] text-rose-400 font-mono text-center pt-1 font-bold">
                  Final Token A attends directly (94%) to Token B!
                </div>
              </div>
            </div>
          </div>

          {/* Right: Step-by-Step Circuit Anatomy (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Microscope className="w-4 h-4 text-rose-400" />
                <span>The 2-Layer Induction Mechanism</span>
              </div>

              <div className="space-y-3 text-xs text-slate-300">
                <div className="bg-slate-950 p-3 rounded-xl border border-blue-500/30 space-y-1">
                  <div className="font-bold text-blue-400 flex items-center gap-1.5">
                    <span>1. Layer 0 Previous-Token Head (H₀)</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed font-mono">
                    Token at position 1 ({activeSeq.tokens[1]}) attends to position 0 ({activeSeq.tokens[0]}).
                    It writes token {activeSeq.tokens[0]}'s identity into token {activeSeq.tokens[1]}'s residual stream.
                  </p>
                </div>

                <div className="bg-slate-950 p-3 rounded-xl border border-rose-500/30 space-y-1">
                  <div className="font-bold text-rose-400 flex items-center gap-1.5">
                    <span>2. Layer 1 Induction Head (H₁)</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed font-mono">
                    At position 4, token {activeSeq.tokens[4]} emits query: <em>"Find where I previously appeared!"</em>.
                    Key at position 1 holds tag <em>"preceded by {activeSeq.tokens[0]}"</em>.
                    Dot product Q·K spikes to 94%!
                  </p>
                </div>

                <div className="bg-slate-950 p-3 rounded-xl border border-emerald-500/30 space-y-1">
                  <div className="font-bold text-emerald-400 flex items-center gap-1.5">
                    <span>3. OV Circuit Logit Boost</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed font-mono">
                    The OV circuit reads token {activeSeq.target}'s vector from position 1 and directly boosts
                    {activeSeq.target}'s logit in the unembedding layer W_U!
                  </p>
                </div>
              </div>
            </div>

            {/* In-Context Learning Fact */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2 text-xs text-slate-300">
              <span className="text-amber-400 font-bold flex items-center gap-1.5">
                <Sparkles className="w-4 h-4" />
                <span>The Sudden "Induction Bump"</span>
              </span>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                During LLM pretraining, in-context learning doesn't develop smoothly. An abrupt macroscopic phase change
                (the <em>induction bump</em>) occurs the exact moment when Layer 0 and Layer 1 attention heads form this synchronized 2-layer circuit!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: SPARSE AUTOENCODER (SAE) SUPERPOSITION */}
      {activeTab === 'SAE_SUPERPOSITION' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: 2D Superposition Activation Canvas (7 Cols) */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center">
            <div className="w-full flex items-center justify-between pb-3 border-b border-slate-800 text-xs">
              <div className="flex items-center gap-2 text-slate-300 font-semibold">
                <Brain className="w-4 h-4 text-cyan-400" />
                <span>2D Activation Plane (Superposition vs Monosemantic Dictionary)</span>
              </div>
              <span className="font-mono text-cyan-400 font-bold">5 Features in 2 Neurons</span>
            </div>

            {/* Canvas */}
            <div className="relative my-4 rounded-xl overflow-hidden border border-slate-800/80 shadow-2xl">
              <canvas
                ref={canvasRef}
                width={500}
                height={400}
                className="block w-full max-w-[500px] h-auto bg-[#080d1a]"
              />
              <div className="absolute bottom-3 left-3 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800 text-[11px] font-mono text-slate-400">
                Colored Vectors: Learned SAE Decoder Columns (||W_dec|| = 1.0)
              </div>
            </div>

            {/* Sparsity Slider */}
            <div className="w-full space-y-2 mt-2 px-2">
              <div className="flex justify-between text-xs text-slate-400 font-mono">
                <span>Dense Superposition (Polysemantic)</span>
                <span className="text-cyan-400 font-bold">L1 Penalty λ = {lambdaSparsity.toFixed(3)}</span>
                <span>Sparse Dictionary (Monosemantic)</span>
              </div>
              <input
                type="range"
                min="0.005"
                max="0.10"
                step="0.005"
                value={lambdaSparsity}
                onChange={(e) => setLambdaSparsity(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 rounded-lg cursor-pointer h-2"
              />
            </div>
          </div>

          {/* Right: Monosemantic Dictionary Explorer (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-bold text-sm">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  <span>Disentangled Monosemantic Features</span>
                </div>
                <span className="text-xs font-mono text-emerald-400 font-bold">
                  R² = {(0.99 - lambdaSparsity * 0.1).toFixed(2)}
                </span>
              </div>

              <div className="space-y-2">
                {SAE_DICTIONARY_FEATURES.map((feat, idx) => {
                  const isSelected = selectedSaeFeature === idx;
                  return (
                    <div
                      key={feat.id}
                      onClick={() => setSelectedSaeFeature(isSelected ? null : idx)}
                      className={`p-3 rounded-xl border cursor-pointer transition ${
                        isSelected
                          ? 'bg-slate-800 border-white text-white shadow-lg'
                          : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2 font-bold" style={{ color: feat.color }}>
                          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: feat.color }} />
                          <span>{feat.label.split(':')[1]}</span>
                        </div>
                        <span className="text-[10px] font-mono text-slate-500">{feat.domain}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Why SAEs Work Fact Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2 text-xs text-slate-300">
              <div className="flex items-center gap-2 text-amber-400 font-bold">
                <Sparkles className="w-4 h-4" />
                <span>The Claude 3 "Golden Gate" Feature</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                In 2024, Anthropic trained an SAE on Claude 3 Sonnet and extracted millions of monosemantic features.
                By clamping the "Golden Gate Bridge" feature activation to high values, Claude identified itself as the Golden Gate Bridge in every response!
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
