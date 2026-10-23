import React, { useState, useEffect, useMemo } from 'react';
import {
  Brain,
  Zap,
  Sparkles,
  Layers,
  Eye,
  Sliders,
  RotateCw,
  Play,
  RotateCcw,
  CheckCircle2,
  Info,
  Lock,
  Unlock,
  Cpu,
  ArrowRight,
  Database,
  Compass,
} from 'lucide-react';

export default function Module06Visualizer() {
  const [activeSubTab, setActiveSubTab] = useState('qkv');

  // --- SubTab 1: QKV Attention & Spotlight ---
  const PRESETS = [
    {
      id: 'winograd',
      name: 'Winograd Coreference ("it" -> "animal")',
      tokens: ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'tired'],
      defaultQuery: 7, // "it"
    },
    {
      id: 'polysemy',
      name: 'Polysemy ("bank robber" vs "river bank")',
      tokens: ['The', 'bank', 'robber', 'fled', 'down', 'the', 'river', 'bank'],
      defaultQuery: 1, // first "bank"
    },
    {
      id: 'vaswani',
      name: 'Attention Is All You Need',
      tokens: ['Attention', 'is', 'all', 'you', 'really', 'need'],
      defaultQuery: 0,
    },
  ];

  const [selectedPresetId, setSelectedPresetId] = useState('winograd');
  const activePreset = useMemo(
    () => PRESETS.find((p) => p.id === selectedPresetId) || PRESETS[0],
    [selectedPresetId]
  );

  const [tokens, setTokens] = useState(activePreset.tokens);
  const [selectedQueryIdx, setSelectedQueryIdx] = useState(activePreset.defaultQuery);
  const [isCausal, setIsCausal] = useState(false);
  const [useSqrtScaling, setUseSqrtScaling] = useState(true);
  const d_k = 64; // standard head dimension
  const sqrtDk = Math.sqrt(d_k);

  // Sync when preset changes
  useEffect(() => {
    setTokens(activePreset.tokens);
    setSelectedQueryIdx(Math.min(activePreset.defaultQuery, activePreset.tokens.length - 1));
  }, [selectedPresetId, activePreset]);

  // Deterministic pseudo-attention weight simulator based on semantic resonance
  const attentionWeights = useMemo(() => {
    const N = tokens.length;
    const matrix = [];

    for (let i = 0; i < N; i++) {
      const row = [];
      let sumExp = 0;
      const scores = [];

      for (let j = 0; j < N; j++) {
        let rawScore = 0;

        // Custom semantic affinities
        const ti = tokens[i].toLowerCase();
        const tj = tokens[j].toLowerCase();

        if (ti === 'it' && tj === 'animal') rawScore = 8.5;
        else if (ti === 'it' && tj === 'tired') rawScore = 6.2;
        else if (ti === 'it' && tj === 'street') rawScore = 1.0;
        else if (i === 1 && ti === 'bank' && tj === 'robber') rawScore = 9.0;
        else if (i === 7 && ti === 'bank' && tj === 'river') rawScore = 9.2;
        else if (i === j) rawScore = 4.5; // Self-attention baseline
        else if (Math.abs(i - j) === 1) rawScore = 3.2; // Local syntactic neighbor
        else rawScore = 1.0 + ((i * 3 + j * 7) % 5) * 0.4;

        // Causal mask: future tokens (j > i) blocked
        if (isCausal && j > i) {
          rawScore = -1e9;
        }

        // Scaling factor
        const scaledScore = isCausal && j > i ? -1e9 : (useSqrtScaling ? rawScore / (sqrtDk / 8) : rawScore);
        scores.push(scaledScore);
      }

      // Softmax
      const maxScore = Math.max(...scores.filter((s) => s > -1e8), 0);
      for (let j = 0; j < N; j++) {
        if (isCausal && j > i) {
          row.push(0.0);
        } else {
          const expVal = Math.exp(scores[j] - maxScore);
          row.push(expVal);
          sumExp += expVal;
        }
      }

      // Normalize
      const normalizedRow = row.map((v) => (sumExp > 0 ? v / sumExp : 0.0));
      matrix.push(normalizedRow);
    }

    return matrix;
  }, [tokens, isCausal, useSqrtScaling, sqrtDk]);

  // Current query's attention distribution over all keys
  const currentAttnDistribution = attentionWeights[selectedQueryIdx] || [];

  // --- SubTab 2: KV-Cache Simulator ---
  const promptTokens = ['Once', 'upon', 'a', 'time'];
  const generatedTokenPool = ['there', 'lived', 'a', 'brave', 'syntactic', 'dragon', 'in', 'SRAM'];
  const [cachedTokenCount, setCachedTokenCount] = useState(4); // initial prompt length = 4

  const handleStepKVCache = () => {
    if (cachedTokenCount < promptTokens.length + generatedTokenPool.length) {
      setCachedTokenCount((prev) => prev + 1);
    }
  };

  const handleResetKVCache = () => {
    setCachedTokenCount(promptTokens.length);
  };

  const allActiveTokens = useMemo(() => {
    return [
      ...promptTokens,
      ...generatedTokenPool.slice(0, Math.max(0, cachedTokenCount - promptTokens.length)),
    ];
  }, [cachedTokenCount]);

  // Compute FLOPs comparison
  const flopsWithoutCache = useMemo(() => {
    // Sum of t^2 for all decoding steps
    let sum = 0;
    for (let t = promptTokens.length; t <= cachedTokenCount; t++) {
      sum += t * t * d_k;
    }
    return sum;
  }, [cachedTokenCount, d_k]);

  const flopsWithCache = useMemo(() => {
    // Only 1 query row @ t keys
    let sum = 0;
    for (let t = promptTokens.length; t <= cachedTokenCount; t++) {
      sum += 1 * t * d_k;
    }
    return sum;
  }, [cachedTokenCount, d_k]);

  // --- SubTab 3: RoPE 2D Rotation Dial ---
  const [posM, setPosM] = useState(3);
  const [posN, setPosN] = useState(1);
  const [freqTheta, setFreqTheta] = useState(0.4);

  // Compute 2D rotation for vectors q and k
  const angleM = posM * freqTheta;
  const angleN = posN * freqTheta;
  const angleDiff = (posM - posN) * freqTheta;

  // Initial unit vectors at angle 0
  const q_x = Math.cos(angleM);
  const q_y = Math.sin(angleM);
  const k_x = Math.cos(angleN);
  const k_y = Math.sin(angleN);
  const ropeInnerProduct = q_x * k_x + q_y * k_y; // equals cos(angleM - angleN) = cos((m-n)*theta)

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Historical Context */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 relative overflow-hidden backdrop-blur-sm">
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono font-semibold text-purple-400 mb-2">
              <span>MODULE 06</span>
              <span>•</span>
              <span>ERA: 2017 - 2024</span>
              <span>•</span>
              <span className="text-cyan-400">VASWANI → ROPE → FLASHATTENTION → GQA</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
              The Modern Transformer Revolution & Attention Mechanics
            </h2>
            <p className="text-slate-300 text-sm max-w-3xl mt-1.5 leading-relaxed">
              Explore the mathematical engine of modern Large Language Models (LLMs). From{' '}
              <strong className="text-white">Scaled Dot-Product Attention</strong> and the{' '}
              <span className="text-cyan-400 font-semibold">Summit of Diplomats</span> analogy to{' '}
              <span className="text-purple-400 font-semibold">Rotary Position Embeddings (RoPE)</span>,{' '}
              <span className="text-emerald-400 font-semibold">Autoregressive KV-Caching</span>, and{' '}
              <span className="text-amber-400 font-semibold">FlashAttention IO-awareness</span>.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <span className="px-3 py-1 rounded-xl bg-purple-500/10 border border-purple-500/30 text-xs font-mono text-purple-300">
              Pure NumPy MHA
            </span>
            <span className="px-3 py-1 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-xs font-mono text-cyan-300">
              PyTorch 2.x SDPA
            </span>
          </div>
        </div>

        {/* Sub-Tabs Navigation */}
        <div className="flex items-center gap-2 mt-6 border-t border-slate-800/80 pt-4 overflow-x-auto pb-1 scrollbar-none">
          {[
            { id: 'qkv', label: '1. QKV Matrix & Spotlight Beams', icon: Eye, color: 'text-purple-400' },
            { id: 'kvcache', label: '2. Autoregressive KV-Cache', icon: Database, color: 'text-emerald-400' },
            { id: 'rope', label: '3. RoPE 2D Rotation Dial', icon: Compass, color: 'text-cyan-400' },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeSubTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${
                  active
                    ? 'bg-slate-800 text-white border border-slate-700 shadow-md shadow-purple-500/5'
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

      {/* --- SUBTAB 1: QKV ATTENTION MATRIX & LIVE SPOTLIGHT BEAMS --- */}
      {activeSubTab === 'qkv' && (
        <div className="space-y-6">
          {/* Controls Bar: Presets & Causal Toggle */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-mono text-slate-400 font-semibold mr-1">Semantic Presets:</span>
              {PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => setSelectedPresetId(preset.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-medium transition ${
                    selectedPresetId === preset.id
                      ? 'bg-purple-600 text-white shadow-md shadow-purple-500/20'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {preset.name}
                </button>
              ))}
            </div>

            {/* Causal Mask Toggle */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsCausal(!isCausal)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition border ${
                  isCausal
                    ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-sm'
                    : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-750'
                }`}
              >
                {isCausal ? <Lock className="w-3.5 h-3.5 text-rose-400" /> : <Unlock className="w-3.5 h-3.5 text-slate-400" />}
                <span>{isCausal ? 'Causal Mask (GPT / Decoder)' : 'Bidirectional (BERT / Encoder)'}</span>
              </button>

              <button
                onClick={() => setUseSqrtScaling(!useSqrtScaling)}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono transition border ${
                  useSqrtScaling
                    ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
                title="Scale scores by 1/sqrt(d_k)"
              >
                1/√d_k Scaling: {useSqrtScaling ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>

          {/* Interactive Token Sequence & Live Spotlight Beams */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Eye className="w-5 h-5 text-purple-400" />
                  Live Attention Spotlight Beams
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Click any token as the <strong className="text-purple-300">Query (Q)</strong> to illuminate radiant attention beams to all <strong className="text-cyan-300">Keys (K)</strong>.
                </p>
              </div>

              <div className="text-xs font-mono text-slate-400">
                Selected Query: <span className="font-bold text-white">"{tokens[selectedQueryIdx]}"</span> (Pos {selectedQueryIdx})
              </div>
            </div>

            {/* Token Pills */}
            <div className="flex flex-wrap gap-2.5 p-4 bg-slate-950 rounded-2xl border border-slate-800 my-4">
              {tokens.map((token, idx) => {
                const isQuery = selectedQueryIdx === idx;
                const attnWeight = currentAttnDistribution[idx] || 0.0;
                const isHighAttn = attnWeight > 0.25;

                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedQueryIdx(idx)}
                    className={`relative group px-4 py-2.5 rounded-xl text-xs font-mono font-bold transition-all duration-300 flex flex-col items-center ${
                      isQuery
                        ? 'bg-gradient-to-tr from-purple-600 to-indigo-600 text-white ring-2 ring-purple-300 shadow-xl shadow-purple-500/40 scale-105 z-20'
                        : isHighAttn
                        ? 'bg-cyan-950/60 border border-cyan-400/80 text-cyan-200 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-900/80 border border-slate-800 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    <span className="text-[9px] uppercase tracking-wider text-slate-400 font-normal">
                      {isQuery ? 'Query' : `Pos ${idx}`}
                    </span>
                    <span className="text-sm mt-0.5">{token}</span>

                    {/* Attention weight mini-badge */}
                    {!isQuery && (
                      <span
                        className={`text-[10px] mt-1 px-1.5 py-0.2 rounded font-mono font-bold ${
                          isHighAttn ? 'bg-cyan-500/30 text-cyan-300' : 'text-slate-500'
                        }`}
                      >
                        {(attnWeight * 100).toFixed(1)}%
                      </span>
                    )}

                    {/* Beam Glow Indicator Underneath */}
                    {!isQuery && attnWeight > 0.05 && (
                      <div
                        className="absolute -bottom-1 left-2 right-2 h-1 rounded-full bg-cyan-400 blur-sm transition-all"
                        style={{ opacity: attnWeight }}
                      />
                    )}
                  </button>
                );
              })}
            </div>

            {/* Sensory Physical Analogy Callout */}
            <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-500/30 text-xs text-slate-300 leading-relaxed flex items-start gap-3">
              <Sparkles className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
              <div>
                <strong className="text-purple-300">The Summit of Diplomats Metaphor:</strong> Token{' '}
                <code className="text-white font-bold">"{tokens[selectedQueryIdx]}"</code> steps up to the podium, opening its{' '}
                <strong className="text-purple-400">Query envelope</strong>: <em>"What information explains me?"</em> Every other diplomat presents their{' '}
                <strong className="text-cyan-400">Key envelope</strong>.
                The dot product measures resonance.
                {tokens[selectedQueryIdx].toLowerCase() === 'it' && (
                  <span className="text-emerald-300 font-semibold ml-1">
                    Notice how "it" overwhelmingly attends to "animal" (coreference resolved)!
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Matrix Tiles Pipeline ($Q \times K^\top \to \text{Softmax} \times V$) */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-base font-bold text-white flex items-center gap-2 mb-4">
              <Layers className="w-4 h-4 text-cyan-400" />
              The Matrix Tile Pipeline ($Q \times K^\top \rightarrow \text{Softmax} \rightarrow \times V$)
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono border-collapse min-w-[600px]">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-slate-500">Query (Tokens ↓) \ Key (Tokens →)</th>
                    {tokens.map((t, i) => (
                      <th
                        key={i}
                        className={`p-2 text-center transition ${
                          selectedQueryIdx === i ? 'text-purple-400 font-black underline' : 'text-slate-400'
                        }`}
                      >
                        {t}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tokens.map((qToken, rowIdx) => (
                    <tr
                      key={rowIdx}
                      onClick={() => setSelectedQueryIdx(rowIdx)}
                      className={`cursor-pointer transition ${
                        selectedQueryIdx === rowIdx ? 'bg-purple-950/40 ring-1 ring-purple-500/50' : 'hover:bg-slate-850'
                      }`}
                    >
                      <td className="p-2 font-bold text-white border-t border-slate-800">
                        <span className="text-slate-500 text-[10px] mr-1">{rowIdx}:</span>
                        {qToken}
                      </td>
                      {attentionWeights[rowIdx].map((val, colIdx) => {
                        const isMasked = isCausal && colIdx > rowIdx;
                        const isRowSelected = selectedQueryIdx === rowIdx;

                        return (
                          <td
                            key={colIdx}
                            className="p-2 text-center border-t border-slate-800 relative transition-all"
                            style={{
                              backgroundColor: isMasked
                                ? 'rgba(15, 23, 42, 0.9)'
                                : isRowSelected
                                ? `rgba(168, 85, 247, ${Math.max(0.08, val * 0.85)})`
                                : `rgba(6, 182, 212, ${Math.max(0.04, val * 0.5)})`,
                            }}
                          >
                            {isMasked ? (
                              <span className="text-slate-600 text-[10px] font-bold">-∞</span>
                            ) : (
                              <span
                                className={`font-mono ${
                                  val > 0.25 ? 'text-white font-bold' : val > 0.05 ? 'text-slate-300' : 'text-slate-500'
                                }`}
                              >
                                {val.toFixed(2)}
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between text-xs text-slate-400 font-mono mt-4 pt-4 border-t border-slate-800">
              <div>
                Formula:{' '}
                <code className="text-cyan-300">
                  Attention(Q, K, V) = Softmax( (Q Kᵀ) / √{d_k} {isCausal ? '+ M_causal' : ''} ) V
                </code>
              </div>
              <div>Sum per row = 1.000</div>
            </div>
          </div>
        </div>
      )}

      {/* --- SUBTAB 2: AUTOREGRESSIVE KV-CACHE SIMULATOR --- */}
      {activeSubTab === 'kvcache' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Database className="w-5 h-5 text-emerald-400" />
                    Autoregressive KV-Cache Buffer Simulator
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Observe how past Keys and Values persist in high-speed SRAM/HBM cache, while only 1 new Query row is calculated.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleStepKVCache}
                    disabled={cachedTokenCount >= promptTokens.length + generatedTokenPool.length}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs transition disabled:opacity-40 shadow-lg shadow-emerald-500/20"
                  >
                    <Play className="w-3.5 h-3.5" />
                    <span>Generate Next Token</span>
                  </button>

                  <button
                    onClick={handleResetKVCache}
                    className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition"
                    title="Reset Cache"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Cache Memory Visualizer */}
              <div className="p-6 bg-slate-950 rounded-2xl border border-slate-800 space-y-4">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <span className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-emerald-400" />
                    <span>SRAM / HBM KV-Cache Ledger</span>
                  </span>
                  <span>
                    Cached Tokens: <strong className="text-white">{cachedTokenCount}</strong> /{' '}
                    {promptTokens.length + generatedTokenPool.length}
                  </span>
                </div>

                {/* Token Slots */}
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                  {allActiveTokens.map((token, idx) => {
                    const isPrompt = idx < promptTokens.length;
                    const isLatest = idx === cachedTokenCount - 1;

                    return (
                      <div
                        key={idx}
                        className={`p-3 rounded-xl border flex flex-col items-center text-center transition-all duration-300 ${
                          isLatest
                            ? 'bg-emerald-500/20 border-emerald-400 text-white shadow-lg shadow-emerald-500/30 ring-2 ring-emerald-400 scale-105'
                            : isPrompt
                            ? 'bg-slate-900 border-slate-700 text-slate-200'
                            : 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
                        }`}
                      >
                        <span className="text-[9px] font-mono text-slate-500">
                          {isLatest ? '⚡ ACTIVE Q_t' : isPrompt ? 'Prompt' : `Step ${idx - promptTokens.length + 1}`}
                        </span>
                        <span className="text-sm font-bold font-mono my-1.5">"{token}"</span>
                        <div className="flex items-center gap-1 text-[9px] font-mono text-emerald-400">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>K, V Cached</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Execution FLOPs & Speedup Comparison */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-rose-400 font-bold">Without KV-Cache</div>
                  <div className="text-xl font-black text-rose-300 font-mono mt-1">
                    {flopsWithoutCache.toLocaleString()} FLOPs
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Recomputing O(T³) FLOPs</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-emerald-400 font-bold">With KV-Cache</div>
                  <div className="text-xl font-black text-emerald-300 font-mono mt-1">
                    {flopsWithCache.toLocaleString()} FLOPs
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Linear O(T²) generation</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
                  <div className="text-[11px] font-mono text-cyan-400 font-bold">Inference Speedup</div>
                  <div className="text-xl font-black text-cyan-300 font-mono mt-1">
                    {(flopsWithoutCache / Math.max(1, flopsWithCache)).toFixed(1)}x
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Memory traffic saved</div>
                </div>
              </div>
            </div>
          </div>

          {/* Theoretical Card */}
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-base font-bold text-white flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-emerald-400" />
                The Stenographer's Ledger Analogy
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Imagine a courtroom stenographer. When someone speaks a 500th sentence:
                <br /><br />
                <strong>Without Cache:</strong> The stenographer would re-read all 499 previous pages from the file cabinet,
                re-analyzing every grammar rule from scratch.
                <br /><br />
                <strong className="text-emerald-400">With KV-Cache:</strong> The past 499 sentences are already indexed on the desk.
                The stenographer analyzes only the ONE new word and cross-references it with the existing ledger!
              </p>

              <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs space-y-2">
                <div className="text-emerald-400 text-[11px] uppercase tracking-wider font-semibold">
                  Generation Step Complexity:
                </div>
                <div className="text-white font-bold">
                  Q_t (1 × d) @ K_{'{1:t}'}^T (d × t) = 1 × t
                </div>
                <div className="text-slate-400 text-[11px]">
                  Reduces total token generation from <span className="text-rose-400">O(T³)</span> down to{' '}
                  <span className="text-emerald-400">O(T²)</span>!
                </div>
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-3">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                FlashAttention & IO-Awareness (Dao et al. 2022)
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Standard attention materializes an N × N matrix into slow GPU High-Bandwidth Memory (HBM).
                <strong>FlashAttention</strong> tiles the computation in fast On-Chip SRAM using an online Softmax algorithm,
                eliminating massive memory read/writes and giving a 2x-4x wall-clock speedup!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* --- SUBTAB 3: ROPE 2D ROTATION DIAL --- */}
      {activeSubTab === 'rope' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-4 bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Rotary Position Controls
            </h3>
            <p className="text-xs text-slate-400">
              Adjust token positions m and n to observe how 2D subspace rotation preserves relative distance (m - n).
            </p>

            {/* Position m */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-purple-400">Query Token Position (m)</span>
                <span className="font-mono text-purple-300">m = {posM}</span>
              </div>
              <input
                type="range"
                min="0"
                max="10"
                value={posM}
                onChange={(e) => setPosM(parseInt(e.target.value))}
                className="w-full accent-purple-400 cursor-pointer"
              />
              <div className="text-[10px] text-slate-500">Angle: m · θ = {angleM.toFixed(2)} rad</div>
            </div>

            {/* Position n */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-cyan-400">Key Token Position (n)</span>
                <span className="font-mono text-cyan-300">n = {posN}</span>
              </div>
              <input
                type="range"
                min="0"
                max="10"
                value={posN}
                onChange={(e) => setPosN(parseInt(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="text-[10px] text-slate-500">Angle: n · θ = {angleN.toFixed(2)} rad</div>
            </div>

            {/* Subspace Frequency Theta */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-amber-400">Harmonic Frequency (θ_i)</span>
                <span className="font-mono text-amber-300">{freqTheta.toFixed(2)} rad/pos</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={freqTheta}
                onChange={(e) => setFreqTheta(parseFloat(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer"
              />
            </div>

            {/* Relative Shift Proof Buttons */}
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="text-[11px] font-mono text-slate-400 font-semibold uppercase">
                Test Relative Invariance:
              </div>
              <button
                onClick={() => {
                  setPosM(posM + 2);
                  setPosN(posN + 2);
                }}
                className="w-full py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-200 transition"
              >
                Shift Both by +2 Positions
              </button>
            </div>
          </div>

          {/* Visual 2D Plane & Dial */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                <Compass className="w-5 h-5 text-cyan-400" />
                2D Subspace Unit Circle Rotation (Su et al. 2021)
              </h3>

              <div className="flex flex-col md:flex-row items-center justify-around gap-6 p-6 bg-slate-950 rounded-2xl border border-slate-800">
                {/* SVG 2D Dial */}
                <svg width="240" height="240" viewBox="-120 -120 240 240" className="overflow-visible">
                  {/* Outer Unit Circle */}
                  <circle cx="0" cy="0" r="90" fill="none" stroke="#334155" strokeWidth="1.5" strokeDasharray="4 4" />
                  <line x1="-105" y1="0" x2="105" y2="0" stroke="#1e293b" strokeWidth="1.5" />
                  <line x1="0" y1="-105" x2="0" y2="105" stroke="#1e293b" strokeWidth="1.5" />

                  {/* Key Vector Arrow (Cyan) */}
                  <line
                    x1="0"
                    y1="0"
                    x2={k_x * 90}
                    y2={-k_y * 90}
                    stroke="#06b6d4"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                  />
                  <circle cx={k_x * 90} cy={-k_y * 90} r="5" fill="#06b6d4" />

                  {/* Query Vector Arrow (Purple) */}
                  <line
                    x1="0"
                    y1="0"
                    x2={q_x * 90}
                    y2={-q_y * 90}
                    stroke="#a855f7"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                  />
                  <circle cx={q_x * 90} cy={-q_y * 90} r="5" fill="#a855f7" />

                  {/* Origin */}
                  <circle cx="0" cy="0" r="4" fill="#94a3b8" />
                </svg>

                {/* Mathematical Proof Card */}
                <div className="space-y-3 font-mono text-xs max-w-sm">
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
                    <div className="text-slate-400 text-[11px] uppercase tracking-wider">Rotated Angle Breakdown:</div>
                    <div className="text-purple-400 font-bold">Query R_m q: {angleM.toFixed(2)} rad (pos {posM})</div>
                    <div className="text-cyan-400 font-bold">Key R_n k:   {angleN.toFixed(2)} rad (pos {posN})</div>
                    <div className="text-amber-300 font-bold pt-1 border-t border-slate-800">
                      Relative Angle Δ = (m - n)·θ = {angleDiff.toFixed(2)} rad
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-purple-950/30 border border-purple-500/40 space-y-1">
                    <div className="text-purple-300 font-bold text-sm">
                      〈R_m q, R_n k〉 = {ropeInnerProduct.toFixed(4)}
                    </div>
                    <div className="text-[10px] text-slate-400">
                      Notice: Even if you shift m and n by +100 positions, as long as (m - n) remains constant, the dot product is
                      <strong> 100% IDENTICAL</strong>!
                    </div>
                  </div>
                </div>
              </div>

              {/* Formula & Theorem */}
              <div className="mt-6 pt-4 border-t border-slate-800 text-xs font-mono text-slate-400">
                Fundamental Theorem:{' '}
                <code className="text-cyan-300">
                  〈R_m q, R_n k〉 = qᵀ R_mᵀ R_n k = qᵀ R_{'{n-m}'} k = g(q, k, n - m)
                </code>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
