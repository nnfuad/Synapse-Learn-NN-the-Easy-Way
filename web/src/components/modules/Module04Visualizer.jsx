import React, { useState, useEffect, useMemo } from 'react';
import {
  Sliders,
  Play,
  RotateCcw,
  Zap,
  Eye,
  Layers,
  Sparkles,
  BookOpen,
  ArrowRight,
  Maximize2,
  GitBranch,
  ShieldAlert,
  CheckCircle2,
  Activity,
  Cpu,
} from 'lucide-react';

const KERNEL_PRESETS = {
  sobelV: {
    name: 'Sobel Vertical (Edges)',
    matrix: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
    bias: 0.0,
  },
  sobelH: {
    name: 'Sobel Horizontal (Edges)',
    matrix: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
    bias: 0.0,
  },
  sharpen: {
    name: 'Sharpen Kernel',
    matrix: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
    bias: 0.0,
  },
  gaussian: {
    name: 'Gaussian Blur (1/16)',
    matrix: [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ].map((row) => row.map((v) => +(v / 16).toFixed(2))),
    bias: 0.0,
  },
  identity: {
    name: 'Identity Filter',
    matrix: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
    bias: 0.0,
  },
};

const INITIAL_IMAGE = [
  [0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 0],
  [0, 1, 0, 0, 1, 0],
  [0, 1, 0, 0, 1, 0],
  [0, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 0],
];

export default function Module04Visualizer() {
  const [activeTab, setActiveTab] = useState('convScanner');

  // Conv Scanner State
  const [imageGrid, setImageGrid] = useState(INITIAL_IMAGE);
  const [selectedKernelKey, setSelectedKernelKey] = useState('sobelV');
  const [paddingMode, setPaddingMode] = useState('valid'); // 'valid' (0) or 'same' (1)
  const [stride, setStride] = useState(1);
  const [scanStep, setScanStep] = useState(0);
  const [isAutoScanning, setIsAutoScanning] = useState(false);

  // Receptive Field Explorer State
  const [rfLayers, setRfLayers] = useState(3);
  const [rfKernelSize, setRfKernelSize] = useState(3);
  const [rfStride, setRfStride] = useState(1);

  // ResNet Highway State
  const [resDepth, setResDepth] = useState(30);
  const [gradientPulseActive, setGradientPulseActive] = useState(false);

  // Kernel definition
  const currentKernel = KERNEL_PRESETS[selectedKernelKey];

  // Padded image computation
  const paddedGrid = useMemo(() => {
    if (paddingMode === 'valid') return imageGrid;
    const H = imageGrid.length;
    const W = imageGrid[0].length;
    const padded = Array.from({ length: H + 2 }, () => new Array(W + 2).fill(0));
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        padded[r + 1][c + 1] = imageGrid[r][c];
      }
    }
    return padded;
  }, [imageGrid, paddingMode]);

  // Output feature map dimension
  const { outH, outW, scanPositions } = useMemo(() => {
    const H = paddedGrid.length;
    const W = paddedGrid[0].length;
    const kH = 3;
    const kW = 3;
    const oH = Math.floor((H - kH) / stride) + 1;
    const oW = Math.floor((W - kW) / stride) + 1;

    const positions = [];
    for (let r = 0; r < oH; r++) {
      for (let c = 0; c < oW; c++) {
        positions.push({
          outR: r,
          outC: c,
          inR: r * stride,
          inC: c * stride,
        });
      }
    }
    return { outH: oH, outW: oW, scanPositions: positions };
  }, [paddedGrid, stride]);

  // Clamp current scan position
  const currentScan = scanPositions[scanStep % Math.max(1, scanPositions.length)] || {
    outR: 0,
    outC: 0,
    inR: 0,
    inC: 0,
  };

  // Compute full output feature map
  const outputFeatureMap = useMemo(() => {
    const out = Array.from({ length: outH }, () => new Array(outW).fill(0));
    scanPositions.forEach((pos) => {
      let sum = 0;
      for (let kr = 0; kr < 3; kr++) {
        for (let kc = 0; kc < 3; kc++) {
          const pixelVal = paddedGrid[pos.inR + kr]?.[pos.inC + kc] ?? 0;
          const weightVal = currentKernel.matrix[kr][kc];
          sum += pixelVal * weightVal;
        }
      }
      out[pos.outR][pos.outC] = +(sum + currentKernel.bias).toFixed(2);
    });
    return out;
  }, [paddedGrid, currentKernel, scanPositions, outH, outW]);

  // Current calculation details
  const currentCalc = useMemo(() => {
    const terms = [];
    let sum = 0;
    for (let kr = 0; kr < 3; kr++) {
      for (let kc = 0; kc < 3; kc++) {
        const p = paddedGrid[currentScan.inR + kr]?.[currentScan.inC + kc] ?? 0;
        const w = currentKernel.matrix[kr][kc];
        const prod = p * w;
        sum += prod;
        terms.push({ p, w, prod });
      }
    }
    const rawVal = sum + currentKernel.bias;
    const reluVal = Math.max(0, rawVal);
    return { terms, rawVal: +rawVal.toFixed(2), reluVal: +reluVal.toFixed(2) };
  }, [paddedGrid, currentScan, currentKernel]);

  // Auto scan interval
  useEffect(() => {
    let timer;
    if (isAutoScanning) {
      timer = setInterval(() => {
        setScanStep((s) => (s + 1) % scanPositions.length);
      }, 500);
    }
    return () => clearInterval(timer);
  }, [isAutoScanning, scanPositions.length]);

  // Receptive Field calculation
  const rfMetrics = useMemo(() => {
    let size = 32;
    let rf = 1;
    let jump = 1;
    const layerHistory = [];

    for (let l = 1; l <= rfLayers; l++) {
      rf = rf + (rfKernelSize - 1) * jump;
      jump = jump * rfStride;
      size = Math.floor((size - rfKernelSize) / rfStride) + 1;
      layerHistory.push({
        layer: l,
        rf,
        jump,
        outSize: Math.max(1, size),
      });
    }

    return {
      currentRF: rf,
      jump,
      layerHistory,
    };
  }, [rfLayers, rfKernelSize, rfStride]);

  // Toggle pixel on click
  const togglePixel = (r, c) => {
    setImageGrid((prev) => {
      const copy = prev.map((row) => [...row]);
      copy[r][c] = copy[r][c] === 1 ? 0 : 1;
      return copy;
    });
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Sub-navigation menu */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-4 overflow-x-auto">
        {[
          { id: 'convScanner', label: '1. Interactive 2D Convolution Scanner', icon: Eye },
          { id: 'receptiveField', label: '2. Receptive Field Cone Calculator', icon: Layers },
          { id: 'resnetHighway', label: '3. ResNet Identity Shortcut Highway', icon: GitBranch },
          { id: 'citations', label: '4. Spatial Research Hall of Fame', icon: BookOpen },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${
                isActive
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/40 shadow-sm shadow-emerald-500/10'
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
      {/* TAB 1: 2D CONVOLUTION SCANNER */}
      {/* ========================================================================================= */}
      {activeTab === 'convScanner' && (
        <div className="space-y-6">
          {/* Controls Bar */}
          <div className="bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              {/* Kernel Selector */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-400 font-semibold">Filter:</span>
                <select
                  value={selectedKernelKey}
                  onChange={(e) => setSelectedKernelKey(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-medium focus:outline-none focus:border-emerald-500"
                >
                  {Object.entries(KERNEL_PRESETS).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Padding Selector */}
              <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-lg text-xs">
                <button
                  onClick={() => {
                    setPaddingMode('valid');
                    setScanStep(0);
                  }}
                  className={`px-2.5 py-1 rounded font-bold transition ${
                    paddingMode === 'valid' ? 'bg-emerald-500 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Valid (0-pad)
                </button>
                <button
                  onClick={() => {
                    setPaddingMode('same');
                    setScanStep(0);
                  }}
                  className={`px-2.5 py-1 rounded font-bold transition ${
                    paddingMode === 'same' ? 'bg-emerald-500 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Same (1-pad)
                </button>
              </div>

              {/* Stride Selector */}
              <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-lg text-xs">
                <button
                  onClick={() => {
                    setStride(1);
                    setScanStep(0);
                  }}
                  className={`px-2 py-1 rounded font-bold transition ${
                    stride === 1 ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Stride 1
                </button>
                <button
                  onClick={() => {
                    setStride(2);
                    setScanStep(0);
                  }}
                  className={`px-2 py-1 rounded font-bold transition ${
                    stride === 2 ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Stride 2
                </button>
              </div>
            </div>

            {/* Play/Step Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsAutoScanning(!isAutoScanning)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition shadow ${
                  isAutoScanning
                    ? 'bg-rose-600 text-white shadow-rose-600/30'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/20'
                }`}
              >
                <Play className="w-3.5 h-3.5" />
                <span>{isAutoScanning ? 'Pause Auto Scan' : 'Auto Scan'}</span>
              </button>

              <button
                onClick={() => setScanStep((s) => (s + 1) % scanPositions.length)}
                className="px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
              >
                Step Forward ➔
              </button>

              <button
                onClick={() => {
                  setScanStep(0);
                  setIsAutoScanning(false);
                }}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white bg-slate-900 border border-slate-800"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Grids Display: Input -> Kernel -> Output */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Input Canvas (6x6 or 8x8 padded) */}
            <div className="lg:col-span-5 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-white flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                  <span>Input Canvas {paddingMode === 'same' ? '(8×8 Padded)' : '(6×6)'}</span>
                </span>
                <span className="text-[11px] font-mono text-cyan-400">Click cells to toggle</span>
              </div>

              {/* Grid with scanning flashlight overlay */}
              <div className="relative inline-block mx-auto bg-[#070b14] p-3 rounded-xl border border-slate-800/60">
                <div
                  className="grid gap-1"
                  style={{
                    gridTemplateColumns: `repeat(${paddedGrid[0].length}, minmax(0, 1fr))`,
                  }}
                >
                  {paddedGrid.map((row, r) =>
                    row.map((val, c) => {
                      const isKernelFocus =
                        r >= currentScan.inR &&
                        r < currentScan.inR + 3 &&
                        c >= currentScan.inC &&
                        c < currentScan.inC + 3;

                      const isPadCell =
                        paddingMode === 'same' &&
                        (r === 0 || r === paddedGrid.length - 1 || c === 0 || c === row.length - 1);

                      return (
                        <div
                          key={`cell-${r}-${c}`}
                          onClick={() => {
                            if (!isPadCell) {
                              const origR = paddingMode === 'same' ? r - 1 : r;
                              const origC = paddingMode === 'same' ? c - 1 : c;
                              togglePixel(origR, origC);
                            }
                          }}
                          className={`w-9 h-9 rounded flex items-center justify-center font-mono text-xs font-bold transition-all cursor-pointer select-none ${
                            isKernelFocus
                              ? 'ring-2 ring-emerald-400 bg-emerald-500/20 shadow-lg shadow-emerald-500/20 z-10'
                              : isPadCell
                              ? 'bg-slate-900/40 text-slate-600 border border-dashed border-slate-800'
                              : val === 1
                              ? 'bg-cyan-500 text-white shadow-sm'
                              : 'bg-slate-900 text-slate-400 border border-slate-800/80 hover:border-slate-700'
                          }`}
                        >
                          {val}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              <div className="text-[11px] text-slate-400 flex items-center gap-2">
                <span className="w-3 h-3 rounded bg-emerald-500/30 border border-emerald-400 inline-block"></span>
                <span>Active 3×3 Receptive Scanning Window (Flashlight)</span>
              </div>
            </div>

            {/* Middle: 3x3 Filter Kernel & Live Dot-Product */}
            <div className="lg:col-span-3 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-5 space-y-4 flex flex-col justify-between">
              <div>
                <span className="text-sm font-bold text-white flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-400"></span>
                  <span>3×3 Filter Weights (K)</span>
                </span>

                <div className="grid grid-cols-3 gap-1.5 mt-4 p-2 bg-[#070b14] rounded-xl border border-slate-800/60 max-w-[180px] mx-auto">
                  {currentKernel.matrix.map((row, r) =>
                    row.map((weight, c) => (
                      <div
                        key={`k-${r}-${c}`}
                        className="w-12 h-12 rounded bg-purple-950/40 border border-purple-500/40 flex items-center justify-center font-mono text-xs font-bold text-purple-200"
                      >
                        {weight}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Real-time Dot Product calculation box */}
              <div className="bg-slate-900/90 border border-slate-800 p-3.5 rounded-xl space-y-2 text-xs">
                <div className="font-mono text-cyan-400 font-bold text-[11px]">
                  Pos: ({currentScan.outR}, {currentScan.outC})
                </div>
                <div className="text-slate-300 font-mono text-[11px]">
                  Σ (Pixel · Weight) = {currentCalc.rawVal}
                </div>
                <div className="text-emerald-400 font-mono font-bold text-[11px]">
                  ReLU Output = {currentCalc.reluVal}
                </div>
              </div>
            </div>

            {/* Output Feature Map */}
            <div className="lg:col-span-4 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-white flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                  <span>Feature Map ({outH}×{outW})</span>
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  Step {scanStep + 1}/{scanPositions.length}
                </span>
              </div>

              <div className="relative inline-block mx-auto bg-[#070b14] p-3 rounded-xl border border-slate-800/60">
                <div
                  className="grid gap-1.5"
                  style={{
                    gridTemplateColumns: `repeat(${outW}, minmax(0, 1fr))`,
                  }}
                >
                  {outputFeatureMap.map((row, r) =>
                    row.map((val, c) => {
                      const isCurrent = r === currentScan.outR && c === currentScan.outC;
                      return (
                        <div
                          key={`out-${r}-${c}`}
                          className={`w-11 h-11 rounded flex items-center justify-center font-mono text-xs font-bold transition-all ${
                            isCurrent
                              ? 'ring-2 ring-emerald-400 bg-emerald-500 text-white shadow-lg shadow-emerald-500/30'
                              : val > 0
                              ? 'bg-slate-850 text-emerald-400 border border-emerald-900/60'
                              : 'bg-slate-900 text-slate-500 border border-slate-800'
                          }`}
                        >
                          {val}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed">
                Translation Equivariance: As patterns shift on the input grid, the corresponding activations shift in the output feature map identically.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================================= */}
      {/* TAB 2: RECEPTIVE FIELD EXPANSION CALCULATOR */}
      {/* ========================================================================================= */}
      {activeTab === 'receptiveField' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls */}
          <div className="lg:col-span-4 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 space-y-6">
            <div>
              <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono font-bold uppercase tracking-wider">
                <Layers className="w-4 h-4" />
                <span>Exact Mathematical Propagation</span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">Receptive Field Expander</h3>
              <p className="text-xs text-slate-400 mt-1">
                Calculate how a single deep neuron's field of view grows across stacked convolutional layers.
              </p>
            </div>

            {/* Depth Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Stack Depth (Layers):</span>
                <span className="font-mono font-bold text-cyan-400">{rfLayers} layers</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={rfLayers}
                onChange={(e) => setRfLayers(parseInt(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
            </div>

            {/* Kernel Size Selector */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Kernel Size (k):</span>
                <span className="font-mono font-bold text-purple-400">{rfKernelSize}×{rfKernelSize}</span>
              </div>
              <div className="flex gap-2">
                {[3, 5, 7].map((k) => (
                  <button
                    key={k}
                    onClick={() => setRfKernelSize(k)}
                    className={`flex-1 py-1.5 rounded-lg border text-xs font-bold transition ${
                      rfKernelSize === k
                        ? 'bg-purple-600/30 border-purple-500 text-white'
                        : 'bg-slate-900 border-slate-800 text-slate-400'
                    }`}
                  >
                    {k}×{k}
                  </button>
                ))}
              </div>
            </div>

            {/* Stride Selector */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Layer Stride (s):</span>
                <span className="font-mono font-bold text-emerald-400">{rfStride}</span>
              </div>
              <div className="flex gap-2">
                {[1, 2].map((s) => (
                  <button
                    key={s}
                    onClick={() => setRfStride(s)}
                    className={`flex-1 py-1.5 rounded-lg border text-xs font-bold transition ${
                      rfStride === s
                        ? 'bg-emerald-600/30 border-emerald-500 text-white'
                        : 'bg-slate-900 border-slate-800 text-slate-400'
                    }`}
                  >
                    Stride {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Mathematical Formulas Card */}
            <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-xl space-y-2 text-xs">
              <div className="font-bold text-cyan-400">Receptive Field Recurrence:</div>
              <div className="font-mono text-[11px] text-slate-300 bg-slate-950 p-2 rounded border border-slate-800">
                r_l = r_{'{l-1}'} + (k - 1) · j_{'{l-1}'}
                <br />
                j_l = j_{'{l-1}'} · s
              </div>
              <div className="text-[11px] text-slate-400 pt-1">
                Current Total Receptive Field: <span className="font-bold text-emerald-400">{rfMetrics.currentRF}×{rfMetrics.currentRF} pixels</span>
              </div>
            </div>
          </div>

          {/* Right: Cross-Sectional Cone Visualization */}
          <div className="lg:col-span-8 bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Maximize2 className="w-5 h-5 text-cyan-400" />
                <span>Spatial Hierarchy & Bounding Field of View</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Each layer expands coverage of the input image, building features from local edges to high-level semantic objects.
              </p>

              {/* Table of layer metrics */}
              <div className="mt-6 overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="pb-2">Layer</th>
                      <th className="pb-2">Spatial Map</th>
                      <th className="pb-2">Effective Jump</th>
                      <th className="pb-2 text-emerald-400">Receptive Field</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {rfMetrics.layerHistory.map((m) => (
                      <tr key={m.layer} className="hover:bg-slate-900/40">
                        <td className="py-2.5 text-white font-bold">Conv Stage {m.layer}</td>
                        <td className="py-2.5 text-slate-300">{m.outSize}×{m.outSize}</td>
                        <td className="py-2.5 text-cyan-400">j = {m.jump}</td>
                        <td className="py-2.5 text-emerald-400 font-bold">{m.rf}×{m.rf} pixels</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* The VGG-16 Insight Card */}
              <div className="mt-6 p-4 rounded-xl bg-purple-950/20 border border-purple-800/40 space-y-2 text-xs">
                <div className="font-bold text-purple-300 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4 text-purple-400" />
                  <span>The VGG-16 Breakthrough (Simonyan & Zisserman 2014)</span>
                </div>
                <p className="text-slate-300 leading-relaxed">
                  Stacking two 3×3 convolutions covers an exact <span className="font-bold text-purple-300">5×5 receptive field</span> (3 + 2 = 5). However, two 3×3 filters use only 18 C² parameters vs 25 C² for a single 5×5 filter — saving <span className="font-bold text-emerald-400">28% of parameters</span> while introducing two non-linear activations!
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================================= */}
      {/* TAB 3: RESNET IDENTITY SHORTCUT HIGHWAY */}
      {/* ========================================================================================= */}
      {activeTab === 'resnetHighway' && (
        <div className="bg-[#0a0f1d] border border-slate-800/80 rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono font-bold uppercase tracking-wider">
                <GitBranch className="w-4 h-4" />
                <span>He et al. (CVPR 2016 Best Paper)</span>
              </div>
              <h3 className="text-xl font-bold text-white mt-1">The ResNet Revolution & The Identity Gradient Highway</h3>
              <p className="text-xs text-slate-400 mt-1">
                Resolving the 2015 Deep Degradation crisis via residual learning: H(x) = F(x) + x.
              </p>
            </div>

            <button
              onClick={() => {
                setGradientPulseActive(true);
                setTimeout(() => setGradientPulseActive(false), 2000);
              }}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 transition"
            >
              <Zap className="w-4 h-4" />
              <span>Simulate Backprop Pulse</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Plain Deep CNN Block (Degraded) */}
            <div className="p-5 rounded-2xl border border-rose-900/40 bg-rose-950/10 space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-bold text-rose-300 text-sm">56-Layer Plain CNN (No Shortcut)</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">
                  Optimization Collapse
                </span>
              </div>

              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-center space-y-3 font-mono text-xs">
                <div className="text-slate-400">Input: x</div>
                <div className="w-px h-5 bg-slate-700 mx-auto"></div>
                <div className="p-2 rounded bg-slate-900 border border-slate-800 text-slate-300">
                  Conv 3×3 + BN + ReLU
                </div>
                <div className="w-px h-5 bg-slate-700 mx-auto"></div>
                <div className="p-2 rounded bg-slate-900 border border-slate-800 text-slate-300">
                  Conv 3×3 + BN + ReLU
                </div>
                <div className="w-px h-5 bg-slate-700 mx-auto"></div>
                <div className="text-rose-400 font-bold">Output: H(x) = F(x)</div>
              </div>

              <div className="p-3 bg-rose-950/30 rounded-lg border border-rose-800/40 text-xs text-slate-300 space-y-1">
                <div className="font-bold text-rose-200">Backward Gradient Signal:</div>
                <div className="font-mono text-rose-400">dL/dx_1 ≈ 0.0003 (Severely Attenuated)</div>
                <p className="text-[11px] text-slate-400 pt-1">
                  Without skip connections, pushing non-linear layers to learn an identity transformation is mathematically intractable for SGD.
                </p>
              </div>
            </div>

            {/* ResNet Residual Block (Highway) */}
            <div className="p-5 rounded-2xl border border-emerald-900/40 bg-emerald-950/10 space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-bold text-emerald-300 text-sm">ResNet Residual Block (Identity Skip)</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  Gradient Highway
                </span>
              </div>

              <div className="relative p-4 bg-slate-950 rounded-xl border border-slate-800 text-center space-y-3 font-mono text-xs">
                {/* Skip bypass line */}
                <div
                  className={`absolute left-4 top-5 bottom-12 w-0.5 rounded transition-all duration-300 ${
                    gradientPulseActive ? 'bg-emerald-400 shadow-md shadow-emerald-400' : 'bg-emerald-500/40'
                  }`}
                ></div>
                <div className="absolute left-6 top-1/2 -translate-y-1/2 text-[10px] text-emerald-400 font-bold bg-slate-900 px-1 rounded">
                  Identity Shortcut (+x)
                </div>

                <div className="text-slate-400">Input: x</div>
                <div className="w-px h-5 bg-slate-700 mx-auto"></div>
                <div className="p-2 rounded bg-slate-900 border border-slate-800 text-slate-300 ml-12">
                  Weight Layer F(x)
                </div>
                <div className="w-px h-5 bg-slate-700 mx-auto"></div>
                <div className="p-2 rounded bg-emerald-950/50 border border-emerald-600/60 text-emerald-300 font-bold">
                  Addition: F(x) + x
                </div>
                <div className="w-px h-5 bg-slate-700 mx-auto"></div>
                <div className="text-emerald-400 font-bold">Output: ReLU(F(x) + x)</div>
              </div>

              <div className="p-3 bg-emerald-950/30 rounded-lg border border-emerald-800/40 text-xs text-slate-300 space-y-1">
                <div className="font-bold text-emerald-200">The ResNet Backward Formula:</div>
                <div className="font-mono text-emerald-400">dL/dx_l = dL/dx_L · (I + dF/dx)</div>
                <p className="text-[11px] text-slate-400 pt-1">
                  The identity term <span className="font-bold text-emerald-300">I</span> ensures that gradient signals flow directly back to layer 1 undiminished, allowing 152+ layer networks to converge effortlessly.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================================= */}
      {/* TAB 4: RESEARCH HALL OF FAME */}
      {/* ========================================================================================= */}
      {activeTab === 'citations' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            {
              title: 'Gradient-Based Learning Applied to Document Recognition',
              authors: 'Yann LeCun, Léon Bottou, Yoshua Bengio, & Patrick Haffner (1998)',
              venue: 'Proceedings of the IEEE',
              badge: 'The CNN Foundation',
              color: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
              insight:
                'Introduced LeNet-5, establishing local receptive fields, shared weights (translation invariance), and spatial subsampling as the foundational pillars of visual deep learning.',
            },
            {
              title: 'Deep Residual Learning for Image Recognition',
              authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, & Jian Sun (2015)',
              venue: 'CVPR 2016 (Best Paper Award)',
              badge: 'The ResNet Revolution',
              color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
              insight:
                'Solved the degradation problem by introducing identity shortcut connections. Won 1st place in all five ImageNet/COCO 2015 competitions and unlocked 100+ layer deep models.',
            },
            {
              title: 'Identity Mappings in Deep Residual Networks',
              authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, & Jian Sun (2016)',
              venue: 'ECCV 2016',
              badge: 'Pre-Activation ResNet',
              color: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
              insight:
                'Analyzed the mathematics of propagation. Placed BN and ReLU prior to convolution (pre-activation), ensuring that the skip connection remains an unobstructed gradient highway.',
            },
            {
              title: 'A ConvNet for the 2020s (ConvNeXt)',
              authors: 'Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, & Saining Xie (2022)',
              venue: 'CVPR 2022',
              badge: 'Modern CNN SOTA',
              color: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
              insight:
                'Modernized classical CNNs by incorporating Vision Transformer (ViT) designs: 7x7 depthwise convolutions, inverted bottlenecks, and LayerNorm instead of BatchNorm.',
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
