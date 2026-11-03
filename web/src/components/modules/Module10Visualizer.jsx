import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import confetti from 'canvas-confetti';
import {
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  ArrowRight,
  ArrowLeft,
  Check,
  CheckCircle2,
  AlertCircle,
  Info,
  Sliders,
  Eye,
  Activity,
  Layers,
  Cpu,
  Award,
  Trophy,
  Target,
  Flame,
  Zap,
  BookOpen,
  Binary,
  Compass,
  HelpCircle,
  Maximize2,
  ChevronRight,
  RefreshCw,
  Crosshair,
  TrendingDown,
  Grid,
  Plus,
  Minus,
  MousePointer,
  FastForward,
  Star,
  ShieldCheck,
  Calculator,
  Shuffle,
  Lightbulb,
} from 'lucide-react';

// ==========================================
// 1. DATASET GENERATORS & PRESETS
// ==========================================
const PRESET_DATASETS = {
  XOR: {
    id: 'XOR',
    name: 'XOR Non-Linear Crisis (1969)',
    description: 'The iconic problem that broke single-layer perceptrons. Requires at least 2 hidden neurons to fold space.',
    points: [
      // Quadrant 3: (-1, -1) -> 0
      { x1: -1.0, x2: -1.0, y: 0, id: 'xor-0-1' },
      { x1: -1.3, x2: -0.9, y: 0, id: 'xor-0-2' },
      { x1: -0.9, x2: -1.3, y: 0, id: 'xor-0-3' },
      { x1: -1.2, x2: -1.2, y: 0, id: 'xor-0-4' },
      // Quadrant 1: (1, 1) -> 0
      { x1: 1.0, x2: 1.0, y: 0, id: 'xor-0-5' },
      { x1: 1.3, x2: 0.9, y: 0, id: 'xor-0-6' },
      { x1: 0.9, x2: 1.3, y: 0, id: 'xor-0-7' },
      { x1: 1.2, x2: 1.2, y: 0, id: 'xor-0-8' },
      // Quadrant 2: (-1, 1) -> 1
      { x1: -1.0, x2: 1.0, y: 1, id: 'xor-1-1' },
      { x1: -1.3, x2: 0.9, y: 1, id: 'xor-1-2' },
      { x1: -0.9, x2: 1.3, y: 1, id: 'xor-1-3' },
      { x1: -1.2, x2: 1.2, y: 1, id: 'xor-1-4' },
      // Quadrant 4: (1, -1) -> 1
      { x1: 1.0, x2: -1.0, y: 1, id: 'xor-1-5' },
      { x1: 1.3, x2: -0.9, y: 1, id: 'xor-1-6' },
      { x1: 0.9, x2: -1.3, y: 1, id: 'xor-1-7' },
      { x1: 1.2, x2: -1.2, y: 1, id: 'xor-1-8' },
    ],
  },

  CIRCLES: {
    id: 'CIRCLES',
    name: 'Concentric Circles (Island Trap)',
    description: 'An inner circular cluster surrounded by an outer ring. Demonstrates multi-hyperplane enclosure.',
    points: (() => {
      const pts = [];
      // Inner Circle (Class 1) - radius ~ 0.55
      for (let i = 0; i < 12; i++) {
        const theta = (i / 12) * Math.PI * 2;
        const r = 0.5 + ((i % 3) - 1) * 0.12;
        pts.push({
          x1: parseFloat((r * Math.cos(theta)).toFixed(2)),
          x2: parseFloat((r * Math.sin(theta)).toFixed(2)),
          y: 1,
          id: `circ-1-${i}`,
        });
      }
      // Outer Circle (Class 0) - radius ~ 1.55
      for (let i = 0; i < 18; i++) {
        const theta = (i / 18) * Math.PI * 2;
        const r = 1.55 + ((i % 4) - 1.5) * 0.15;
        pts.push({
          x1: parseFloat((r * Math.cos(theta)).toFixed(2)),
          x2: parseFloat((r * Math.sin(theta)).toFixed(2)),
          y: 0,
          id: `circ-0-${i}`,
        });
      }
      return pts;
    })(),
  },

  MOONS: {
    id: 'MOONS',
    name: 'Interlocking Two Moons',
    description: 'Two interlocking crescent manifolds. Tests the network’s ability to bend smooth continuous boundaries.',
    points: (() => {
      const pts = [];
      const N = 12;
      // Top Moon (Class 1)
      for (let i = 0; i < N; i++) {
        const theta = (i / (N - 1)) * Math.PI;
        pts.push({
          x1: parseFloat((Math.cos(theta) - 0.5).toFixed(2)),
          x2: parseFloat((Math.sin(theta) + 0.2).toFixed(2)),
          y: 1,
          id: `moon-1-${i}`,
        });
      }
      // Bottom Moon (Class 0)
      for (let i = 0; i < N; i++) {
        const theta = (i / (N - 1)) * Math.PI;
        pts.push({
          x1: parseFloat((1.0 - Math.cos(theta) - 0.5).toFixed(2)),
          x2: parseFloat((1.0 - Math.sin(theta) - 0.7).toFixed(2)),
          y: 0,
          id: `moon-0-${i}`,
        });
      }
      return pts;
    })(),
  },

  GAUSSIAN: {
    id: 'GAUSSIAN',
    name: 'Dual Bivariate Gaussian',
    description: 'Two cleanly separated probability clouds with minor variance overlap.',
    points: [
      // Cluster 0 (Center: -1.2, -1.0)
      { x1: -1.2, x2: -1.0, y: 0, id: 'gauss-0-1' },
      { x1: -1.6, x2: -0.8, y: 0, id: 'gauss-0-2' },
      { x1: -0.9, x2: -1.4, y: 0, id: 'gauss-0-3' },
      { x1: -1.4, x2: -1.3, y: 0, id: 'gauss-0-4' },
      { x1: -0.8, x2: -0.7, y: 0, id: 'gauss-0-5' },
      { x1: -1.5, x2: -0.5, y: 0, id: 'gauss-0-6' },
      { x1: -1.0, x2: -1.1, y: 0, id: 'gauss-0-7' },
      { x1: -1.3, x2: -1.5, y: 0, id: 'gauss-0-8' },
      // Cluster 1 (Center: 1.2, 1.0)
      { x1: 1.2, x2: 1.0, y: 1, id: 'gauss-1-1' },
      { x1: 1.6, x2: 0.8, y: 1, id: 'gauss-1-2' },
      { x1: 0.9, x2: 1.4, y: 1, id: 'gauss-1-3' },
      { x1: 1.4, x2: 1.3, y: 1, id: 'gauss-1-4' },
      { x1: 0.8, x2: 0.7, y: 1, id: 'gauss-1-5' },
      { x1: 1.5, x2: 0.5, y: 1, id: 'gauss-1-6' },
      { x1: 1.0, x2: 1.1, y: 1, id: 'gauss-1-7' },
      { x1: 1.3, x2: 1.5, y: 1, id: 'gauss-1-8' },
    ],
  },

  CUSTOM: {
    id: 'CUSTOM',
    name: 'Custom Interactive Canvas Points',
    description: 'Click anywhere on the 2D decision boundary to place your own custom points.',
    points: [
      { x1: -1.0, x2: -0.5, y: 0, id: 'c-1' },
      { x1: 1.0, x2: 0.5, y: 1, id: 'c-2' },
      { x1: -0.5, x2: 1.2, y: 1, id: 'c-3' },
      { x1: 0.5, x2: -1.2, y: 0, id: 'c-4' },
    ],
  },
};

// ==========================================
// 2. NEURAL NETWORK MATH UTILITIES
// ==========================================
function createInitialModel(hiddenSize = 4) {
  // Xavier / He initialization
  const scale1 = Math.sqrt(2.0 / (2 + hiddenSize));
  const scale2 = Math.sqrt(2.0 / (hiddenSize + 1));

  const W1 = [];
  const b1 = [];
  for (let j = 0; j < hiddenSize; j++) {
    W1.push([
      (Math.random() * 2 - 1) * scale1 * 1.5,
      (Math.random() * 2 - 1) * scale1 * 1.5,
    ]);
    b1.push((Math.random() * 2 - 1) * 0.2);
  }

  const W2 = [];
  for (let j = 0; j < hiddenSize; j++) {
    W2.push((Math.random() * 2 - 1) * scale2 * 1.5);
  }
  const b2 = (Math.random() * 2 - 1) * 0.1;

  return { W1, b1, W2, b2 };
}

// Forward evaluation of single coordinate (x1, x2)
function forwardPass(x1, x2, model, actType = 'sigmoid') {
  const { W1, b1, W2, b2 } = model;
  const H = b1.length;
  const z1 = new Array(H);
  const a1 = new Array(H);

  for (let j = 0; j < H; j++) {
    const sum = W1[j][0] * x1 + W1[j][1] * x2 + b1[j];
    z1[j] = Math.max(-30, Math.min(30, sum));
    if (actType === 'sigmoid') {
      a1[j] = 1.0 / (1.0 + Math.exp(-z1[j]));
    } else if (actType === 'tanh') {
      a1[j] = Math.tanh(z1[j]);
    } else if (actType === 'relu') {
      a1[j] = Math.max(0, z1[j]);
    }
  }

  let z2 = b2;
  for (let j = 0; j < H; j++) {
    z2 += W2[j] * a1[j];
  }
  z2 = Math.max(-30, Math.min(30, z2));
  const yHat = 1.0 / (1.0 + Math.exp(-z2)); // Output is always Sigmoid for binary probability

  return { z1, a1, z2, yHat };
}

// Full forward + analytical backprop for 1 sample
function computeSampleDerivatives(sample, model, actType = 'sigmoid') {
  const { x1, x2, y } = sample;
  const { z1, a1, z2, yHat } = forwardPass(x1, x2, model, actType);
  const H = model.b1.length;

  const eps = 1e-7;
  const clampedYHat = Math.max(eps, Math.min(1 - eps, yHat));
  const loss = -(y * Math.log(clampedYHat) + (1 - y) * Math.log(1 - clampedYHat));

  // Output gradient: dL/dz2 = yHat - y
  const delta2 = yHat - y;

  // Output layer weight and bias gradients
  const dW2 = new Array(H);
  for (let j = 0; j < H; j++) {
    dW2[j] = delta2 * a1[j];
  }
  const db2 = delta2;

  // Backpressure into hidden layer
  const da1 = new Array(H);
  const delta1 = new Array(H);
  const dW1 = [];
  const db1 = new Array(H);

  for (let j = 0; j < H; j++) {
    da1[j] = delta2 * model.W2[j];

    let dAct = 0;
    if (actType === 'sigmoid') {
      dAct = a1[j] * (1.0 - a1[j]);
    } else if (actType === 'tanh') {
      dAct = 1.0 - a1[j] * a1[j];
    } else if (actType === 'relu') {
      dAct = z1[j] > 0 ? 1.0 : 0.0;
    }

    delta1[j] = da1[j] * dAct;
    dW1.push([delta1[j] * x1, delta1[j] * x2]);
    db1[j] = delta1[j];
  }

  return {
    x1,
    x2,
    y,
    z1,
    a1,
    z2,
    yHat,
    loss,
    delta2,
    dW2,
    db2,
    da1,
    delta1,
    dW1,
    db1,
  };
}

// Evaluate entire dataset
function evaluateDataset(dataset, model, actType = 'sigmoid') {
  if (!dataset || dataset.length === 0) {
    return { loss: 0, accuracy: 0, predictions: [] };
  }
  let totalLoss = 0;
  let correct = 0;
  const predictions = [];

  for (const pt of dataset) {
    const { yHat } = forwardPass(pt.x1, pt.x2, model, actType);
    const eps = 1e-7;
    const clampedYHat = Math.max(eps, Math.min(1 - eps, yHat));
    const loss = -(pt.y * Math.log(clampedYHat) + (1 - pt.y) * Math.log(1 - clampedYHat));
    totalLoss += loss;
    const predClass = yHat >= 0.5 ? 1 : 0;
    if (predClass === pt.y) correct++;
    predictions.push({ ...pt, yHat, predClass });
  }

  return {
    loss: totalLoss / dataset.length,
    accuracy: (correct / dataset.length) * 100,
    predictions,
  };
}

// Full Batch Gradient Descent Epoch
function trainEpoch(dataset, model, lr = 0.1, actType = 'sigmoid') {
  const H = model.b1.length;
  const N = dataset.length;
  if (N === 0) return model;

  const avgDW1 = Array.from({ length: H }, () => [0, 0]);
  const avgDb1 = new Array(H).fill(0);
  const avgDW2 = new Array(H).fill(0);
  let avgDb2 = 0;

  for (const pt of dataset) {
    const grads = computeSampleDerivatives(pt, model, actType);
    for (let j = 0; j < H; j++) {
      avgDW1[j][0] += grads.dW1[j][0] / N;
      avgDW1[j][1] += grads.dW1[j][1] / N;
      avgDb1[j] += grads.db1[j] / N;
      avgDW2[j] += grads.dW2[j] / N;
    }
    avgDb2 += grads.db2 / N;
  }

  // Parameter update rule: theta <- theta - lr * grad
  const newW1 = model.W1.map((row, j) => [
    row[0] - lr * avgDW1[j][0],
    row[1] - lr * avgDW1[j][1],
  ]);
  const newB1 = model.b1.map((b, j) => b - lr * avgDb1[j]);
  const newW2 = model.W2.map((w, j) => w - lr * avgDW2[j]);
  const newB2 = model.b2 - lr * avgDb2;

  return { W1: newW1, b1: newB1, W2: newW2, b2: newB2 };
}

// Single Sample SGD Step
function trainSingleSample(sample, model, lr = 0.1, actType = 'sigmoid') {
  const grads = computeSampleDerivatives(sample, model, actType);
  const H = model.b1.length;

  const newW1 = model.W1.map((row, j) => [
    row[0] - lr * grads.dW1[j][0],
    row[1] - lr * grads.dW1[j][1],
  ]);
  const newB1 = model.b1.map((b, j) => b - lr * grads.db1[j]);
  const newW2 = model.W2.map((w, j) => w - lr * grads.dW2[j]);
  const newB2 = model.b2 - lr * grads.db2;

  return {
    model: { W1: newW1, b1: newB1, W2: newW2, b2: newB2 },
    grads,
  };
}

// ==========================================
// 3. MAIN COMPONENT
// ==========================================
export default function Module10Visualizer() {
  // Navigation Tabs
  const [activeTab, setActiveTab] = useState('GLASS_BOX'); // 'GLASS_BOX' | 'STEP_INSPECTOR' | 'OPTIMIZER_GAME' | 'FORMULAS'

  // Model & Training State (Shared between Tab 1 & Tab 2)
  const [hiddenSize, setHiddenSize] = useState(4);
  const [actFunction, setActFunction] = useState('sigmoid'); // 'sigmoid' | 'tanh' | 'relu'
  const [model, setModel] = useState(() => createInitialModel(4));
  const [learningRate, setLearningRate] = useState(0.2);
  const [isTraining, setIsTraining] = useState(false);
  const [trainSpeed, setTrainSpeed] = useState(2); // Steps per frame
  const [epochCount, setEpochCount] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);

  // Dataset State
  const [datasetKey, setDatasetKey] = useState('XOR');
  const [customPoints, setCustomPoints] = useState(PRESET_DATASETS.CUSTOM.points);
  const [customPlacementClass, setCustomPlacementClass] = useState(1); // 0 or 1

  // Active Dataset
  const activeDataset = useMemo(() => {
    if (datasetKey === 'CUSTOM') {
      return customPoints;
    }
    return PRESET_DATASETS[datasetKey]?.points || [];
  }, [datasetKey, customPoints]);

  // Telemetry (Current Loss & Accuracy)
  const telemetry = useMemo(() => {
    return evaluateDataset(activeDataset, model, actFunction);
  }, [activeDataset, model, actFunction]);

  // Selected Sample for By-Hand Inspector
  const [selectedSampleIndex, setSelectedSampleIndex] = useState(0);
  const activeSample = useMemo(() => {
    if (activeDataset.length === 0) return { x1: 1.0, x2: -1.0, y: 1, id: 'manual' };
    const safeIdx = Math.max(0, Math.min(activeDataset.length - 1, selectedSampleIndex));
    return activeDataset[safeIdx];
  }, [activeDataset, selectedSampleIndex]);

  // Inspector manual point coordinates (if user adjusts sliders directly)
  const [inspectorCustomX1, setInspectorCustomX1] = useState(1.0);
  const [inspectorCustomX2, setInspectorCustomX2] = useState(-1.0);
  const [inspectorCustomY, setInspectorCustomY] = useState(1);
  const [useManualInspectorPoint, setUseManualInspectorPoint] = useState(false);

  const inspectedPoint = useMemo(() => {
    if (useManualInspectorPoint) {
      return { x1: inspectorCustomX1, x2: inspectorCustomX2, y: inspectorCustomY, id: 'custom-inspect' };
    }
    return activeSample;
  }, [useManualInspectorPoint, inspectorCustomX1, inspectorCustomX2, inspectorCustomY, activeSample]);

  // Analytical gradients for the inspected point
  const inspectedDerivatives = useMemo(() => {
    return computeSampleDerivatives(inspectedPoint, model, actFunction);
  }, [inspectedPoint, model, actFunction]);

  // Manual Nudge Feedback Alert
  const [nudgeFeedback, setNudgeFeedback] = useState(null);

  // Network Graph Active Hover Element
  const [hoveredElement, setHoveredElement] = useState(null); // { type: 'synapse' | 'node', id: ... }

  // Canvas Refs
  const canvasRef = useRef(null);
  const animFrameIdRef = useRef(null);

  // Re-initialize model when hidden size changes
  const handleResetModel = useCallback(() => {
    setIsTraining(false);
    setModel(createInitialModel(hiddenSize));
    setEpochCount(0);
    setLossHistory([]);
  }, [hiddenSize]);

  // Continuous Training Animation Loop (60 FPS)
  useEffect(() => {
    if (!isTraining) {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      return;
    }

    let isMounted = true;

    const loop = () => {
      if (!isMounted) return;

      setModel((prevModel) => {
        let currentModel = prevModel;
        for (let s = 0; s < trainSpeed; s++) {
          currentModel = trainEpoch(activeDataset, currentModel, learningRate, actFunction);
        }
        return currentModel;
      });

      setEpochCount((prev) => prev + trainSpeed);

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);

    return () => {
      isMounted = false;
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [isTraining, activeDataset, learningRate, actFunction, trainSpeed]);

  // Record Loss History periodically
  useEffect(() => {
    setLossHistory((prev) => {
      const next = [...prev, telemetry.loss];
      if (next.length > 50) next.shift();
      return next;
    });
  }, [telemetry.loss, epochCount]);

  // ----------------------------------------------------
  // 2D Decision Boundary Canvas Rendering (60 FPS Engine)
  // ----------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 1. Offscreen Grid Rendering (64x64) for ultra-fast bilinear interpolation
    const gridRes = 64;
    const offscreen = document.createElement('canvas');
    offscreen.width = gridRes;
    offscreen.height = gridRes;
    const offCtx = offscreen.getContext('2d');
    const imgData = offCtx.createImageData(gridRes, gridRes);
    const data = imgData.data;

    for (let gy = 0; gy < gridRes; gy++) {
      // Invert Y coordinate so +2.5 is at the top of the canvas
      const wy = 2.5 - (gy / (gridRes - 1)) * 5.0;
      for (let gx = 0; gx < gridRes; gx++) {
        const wx = -2.5 + (gx / (gridRes - 1)) * 5.0;
        const { yHat } = forwardPass(wx, wy, model, actFunction);
        const p = Math.max(0, Math.min(1, yHat));
        const idx = (gy * gridRes + gx) * 4;

        // Class 0 = Cyan (#06b6d4: rgb 6, 182, 212)
        // Class 1 = Rose / Fuchsia (#f43f5e: rgb 244, 63, 94)
        // Background theme: deep dark slate
        const r = Math.round(10 * (1 - p) + 140 * p + 30 * Math.pow(p, 2));
        const g = Math.round(40 * (1 - p) + 16 * p + 45 * Math.sin(p * Math.PI));
        const b = Math.round(75 * (1 - p) + 60 * p);

        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 230;
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    // Draw stretched offscreen buffer to full resolution canvas with smooth antialiasing
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreen, 0, 0, width, height);

    // 2. Draw Subtle Grid Coordinate System
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    const toCanvasX = (wx) => ((wx + 2.5) / 5.0) * width;
    const toCanvasY = (wy) => ((2.5 - wy) / 5.0) * height;

    // Grid ticks: -2, -1, 0, 1, 2
    [-2, -1, 1, 2].forEach((tick) => {
      // vertical line
      ctx.beginPath();
      ctx.moveTo(toCanvasX(tick), 0);
      ctx.lineTo(toCanvasX(tick), height);
      ctx.stroke();

      // horizontal line
      ctx.beginPath();
      ctx.moveTo(0, toCanvasY(tick));
      ctx.lineTo(width, toCanvasY(tick));
      ctx.stroke();
    });

    // Central Axes (x=0, y=0)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.stroke();

    // 3. Draw Axis Labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
    ctx.font = '10px monospace';
    ctx.fillText('x₁', width - 16, toCanvasY(0) - 6);
    ctx.fillText('x₂', toCanvasX(0) + 6, 14);

    // 4. Draw Data Points
    activeDataset.forEach((pt, idx) => {
      const cx = toCanvasX(pt.x1);
      const cy = toCanvasY(pt.x2);
      const isSelected = !useManualInspectorPoint && selectedSampleIndex === idx;

      // Evaluation for accuracy status dot
      const { yHat } = forwardPass(pt.x1, pt.x2, model, actFunction);
      const isCorrect = (yHat >= 0.5 ? 1 : 0) === pt.y;

      // Outer Glow
      ctx.fillStyle = pt.y === 1 ? 'rgba(244, 63, 94, 0.25)' : 'rgba(6, 182, 212, 0.25)';
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 14 : 9, 0, Math.PI * 2);
      ctx.fill();

      // Core Circle
      ctx.fillStyle = pt.y === 1 ? '#f43f5e' : '#06b6d4';
      ctx.strokeStyle = isCorrect ? '#ffffff' : '#f59e0b';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 7 : 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Center dot
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(cx, cy, 1.8, 0, Math.PI * 2);
      ctx.fill();

      // Selected Crosshair / Halo
      if (isSelected) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(cx, cy, 17, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // 5. Draw Inspected Point if custom manual point
    if (useManualInspectorPoint) {
      const cx = toCanvasX(inspectorCustomX1);
      const cy = toCanvasY(inspectorCustomX2);
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      ctx.fillStyle = inspectorCustomY === 1 ? '#f43f5e' : '#06b6d4';
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [model, activeDataset, actFunction, selectedSampleIndex, useManualInspectorPoint, inspectorCustomX1, inspectorCustomX2, inspectorCustomY]);

  // Click handler on 2D Canvas
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // Convert pixel to world coords (-2.5 to +2.5)
    const wx = parseFloat((-2.5 + (px / rect.width) * 5.0).toFixed(2));
    const wy = parseFloat((2.5 - (py / rect.height) * 5.0).toFixed(2));

    if (datasetKey === 'CUSTOM') {
      // Add custom point
      const newPt = {
        x1: wx,
        x2: wy,
        y: customPlacementClass,
        id: `custom-${Date.now()}`,
      };
      setCustomPoints((prev) => [...prev, newPt]);
    } else {
      // Find nearest point to select for inspection
      let nearestIdx = 0;
      let minDist = Infinity;
      activeDataset.forEach((pt, idx) => {
        const d = Math.hypot(pt.x1 - wx, pt.x2 - wy);
        if (d < minDist) {
          minDist = d;
          nearestIdx = idx;
        }
      });
      setSelectedSampleIndex(nearestIdx);
      setUseManualInspectorPoint(false);
    }
  };

  // Manual Nudge action from Step Inspector
  const handleApplyNudge = (lrStep = 0.2) => {
    const prevLoss = inspectedDerivatives.loss;
    const { model: updatedModel, grads } = trainSingleSample(inspectedPoint, model, lrStep, actFunction);
    setModel(updatedModel);

    // Compute updated loss
    const postPass = forwardPass(inspectedPoint.x1, inspectedPoint.x2, updatedModel, actFunction);
    const eps = 1e-7;
    const clampedYHat = Math.max(eps, Math.min(1 - eps, postPass.yHat));
    const newLoss = -(inspectedPoint.y * Math.log(clampedYHat) + (1 - inspectedPoint.y) * Math.log(1 - clampedYHat));
    const deltaLoss = newLoss - prevLoss;

    setNudgeFeedback({
      prevLoss: prevLoss.toFixed(4),
      newLoss: newLoss.toFixed(4),
      delta: deltaLoss.toFixed(4),
      isDrop: deltaLoss < 0,
    });

    setTimeout(() => {
      setNudgeFeedback(null);
    }, 4000);
  };

  // ==========================================
  // 4. GAME MODE STATE & LOGIC
  // ==========================================
  const GAME_LEVELS = useMemo(
    () => [
      {
        id: 1,
        title: 'Level 1: Single Neuron Gatekeeper',
        subtitle: 'Linearly Separable OR Gate',
        description: 'Tilt the single hyperplane slope and bias knob to achieve 100% classification accuracy on this OR pattern.',
        arch: '2 -> 1 (Single Perceptron)',
        targetAcc: 100,
        targetLoss: 0.15,
        parTurns: 6,
        dataset: [
          { x1: -1.0, x2: -1.0, y: 0, id: 'g1-1' }, // (0,0) -> 0
          { x1: -1.0, x2: 1.0, y: 1, id: 'g1-2' },  // (0,1) -> 1
          { x1: 1.0, x2: -1.0, y: 1, id: 'g1-3' },  // (1,0) -> 1
          { x1: 1.0, x2: 1.0, y: 1, id: 'g1-4' },   // (1,1) -> 1
        ],
        initialModel: {
          W1: [[0.1, -0.2]],
          b1: [-0.4],
          W2: [1.0],
          b2: 0.0,
        },
        hint: 'Both x₁ and x₂ should contribute positively to the output. Increase W₁[0] and W₁[1] to positive values and push bias b₁ above 0.3.',
      },
      {
        id: 2,
        title: 'Level 2: The 1969 XOR Breakthrough',
        subtitle: 'Minsky & Papert Historical Challenge',
        description: 'Single hyperplanes fail completely on XOR. Use 2 hidden neurons to slice the space diagonally and isolate both classes!',
        arch: '2 -> 2 -> 1',
        targetAcc: 100,
        targetLoss: 0.22,
        parTurns: 10,
        dataset: [
          { x1: -1.0, x2: -1.0, y: 0, id: 'g2-1' },
          { x1: 1.0, x2: 1.0, y: 0, id: 'g2-2' },
          { x1: -1.0, x2: 1.0, y: 1, id: 'g2-3' },
          { x1: 1.0, x2: -1.0, y: 1, id: 'g2-4' },
        ],
        initialModel: {
          W1: [
            [1.2, 1.2],
            [-1.1, -1.1],
          ],
          b1: [0.5, 0.5],
          W2: [0.8, -0.9],
          b2: -0.2,
        },
        hint: 'Neuron 1 should detect when inputs are both positive (x₁+x₂ > 0.5), while Neuron 2 detects when both are negative. Output weights combine them with opposite signs!',
      },
      {
        id: 3,
        title: 'Level 3: Concentric Circle Trap',
        subtitle: 'Multi-Hyperplane Geometric Enclosure',
        description: 'Capture the inner circular island using 4 hidden neurons forming a polygonal cage!',
        arch: '2 -> 4 -> 1',
        targetAcc: 95,
        targetLoss: 0.25,
        parTurns: 12,
        dataset: PRESET_DATASETS.CIRCLES.points,
        initialModel: {
          W1: [
            [1.5, 0.0],
            [-1.5, 0.0],
            [0.0, 1.5],
            [0.0, -1.5],
          ],
          b1: [0.3, 0.3, 0.3, 0.3],
          W2: [1.2, 1.2, 1.2, 1.2],
          b2: -3.5,
        },
        hint: 'Hidden neurons act as four walls (Right, Left, Top, Bottom). Tune the four biases b₁ to pull the walls tight around radius r ≈ 0.6!',
      },
      {
        id: 4,
        title: 'Level 4: Human vs SGD Speedrun',
        subtitle: 'The Non-Convex Two Moons Race',
        description: 'Bite into the curved valleys of Two Moons. Race to beat target loss < 0.22 in the fewest turns!',
        arch: '2 -> 4 -> 1',
        targetAcc: 92,
        targetLoss: 0.22,
        parTurns: 15,
        dataset: PRESET_DATASETS.MOONS.points,
        initialModel: createInitialModel(4),
        hint: 'Use the "SGD Assist" button to apply automated gradient nudges when fine-tuning multiple weights at once!',
      },
    ],
    []
  );

  const [currentLevelId, setCurrentLevelId] = useState(1);
  const activeLevel = GAME_LEVELS.find((l) => l.id === currentLevelId) || GAME_LEVELS[0];

  const [gameModel, setGameModel] = useState(activeLevel.initialModel);
  const [gameTurns, setGameTurns] = useState(0);
  const [levelCleared, setLevelCleared] = useState(false);
  const [showGameHint, setShowGameHint] = useState(false);

  // Evaluate game state
  const gameTelemetry = useMemo(() => {
    return evaluateDataset(activeLevel.dataset, gameModel, 'sigmoid');
  }, [activeLevel, gameModel]);

  // Check victory condition
  useEffect(() => {
    if (
      !levelCleared &&
      gameTelemetry.accuracy >= activeLevel.targetAcc &&
      gameTelemetry.loss <= activeLevel.targetLoss
    ) {
      setLevelCleared(true);
      confetti({
        particleCount: 120,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#06b6d4', '#a855f7', '#10b981', '#f43f5e', '#facc15'],
      });
    }
  }, [gameTelemetry, activeLevel, levelCleared]);

  // Reset Game Level
  const handleResetLevel = (level) => {
    setGameModel(JSON.parse(JSON.stringify(level.initialModel)));
    setGameTurns(0);
    setLevelCleared(false);
    setShowGameHint(false);
  };

  // Select Level
  const handleSelectLevel = (levelId) => {
    const nextLevel = GAME_LEVELS.find((l) => l.id === levelId) || GAME_LEVELS[0];
    setCurrentLevelId(levelId);
    handleResetLevel(nextLevel);
  };

  // Direct manual weight adjustment in Game Mode
  const handleGameWeightChange = (layer, j, i, delta) => {
    setGameTurns((prev) => prev + 1);
    setGameModel((prev) => {
      const next = JSON.parse(JSON.stringify(prev));
      if (layer === 'W1') {
        next.W1[j][i] = parseFloat((next.W1[j][i] + delta).toFixed(2));
      } else if (layer === 'b1') {
        next.b1[j] = parseFloat((next.b1[j] + delta).toFixed(2));
      } else if (layer === 'W2') {
        next.W2[j] = parseFloat((next.W2[j] + delta).toFixed(2));
      } else if (layer === 'b2') {
        next.b2 = parseFloat((next.b2 + delta).toFixed(2));
      }
      return next;
    });
  };

  // Game Mode SGD Assist Turn
  const handleGameSgdAssist = () => {
    setGameTurns((prev) => prev + 1);
    const updated = trainEpoch(activeLevel.dataset, gameModel, 0.35, 'sigmoid');
    setGameModel(updated);
  };

  // Render Mini Game Canvas
  const gameCanvasRef = useRef(null);
  useEffect(() => {
    const canvas = gameCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const gridRes = 48;
    const offscreen = document.createElement('canvas');
    offscreen.width = gridRes;
    offscreen.height = gridRes;
    const offCtx = offscreen.getContext('2d');
    const imgData = offCtx.createImageData(gridRes, gridRes);
    const data = imgData.data;

    for (let gy = 0; gy < gridRes; gy++) {
      const wy = 2.5 - (gy / (gridRes - 1)) * 5.0;
      for (let gx = 0; gx < gridRes; gx++) {
        const wx = -2.5 + (gx / (gridRes - 1)) * 5.0;
        const { yHat } = forwardPass(wx, wy, gameModel, 'sigmoid');
        const p = Math.max(0, Math.min(1, yHat));
        const idx = (gy * gridRes + gx) * 4;

        data[idx] = Math.round(15 * (1 - p) + 150 * p);
        data[idx + 1] = Math.round(35 * (1 - p) + 20 * p);
        data[idx + 2] = Math.round(75 * (1 - p) + 65 * p);
        data[idx + 3] = 230;
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, width, height);

    // Coordinate conversion
    const toCanvasX = (wx) => ((wx + 2.5) / 5.0) * width;
    const toCanvasY = (wy) => ((2.5 - wy) / 5.0) * height;

    // Center axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.stroke();

    // Data points
    activeLevel.dataset.forEach((pt) => {
      const cx = toCanvasX(pt.x1);
      const cy = toCanvasY(pt.x2);
      const { yHat } = forwardPass(pt.x1, pt.x2, gameModel, 'sigmoid');
      const isCorrect = (yHat >= 0.5 ? 1 : 0) === pt.y;

      ctx.fillStyle = pt.y === 1 ? '#f43f5e' : '#06b6d4';
      ctx.strokeStyle = isCorrect ? '#ffffff' : '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }, [gameModel, activeLevel]);

  // Compute 3-Star Rating for Game Mode
  const starRating = useMemo(() => {
    if (!levelCleared) return 0;
    if (gameTurns <= activeLevel.parTurns) return 3;
    if (gameTurns <= activeLevel.parTurns + 4) return 2;
    return 1;
  }, [levelCleared, gameTurns, activeLevel]);

  // ==========================================
  // RENDER JSX
  // ==========================================
  return (
    <div className="space-y-6">
      {/* Module 10 Top Header Banner */}
      <div className="bg-gradient-to-r from-cyan-950/40 via-purple-950/20 to-slate-900 border border-cyan-800/40 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Module 10 • Glass-Box Neural Network Sandbox
              </span>
              <span className="text-xs text-slate-400 font-mono">
                First Principles • Analytical Reverse Chain Rule
              </span>
            </div>
            <h2 className="text-xl font-black text-white flex items-center gap-2">
              <Eye className="w-5 h-5 text-cyan-400" />
              Train a NN by Hand and See What Happens Inside
            </h2>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Open the black box completely: see how non-linear neurons warp 2D hyperplanes at 60 FPS, trace
              every scalar arithmetic step with analytical backpropagation derivatives, and challenge yourself in
              the <span className="text-purple-400 font-semibold font-mono">"Be The Optimizer!"</span> puzzle mode.
            </p>
          </div>

          {/* Quick Telemetry Pill */}
          <div className="flex items-center gap-3 bg-slate-900/90 border border-slate-800 px-4 py-2.5 rounded-xl text-xs font-mono">
            <div>
              <div className="text-[10px] text-slate-500 uppercase">Loss (BCE)</div>
              <div className="text-sm font-bold text-rose-400">
                {telemetry.loss.toFixed(4)}
              </div>
            </div>
            <div className="h-6 w-px bg-slate-800" />
            <div>
              <div className="text-[10px] text-slate-500 uppercase">Accuracy</div>
              <div className="text-sm font-bold text-emerald-400">
                {telemetry.accuracy.toFixed(1)}%
              </div>
            </div>
            <div className="h-6 w-px bg-slate-800" />
            <div>
              <div className="text-[10px] text-slate-500 uppercase">Epoch</div>
              <div className="text-sm font-bold text-cyan-400">{epochCount}</div>
            </div>
          </div>
        </div>

        {/* Sub-Tabs Navigation */}
        <div className="flex items-center gap-2 mt-5 overflow-x-auto pb-1 border-t border-slate-800/80 pt-4">
          <button
            onClick={() => setActiveTab('GLASS_BOX')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
              activeTab === 'GLASS_BOX'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-md shadow-cyan-500/10'
                : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-850 hover:text-white'
            }`}
          >
            <Eye className="w-4 h-4 text-cyan-400" />
            <span>Tab 1: Glass Box Visualizer</span>
          </button>

          <button
            onClick={() => setActiveTab('STEP_INSPECTOR')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
              activeTab === 'STEP_INSPECTOR'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50 shadow-md shadow-purple-500/10'
                : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-850 hover:text-white'
            }`}
          >
            <Calculator className="w-4 h-4 text-purple-400" />
            <span>Tab 2: By-Hand Step Inspector</span>
          </button>

          <button
            onClick={() => setActiveTab('OPTIMIZER_GAME')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
              activeTab === 'OPTIMIZER_GAME'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-md shadow-amber-500/10'
                : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-850 hover:text-white'
            }`}
          >
            <Trophy className="w-4 h-4 text-amber-400" />
            <span>Tab 3: Be The Optimizer! (Game)</span>
          </button>

          <button
            onClick={() => setActiveTab('FORMULAS')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
              activeTab === 'FORMULAS'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-md shadow-emerald-500/10'
                : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-850 hover:text-white'
            }`}
          >
            <BookOpen className="w-4 h-4 text-emerald-400" />
            <span>Tab 4: Formulas & Glass Box Guide</span>
          </button>
        </div>
      </div>

      {/* ============================================================== */}
      {/* TAB 1: GLASS BOX VISUALIZER (60 FPS Canvas & Network Graph) */}
      {/* ============================================================== */}
      {activeTab === 'GLASS_BOX' && (
        <div className="space-y-6">
          {/* Controls & Presets Ribbon */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4">
            {/* Dataset Selector */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-semibold text-slate-400 font-mono flex items-center gap-1.5">
                <Grid className="w-3.5 h-3.5 text-cyan-400" /> Dataset:
              </span>
              {Object.keys(PRESET_DATASETS).map((key) => (
                <button
                  key={key}
                  onClick={() => setDatasetKey(key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition ${
                    datasetKey === key
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : 'bg-slate-800/80 text-slate-400 hover:text-white border border-slate-700/60'
                  }`}
                >
                  {key}
                </button>
              ))}
            </div>

            {/* Training Buttons */}
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => setIsTraining((prev) => !prev)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold transition shadow-lg ${
                  isTraining
                    ? 'bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-amber-500/20'
                    : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-emerald-500/20'
                }`}
              >
                {isTraining ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isTraining ? 'Pause Training' : 'Start 60 FPS'}</span>
              </button>

              <button
                onClick={() => {
                  setModel((prev) => trainEpoch(activeDataset, prev, learningRate, actFunction));
                  setEpochCount((prev) => prev + 1);
                }}
                disabled={isTraining}
                className="flex items-center gap-1 px-3 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700 disabled:opacity-40 transition"
                title="Run 1 full batch gradient epoch"
              >
                <FastForward className="w-3.5 h-3.5 text-cyan-400" />
                <span>1-Epoch Step</span>
              </button>

              <button
                onClick={() => {
                  const { model: nextM } = trainSingleSample(activeSample, model, learningRate, actFunction);
                  setModel(nextM);
                  setEpochCount((prev) => prev + 1);
                }}
                disabled={isTraining}
                className="flex items-center gap-1 px-3 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700 disabled:opacity-40 transition"
                title="Run stochastic gradient descent on 1 selected point"
              >
                <Target className="w-3.5 h-3.5 text-purple-400" />
                <span>1-Sample SGD</span>
              </button>

              <button
                onClick={handleResetModel}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-400 hover:text-white border border-slate-700 transition"
                title="Randomize weights (He/Xavier Initialization)"
              >
                <Shuffle className="w-4 h-4 text-rose-400" />
              </button>
            </div>
          </div>

          {/* Main 2-Column Visualization Stage */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: 2D Decision Boundary Canvas */}
            <div className="lg:col-span-6 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-bold text-white font-mono">2D Continuous Decision Field</h3>
                </div>
                <div className="flex items-center gap-3 text-[11px] font-mono">
                  <span className="flex items-center gap-1 text-cyan-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" /> Class 0
                  </span>
                  <span className="flex items-center gap-1 text-rose-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-400 inline-block" /> Class 1
                  </span>
                </div>
              </div>

              {/* Canvas Container */}
              <div className="relative aspect-square w-full max-w-[420px] mx-auto rounded-xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950 flex items-center justify-center">
                <canvas
                  ref={canvasRef}
                  width={380}
                  height={380}
                  onClick={handleCanvasClick}
                  className="w-full h-full cursor-crosshair block"
                />

                {/* Custom Point Click Mode Banner */}
                {datasetKey === 'CUSTOM' && (
                  <div className="absolute top-3 left-3 right-3 bg-slate-900/90 backdrop-blur-md border border-cyan-500/40 rounded-lg p-2 flex items-center justify-between text-xs">
                    <span className="text-slate-300 font-mono text-[11px]">Click to add:</span>
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setCustomPlacementClass(0);
                        }}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition ${
                          customPlacementClass === 0
                            ? 'bg-cyan-500 text-slate-950'
                            : 'bg-slate-800 text-cyan-400 border border-cyan-800'
                        }`}
                      >
                        Class 0 (Cyan)
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setCustomPlacementClass(1);
                        }}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition ${
                          customPlacementClass === 1
                            ? 'bg-rose-500 text-slate-950'
                            : 'bg-slate-800 text-rose-400 border border-rose-800'
                        }`}
                      >
                        Class 1 (Rose)
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setCustomPoints([]);
                        }}
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 hover:bg-rose-900/40 text-slate-400 hover:text-rose-300 border border-slate-700 transition"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Canvas Helper Footer */}
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-1">
                <div className="flex items-center gap-1.5">
                  <MousePointer className="w-3.5 h-3.5 text-cyan-400" />
                  <span>
                    {datasetKey === 'CUSTOM'
                      ? 'Click canvas to place data points'
                      : 'Click any point on canvas to inspect in Tab 2'}
                  </span>
                </div>
                <div className="text-[11px] text-slate-500">Range: [-2.5, +2.5]</div>
              </div>
            </div>

            {/* Right Column: Dynamic Network Graph & Synapses */}
            <div className="lg:col-span-6 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  <h3 className="text-sm font-bold text-white font-mono">Dynamic Glass-Box Synaptic Graph</h3>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-400 uppercase">Hidden:</span>
                  {[2, 3, 4, 6].map((num) => (
                    <button
                      key={num}
                      onClick={() => {
                        setHiddenSize(num);
                        setModel(createInitialModel(num));
                      }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition ${
                        hiddenSize === num
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                          : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'
                      }`}
                    >
                      {num}
                    </button>
                  ))}
                </div>
              </div>

              {/* Dynamic SVG Network Graph */}
              <div className="relative aspect-square w-full max-w-[420px] mx-auto rounded-xl overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center p-2">
                <svg viewBox="0 0 400 360" className="w-full h-full select-none">
                  <defs>
                    <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#0891b2" stopOpacity="0.3" />
                    </linearGradient>
                    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#7e22ce" stopOpacity="0.3" />
                    </linearGradient>
                    <linearGradient id="roseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#be123c" stopOpacity="0.3" />
                    </linearGradient>
                  </defs>

                  {/* Synapse Lines: Layer 0 (Input) -> Layer 1 (Hidden) */}
                  {model.W1.map((row, j) => {
                    const hy = 40 + ((360 - 80) / (hiddenSize - 1 || 1)) * j;
                    return row.map((w, i) => {
                      const iy = 110 + i * 140;
                      const isPos = w >= 0;
                      const strokeWidth = Math.min(5.5, Math.max(1.2, Math.abs(w) * 1.5));
                      const isHovered =
                        hoveredElement?.type === 'synapse' &&
                        hoveredElement.id === `w1-${j}-${i}`;

                      return (
                        <g key={`w1-${j}-${i}`}>
                          <line
                            x1={60}
                            y1={iy}
                            x2={200}
                            y2={hy}
                            stroke={
                              isHovered
                                ? '#facc15'
                                : isPos
                                ? '#06b6d4'
                                : '#f43f5e'
                            }
                            strokeWidth={strokeWidth}
                            strokeOpacity={isHovered ? 1 : 0.4 + Math.min(0.5, Math.abs(w) * 0.2)}
                            strokeDasharray={isPos ? 'none' : '4, 2'}
                            className="cursor-pointer transition-all duration-150"
                            onMouseEnter={() =>
                              setHoveredElement({
                                type: 'synapse',
                                id: `w1-${j}-${i}`,
                                label: `W₁[${j}][${i}] = ${w.toFixed(3)}`,
                                sub: `Input x${i + 1} -> Hidden h${j + 1}`,
                              })
                            }
                            onMouseLeave={() => setHoveredElement(null)}
                          />
                        </g>
                      );
                    });
                  })}

                  {/* Synapse Lines: Layer 1 (Hidden) -> Layer 2 (Output) */}
                  {model.W2.map((w, j) => {
                    const hy = 40 + ((360 - 80) / (hiddenSize - 1 || 1)) * j;
                    const oy = 180;
                    const isPos = w >= 0;
                    const strokeWidth = Math.min(5.5, Math.max(1.2, Math.abs(w) * 1.5));
                    const isHovered =
                      hoveredElement?.type === 'synapse' &&
                      hoveredElement.id === `w2-${j}`;

                    return (
                      <g key={`w2-${j}`}>
                        <line
                          x1={200}
                          y1={hy}
                          x2={340}
                          y2={oy}
                          stroke={
                            isHovered
                              ? '#facc15'
                              : isPos
                              ? '#10b981'
                              : '#f43f5e'
                          }
                          strokeWidth={strokeWidth}
                          strokeOpacity={isHovered ? 1 : 0.4 + Math.min(0.5, Math.abs(w) * 0.2)}
                          strokeDasharray={isPos ? 'none' : '4, 2'}
                          className="cursor-pointer transition-all duration-150"
                          onMouseEnter={() =>
                            setHoveredElement({
                              type: 'synapse',
                              id: `w2-${j}`,
                              label: `W₂[${j}] = ${w.toFixed(3)}`,
                              sub: `Hidden h${j + 1} -> Output ŷ`,
                            })
                          }
                          onMouseLeave={() => setHoveredElement(null)}
                        />
                      </g>
                    );
                  })}

                  {/* Layer 0 Input Nodes */}
                  {[0, 1].map((i) => {
                    const iy = 110 + i * 140;
                    const val = i === 0 ? inspectedPoint.x1 : inspectedPoint.x2;
                    return (
                      <g
                        key={`in-${i}`}
                        className="cursor-pointer"
                        onMouseEnter={() =>
                          setHoveredElement({
                            type: 'node',
                            id: `in-${i}`,
                            label: `Input x${i + 1} = ${val.toFixed(2)}`,
                            sub: `Feature dimension ${i + 1}`,
                          })
                        }
                        onMouseLeave={() => setHoveredElement(null)}
                      >
                        <circle cx={60} cy={iy} r={18} fill="#090d16" stroke="#06b6d4" strokeWidth={2.5} />
                        <text
                          x={60}
                          y={iy + 4}
                          textAnchor="middle"
                          fill="#ffffff"
                          fontSize="11"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          x{i + 1}
                        </text>
                        <text
                          x={60}
                          y={iy + 28}
                          textAnchor="middle"
                          fill="#94a3b8"
                          fontSize="9"
                          fontFamily="monospace"
                        >
                          {val.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}

                  {/* Layer 1 Hidden Nodes */}
                  {model.b1.map((b, j) => {
                    const hy = 40 + ((360 - 80) / (hiddenSize - 1 || 1)) * j;
                    const actVal = inspectedDerivatives.a1[j] || 0.5;
                    const zVal = inspectedDerivatives.z1[j] || 0.0;
                    return (
                      <g
                        key={`hid-${j}`}
                        className="cursor-pointer"
                        onMouseEnter={() =>
                          setHoveredElement({
                            type: 'node',
                            id: `hid-${j}`,
                            label: `Hidden h${j + 1}: a = ${actVal.toFixed(3)}`,
                            sub: `Pre-activation z = ${zVal.toFixed(3)}, bias b = ${b.toFixed(3)}`,
                          })
                        }
                        onMouseLeave={() => setHoveredElement(null)}
                      >
                        {/* Glow halo */}
                        <circle
                          cx={200}
                          cy={hy}
                          r={18}
                          fill="rgba(168, 85, 247, 0.2)"
                          stroke="#a855f7"
                          strokeWidth={2}
                        />
                        {/* Active core fill */}
                        <circle
                          cx={200}
                          cy={hy}
                          r={14 * Math.max(0.2, actVal)}
                          fill="rgba(168, 85, 247, 0.6)"
                        />
                        <text
                          x={200}
                          y={hy + 4}
                          textAnchor="middle"
                          fill="#ffffff"
                          fontSize="10"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          h{j + 1}
                        </text>
                        {/* Bias Pill Badge */}
                        <rect
                          x={218}
                          y={hy - 8}
                          width={42}
                          height={16}
                          rx={4}
                          fill="#1e1b4b"
                          stroke="#6366f1"
                          strokeWidth={0.8}
                        />
                        <text
                          x={239}
                          y={hy + 4}
                          textAnchor="middle"
                          fill="#c7d2fe"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          b={b.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}

                  {/* Layer 2 Output Node */}
                  {(() => {
                    const oy = 180;
                    const pred = inspectedDerivatives.yHat || 0.5;
                    return (
                      <g
                        className="cursor-pointer"
                        onMouseEnter={() =>
                          setHoveredElement({
                            type: 'node',
                            id: 'out',
                            label: `Output ŷ = ${pred.toFixed(4)}`,
                            sub: `Target y = ${inspectedPoint.y}, Loss = ${inspectedDerivatives.loss.toFixed(4)}`,
                          })
                        }
                        onMouseLeave={() => setHoveredElement(null)}
                      >
                        <circle
                          cx={340}
                          cy={oy}
                          r={22}
                          fill="rgba(244, 63, 94, 0.25)"
                          stroke="#f43f5e"
                          strokeWidth={2.5}
                        />
                        <circle
                          cx={340}
                          cy={oy}
                          r={16 * Math.max(0.2, pred)}
                          fill="rgba(244, 63, 94, 0.7)"
                        />
                        <text
                          x={340}
                          y={oy + 4}
                          textAnchor="middle"
                          fill="#ffffff"
                          fontSize="11"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          ŷ
                        </text>
                        <rect
                          x={318}
                          y={oy + 26}
                          width={44}
                          height={16}
                          rx={4}
                          fill="#4c0519"
                          stroke="#f43f5e"
                          strokeWidth={0.8}
                        />
                        <text
                          x={340}
                          y={oy + 38}
                          textAnchor="middle"
                          fill="#fecdd3"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          {pred.toFixed(2)}
                        </text>
                      </g>
                    );
                  })()}
                </svg>

                {/* Floating Interactive Hover Tooltip */}
                {hoveredElement && (
                  <div className="absolute bottom-3 left-3 right-3 bg-slate-900/95 backdrop-blur-md border border-amber-500/40 rounded-lg p-2.5 text-xs font-mono shadow-xl animate-fadeIn">
                    <div className="text-amber-300 font-bold">{hoveredElement.label}</div>
                    <div className="text-slate-400 text-[11px]">{hoveredElement.sub}</div>
                  </div>
                )}
              </div>

              {/* Hyperparameters Controls Bar */}
              <div className="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
                <div className="space-y-1">
                  <div className="flex justify-between text-slate-400 text-[11px]">
                    <span>Learning Rate (η):</span>
                    <span className="text-cyan-400 font-bold">{learningRate.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="1.0"
                    step="0.01"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-slate-400 text-[11px]">
                    <span>Activation Func:</span>
                    <span className="text-purple-400 font-bold uppercase">{actFunction}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    {['sigmoid', 'tanh', 'relu'].map((act) => (
                      <button
                        key={act}
                        onClick={() => setActFunction(act)}
                        className={`flex-1 py-1 rounded text-[10px] uppercase font-bold transition ${
                          actFunction === act
                            ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                            : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'
                        }`}
                      >
                        {act}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 2: BY-HAND STEP INSPECTOR (Exact Scalar Arithmetic) */}
      {/* ============================================================== */}
      {activeTab === 'STEP_INSPECTOR' && (
        <div className="space-y-6">
          {/* Sample Selector Bar */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-1">
              <span className="text-xs font-semibold text-purple-400 font-mono flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5" /> Selected Training Sample
              </span>
              <p className="text-xs text-slate-400">
                Pick a point from the dataset or adjust the coordinates manually to inspect the exact numerical chain rule.
              </p>
            </div>

            <div className="flex items-center gap-3 flex-wrap">
              {/* Dropdown to pick from dataset */}
              <select
                value={useManualInspectorPoint ? 'custom' : selectedSampleIndex}
                onChange={(e) => {
                  if (e.target.value === 'custom') {
                    setUseManualInspectorPoint(true);
                  } else {
                    setUseManualInspectorPoint(false);
                    setSelectedSampleIndex(parseInt(e.target.value, 10));
                  }
                }}
                className="bg-slate-850 border border-slate-700 text-slate-200 text-xs font-mono rounded-xl px-3 py-2 focus:outline-none focus:border-purple-500"
              >
                {activeDataset.map((pt, idx) => (
                  <option key={pt.id || idx} value={idx}>
                    Point #{idx + 1}: x=({pt.x1}, {pt.x2}) → y={pt.y}
                  </option>
                ))}
                <option value="custom">Custom Arbitrary Point (Sliders)</option>
              </select>

              {/* Nudge Button */}
              <button
                onClick={() => handleApplyNudge(0.25)}
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-lg shadow-purple-500/20 transition"
              >
                <Zap className="w-3.5 h-3.5 text-amber-300" />
                <span>Nudge Weights (-η ∇L)</span>
              </button>
            </div>
          </div>

          {/* Optional Manual Sliders (if custom point chosen) */}
          {useManualInspectorPoint && (
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Input x₁:</span>
                  <span className="text-cyan-400 font-bold">{inspectorCustomX1.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-2.5"
                  max="2.5"
                  step="0.05"
                  value={inspectorCustomX1}
                  onChange={(e) => setInspectorCustomX1(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Input x₂:</span>
                  <span className="text-cyan-400 font-bold">{inspectorCustomX2.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-2.5"
                  max="2.5"
                  step="0.05"
                  value={inspectorCustomX2}
                  onChange={(e) => setInspectorCustomX2(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Ground Truth y:</span>
                  <span className="text-rose-400 font-bold">{inspectorCustomY}</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setInspectorCustomY(0)}
                    className={`flex-1 py-1 rounded text-xs font-bold transition ${
                      inspectorCustomY === 0
                        ? 'bg-cyan-500 text-slate-950'
                        : 'bg-slate-800 text-cyan-400 border border-slate-700'
                    }`}
                  >
                    Class 0
                  </button>
                  <button
                    onClick={() => setInspectorCustomY(1)}
                    className={`flex-1 py-1 rounded text-xs font-bold transition ${
                      inspectorCustomY === 1
                        ? 'bg-rose-500 text-slate-950'
                        : 'bg-slate-800 text-rose-400 border border-slate-700'
                    }`}
                  >
                    Class 1
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Nudge Feedback Alert Banner */}
          {nudgeFeedback && (
            <div
              className={`p-4 rounded-xl border flex items-center justify-between text-xs font-mono transition-all animate-fadeIn ${
                nudgeFeedback.isDrop
                  ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
                  : 'bg-rose-950/40 border-rose-500/50 text-rose-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>
                  <strong>Manual Gradient Nudge Applied!</strong> Loss shifted from{' '}
                  <span className="font-bold">{nudgeFeedback.prevLoss}</span> →{' '}
                  <span className="font-bold">{nudgeFeedback.newLoss}</span> (Δ {nudgeFeedback.delta}).
                </span>
              </div>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30">
                1-Step SGD Complete
              </span>
            </div>
          )}

          {/* Stepper Pipeline Cards */}
          <div className="space-y-4">
            {/* Stage 1: Input Vector & Label */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-cyan-400 font-mono uppercase tracking-wider">
                  Stage 1: Input Vector & Target
                </span>
                <span className="text-xs text-slate-500 font-mono">Dendrite Reception</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">FEATURE x₁</div>
                  <div className="text-lg font-bold text-white mt-1">
                    {inspectedPoint.x1.toFixed(3)}
                  </div>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">FEATURE x₂</div>
                  <div className="text-lg font-bold text-white mt-1">
                    {inspectedPoint.x2.toFixed(3)}
                  </div>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">GROUND TRUTH y</div>
                  <div className="text-lg font-bold text-amber-400 mt-1">
                    {inspectedPoint.y} (Class {inspectedPoint.y})
                  </div>
                </div>
              </div>
            </div>

            {/* Stage 2: Hidden Pre-activations z1 = W1*x + b1 */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-purple-400 font-mono uppercase tracking-wider">
                  Stage 2: Hidden Pre-activations (Affine Affiliation z₁ = W₁·x + b₁)
                </span>
                <span className="text-xs text-slate-500 font-mono">Linear Summation</span>
              </div>
              <div className="space-y-2">
                {model.b1.map((b, j) => {
                  const w1_val = model.W1[j][0];
                  const w2_val = model.W1[j][1];
                  const x1_val = inspectedPoint.x1;
                  const x2_val = inspectedPoint.x2;
                  const z_val = inspectedDerivatives.z1[j];

                  return (
                    <div
                      key={`stage2-${j}`}
                      className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs flex flex-col md:flex-row md:items-center justify-between gap-2"
                    >
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 font-bold">
                          z₁[{j + 1}]
                        </span>
                        <span className="text-slate-300">
                          = (W₁[{j}][0] · x₁) + (W₁[{j}][1] · x₂) + b₁[{j}]
                        </span>
                      </div>
                      <div className="text-slate-400 text-[11px]">
                        = ({w1_val.toFixed(2)} × {x1_val.toFixed(2)}) + ({w2_val.toFixed(2)} × {x2_val.toFixed(2)}) + ({b.toFixed(2)})
                        {' → '}
                        <span className="text-purple-300 font-bold text-xs">{z_val.toFixed(4)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Stage 3: Non-Linear Activation a1 = σ(z1) */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-indigo-400 font-mono uppercase tracking-wider">
                  Stage 3: Non-Linear Activation (Space Warping a₁ = σ(z₁))
                </span>
                <span className="text-xs text-slate-500 font-mono">Sigmoidal Squashing</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 font-mono text-xs">
                {inspectedDerivatives.a1.map((a, j) => (
                  <div key={`stage3-${j}`} className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5">
                    <div className="flex justify-between text-slate-500 text-[10px]">
                      <span>NEURON h{j + 1}</span>
                      <span>σ(z={inspectedDerivatives.z1[j].toFixed(2)})</span>
                    </div>
                    <div className="text-base font-bold text-indigo-300">{a.toFixed(4)}</div>
                    {/* Visual Meter Bar */}
                    <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                        style={{ width: `${Math.max(0, Math.min(100, a * 100))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Stage 4: Output Layer Pre-activation & Prediction */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-rose-400 font-mono uppercase tracking-wider">
                  Stage 4: Output Synthesis & Binary Cross-Entropy Loss
                </span>
                <span className="text-xs text-slate-500 font-mono">Prediction & Seed</span>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3 font-mono text-xs">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-850 pb-2">
                  <span className="text-slate-400">
                    Output Pre-activation: z₂ = Σ (W₂[j] · a₁[j]) + b₂
                  </span>
                  <span className="text-white font-bold">
                    z₂ = {inspectedDerivatives.z2.toFixed(4)}
                  </span>
                </div>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-850 pb-2">
                  <span className="text-slate-400">
                    Predicted Probability: ŷ = σ(z₂) = 1 / (1 + e^(-z₂))
                  </span>
                  <span className="text-rose-400 font-bold text-sm">
                    ŷ = {inspectedDerivatives.yHat.toFixed(4)}{' '}
                    <span className="text-[11px] text-slate-500">
                      ({(inspectedDerivatives.yHat * 100).toFixed(1)}% prob of Class 1)
                    </span>
                  </span>
                </div>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                  <span className="text-slate-400">
                    BCE Loss: L = -[y ln(ŷ) + (1-y) ln(1-ŷ)]
                  </span>
                  <span className="text-amber-400 font-bold text-sm">
                    L = {inspectedDerivatives.loss.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>

            {/* Stage 5: Reverse Chain Rule Derivatives Breakdown */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-amber-400 font-mono uppercase tracking-wider flex items-center gap-1.5">
                  <Flame className="w-3.5 h-3.5" /> Stage 5: Reverse Mode Chain Rule Derivatives
                </span>
                <span className="text-xs text-slate-500 font-mono">Backpressure Waves</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
                {/* Output Layer Gradients */}
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2.5">
                  <div className="text-amber-400 font-bold text-[11px] uppercase border-b border-slate-850 pb-1">
                    Output Layer Gradients (∂L/∂W₂, ∂L/∂b₂)
                  </div>
                  <div className="text-slate-300 text-[11px] space-y-1">
                    <p className="text-slate-400">
                      Output delta: δ₂ = ∂L/∂z₂ = (ŷ - y) = ({inspectedDerivatives.yHat.toFixed(3)} - {inspectedPoint.y}) ={' '}
                      <span className="text-rose-400 font-bold">
                        {inspectedDerivatives.delta2.toFixed(4)}
                      </span>
                    </p>
                    <p className="text-slate-400">
                      ∂L/∂b₂ = δ₂ ={' '}
                      <span className="text-rose-400 font-bold">
                        {inspectedDerivatives.db2.toFixed(4)}
                      </span>
                    </p>
                  </div>

                  <div className="space-y-1 pt-1">
                    {inspectedDerivatives.dW2.map((dw, j) => (
                      <div key={`dw2-${j}`} className="flex justify-between text-[11px]">
                        <span className="text-slate-400">∂L/∂W₂[{j}] = δ₂ · a₁[{j}]</span>
                        <span className="text-amber-300 font-bold">{dw.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Hidden Layer Backpropagated Gradients */}
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2.5">
                  <div className="text-cyan-400 font-bold text-[11px] uppercase border-b border-slate-850 pb-1">
                    Hidden Layer Deltas (δ₁ = ∂L/∂a₁ ⊙ σ'(z₁))
                  </div>
                  <div className="space-y-1.5">
                    {inspectedDerivatives.delta1.map((del, j) => (
                      <div key={`del1-${j}`} className="text-[11px] border-b border-slate-900 pb-1">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Hidden δ₁[{j + 1}]:</span>
                          <span className="text-cyan-300 font-bold">{del.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between text-slate-500 text-[10px]">
                          <span>∂L/∂W₁: [{inspectedDerivatives.dW1[j][0].toFixed(3)}, {inspectedDerivatives.dW1[j][1].toFixed(3)}]</span>
                          <span>∂L/∂b₁: {inspectedDerivatives.db1[j].toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Action Button: Apply Manual Gradient Nudge */}
              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => handleApplyNudge(0.25)}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-gradient-to-r from-amber-500 to-rose-500 hover:from-amber-400 hover:to-rose-400 text-slate-950 shadow-lg shadow-amber-500/20 transition"
                >
                  <Zap className="w-4 h-4 text-slate-950 fill-current" />
                  <span>Execute Parameter Nudge (θ ← θ - η·∇L)</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 3: BE THE OPTIMIZER! (Interactive Game Mode) */}
      {/* ============================================================== */}
      {activeTab === 'OPTIMIZER_GAME' && (
        <div className="space-y-6">
          {/* Level Selection Bar */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-semibold text-amber-400 font-mono flex items-center gap-1.5">
                <Trophy className="w-3.5 h-3.5" /> Challenge Levels:
              </span>
              {GAME_LEVELS.map((lvl) => (
                <button
                  key={lvl.id}
                  onClick={() => handleSelectLevel(lvl.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition flex items-center gap-1.5 ${
                    currentLevelId === lvl.id
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-md shadow-amber-500/10'
                      : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'
                  }`}
                >
                  <span>Lvl {lvl.id}</span>
                </button>
              ))}
            </div>

            {/* Level Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowGameHint((prev) => !prev)}
                className="flex items-center gap-1 px-3 py-1.5 rounded-xl text-xs font-mono bg-slate-800 hover:bg-slate-750 text-amber-300 border border-slate-700 transition"
              >
                <Lightbulb className="w-3.5 h-3.5 text-amber-400" />
                <span>{showGameHint ? 'Hide Hint' : 'Take Hint'}</span>
              </button>

              <button
                onClick={handleGameSgdAssist}
                className="flex items-center gap-1 px-3 py-1.5 rounded-xl text-xs font-mono bg-slate-800 hover:bg-slate-750 text-cyan-300 border border-slate-700 transition"
                title="Spend 1 turn to take an automated SGD step"
              >
                <FastForward className="w-3.5 h-3.5 text-cyan-400" />
                <span>SGD Assist (1 Turn)</span>
              </button>

              <button
                onClick={() => handleResetLevel(activeLevel)}
                className="p-1.5 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-400 hover:text-white border border-slate-700 transition"
                title="Reset Level"
              >
                <RotateCcw className="w-4 h-4 text-rose-400" />
              </button>
            </div>
          </div>

          {/* Level Header HUD */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-2">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <span className="text-[10px] font-mono font-bold text-amber-400 uppercase tracking-widest">
                  {activeLevel.subtitle}
                </span>
                <h3 className="text-lg font-black text-white">{activeLevel.title}</h3>
                <p className="text-xs text-slate-300 max-w-2xl mt-0.5">{activeLevel.description}</p>
              </div>

              {/* Game Stats Telemetry */}
              <div className="flex items-center gap-3 bg-slate-950 border border-slate-800 px-4 py-2.5 rounded-xl text-xs font-mono flex-shrink-0">
                <div>
                  <div className="text-[10px] text-slate-500 uppercase">Turns Used</div>
                  <div className="text-sm font-bold text-white">
                    {gameTurns}{' '}
                    <span className="text-slate-500 text-[10px]">/ Par {activeLevel.parTurns}</span>
                  </div>
                </div>
                <div className="h-6 w-px bg-slate-800" />
                <div>
                  <div className="text-[10px] text-slate-500 uppercase">Loss Target</div>
                  <div
                    className={`text-sm font-bold ${
                      gameTelemetry.loss <= activeLevel.targetLoss ? 'text-emerald-400' : 'text-rose-400'
                    }`}
                  >
                    {gameTelemetry.loss.toFixed(3)}{' '}
                    <span className="text-slate-500 text-[10px]">
                      (&lt; {activeLevel.targetLoss.toFixed(2)})
                    </span>
                  </div>
                </div>
                <div className="h-6 w-px bg-slate-800" />
                <div>
                  <div className="text-[10px] text-slate-500 uppercase">Accuracy</div>
                  <div
                    className={`text-sm font-bold ${
                      gameTelemetry.accuracy >= activeLevel.targetAcc ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                  >
                    {gameTelemetry.accuracy.toFixed(0)}%{' '}
                    <span className="text-slate-500 text-[10px]">
                      (Goal: {activeLevel.targetAcc}%)
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Hint Box */}
            {showGameHint && (
              <div className="mt-3 p-3 bg-amber-950/40 border border-amber-500/40 rounded-xl text-xs font-mono text-amber-200 flex items-start gap-2 animate-fadeIn">
                <Lightbulb className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-amber-300">Optimizer Hint:</strong> {activeLevel.hint}
                </div>
              </div>
            )}
          </div>

          {/* Victory Modal Banner */}
          {levelCleared && (
            <div className="bg-gradient-to-r from-emerald-950/80 via-slate-900 to-amber-950/60 border border-emerald-500/60 rounded-2xl p-6 shadow-2xl flex flex-col md:flex-row items-center justify-between gap-4 animate-fadeIn">
              <div className="space-y-1 text-center md:text-left">
                <div className="flex items-center justify-center md:justify-start gap-2">
                  <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                  <h3 className="text-xl font-black text-white">LEVEL OBJECTIVE ACHIEVED!</h3>
                </div>
                <p className="text-xs text-slate-300">
                  Spectacular! You warped the internal representation and solved the classification goal in{' '}
                  <span className="text-emerald-300 font-bold">{gameTurns} turns</span>.
                </p>
                <div className="flex items-center justify-center md:justify-start gap-1 pt-1">
                  {[1, 2, 3].map((star) => (
                    <Star
                      key={star}
                      className={`w-5 h-5 ${
                        star <= starRating ? 'text-amber-400 fill-amber-400' : 'text-slate-700'
                      }`}
                    />
                  ))}
                  <span className="text-xs font-mono text-amber-300 ml-2 font-bold">
                    {starRating === 3 ? '★ 3 Stars (Under Par!)' : starRating === 2 ? '★ 2 Stars' : '★ 1 Star'}
                  </span>
                </div>
              </div>

              {currentLevelId < GAME_LEVELS.length && (
                <button
                  onClick={() => handleSelectLevel(currentLevelId + 1)}
                  className="px-6 py-3 rounded-xl text-xs font-bold bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-lg shadow-emerald-500/20 transition flex items-center gap-2 flex-shrink-0"
                >
                  <span>Next Level Challenge</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              )}
            </div>
          )}

          {/* Game Stage: Canvas on Left, Interactive Weight Knobs on Right */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Mini Level Canvas */}
            <div className="lg:col-span-5 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white font-mono flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" /> Live Target Boundary
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  {gameTelemetry.predictions.filter((p) => p.predClass === p.y).length} /{' '}
                  {activeLevel.dataset.length} Solved
                </span>
              </div>

              <div className="relative aspect-square w-full max-w-[340px] mx-auto rounded-xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950 flex items-center justify-center">
                <canvas
                  ref={gameCanvasRef}
                  width={340}
                  height={340}
                  className="w-full h-full block"
                />
              </div>

              <div className="text-center text-[11px] text-slate-400 font-mono">
                Observe the boundary reshape live as you adjust the weights below!
              </div>
            </div>

            {/* Right Column: Weight Adjustment Knobs */}
            <div className="lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-amber-400 font-mono uppercase tracking-wider flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5" /> Interactive Synaptic Knobs
                </span>
                <span className="text-xs text-slate-400 font-mono">Click [+] / [-] or Drag</span>
              </div>

              <div className="space-y-4 max-h-[380px] overflow-y-auto pr-1">
                {/* Input Layer Weights & Biases */}
                <div className="space-y-2.5">
                  <div className="text-[11px] font-mono text-slate-400 uppercase font-semibold">
                    Layer 1: Input → Hidden Synapses
                  </div>
                  {gameModel.W1.map((row, j) => (
                    <div key={`gw1-${j}`} className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-purple-300 font-bold">Neuron h{j + 1}</span>
                        {/* Bias b1[j] knob */}
                        <div className="flex items-center gap-1.5">
                          <span className="text-slate-400 text-[10px]">bias b₁[{j}]:</span>
                          <span className="text-white font-bold">{gameModel.b1[j].toFixed(2)}</span>
                          <button
                            onClick={() => handleGameWeightChange('b1', j, null, -0.2)}
                            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px]"
                          >
                            <Minus className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => handleGameWeightChange('b1', j, null, 0.2)}
                            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px]"
                          >
                            <Plus className="w-3 h-3" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                        {row.map((w, i) => (
                          <div key={`gw1-${j}-${i}`} className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 flex items-center justify-between">
                            <span className="text-slate-400 text-[10px]">W₁[{j}][{i}]:</span>
                            <span className="font-bold text-cyan-300">{w.toFixed(2)}</span>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => handleGameWeightChange('W1', j, i, -0.2)}
                                className="p-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                              >
                                <Minus className="w-3 h-3" />
                              </button>
                              <button
                                onClick={() => handleGameWeightChange('W1', j, i, 0.2)}
                                className="p-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                              >
                                <Plus className="w-3 h-3" />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Output Layer Weights & Bias */}
                <div className="space-y-2.5 pt-2">
                  <div className="text-[11px] font-mono text-slate-400 uppercase font-semibold">
                    Layer 2: Hidden → Output Synapses
                  </div>
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-rose-300 font-bold">Output Neuron ŷ</span>
                      {/* Output bias b2 knob */}
                      <div className="flex items-center gap-1.5">
                        <span className="text-slate-400 text-[10px]">bias b₂:</span>
                        <span className="text-white font-bold">{gameModel.b2.toFixed(2)}</span>
                        <button
                          onClick={() => handleGameWeightChange('b2', null, null, -0.2)}
                          className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px]"
                        >
                          <Minus className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => handleGameWeightChange('b2', null, null, 0.2)}
                          className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px]"
                        >
                          <Plus className="w-3 h-3" />
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      {gameModel.W2.map((w, j) => (
                        <div key={`gw2-${j}`} className="bg-slate-900/80 p-2 rounded-lg border border-slate-800 flex items-center justify-between">
                          <span className="text-slate-400 text-[10px]">W₂[{j}]:</span>
                          <span className="font-bold text-amber-300">{w.toFixed(2)}</span>
                          <div className="flex items-center gap-1">
                            <button
                              onClick={() => handleGameWeightChange('W2', j, null, -0.2)}
                              className="p-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                            >
                              <Minus className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => handleGameWeightChange('W2', j, null, 0.2)}
                              className="p-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                            >
                              <Plus className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 4: MATHEMATICAL FORMULAS & GLASS BOX GUIDE */}
      {/* ============================================================== */}
      {activeTab === 'FORMULAS' && (
        <div className="space-y-6 text-slate-300">
          {/* Section 1: What Can This Simple NN Do? */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <Compass className="w-5 h-5 text-cyan-400" />
              <h3 className="text-base font-bold text-white font-mono">
                1. What Can This Simple Neural Network Do?
              </h3>
            </div>
            <p className="text-xs leading-relaxed">
              A single artificial neuron is strictly limited to drawing a single flat linear hyperplane:{' '}
              <span className="font-mono text-cyan-300">w₁x₁ + w₂x₂ + b = 0</span>. As proven by Marvin Minsky
              and Seymour Papert in their historic 1969 book <em>Perceptrons</em>, no single hyperplane can
              separate the four vertices of an XOR gate.
            </p>
            <p className="text-xs leading-relaxed">
              However, inserting just <strong>one hidden layer</strong> of non-linear neurons completely breaks
              this barrier:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs pt-2">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold text-[11px]">1. Multi-Slicing</span>
                <p className="text-slate-400 text-[11px]">
                  Each hidden neuron acts as a distinct fence dividing the 2D plane into half-spaces.
                </p>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                <span className="text-purple-400 font-bold text-[11px]">2. Convex Polytopes</span>
                <p className="text-slate-400 text-[11px]">
                  Combining 3 or 4 hidden neurons can enclose a closed geometric polygon, capturing concentric circles.
                </p>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
                <span className="text-emerald-400 font-bold text-[11px]">3. Continuous Smoothness</span>
                <p className="text-slate-400 text-[11px]">
                  Non-linear sigmoids soften hard boundaries into differentiable probability distributions.
                </p>
              </div>
            </div>
          </div>

          {/* Section 2: Universal Approximation Theorem */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-400" />
              <h3 className="text-base font-bold text-white font-mono">
                2. The Universal Approximation Theorem (Cybenko 1989 & Hornik 1991)
              </h3>
            </div>
            <p className="text-xs leading-relaxed">
              George Cybenko (1989) for sigmoid activations and Kurt Hornik (1991) for general non-polynomial
              activations proved one of the foundational theorems of deep learning:
            </p>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-amber-200">
              <strong>Theorem:</strong> Let σ(·) be any continuous, non-constant, bounded activation function.
              Then for any continuous function f: [0, 1]ⁿ → ℝ and any error threshold ε &gt; 0, there exists a
              finite integer H &gt; 0 and real parameters αⱼ, wⱼ, bⱼ such that:
              <div className="text-center py-2 text-white font-bold text-sm">
                | f(x) - Σⱼ₌₁ᴴ αⱼ · σ(wⱼᵀ x + bⱼ) | &lt; ε
              </div>
              for all x ∈ [0, 1]ⁿ.
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              <strong>Geometric Intuition:</strong> By pairing two sigmoids facing opposite directions with
              different offsets, the network creates a localized "bump" or ridge wavelet. Summing sufficiently
              many scaled bumps allows the network to carve any continuous landscape imaginable!
            </p>
          </div>

          {/* Section 3: Space Warping */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-purple-400" />
              <h3 className="text-base font-bold text-white font-mono">
                3. Space Warping: How Hidden Layers Untangle Entangled Data
              </h3>
            </div>
            <p className="text-xs leading-relaxed">
              Instead of thinking about a neural network as drawing curved boundaries in the input space, it is
              often far more profound to think in reverse: the neural network{' '}
              <strong>deforms the coordinate space itself</strong> until the problem becomes linearly separable!
            </p>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2 font-mono text-xs">
              <div className="text-purple-300 font-bold">The XOR Space Warping Mapping:</div>
              <p className="text-slate-400 text-[11px]">
                In the original input space (x₁, x₂), the points (0,0) and (1,1) are entangled with (0,1) and
                (1,0). But passing each point through the two hidden neurons maps each point into a new hidden
                activation coordinate system (a₁, a₂):
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center pt-1 text-[11px]">
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-400">(0, 0)</span> →{' '}
                  <span className="text-cyan-400 font-bold">(0, 0)</span>
                </div>
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-400">(1, 1)</span> →{' '}
                  <span className="text-cyan-400 font-bold">(1, 1)</span>
                </div>
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-400">(0, 1)</span> →{' '}
                  <span className="text-rose-400 font-bold">(0, 1)</span>
                </div>
                <div className="bg-slate-900 p-2 rounded border border-slate-800">
                  <span className="text-slate-400">(1, 0)</span> →{' '}
                  <span className="text-rose-400 font-bold">(1, 0)</span>
                </div>
              </div>
              <p className="text-slate-400 text-[11px] pt-1">
                In this new (a₁, a₂) feature space, the output layer neuron simply draws a single straight line
                cutting Class 0 from Class 1!
              </p>
            </div>
          </div>

          {/* Section 4: Why Depth Matters vs Width */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-bold text-white font-mono">
                4. Why Depth Matters: The Curse of Shallow Width
              </h3>
            </div>
            <p className="text-xs leading-relaxed">
              If a single hidden layer can approximate any function by the Universal Approximation Theorem, why
              do modern architectures use hundreds of layers instead of one colossal layer?
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5">
                <span className="text-rose-400 font-bold text-[11px] uppercase">
                  Shallow Network (Width O(2ⁿ))
                </span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Approximating high-dimensional parity, composition, or oscillations with only 1 hidden layer
                  requires an <strong>exponential number of neurons</strong>. The network essentially memorizes a
                  giant lookup table of localized bumps without compositional generalization.
                </p>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5">
                <span className="text-emerald-400 font-bold text-[11px] uppercase">
                  Deep Network (Depth O(n))
                </span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Deep architectures reuse intermediate abstractions hierarchically. Each additional layer can
                  fold space exponentially: a network with L layers can partition space into up to{' '}
                  <span className="text-emerald-300 font-bold">2ᴸ linear regions</span> with only linear parameter
                  growth!
                </p>
              </div>
            </div>
          </div>

          {/* Section 5: Scalar vs Matrix Equations Rosetta Stone */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2">
              <Binary className="w-5 h-5 text-emerald-400" />
              <h3 className="text-base font-bold text-white font-mono">
                5. Scalar Arithmetic vs Vectorized Matrix Notation (Rosetta Stone)
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                    <th className="py-2.5 px-3">Operation Stage</th>
                    <th className="py-2.5 px-3 text-cyan-300">Scalar Notation (Hand Engine)</th>
                    <th className="py-2.5 px-3 text-purple-300">Vectorized Matrix (NumPy / PyTorch)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850 text-[11px]">
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Hidden Affine</td>
                    <td className="py-2.5 px-3 text-slate-400">z₁[j] = Σᵢ W₁[j][i] · x[i] + b₁[j]</td>
                    <td className="py-2.5 px-3 text-slate-200">z₁ = W₁ · x + b₁</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Hidden Activation</td>
                    <td className="py-2.5 px-3 text-slate-400">a₁[j] = 1 / (1 + e^(-z₁[j]))</td>
                    <td className="py-2.5 px-3 text-slate-200">a₁ = σ(z₁)</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Output Affine</td>
                    <td className="py-2.5 px-3 text-slate-400">z₂ = Σⱼ W₂[j] · a₁[j] + b₂</td>
                    <td className="py-2.5 px-3 text-slate-200">z₂ = W₂ · a₁ + b₂</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Output Error Signal</td>
                    <td className="py-2.5 px-3 text-slate-400">δ₂ = ŷ - y</td>
                    <td className="py-2.5 px-3 text-slate-200">δ₂ = ŷ - y</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Output Weight Grad</td>
                    <td className="py-2.5 px-3 text-slate-400">∂L/∂W₂[j] = δ₂ · a₁[j]</td>
                    <td className="py-2.5 px-3 text-slate-200">∇_W₂ L = δ₂ · a₁ᵀ</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Hidden Backpressure</td>
                    <td className="py-2.5 px-3 text-slate-400">δ₁[j] = (δ₂ · W₂[j]) · a₁[j](1 - a₁[j])</td>
                    <td className="py-2.5 px-3 text-slate-200">δ₁ = (W₂ᵀ · δ₂) ⊙ σ'(z₁)</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">Input Weight Grad</td>
                    <td className="py-2.5 px-3 text-slate-400">∂L/∂W₁[j][i] = δ₁[j] · x[i]</td>
                    <td className="py-2.5 px-3 text-slate-200">∇_W₁ L = δ₁ · xᵀ</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
