import React, { useState } from 'react';
import { Clock, BookOpen, AlertTriangle, Sparkles, CheckCircle2, ChevronRight, Award } from 'lucide-react';

const milestones = [
  {
    year: "1943",
    title: "The McCulloch-Pitts Neuron",
    subtitle: "The First Mathematical Model of Brain Activity",
    authors: "Warren McCulloch & Walter Pitts",
    paper: "A Logical Calculus of the Ideas Immanent in Nervous Activity",
    analogy: "Biological wire with an emergency brake. Excitatory inputs add up; if any inhibitory wire triggers, it halts.",
    breakthrough: "Proved that networks of simplified binary neurons could calculate any logical expression (AND, OR, NOT) that a Turing machine can compute.",
    crisis: "Weights were static and hand-wired by humans. The neuron had no mechanism to LEARN from errors.",
    tag: "Genesis",
    color: "from-blue-500 to-cyan-500"
  },
  {
    year: "1951 - 1964",
    title: "SGD & Polyak Heavy-Ball Momentum",
    subtitle: "Navigating Noisy & Anisotropic Loss Landscapes",
    authors: "Herbert Robbins & Sutton Monro (1951) • Boris T. Polyak (1964)",
    paper: "A Stochastic Approximation Method (1951) • Some methods of speeding up the convergence of iteration methods (1964)",
    analogy: "Skiing down an ill-conditioned mountain ravine. Vanilla SGD violently bounces off canyon walls; Polyak heavy-ball momentum builds forward inertia to glide along the ravine floor.",
    breakthrough: "Robbins & Monro established convergence of SGD with noisy expectations. Polyak added kinetic mass/momentum to cancel orthogonal oscillations and accelerate convergence on ill-conditioned quadratics.",
    crisis: "Fixed learning rates struggled when landscape curvature varied wildly across different parameter axes.",
    tag: "Optimization Origins",
    color: "from-amber-500 to-red-500"
  },
  {
    year: "1957",
    title: "Rosenblatt's Perceptron",
    subtitle: "The First Machine That Could Learn From Experience",
    authors: "Frank Rosenblatt (Cornell Aeronautical Lab)",
    paper: "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain",
    analogy: "An archer adjusting bow tension. If an arrow shoots too high, dial down the weight; if too low, dial it up.",
    breakthrough: "Invented the Perceptron Learning Rule. In 1960, the U.S. Navy demonstrated the Mark I Perceptron machine with 400 photocells and motor-driven potentiometers.",
    crisis: "The media overhyped it as 'the embryo of an electronic computer that will walk, talk, and see'. This extreme hype set the stage for an equally extreme crash.",
    tag: "First Learning Rule",
    color: "from-emerald-500 to-teal-500"
  },
  {
    year: "1969",
    title: "The Minsky & Papert XOR Barrier",
    subtitle: "The Crash and The First AI Winter",
    authors: "Marvin Minsky & Seymour Papert (MIT)",
    paper: "Perceptrons: An Introduction to Computational Geometry",
    analogy: "Trying to separate a checkerboard into black and white using only a single straight ruler. It's geometrically impossible.",
    breakthrough: "Rigorous mathematical proof that a single-layer perceptron cannot solve non-linearly separable problems like Exclusive-OR (XOR).",
    crisis: "DARPA pulled funding. Research ground to a near-total halt for over a decade. The 'First AI Winter' began.",
    tag: "The AI Winter",
    color: "from-rose-500 to-amber-500"
  },
  {
    year: "1970 - 1974",
    title: "Linnainmaa & Werbos: Reverse AD",
    subtitle: "The Discovery of General Reverse-Mode Automatic Differentiation",
    authors: "Seppo Linnainmaa (1970) • Paul J. Werbos (1974)",
    paper: "The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors (1970) • Beyond Regression (1974)",
    analogy: "A water pipe network where pressure pulse is injected at the single outlet reservoir (dL/dL = 1.0) and propagates backward, splitting and adding at pipe junctions.",
    breakthrough: "Linnainmaa invented O(V + E) reverse-mode automatic differentiation on arbitrary computational DAGs. Werbos first proposed applying reverse AD to train multi-stage neural networks.",
    crisis: "The AI Winter was in full freeze; mainstream computing ignored neural networks and automatic differentiation for over a decade.",
    tag: "Autodiff Genesis",
    color: "from-cyan-500 to-indigo-500"
  },
  {
    year: "1986",
    title: "Rumelhart, Hinton & Backpropagation",
    subtitle: "The Renaissance of Multilayer Networks",
    authors: "David Rumelhart, Geoffrey Hinton & Ronald Williams",
    paper: "Learning representations by back-propagating errors (Nature)",
    analogy: "A bucket brigade passing water down a line, but when water spills, blame is calculated precisely backward to adjust each person's grip.",
    breakthrough: "Showed that hidden layers with smooth non-linear activations (Sigmoid) can be trained efficiently using the calculus chain rule.",
    crisis: "Deep networks suffered from the Vanishing Gradient problem: gradients shrank exponentially toward zero as they propagated back through multiple layers.",
    tag: "The Renaissance",
    color: "from-purple-500 to-pink-500"
  },
  {
    year: "2014 - 2017",
    title: "Kingma, Ba & Loshchilov: Adam to AdamW",
    subtitle: "Adaptive Moments & Decoupled Weight Decay",
    authors: "Diederik Kingma & Jimmy Ba (2014) • Ilya Loshchilov & Frank Hutter (2017)",
    paper: "Adam: A Method for Stochastic Optimization (ICLR 2015) • Decoupled Weight Decay Regularization (ICLR 2019)",
    analogy: "Skier wearing smart crampons that automatically adjust friction independently along x and y, plus a gentle decoupled bungee tether (AdamW) preventing weight explosion.",
    breakthrough: "Adam combined 1st-moment momentum and 2nd-moment RMSprop with bias correction. AdamW fixed Adam's fundamental flaw with L2 regularization by decoupling weight decay from adaptive gradient scaling, creating the universal default optimizer for modern Transformers.",
    crisis: "High memory footprint storing two full state tensors (m and v) per parameter, prompting memory-efficient optimizers (8-bit Adam, Adafactor).",
    tag: "SOTA Optimization",
    color: "from-emerald-400 to-cyan-500"
  },
  {
    year: "2012",
    title: "AlexNet & The GPU Revolution",
    subtitle: "Deep Learning Conquers Computer Vision",
    authors: "Alex Krizhevsky, Ilya Sutskever & Geoffrey Hinton",
    paper: "ImageNet Classification with Deep Convolutional Neural Networks",
    analogy: "Parallel processing engines scanning millions of images at once, learning hierarchically from edges to textures to dog faces.",
    breakthrough: "Crushed the ImageNet challenge with 8-layer CNN trained on dual NVIDIA GTX 580 GPUs, popularizing ReLU and Dropout.",
    crisis: "Recurrent networks (RNNs/LSTMs) for language were strictly sequential and could not scale across massive internet corpora.",
    tag: "Deep Era",
    color: "from-cyan-500 to-blue-600"
  },
  {
    year: "2017",
    title: "The Transformer & Attention",
    subtitle: "Attention Is All You Need",
    authors: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin (Google Brain)",
    paper: "Attention Is All You Need",
    analogy: "A room full of diplomats where everyone can instantly look up anyone else's badge via Query, Key, and Value vectors.",
    breakthrough: "Replaced sequential recurrence with parallel self-attention, unlocking today's Foundation Models (LLaMA, GPT-4, Gemini).",
    crisis: "Quadratic memory complexity O(N^2) with sequence length, hallucination, and lack of internal reasoning verification.",
    tag: "Modern LLM Era",
    color: "from-amber-500 to-orange-500"
  },
  {
    year: "2024-2026",
    title: "Reasoning & Mechanistic Interpretability",
    subtitle: "Test-Time Search, MoE & Internal Brain Scans",
    authors: "OpenAI, DeepSeek, Anthropic",
    paper: "DeepSeek-V3 / R1 & Sparse Autoencoders for Transformer Circuits",
    analogy: "Thinking before speaking: exploring multiple reasoning trajectories and pruning wrong steps, while peering into artificial neurons with MRI scans.",
    breakthrough: "Reinforcement learning for long chain-of-thought reasoning (GRPO), Mixture of Experts (MoE), and Sparse Autoencoders (SAEs) demystifying polysemantic neurons.",
    crisis: "Alignment guarantees, safety verification under deceptive alignment, and energy scaling limits.",
    tag: "SOTA Frontier",
    color: "from-fuchsia-500 to-indigo-500"
  }
];

export default function HistoryTimeline() {
  const [selectedIdx, setSelectedIdx] = useState(2); // Default to 1957 Perceptron
  const active = milestones[selectedIdx];

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-slate-700/60 rounded-2xl p-6 sm:p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-cyan-400 mb-2">
            <Clock className="w-4 h-4" />
            <span>THE 80-YEAR NEURAL ODYSSEY (1943 → 2026)</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
            From Biological Switches to SOTA Foundation Intelligence
          </h2>
          <p className="text-slate-300 text-xs sm:text-sm mt-2 leading-relaxed">
            Click through each historical milestone to understand the philosophical breakthroughs, the crushing crises, and the intuitive analogies behind eight decades of neural network research.
          </p>
        </div>
      </div>

      {/* Horizontal Interactive Timeline Scroller */}
      <div className="relative">
        <div className="flex items-center gap-3 overflow-x-auto pb-4 pt-1 scrollbar-thin scrollbar-thumb-slate-700">
          {milestones.map((m, idx) => {
            const isSelected = selectedIdx === idx;
            return (
              <button
                key={idx}
                onClick={() => setSelectedIdx(idx)}
                className={`flex-shrink-0 text-left p-3.5 rounded-xl border transition duration-200 w-48 ${
                  isSelected
                    ? 'bg-slate-800 border-cyan-500/80 shadow-lg shadow-cyan-500/10 scale-102 ring-1 ring-cyan-500/40'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-850'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-mono font-extrabold ${isSelected ? 'text-cyan-400' : 'text-slate-500'}`}>
                    {m.year}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-950 text-slate-400">
                    {m.tag}
                  </span>
                </div>
                <div className="text-xs font-bold text-slate-200 mt-1 truncate">
                  {m.title}
                </div>
                <div className="text-[11px] text-slate-400 truncate mt-0.5">
                  {m.authors.split(' & ')[0].split(' (')[0]}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Milestone Deep-Dive Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 shadow-2xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 pb-6 border-b border-slate-800">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-sm font-mono font-extrabold text-cyan-400 bg-cyan-950/60 px-2.5 py-0.5 rounded-md border border-cyan-800/60">
                {active.year}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {active.authors}
              </span>
            </div>
            <h3 className="text-2xl sm:text-3xl font-black text-white pt-1">
              {active.title}
            </h3>
            <p className="text-sm text-slate-300 font-medium">
              {active.subtitle}
            </p>
          </div>

          <div className="flex-shrink-0 bg-slate-950/80 border border-slate-800 p-3 rounded-xl text-xs font-mono max-w-sm">
            <div className="text-slate-500 flex items-center gap-1.5 mb-1 font-bold">
              <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
              <span>Seminal Publication:</span>
            </div>
            <div className="text-slate-300 italic">"{active.paper}"</div>
          </div>
        </div>

        {/* 3 Pillars: Analogy, Breakthrough, Crisis */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Analogy */}
          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2 text-cyan-400 font-bold text-xs uppercase tracking-wider font-mono">
              <Sparkles className="w-4 h-4" />
              <span>Physical Metaphor</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {active.analogy}
            </p>
          </div>

          {/* Breakthrough */}
          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase tracking-wider font-mono">
              <CheckCircle2 className="w-4 h-4" />
              <span>Core Breakthrough</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {active.breakthrough}
            </p>
          </div>

          {/* Crisis */}
          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-2">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-xs uppercase tracking-wider font-mono">
              <AlertTriangle className="w-4 h-4" />
              <span>Historical Crisis / Barrier</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {active.crisis}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
