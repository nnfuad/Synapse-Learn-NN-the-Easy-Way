import React, { useState } from 'react';
import {
  Brain,
  Cpu,
  GitBranch,
  Network,
  Zap,
  Sparkles,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Layers,
  ArrowRight,
  RotateCcw,
  Sliders,
  Scale,
  Compass,
} from 'lucide-react';

// Preset Reasoning Prompts & Steps for Test-Time Search Tree
const REASONING_TREE_DATA = {
  problem: 'Find all prime triplets (p, p+2, p+4) where p is a positive integer.',
  root: {
    id: 'root',
    text: 'Problem: Determine all prime triplets (p, p+2, p+4)',
    score: 1.0,
    type: 'root',
    children: [
      {
        id: 'branch_a',
        rolloutId: 1,
        title: 'Rollout 1: Brute-Force Testing (Dead End)',
        text: 'Step 1: Test small primes: p=2 -> (2,4,6) not prime. p=3 -> (3,5,7) works!',
        score: 0.7,
        children: [
          {
            id: 'branch_a_1',
            text: 'Step 2: Test p=5 -> (5,7,9); 9 is 3×3 (Failed). Test p=7 -> (7,9,11); 9 fails.',
            score: 0.3,
            children: [
              {
                id: 'branch_a_2',
                text: 'Step 3: Keep testing p=11, 13, 17... (Unbounded search, no analytical proof)',
                score: 0.1,
                status: 'PRUNED',
              },
            ],
          },
        ],
      },
      {
        id: 'branch_b',
        rolloutId: 2,
        title: 'Rollout 2: Modular Arithmetic Analysis (Winning CoT)',
        text: 'Step 1: Analyze numbers modulo 3. Any integer p can be written as 3k, 3k+1, or 3k+2.',
        score: 0.98,
        status: 'WINNER',
        children: [
          {
            id: 'branch_b_1',
            text: 'Case 1: If p = 3k+1, then p+2 = 3k+3 = 3(k+1) is divisible by 3 and > 3 (Not prime).',
            score: 1.0,
            status: 'WINNER',
          },
          {
            id: 'branch_b_2',
            text: 'Case 2: If p = 3k+2, then p+4 = 3k+6 = 3(k+2) is divisible by 3 and > 3 (Not prime).',
            score: 1.0,
            status: 'WINNER',
          },
          {
            id: 'branch_b_3',
            text: 'Case 3: If p = 3k, the only prime divisible by 3 is p = 3. Check: (3, 5, 7) are all prime!',
            score: 1.0,
            status: 'WINNER',
            children: [
              {
                id: 'branch_b_conclusion',
                text: 'Q.E.D. The only prime triplet is exactly (3, 5, 7). No other triplets exist!',
                score: 1.0,
                status: 'WINNER',
              },
            ],
          },
        ],
      },
      {
        id: 'branch_c',
        rolloutId: 3,
        title: 'Rollout 3: Hallucination & Self-Correction',
        text: 'Step 1: Assume Twin Prime conjecture implies infinite triplets.',
        score: 0.2,
        children: [
          {
            id: 'branch_c_1',
            text: 'Step 2: [Self-Correction Trigger] Wait! One of any 3 consecutive odd numbers must be divisible by 3!',
            score: 0.9,
            children: [
              {
                id: 'branch_c_2',
                text: 'Backtrack to modular residue proof.',
                score: 0.95,
                status: 'MERGED',
              },
            ],
          },
        ],
      },
      {
        id: 'branch_d',
        rolloutId: 4,
        title: 'Rollout 4: Arithmetic Progression Flaw',
        text: 'Step 1: Check difference d = 2. Try formula p_n = 6k ± 1.',
        score: 0.4,
        children: [
          {
            id: 'branch_d_1',
            text: 'Step 2: 6k+1 + 2 = 6k+3 = 3(2k+1) is composite (Blocked).',
            score: 0.3,
            status: 'PRUNED',
          },
        ],
      },
    ],
  },
};

// Preset Tokens for MoE Routing Demonstration
const SAMPLE_TOKENS = [
  { id: 0, text: 'The', domain: 'syntax' },
  { id: 1, text: 'Riemann', domain: 'math_advanced' },
  { id: 2, text: 'zeta', domain: 'math_advanced' },
  { id: 3, text: 'function', domain: 'code_math' },
  { id: 4, text: 'satisfies', domain: 'syntax' },
  { id: 5, text: 'analytic', domain: 'math_advanced' },
  { id: 6, text: 'continuation', domain: 'math_advanced' },
  { id: 7, text: 'via', domain: 'syntax' },
  { id: 8, text: 'Poisson', domain: 'physics_math' },
  { id: 9, text: 'summation', domain: 'math_advanced' },
];

const EXPERTS = [
  { id: 0, name: 'Expert 0: Pure Math & Topology', color: 'from-blue-500 to-indigo-600', domain: 'math_advanced' },
  { id: 1, name: 'Expert 1: Code & Algorithms', color: 'from-emerald-500 to-teal-600', domain: 'code_math' },
  { id: 2, name: 'Expert 2: Grammar & Discourse', color: 'from-purple-500 to-pink-600', domain: 'syntax' },
  { id: 3, name: 'Expert 3: Physics & Equations', color: 'from-amber-500 to-orange-600', domain: 'physics_math' },
  { id: 4, name: 'Expert 4: Logic & Deductions', color: 'from-rose-500 to-red-600', domain: 'math_advanced' },
  { id: 5, name: 'Expert 5: Common Sense Semantics', color: 'from-cyan-500 to-blue-600', domain: 'syntax' },
];

export default function Module08Visualizer() {
  const [activeSubTab, setActiveSubTab] = useState('MOE'); // 'MOE' or 'GRPO_REASONING'

  // MoE States
  const [routerMode, setRouterMode] = useState('BALANCED'); // 'BALANCED' or 'COLLAPSED'
  const [selectedTokenIdx, setSelectedTokenIdx] = useState(1); // 'Riemann'
  const [topK, setTopK] = useState(2);
  const [enableSharedExpert, setEnableSharedExpert] = useState(true);

  // GRPO States
  const [groupRewards, setGroupRewards] = useState([0.3, 1.0, 0.7, 0.2]); // 4 candidate rollouts
  const [selectedNode, setSelectedNode] = useState(REASONING_TREE_DATA.root.children[1]);

  // Compute MoE Dispatch for current token
  const token = SAMPLE_TOKENS[selectedTokenIdx];
  let routedExpertIds = [];
  let expertProbabilities = [];

  if (routerMode === 'COLLAPSED') {
    // Router Collapse: All tokens disproportionately route to Expert 0 and 1
    routedExpertIds = [0, 1];
    expertProbabilities = [0.65, 0.32, 0.01, 0.01, 0.005, 0.005];
  } else {
    // Balanced Routing: Specialization by domain
    if (token.domain === 'math_advanced') {
      routedExpertIds = [0, 4];
      expertProbabilities = [0.52, 0.08, 0.05, 0.12, 0.20, 0.03];
    } else if (token.domain === 'code_math') {
      routedExpertIds = [1, 0];
      expertProbabilities = [0.18, 0.58, 0.04, 0.08, 0.09, 0.03];
    } else if (token.domain === 'physics_math') {
      routedExpertIds = [3, 0];
      expertProbabilities = [0.15, 0.05, 0.04, 0.62, 0.10, 0.04];
    } else {
      routedExpertIds = [2, 5];
      expertProbabilities = [0.06, 0.04, 0.48, 0.05, 0.05, 0.32];
    }
  }

  // Calculate Load Balancing Loss metric
  const meanProb = 1 / EXPERTS.length;
  const varianceFromUniform = expertProbabilities.reduce((acc, p) => acc + (p - meanProb) ** 2, 0);
  const auxLossScore = routerMode === 'COLLAPSED' ? 0.095 : 0.018;

  // GRPO Relative Advantage Calculations
  const meanReward = groupRewards.reduce((a, b) => a + b, 0) / groupRewards.length;
  const stdReward = Math.sqrt(
    groupRewards.reduce((acc, r) => acc + (r - meanReward) ** 2, 0) / groupRewards.length
  ) || 1e-4;
  const groupAdvantages = groupRewards.map((r) => (r - meanReward) / stdReward);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900/90 to-purple-950/40 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-purple-400 font-mono text-xs font-bold tracking-wider uppercase">
              <Sparkles className="w-4 h-4" />
              <span>Frontier Stage • Module 08</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white mt-1">
              LLMs, Alignment & Reasoning Engines
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Inspect the dual pillars of modern LLM supremacy: <strong>Sparse MoE conditional computation</strong> (DeepSeek-V3 / Mixtral)
              and <strong>Critic-Free GRPO Test-Time Reasoning</strong> (DeepSeek-R1 / OpenAI o1).
            </p>
          </div>

          {/* Sub-Tab Selector */}
          <div className="bg-slate-950 p-1.5 rounded-xl border border-slate-800 flex items-center gap-1 self-start md:self-auto">
            <button
              onClick={() => setActiveSubTab('MOE')}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-2 ${
                activeSubTab === 'MOE'
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-lg shadow-purple-500/10'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Cpu className="w-4 h-4 text-purple-400" />
              <span>Sparse MoE Router</span>
            </button>

            <button
              onClick={() => setActiveSubTab('GRPO_REASONING')}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-2 ${
                activeSubTab === 'GRPO_REASONING'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-lg shadow-emerald-500/10'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <GitBranch className="w-4 h-4 text-emerald-400" />
              <span>GRPO & Test-Time Search</span>
            </button>
          </div>
        </div>
      </div>

      {/* VIEW 1: SPARSE MIXTURE OF EXPERTS (MoE) */}
      {activeSubTab === 'MOE' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Interactive Routing Blueprint (7 Cols) */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Network className="w-4 h-4 text-purple-400" />
                <span>Live MoE Token Dispatcher</span>
              </div>

              {/* Mode Toggle: Balanced vs Collapsed */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-400 font-mono">Router State:</span>
                <button
                  onClick={() => setRouterMode(routerMode === 'BALANCED' ? 'COLLAPSED' : 'BALANCED')}
                  className={`px-3 py-1 rounded-lg font-bold border transition flex items-center gap-1.5 ${
                    routerMode === 'BALANCED'
                      ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                      : 'bg-rose-500/20 border-rose-500/50 text-rose-300 animate-pulse'
                  }`}
                >
                  {routerMode === 'BALANCED' ? (
                    <>
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>Balanced (Aux Loss Active)</span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-3.5 h-3.5" />
                      <span>Collapsed (Zero Aux Loss)</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Token Stream Strip */}
            <div className="space-y-2">
              <span className="text-xs text-slate-400 font-mono block">Select Token in Context Sequence:</span>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_TOKENS.map((t, idx) => (
                  <button
                    key={t.id}
                    onClick={() => setSelectedTokenIdx(idx)}
                    className={`px-3 py-1.5 rounded-xl font-mono text-xs font-semibold transition border ${
                      selectedTokenIdx === idx
                        ? 'bg-purple-600 text-white border-purple-400 shadow-md shadow-purple-600/30'
                        : 'bg-slate-950 text-slate-300 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    "{t.text}"
                  </button>
                ))}
              </div>
            </div>

            {/* Visual Dispatch Diagram */}
            <div className="bg-[#080d1a] border border-slate-800 rounded-xl p-5 relative overflow-hidden">
              <div className="flex flex-col items-center gap-6">
                {/* 1. Incoming Token */}
                <div className="flex flex-col items-center">
                  <div className="px-5 py-2 rounded-xl bg-purple-500/20 border border-purple-500/50 text-purple-200 font-mono font-bold text-sm shadow-lg shadow-purple-500/20">
                    Token: "{token.text}" <span className="text-[10px] text-purple-400 font-normal">({token.domain})</span>
                  </div>
                  <div className="w-0.5 h-6 bg-slate-700" />
                </div>

                {/* 2. Top-K Router Gate */}
                <div className="w-full max-w-md bg-slate-900 border border-purple-500/30 rounded-xl p-3 text-center shadow-lg">
                  <div className="text-xs font-bold text-white flex items-center justify-center gap-1.5">
                    <Scale className="w-4 h-4 text-purple-400" />
                    <span>Top-{topK} Gating Network W_gate (Softmax Normalized)</span>
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono mt-1">
                    Selects Top {topK} of {EXPERTS.length} Routed Experts
                  </div>
                </div>

                {/* 3. Dispatched Experts Grid */}
                <div className="w-full grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {EXPERTS.map((exp) => {
                    const isRouted = routedExpertIds.includes(exp.id);
                    const prob = expertProbabilities[exp.id];

                    return (
                      <div
                        key={exp.id}
                        className={`p-3 rounded-xl border transition relative ${
                          isRouted
                            ? 'bg-purple-950/40 border-purple-500 text-purple-200 shadow-md shadow-purple-500/20'
                            : 'bg-slate-950/60 border-slate-800/80 text-slate-500 opacity-60'
                        }`}
                      >
                        {isRouted && (
                          <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-purple-400 animate-ping" />
                        )}
                        <div className="text-xs font-bold truncate text-slate-200">{exp.name.split(':')[0]}</div>
                        <div className="text-[10px] text-slate-400 truncate mt-0.5">{exp.name.split(':')[1]}</div>
                        <div className="mt-2 flex items-center justify-between text-[11px] font-mono">
                          <span className="text-slate-400">Gate Prob:</span>
                          <span className={isRouted ? 'font-bold text-purple-300' : 'text-slate-500'}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* 4. DeepSeek-Style Shared Isolated Expert */}
                {enableSharedExpert && (
                  <div className="w-full border-t border-dashed border-slate-800 pt-4 flex flex-col items-center">
                    <div className="w-full bg-gradient-to-r from-cyan-950/40 via-blue-950/40 to-cyan-950/40 border border-cyan-500/40 rounded-xl p-3 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-cyan-400" />
                        <div>
                          <div className="text-xs font-bold text-cyan-200">DeepSeek-Style Shared Expert</div>
                          <div className="text-[10px] text-slate-400 font-mono">Always active (100% tokens) for common knowledge</div>
                        </div>
                      </div>
                      <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                        Always Active
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Auxiliary Loss & Expert Utilization (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            {/* Aux Loss Gauge */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-bold text-sm">
                  <Scale className="w-4 h-4 text-purple-400" />
                  <span>Auxiliary Load Balancing Loss</span>
                </div>
                <span className="text-xs font-mono font-bold text-purple-400">
                  L_balance = {auxLossScore.toFixed(4)}
                </span>
              </div>

              {/* Progress meter */}
              <div className="space-y-1">
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full transition-all duration-500 ${
                      routerMode === 'BALANCED' ? 'bg-emerald-500 w-[20%]' : 'bg-rose-500 w-[85%]'
                    }`}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>Balanced (Low Loss)</span>
                  <span>Collapsed (Critical Loss)</span>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed">
                Formula: <code className="text-purple-300 font-mono">α · N · ∑(f_i · P_i)</code>.
                In a collapsed router, tokens overwhelm 1–2 experts while leaving others idle.
                The auxiliary loss forces the router to spread tokens evenly across all experts.
              </p>
            </div>

            {/* Per-Expert Utilization Bar Chart */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Brain className="w-4 h-4 text-cyan-400" />
                <span>Expert Utilization Distribution</span>
              </div>

              <div className="space-y-2 pt-1">
                {EXPERTS.map((exp) => {
                  const prob = expertProbabilities[exp.id];
                  return (
                    <div key={exp.id} className="space-y-1">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="text-slate-300 truncate max-w-[180px]">{exp.name.split(':')[0]}</span>
                        <span className="text-slate-400 font-bold">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            prob > 0.4 ? 'bg-rose-500' : 'bg-purple-500'
                          }`}
                          style={{ width: `${Math.min(100, prob * 120)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Architecture Fact Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2 text-xs text-slate-300">
              <span className="text-amber-400 font-bold flex items-center gap-1.5">
                <Sparkles className="w-4 h-4" />
                <span>DeepSeek-V3 MoE Efficiency Secret</span>
              </span>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                DeepSeek-V3 uses <strong>256 fine-grained routed experts</strong> (activating 8) + 1 shared expert.
                This achieves 671B total parameters while only activating 37B per token, drastically cutting inference cost!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: GRPO & TEST-TIME REASONING SEARCH */}
      {activeSubTab === 'GRPO_REASONING' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Interactive Reasoning Search Tree (7 Cols) */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <GitBranch className="w-4 h-4 text-emerald-400" />
                <span>Test-Time Compute Rollout Search Tree</span>
              </div>
              <span className="text-xs font-mono text-emerald-400 font-bold">
                MCTS / Best-of-N Candidate Paths
              </span>
            </div>

            {/* Problem Statement Card */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-300">
              <span className="text-slate-500 block uppercase text-[10px]">Active Problem:</span>
              <span className="text-white font-bold">{REASONING_TREE_DATA.problem}</span>
            </div>

            {/* Interactive Candidate Rollouts */}
            <div className="space-y-3">
              <span className="text-xs text-slate-400 font-mono block">Sampled Rollout Group (G = 4):</span>
              <div className="space-y-2">
                {REASONING_TREE_DATA.root.children.map((branch, idx) => {
                  const isSelected = selectedNode?.id === branch.id;
                  const reward = groupRewards[idx];
                  const advantage = groupAdvantages[idx];

                  return (
                    <div
                      key={branch.id}
                      onClick={() => setSelectedNode(branch)}
                      className={`p-3 rounded-xl border cursor-pointer transition ${
                        isSelected
                          ? 'bg-emerald-950/40 border-emerald-500 text-emerald-200 shadow-md shadow-emerald-500/10'
                          : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2 font-bold text-slate-200">
                          {branch.status === 'WINNER' ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          ) : (
                            <XCircle className="w-4 h-4 text-rose-400" />
                          )}
                          <span>{branch.title}</span>
                        </div>
                        <div className="flex items-center gap-2 font-mono text-[11px]">
                          <span className="text-slate-500">Reward:</span>
                          <span className="font-bold text-white">{reward.toFixed(2)}</span>
                          <span className="text-slate-500">Advantage:</span>
                          <span className={advantage >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                            {advantage > 0 ? `+${advantage.toFixed(2)}` : advantage.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      <p className="text-[11px] text-slate-400 mt-2 font-mono">{branch.text}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Tree Branch Inspector */}
            {selectedNode && (
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-2">
                <span className="text-[11px] uppercase font-mono text-cyan-400 font-bold block">
                  Detailed Reasoning Trace:
                </span>
                <div className="space-y-1.5 text-xs text-slate-300">
                  {selectedNode.children?.map((child, i) => (
                    <div key={i} className="flex items-start gap-2 pl-2 border-l-2 border-emerald-500/50 py-1">
                      <ArrowRight className="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0" />
                      <span className="font-mono text-[11px]">{child.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right: GRPO Interactive Math & Reward Sliders (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            {/* Interactive Reward Sliders for Group */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-white font-bold text-sm">
                  <Sliders className="w-4 h-4 text-emerald-400" />
                  <span>Group Reward Verifier (G = 4)</span>
                </div>
              </div>

              <div className="space-y-3 pt-1">
                {groupRewards.map((rew, i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-slate-300">Rollout {i + 1} Reward:</span>
                      <span className="text-emerald-400 font-bold">{rew.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={rew}
                      onChange={(e) => {
                        const newR = [...groupRewards];
                        newR[i] = parseFloat(e.target.value);
                        setGroupRewards(newR);
                      }}
                      className="w-full accent-emerald-400 bg-slate-800 rounded-lg cursor-pointer h-1.5"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Normalized Advantages & Gradient Impact */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2 text-white font-bold text-sm">
                <Scale className="w-4 h-4 text-cyan-400" />
                <span>Group Relative Advantages A_i</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                {groupAdvantages.map((adv, i) => (
                  <div key={i} className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">A_{i + 1}</span>
                    <span className={`font-bold text-sm ${adv >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {adv > 0 ? `+${adv.toFixed(3)}` : adv.toFixed(3)}
                    </span>
                    <span className="text-[10px] text-slate-500 block mt-0.5">
                      {adv >= 0 ? 'Policy Boosted ↑' : 'Policy Penalized ↓'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Pedagogical Explanation: Why GRPO eliminated the Critic */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2 text-xs text-slate-300">
              <div className="flex items-center gap-2 text-amber-400 font-bold">
                <Zap className="w-4 h-4" />
                <span>Why Did DeepSeek-R1 Ditch the Critic?</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Standard PPO requires a <strong>Value Model (Critic)</strong> that is the exact same size as the Policy LLM,
                doubling memory consumption and suffering from catastrophic value drift on complex reasoning.
                GRPO replaces the critic by computing relative advantages across a group of rollouts!
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
