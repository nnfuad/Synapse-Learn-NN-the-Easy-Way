import React, { useState } from 'react';
import { HelpCircle, CheckCircle2, XCircle, Award, Sparkles, RefreshCw } from 'lucide-react';
import confetti from 'canvas-confetti';

const MODULE_QUESTIONS = {
  '00': [
    {
      id: 1,
      question: "Why could Rosenblatt's 1957 Perceptron NOT use standard calculus gradient descent to train its weights?",
      options: [
        { text: "Hardware in 1957 was too slow to calculate derivatives.", correct: false },
        { text: "The Heaviside step activation function has a derivative of zero almost everywhere (and undefined at 0), meaning gradients cannot flow.", correct: true },
        { text: "Matrix multiplication had not yet been discovered.", correct: false },
        { text: "It had too many layers for calculus to work.", correct: false },
      ],
      explanation: "Spot on! The step function is completely flat (slope = 0) everywhere except at x = 0 where it jumps discontinuously. Since d/dx is 0, gradient descent gets zero signal! That is why the field had to wait until 1986 for smooth, differentiable activations like Sigmoid and ReLU."
    },
    {
      id: 2,
      question: "What is the theoretical maximum classification accuracy that ANY single straight line can achieve on the 4 XOR points in 2D space?",
      options: [
        { text: "25% (1 out of 4)", correct: false },
        { text: "50% (2 out of 4)", correct: false },
        { text: "75% (3 out of 4)", correct: true },
        { text: "100% (4 out of 4)", correct: false },
      ],
      explanation: "Exactly! Because the two classes form alternating diagonal pairs (like a checkerboard), any straight cut can at most isolate 3 points correctly while misclassifying the 4th point. 3/4 = 75% accuracy."
    },
    {
      id: 3,
      question: "Geometrically, what is the fundamental superpower that hidden layers with non-linear activations provide to a neural network?",
      options: [
        { text: "They make the learning rate 10x faster.", correct: false },
        { text: "They warp, stretch, and fold the coordinate space so tangled data points become linearly separable in the hidden representation.", correct: true },
        { text: "They round decimal weights to clean binary integers.", correct: false },
        { text: "They act as a hardware firewall preventing overfitting.", correct: false },
      ],
      explanation: "Brilliant! Hidden layers perform a continuous non-linear topological warp on space. Points that were trapped diagonally apart in input space are pushed onto the same side or unfolded in hidden feature space!"
    }
  ],

  '01': [
    {
      id: 101,
      question: "Why did Seppo Linnainmaa's 1970 reverse-mode automatic differentiation make deep learning with millions of weights computationally possible?",
      options: [
        { text: "It replaces floating point math with integer arithmetic.", correct: false },
        { text: "Reverse mode computes gradients for all N input parameters in a single backward pass, whereas forward mode requires N separate evaluations.", correct: true },
        { text: "It removes the need for activation functions.", correct: false },
        { text: "It computes exact Hessian second derivatives for free.", correct: false },
      ],
      explanation: "Exactly! When an objective maps from R^N to R (scalar loss with millions of parameters), forward-mode autodiff requires N forward passes. Linnainmaa's reverse AD sweeps backwards from the scalar loss in O(1) passes, evaluating every gradient simultaneously!"
    },
    {
      id: 102,
      question: "In a computational graph, why must variable gradients be accumulated (grad += ...) instead of assigned (grad = ...)?",
      options: [
        { text: "Because Python variables are immutable.", correct: false },
        { text: "When a variable feeds into multiple downstream operations, the multivariable chain rule requires summing the partial derivatives along all branches.", correct: true },
        { text: "To prevent division by zero in floating point registers.", correct: false },
        { text: "It is just a stylistic convention with no mathematical impact.", correct: false },
      ],
      explanation: "Correct! If node x branches to both y and z, dL/dx = (dL/dy)(dy/dx) + (dL/dz)(dz/dx). Overwriting grad would discard signals from one of the computational pathways!"
    }
  ],

  '02': [
    {
      id: 201,
      question: "Why does vanilla SGD violently oscillate back and forth across ill-conditioned ravines rather than moving smoothly down the valley floor?",
      options: [
        { text: "The condition number is large: steep canyon walls generate massive gradients perpendicular to the valley floor, drowning out the gentle forward gradient.", correct: true },
        { text: "SGD uses random numbers on every step.", correct: false },
        { text: "Because the learning rate is automatically set to zero.", correct: false },
        { text: "Ravines always contain local minima traps.", correct: false },
      ],
      explanation: "Bullseye! In an anisotropic ravine, the Hessian eigenvalues are heavily skewed. The gradient points almost entirely sideways into the canyon walls. Polyak Momentum fixes this by building inertia that cancels lateral oscillations!"
    },
    {
      id: 202,
      question: "Why did Loshchilov & Hutter (2017) introduce AdamW by decoupling weight decay from Adam's moment calculations?",
      options: [
        { text: "To make Adam run on Apple Silicon GPUs.", correct: false },
        { text: "In standard Adam, dividing the gradient by sqrt(v) inadvertently divided the L2 weight decay penalty, giving large-gradient weights too little decay and small-gradient weights too much.", correct: true },
        { text: "Because Adam was too fast and overshot the global minimum.", correct: false },
        { text: "To eliminate the need for learning rate schedules.", correct: false },
      ],
      explanation: "Outstanding! In Adam with L2 regularization, the weight penalty lambda*theta is divided by sqrt(v), undermining true weight decay. AdamW subtracts eta*lambda*theta directly from the weight, restoring proper regularization across deep Transformers."
    }
  ],

  '07': [
    {
      id: 1,
      question: "Why did modern generative AI (Flux.1, Stable Diffusion 3, AlphaFold 3) transition from DDPM Diffusion to Optimal Transport Flow Matching?",
      options: [
        { text: "Flow Matching models require no GPUs and can train on standard CPUs.", correct: false },
        { text: "Flow Matching generates straight, deterministic probability trajectories via Optimal Transport, enabling 10x faster ODE generation in 10-20 steps instead of noisy Brownian SDE curves.", correct: true },
        { text: "Diffusion models cannot generate 2D or 3D images.", correct: false },
        { text: "Flow Matching completely eliminates all neural network weights.", correct: false },
      ],
      explanation: "Spot on! Diffusion models generate curved, stochastic paths driven by Brownian motion, which requires 50-1000 Langevin steps. Flow Matching straightens trajectories so standard Runge-Kutta ODE solvers can generate samples in just 10-20 steps!"
    },
    {
      id: 2,
      question: "In DDPM forward noise corruption q(x_t | x_0), what allows sampling at arbitrary timestep t in a single closed-form step without running steps 1...t sequentially?",
      options: [
        { text: "Linear algebra matrix inversion.", correct: false },
        { text: "The sum of independent Gaussian random variables is analytically another Gaussian, parameterized directly by cumulative signal retention ᾱ_t = ∏(1 - β_s).", correct: true },
        { text: "Backpropagation through time.", correct: false },
        { text: "Rejection sampling with Monte Carlo rollouts.", correct: false },
      ],
      explanation: "Exactly! Because the convolution of Gaussians is Gaussian, the cumulative variance across t steps collapses analytically to x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε."
    }
  ],

  '08': [
    {
      id: 1,
      question: "In Sparse Mixture of Experts (MoE), what devastating pathology occurs if the auxiliary load balancing loss α · N · ∑(f_i · P_i) is omitted?",
      options: [
        { text: "The model runs out of memory immediately.", correct: false },
        { text: "Router Collapse: a self-reinforcing feedback loop routes nearly all tokens to 1 or 2 favorite experts while other experts starve and receive zero gradients.", correct: true },
        { text: "All weights instantly become NaN.", correct: false },
        { text: "The model switches automatically from PyTorch to TensorFlow.", correct: false },
      ],
      explanation: "Bullseye! Without load balancing, the router quickly picks early 'favorite' experts that gain better gradients, attracting even more tokens in a vicious cycle until 80%+ of experts are completely idle."
    },
    {
      id: 2,
      question: "What is the primary breakthrough of DeepSeek-R1's Group Relative Policy Optimization (GRPO) over standard PPO RLHF?",
      options: [
        { text: "It uses larger learning rates.", correct: false },
        { text: "It completely eliminates the memory-heavy Critic/Value neural network by sampling groups of candidate completions and calculating relative advantages within the group.", correct: true },
        { text: "It trains without any prompts or data.", correct: false },
        { text: "It only works on chess engines.", correct: false },
      ],
      explanation: "Precisely! PPO requires maintaining a Critic network the same size as the Policy LLM (doubling GPU VRAM). GRPO samples G rollouts and normalizes advantages directly as A_i = (r_i - mean(r)) / (std(r) + eps), saving enormous compute!"
    }
  ],

  '09': [
    {
      id: 1,
      question: "In Anthropic's Transformer Circuits framework, how does the 2-layer Induction Head circuit perform in-context pattern copying [A][B] ... [A] -> [B]?",
      options: [
        { text: "Through a hardcoded Python regex in the tokenizer.", correct: false },
        { text: "Layer 0 previous-token head transfers token A's identity into token B's residual stream; Layer 1 induction head queries for 'preceded by A', attends to B, and the OV circuit boosts B's logit.", correct: true },
        { text: "By storing all previous tokens in an external SQL database.", correct: false },
        { text: "By using recurrent feedback loops between layers.", correct: false },
      ],
      explanation: "Brilliant! This microscopic 2-layer circuit is the fundamental engine of in-context learning in modern Transformers, emerging during the sudden macroscopic 'induction bump' phase change."
    },
    {
      id: 2,
      question: "Why do Sparse Autoencoders (SAEs) enforce a strict unit norm constraint ||W_dec, i||_2 = 1.0 on decoder dictionary columns?",
      options: [
        { text: "To prevent GPU floating point overflow.", correct: false },
        { text: "To prevent the network from trivially shrinking activation features f_i to cheat the L1 sparsity loss while blowing up decoder weights to compensate.", correct: true },
        { text: "To make the matrix symmetric.", correct: false },
        { text: "It is required by the Adam optimizer.", correct: false },
      ],
      explanation: "Exact! If decoder weights were unconstrained, the model could scale W_dec by 1000x and scale feature activations by 1/1000x, achieving near-zero L1 loss without actual sparsity. Unit norm columns enforce genuine sparse dictionary learning!"
    }
  ]
};

export default function StudentQuiz({ currentModule = '00' }) {
  const questions = MODULE_QUESTIONS[currentModule] || MODULE_QUESTIONS['00'];
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);

  // Reset when module changes
  React.useEffect(() => {
    setSelectedAnswers({});
    setSubmitted(false);
  }, [currentModule]);

  const handleSelect = (qId, optionIdx) => {
    if (submitted) return;
    setSelectedAnswers({ ...selectedAnswers, [qId]: optionIdx });
  };

  const calculateScore = () => {
    let correct = 0;
    questions.forEach(q => {
      if (selectedAnswers[q.id] !== undefined && q.options[selectedAnswers[q.id]].correct) {
        correct++;
      }
    });
    return correct;
  };

  const handleSubmit = () => {
    setSubmitted(true);
    const score = calculateScore();
    if (score === questions.length) {
      confetti({
        particleCount: 80,
        spread: 70,
        origin: { y: 0.6 }
      });
    }
  };

  const handleReset = () => {
    setSelectedAnswers({});
    setSubmitted(false);
  };

  const score = calculateScore();
  const allAnswered = questions.every(q => selectedAnswers[q.id] !== undefined);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Interactive Knowledge Check</span>
        </div>
        <h2 className="text-2xl font-black text-white">Module {currentModule} Mastery Assessment</h2>
        <p className="text-xs text-slate-400 max-w-xl mx-auto">
          Test your conceptual understanding of the mathematics, historical breakthroughs, and physical intuitions covered in this module.
        </p>
      </div>

      {/* Questions List */}
      <div className="space-y-4">
        {questions.map((q, idx) => {
          const selectedIdx = selectedAnswers[q.id];
          return (
            <div
              key={q.id}
              className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4 transition"
            >
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-lg bg-cyan-500/20 text-cyan-300 font-mono font-bold text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                  {idx + 1}
                </span>
                <div className="font-bold text-white text-sm sm:text-base leading-snug">
                  {q.question}
                </div>
              </div>

              {/* Options */}
              <div className="space-y-2 pl-9">
                {q.options.map((opt, oIdx) => {
                  const isSelected = selectedIdx === oIdx;
                  let optStyle = "bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-850";

                  if (submitted) {
                    if (opt.correct) {
                      optStyle = "bg-emerald-950/40 border-emerald-500/60 text-emerald-200";
                    } else if (isSelected && !opt.correct) {
                      optStyle = "bg-rose-950/40 border-rose-500/60 text-rose-200";
                    } else {
                      optStyle = "bg-slate-950/30 border-slate-900 text-slate-600";
                    }
                  } else if (isSelected) {
                    optStyle = "bg-cyan-950/40 border-cyan-500/80 text-cyan-200 ring-1 ring-cyan-500/50";
                  }

                  return (
                    <div
                      key={oIdx}
                      onClick={() => handleSelect(q.id, oIdx)}
                      className={`p-3.5 rounded-xl border text-xs sm:text-sm font-medium transition cursor-pointer flex items-center justify-between ${optStyle}`}
                    >
                      <span>{opt.text}</span>
                      {submitted && opt.correct && (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 ml-2" />
                      )}
                      {submitted && isSelected && !opt.correct && (
                        <XCircle className="w-4 h-4 text-rose-400 flex-shrink-0 ml-2" />
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Feedback Explanation */}
              {submitted && (
                <div className="mt-4 ml-9 p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 leading-relaxed animate-fadeIn">
                  <span className="font-bold text-cyan-400 block mb-1">Pedagogical Insight:</span>
                  {q.explanation}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Action Footer */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex items-center justify-between">
        {!submitted ? (
          <button
            onClick={handleSubmit}
            disabled={!allAnswered}
            className={`px-6 py-2.5 rounded-xl font-bold text-xs sm:text-sm transition flex items-center gap-2 ${
              allAnswered
                ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20 hover:scale-102"
                : "bg-slate-800 text-slate-500 cursor-not-allowed"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span>Submit Answers & Calculate Score</span>
          </button>
        ) : (
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
                <Award className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <div className="text-sm font-bold text-white">
                  Score: {score} / {questions.length} ({Math.round((score / questions.length) * 100)}%)
                </div>
                <div className="text-xs text-slate-400">
                  {score === questions.length ? "Flawless score! Full conceptual mastery demonstrated." : "Good attempt! Review the explanations above."}
                </div>
              </div>
            </div>

            <button
              onClick={handleReset}
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Quiz</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
