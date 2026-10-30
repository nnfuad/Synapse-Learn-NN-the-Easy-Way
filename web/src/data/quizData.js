// Comprehensive Master Question Bank for Synapse Neural Explainer
// Categorized by Module (00-09) and Difficulty (basic, hard, pro)

export const QUIZ_DIFFICULTIES = [
  {
    id: 'all',
    label: 'All Modes',
    icon: '⚡',
    badgeClass: 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400',
    description: 'The complete gauntlet across all difficulty tiers'
  },
  {
    id: 'basic',
    label: 'Basic Mode',
    icon: '🟢',
    badgeClass: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    description: 'Gentle intuitions, visual analogies & core definitions'
  },
  {
    id: 'hard',
    label: 'Hard Mode',
    icon: '🟡',
    badgeClass: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
    description: 'Mathematical mechanics, derivations & classical papers'
  },
  {
    id: 'pro',
    label: 'Pro Mode',
    icon: '🟣',
    badgeClass: 'bg-fuchsia-500/10 border-fuchsia-500/30 text-fuchsia-400',
    description: 'SOTA internals, mechanistic circuits & galaxy-brain edge cases'
  }
];

export const MODULE_QUIZ_DATA = {
  // ==========================================
  // MODULE 00: The Genesis & The Atomic Neuron
  // ==========================================
  '00': [
    {
      id: '00-b1',
      difficulty: 'basic',
      question: "What does a single artificial neuron fundamentally calculate before applying its activation function?",
      options: [
        { text: "A random number between 0 and 1.", correct: false },
        { text: "A weighted sum of its inputs plus a bias term: Σ(w_i · x_i) + b.", correct: true },
        { text: "The square root of all incoming features.", correct: false },
        { text: "A sort operation ordering inputs from smallest to largest.", correct: false }
      ],
      explanation: "Spot on! An artificial neuron acts like an audio mixer: it takes multiple input channels, scales each by an adjustable volume slider (weight), adds an offset (bias), and sums them up.",
      hint: "Think of an audio mixer combining channel volume sliders into a single master fader."
    },
    {
      id: '00-b2',
      difficulty: 'basic',
      question: "Why is the XOR (Exclusive OR) problem famous in the history of neural networks?",
      options: [
        { text: "It was the first problem ever solved by Frank Rosenblatt's Perceptron.", correct: false },
        { text: "It showed that a single-layer perceptron cannot solve non-linearly separable problems, triggering the 1969 First AI Winter.", correct: true },
        { text: "It is the only logic gate that modern GPUs can compute.", correct: false },
        { text: "It required quantum computers to solve.", correct: false }
      ],
      explanation: "Exactly! In XOR, diagonally opposite points share identical outputs (0,0->0, 1,1->0 vs 0,1->1, 1,0->1). A single straight knife cut cannot separate them, creating the legendary crisis documented by Minsky & Papert.",
      hint: "Can you draw one single straight line on graph paper that isolates (0,1) and (1,0) from (0,0) and (1,1)?"
    },
    {
      id: '00-b3',
      difficulty: 'basic',
      question: "What essential freedom does the 'bias' (b) parameter give to a neuron's decision boundary?",
      options: [
        { text: "It shifts the boundary away from the origin (0,0), allowing flexible thresholds.", correct: true },
        { text: "It prevents the weights from becoming negative.", correct: false },
        { text: "It forces the output to always equal zero.", correct: false },
        { text: "It speeds up matrix multiplications on GPUs.", correct: false }
      ],
      explanation: "Right on! Without a bias term, w_1·x_1 + w_2·x_2 = 0 is glued to the origin (0,0). The bias term slides the hyperplane along the axes so it can classify data located anywhere in space.",
      hint: "Without bias, the separating line is permanently pinned to (0,0)."
    },
    {
      id: '00-h1',
      difficulty: 'hard',
      question: "Why could Rosenblatt's 1957 Perceptron NOT use standard calculus gradient descent to train its weights?",
      options: [
        { text: "Hardware in 1957 was too slow to calculate derivatives.", correct: false },
        { text: "The Heaviside step activation function has a derivative of zero almost everywhere (and undefined at 0), meaning gradients cannot flow.", correct: true },
        { text: "Matrix multiplication had not yet been discovered.", correct: false },
        { text: "It had too many layers for calculus to work.", correct: false }
      ],
      explanation: "Bullseye! The step function is completely flat (slope = 0) everywhere except at x = 0 where it jumps discontinuously. Since d/dx is 0, gradient descent gets zero signal! That is why the field had to wait until 1986 for smooth, differentiable activations like Sigmoid and ReLU.",
      hint: "What is the slope of a flat horizontal step?"
    },
    {
      id: '00-h2',
      difficulty: 'hard',
      question: "What is the theoretical maximum classification accuracy that ANY single straight line can achieve on the 4 XOR points in 2D space?",
      options: [
        { text: "25% (1 out of 4)", correct: false },
        { text: "50% (2 out of 4)", correct: false },
        { text: "75% (3 out of 4)", correct: true },
        { text: "100% (4 out of 4)", correct: false }
      ],
      explanation: "Exactly! Because the two classes form alternating diagonal pairs (like a checkerboard), any straight cut can at most isolate 3 points correctly while misclassifying the 4th point. 3/4 = 75% accuracy.",
      hint: "Place 4 corners of a square with alternating colors; any straight cut can at most get 3 right."
    },
    {
      id: '00-h3',
      difficulty: 'hard',
      question: "Geometrically, what is the fundamental superpower that hidden layers with non-linear activations provide to a neural network?",
      options: [
        { text: "They make the learning rate 10x faster.", correct: false },
        { text: "They warp, stretch, and fold the coordinate space so tangled data points become linearly separable in the hidden representation.", correct: true },
        { text: "They round decimal weights to clean binary integers.", correct: false },
        { text: "They act as a hardware firewall preventing overfitting.", correct: false }
      ],
      explanation: "Brilliant! Hidden layers perform a continuous non-linear topological warp on space. Points that were trapped diagonally apart in input space are pushed onto the same side or unfolded in hidden feature space!",
      hint: "Imagine folding a sheet of paper so two distant dots touch."
    },
    {
      id: '00-p1',
      difficulty: 'pro',
      question: "In Minsky & Papert's seminal 1969 book 'Perceptrons', what topological property did they prove perceptrons with bounded receptive fields cannot compute?",
      options: [
        { text: "Linear separability.", correct: false },
        { text: "Connectedness (determining if a shape is a single continuous object or composed of disconnected components).", correct: true },
        { text: "Whether an image has more than 10 pixels.", correct: false },
        { text: "Matrix inversion.", correct: false }
      ],
      explanation: "Outstanding! Minsky & Papert proved that determining whether a figure is connected requires a diameter of perception that scales with the entire canvas. Local perceptrons cannot detect global topological connectivity!",
      hint: "A local window cannot tell if two distant spirals connect outside its field of view."
    },
    {
      id: '00-p2',
      difficulty: 'pro',
      question: "According to the Novikoff Perceptron Convergence Theorem, if data has margin γ and lies in a sphere of radius R, what is the maximum number of weight updates before convergence?",
      options: [
        { text: "Updates ≤ (R / γ)²", correct: true },
        { text: "Updates ≤ 2^(R · γ)", correct: false },
        { text: "Updates ≤ log(R · γ)", correct: false },
        { text: "Convergence is not mathematically bounded.", correct: false }
      ],
      explanation: "Precisely! Albert Novikoff proved in 1962 that if a dataset is linearly separable with margin γ and maximum radius R, the perceptron learning rule stops after at most (R/γ)² mistakes, proving finite convergence.",
      hint: "The bound scales inversely with the square of the geometric margin."
    }
  ],

  // ==========================================
  // MODULE 01: Computational Graphs & Autograd
  // ==========================================
  '01': [
    {
      id: '01-b1',
      difficulty: 'basic',
      question: "What is a 'computational graph' in modern deep learning frameworks like PyTorch?",
      options: [
        { text: "A bar chart showing your GPU temperature.", correct: false },
        { text: "A directed acyclic graph (DAG) where nodes represent variables or operations and edges track mathematical dependencies.", correct: true },
        { text: "A database table that stores training images.", correct: false },
        { text: "An electrical schematic of the CPU circuitry.", correct: false }
      ],
      explanation: "Correct! A computational graph maps out every calculation from inputs to loss. Tracing this graph backwards is how backpropagation automatically computes exact derivatives.",
      hint: "Think of an assembly flowchart where each step feeds into the next without circular loops."
    },
    {
      id: '01-b2',
      difficulty: 'basic',
      question: "What does backpropagation calculate for every learnable weight in a neural network?",
      options: [
        { text: "Its optimal learning rate.", correct: false },
        { text: "The partial derivative of the scalar loss with respect to that weight (∂Loss/∂w).", correct: true },
        { text: "Its physical memory address on the RAM chip.", correct: false },
        { text: "The square root of the training error.", correct: false }
      ],
      explanation: "Spot on! ∂Loss/∂w tells us how sensitive the loss is to that specific weight knob: 'If I nudge this weight up by a tiny bit, does the loss go up or down, and by how much?'",
      hint: "Sensitivity analysis: how much does jiggling one knob affect the final score?"
    },
    {
      id: '01-b3',
      difficulty: 'basic',
      question: "Why is the calculus Chain Rule the heartbeat of deep learning?",
      options: [
        { text: "It allows computing derivatives of composite functions step-by-step by multiplying local derivatives.", correct: true },
        { text: "It converts Python scripts into C++ automatically.", correct: false },
        { text: "It encrypts model weights for security.", correct: false },
        { text: "It replaces all activation functions with straight lines.", correct: false }
      ],
      explanation: "Exactly! Because a deep network is a composite function f(g(h(x))), the chain rule lets us compute total derivatives by multiplying the simple local derivatives along the path from output back to input.",
      hint: "If y depends on u, and u depends on x, how do you find dy/dx?"
    },
    {
      id: '01-h1',
      difficulty: 'hard',
      question: "Why did Seppo Linnainmaa's 1970 reverse-mode automatic differentiation make deep learning with millions of weights computationally possible?",
      options: [
        { text: "It replaces floating point math with integer arithmetic.", correct: false },
        { text: "Reverse mode computes gradients for all N input parameters in a single backward pass, whereas forward mode requires N separate evaluations.", correct: true },
        { text: "It removes the need for activation functions.", correct: false },
        { text: "It computes exact Hessian second derivatives for free.", correct: false }
      ],
      explanation: "Exactly! When an objective maps from R^N to R (scalar loss with millions of parameters), forward-mode autodiff requires N forward passes. Linnainmaa's reverse AD sweeps backwards from the scalar loss in O(1) passes, evaluating every gradient simultaneously!",
      hint: "Consider mapping from R^N -> R with N = 10 billion parameters."
    },
    {
      id: '01-h2',
      difficulty: 'hard',
      question: "In a computational graph, why must variable gradients be accumulated (grad += ...) instead of assigned (grad = ...)?",
      options: [
        { text: "Because Python variables are immutable.", correct: false },
        { text: "When a variable feeds into multiple downstream operations, the multivariable chain rule requires summing the partial derivatives along all branches.", correct: true },
        { text: "To prevent division by zero in floating point registers.", correct: false },
        { text: "It is just a stylistic convention with no mathematical impact.", correct: false }
      ],
      explanation: "Correct! If node x branches to both y and z, dL/dx = (dL/dy)(dy/dx) + (dL/dz)(dz/dx). Overwriting grad would discard signals from one of the computational pathways!",
      hint: "If water flows out through two pipes, the total incoming pressure is the sum of both backpressures."
    },
    {
      id: '01-h3',
      difficulty: 'hard',
      question: "In a custom PyTorch autograd Function, what is the role of ctx.save_for_backward(...) during forward pass?",
      options: [
        { text: "It saves the model weights to the hard drive.", correct: false },
        { text: "It preserves intermediate tensors (inputs/outputs) needed by the mathematical formulas in backward().", correct: true },
        { text: "It sends telemetry logs to TensorBoard.", correct: false },
        { text: "It frees GPU VRAM immediately.", correct: false }
      ],
      explanation: "Precisely! For example, the derivative of sigmoid σ(x) is σ(x)(1 - σ(x)). To calculate this in backward(), the function must remember either x or σ(x) from the forward pass!",
      hint: "To compute d/dx of σ(x), what values from the forward pass do you need?"
    },
    {
      id: '01-p1',
      difficulty: 'pro',
      question: "What is the primary memory vs computation trade-off between Forward-Mode and Reverse-Mode autodiff for vector function f: R^N -> R^M?",
      options: [
        { text: "Forward mode needs O(graph size) memory; reverse mode needs O(1) memory.", correct: false },
        { text: "Forward mode takes O(N) passes with O(1) activation memory, while reverse mode takes O(M) passes with O(graph size) activation memory.", correct: true },
        { text: "Both have identical compute and memory bounds regardless of N or M.", correct: false },
        { text: "Forward mode cannot calculate non-linear derivatives.", correct: false }
      ],
      explanation: "Masterful! Reverse mode sweeps from outputs to inputs in O(M) passes (ideal for loss M=1), but must store all intermediate activations in VRAM. Forward mode requires zero activation caching but takes O(N) passes (disastrous when N is billions of weights).",
      hint: "Think about why training large LLMs runs out of VRAM during backprop."
    },
    {
      id: '01-p2',
      difficulty: 'pro',
      question: "How does Gradient Checkpointing (Activation Recomputation) mathematically optimize VRAM during training?",
      options: [
        { text: "It reduces weights to 8-bit integers.", correct: false },
        { text: "It discards intermediate activations during forward pass and recalculates them on-the-fly during backward pass, dropping memory from O(L) to O(√L) for ~33% extra FLOPs.", correct: true },
        { text: "It skips backpropagation for 50% of layers.", correct: false },
        { text: "It increases learning rate automatically.", correct: false }
      ],
      explanation: "Brilliant! Chen et al. (2016) showed that by saving only checkpoints every √L layers, you free up massive GPU memory for larger batch sizes or sequence lengths at the cost of one extra forward pass per segment.",
      hint: "Trading extra computation to avoid storing every single activation in VRAM."
    }
  ],

  // ==========================================
  // MODULE 02: Loss Geometry & Optimizers
  // ==========================================
  '02': [
    {
      id: '02-b1',
      difficulty: 'basic',
      question: "What does the 'learning rate' (η) control in an optimizer like SGD?",
      options: [
        { text: "How many epochs the training will run.", correct: false },
        { text: "The step size taken in the direction of the negative gradient at each iteration.", correct: true },
        { text: "The number of hidden layers in the model.", correct: false },
        { text: "The resolution of input images.", correct: false }
      ],
      explanation: "Spot on! The learning rate dictates stride length. If it's too big, you overshoot the valley floor and diverge; if it's too tiny, training crawls or gets stuck.",
      hint: "Think of taking giant leaps vs tiny baby steps down a mountain trail."
    },
    {
      id: '02-b2',
      difficulty: 'basic',
      question: "What is the key difference between full Batch Gradient Descent and Mini-Batch SGD?",
      options: [
        { text: "Mini-batch SGD estimates gradients on small random subsets of data rather than scanning the entire dataset on every step.", correct: true },
        { text: "Full Batch GD only works on laptops; mini-batch works only on supercomputers.", correct: false },
        { text: "Mini-batch SGD eliminates the need for a loss function.", correct: false },
        { text: "Full Batch GD uses momentum; mini-batch cannot.", correct: false }
      ],
      explanation: "Correct! On a 10-million image dataset, full batch GD would require hours just to take one single step. Mini-batch SGD computes fast, noisy gradient steps every few milliseconds, speeding up training enormously.",
      hint: "Do you survey every citizen in the country before updating a poll, or sample 100 people?"
    },
    {
      id: '02-b3',
      difficulty: 'basic',
      question: "What is the intuitive physical analogy for Polyak Momentum in optimization?",
      options: [
        { text: "A heavy bowling ball rolling downhill that builds velocity in consistent directions and ignores tiny potholes.", correct: true },
        { text: "A leaf blowing randomly in the wind.", correct: false },
        { text: "A spring that pulls weights back to zero.", correct: false },
        { text: "A thermometer measuring temperature.", correct: false }
      ],
      explanation: "Exactly! Momentum accumulates past velocity. In ravines where gradients oscillate violently side-to-side, lateral oscillations cancel out while forward downhill velocity builds up like a heavy rolling ball.",
      hint: "Inertia: mass resists sudden changes in direction."
    },
    {
      id: '02-h1',
      difficulty: 'hard',
      question: "Why does vanilla SGD violently oscillate back and forth across ill-conditioned ravines rather than moving smoothly down the valley floor?",
      options: [
        { text: "The condition number is large: steep canyon walls generate massive gradients perpendicular to the valley floor, drowning out the gentle forward gradient.", correct: true },
        { text: "SGD uses random numbers on every step.", correct: false },
        { text: "Because the learning rate is automatically set to zero.", correct: false },
        { text: "Ravines always contain local minima traps.", correct: false }
      ],
      explanation: "Bullseye! In an anisotropic ravine, the Hessian eigenvalues are heavily skewed. The gradient points almost entirely sideways into the canyon walls. Polyak Momentum fixes this by building inertia that cancels lateral oscillations!",
      hint: "Think about Hessian eigenvalues being 10,000x steeper in one dimension than another."
    },
    {
      id: '02-h2',
      difficulty: 'hard',
      question: "Why did Loshchilov & Hutter (2017) introduce AdamW by decoupling weight decay from Adam's moment calculations?",
      options: [
        { text: "To make Adam run on Apple Silicon GPUs.", correct: false },
        { text: "In standard Adam, dividing the gradient by sqrt(v) inadvertently divided the L2 weight decay penalty, giving large-gradient weights too little decay and small-gradient weights too much.", correct: true },
        { text: "Because Adam was too fast and overshot the global minimum.", correct: false },
        { text: "To eliminate the need for learning rate schedules.", correct: false }
      ],
      explanation: "Outstanding! In Adam with L2 regularization, the weight penalty λ·θ is divided by √(v), undermining true weight decay. AdamW subtracts η·λ·θ directly from the weight, restoring proper regularization across deep Transformers.",
      hint: "Standard L2 regularization added λ·w directly to the gradient vector inside the adaptive scaling."
    },
    {
      id: '02-h3',
      difficulty: 'hard',
      question: "What is the purpose of the bias correction terms (1 - β₁ᵗ) and (1 - β₂ᵗ) in the Adam optimizer?",
      options: [
        { text: "They prevent the moving averages m_t and v_t from being biased toward zero in initial iterations since they are initialized at 0.", correct: true },
        { text: "They adjust the learning rate based on GPU clock speed.", correct: false },
        { text: "They ensure weights never become negative.", correct: false },
        { text: "They replace the need for loss backpropagation.", correct: false }
      ],
      explanation: "Precisely! Because m_0 and v_0 start at zero, without correction at t=1 with β_2=0.999, v_1 would be 0.001·g², artificially shrinking variance and causing an explosive initial step size.",
      hint: "Exponential moving averages initialized at 0 underestimate true magnitudes early on."
    },
    {
      id: '02-p1',
      difficulty: 'pro',
      question: "In loss landscape geometry, why do flat minima tend to generalize significantly better to unseen test distributions than sharp minima?",
      options: [
        { text: "Flat minima require fewer bits to represent in floating point.", correct: false },
        { text: "Distribution shifts between train and test slightly shift the loss surface; sharp minima cause catastrophic loss spikes under tiny shifts, whereas flat minima absorb shifts gracefully.", correct: true },
        { text: "Flat minima always have zero training error.", correct: false },
        { text: "Sharp minima cannot be found by gradient descent.", correct: false }
      ],
      explanation: "Masterful! Hochreiter & Schmidhuber (1997) and Keskar et al. (2016) demonstrated that if the test distribution shifts the landscape by ε, the loss increase in a sharp basin is proportional to λ_max · ε², leading to poor generalization.",
      hint: "What happens to the loss if a bowl is ultra-narrow vs wide and gentle when you slide it 0.01 units?"
    },
    {
      id: '02-p2',
      difficulty: 'pro',
      question: "How does the Muon optimizer (Moon et al. 2024 / Keller Jordan 2025) scale 2D weight updates differently from standard AdamW?",
      options: [
        { text: "It uses Newton-Schulz iterations to orthogonalize matrix updates, constraining the spectral norm rather than per-element RMS.", correct: true },
        { text: "It replaces all matrix multiplication with random projection.", correct: false },
        { text: "It runs exclusively on CPUs without matrix engines.", correct: false },
        { text: "It removes the need for learning rates entirely.", correct: false }
      ],
      explanation: "Brilliant! Muon recognizes that 2D weight matrices are geometric transformations. By computing a polar decomposition (via iterative Newton-Schulz matrix products), it ensures all singular values of the update matrix equal 1.0, supercharging LLM pretraining speed!",
      hint: "Treating weight tensors as geometric operators rather than uncoordinated bags of independent scalars."
    }
  ],

  // ==========================================
  // MODULE 03: Generalization & Normalization
  // ==========================================
  '03': [
    {
      id: '03-b1',
      difficulty: 'basic',
      question: "What is 'overfitting' in machine learning?",
      options: [
        { text: "When a model is too small to learn any patterns at all.", correct: false },
        { text: "When a model memorizes noise and quirks of the training data, scoring high on training but failing on unseen test data.", correct: true },
        { text: "When training finishes too quickly.", correct: false },
        { text: "When a GPU runs out of VRAM.", correct: false }
      ],
      explanation: "Spot on! Overfitting is like a student who memorizes the exact multiple choice answers of a practice test without understanding the underlying concepts, failing the real exam.",
      hint: "Memorizing answers by heart instead of learning general rules."
    },
    {
      id: '03-b2',
      difficulty: 'basic',
      question: "How does Dropout (Srivastava et al. 2014) combat overfitting during training?",
      options: [
        { text: "It randomly sets a fraction of neuron activations to zero on each forward pass, preventing neurons from co-adapting.", correct: true },
        { text: "It drops low-resolution images from the dataset.", correct: false },
        { text: "It deletes random layers from the network permanently.", correct: false },
        { text: "It slows down the GPU clock to prevent overheating.", correct: false }
      ],
      explanation: "Correct! If random neurons are deactivated on every training step, no neuron can rely on a specific peer to do the work. Each neuron must learn robust, independently useful features.",
      hint: "If random coworkers take the day off unpredictably, everyone has to learn how to do the job."
    },
    {
      id: '03-b3',
      difficulty: 'basic',
      question: "What is the primary difference between Batch Normalization and Layer Normalization?",
      options: [
        { text: "BatchNorm normalizes across the batch dimension for each channel; LayerNorm normalizes across channels/features for each individual sample.", correct: true },
        { text: "BatchNorm is only used for text; LayerNorm is only used for audio.", correct: false },
        { text: "LayerNorm requires training labels; BatchNorm is unsupervised.", correct: false },
        { text: "They are mathematically identical in every way.", correct: false }
      ],
      explanation: "Exactly! BatchNorm computes mean and variance across all samples in a mini-batch. LayerNorm calculates mean and variance independently within each single sequence/sample, making it ideal for variable-length sequential models like Transformers.",
      hint: "Does the normalization slice cut across different samples in the batch or across features within one sample?"
    },
    {
      id: '03-h1',
      difficulty: 'hard',
      question: "Why did Santurkar et al. (NeurIPS 2018) debunk the original 'Internal Covariate Shift' explanation for Batch Normalization?",
      options: [
        { text: "BatchNorm actually increases internal covariate shift.", correct: false },
        { text: "Even when intentionally injecting distributional shift noise, BatchNorm trains just as effectively because its true superpower is smoothing the loss landscape (making loss and gradients Lipschitz continuous).", correct: true },
        { text: "BatchNorm is only useful for convolutional networks, not MLPs.", correct: false },
        { text: "Calculus does not apply to normalized distributions.", correct: false }
      ],
      explanation: "Brilliant! Santurkar et al. proved that BatchNorm works not by stabilizing distribution shifts, but by making the loss landscape significantly smoother and predictive, allowing much larger learning rates.",
      hint: "They injected noise into intermediate distributions and found the network trained just as well."
    },
    {
      id: '03-h2',
      difficulty: 'hard',
      question: "Why have modern Foundation LLMs (LLaMA 3, Mistral, Gemma, DeepSeek) universally replaced LayerNorm with RMSNorm?",
      options: [
        { text: "RMSNorm requires 100x more memory.", correct: false },
        { text: "Research discovered mean-centering is computationally redundant; scaling invariance alone stabilizes gradients, saving 10-50% computational overhead.", correct: true },
        { text: "RMSNorm eliminates the need for token embeddings.", correct: false },
        { text: "PyTorch 1.0 could not compile LayerNorm.", correct: false }
      ],
      explanation: "Spot on! Zhang & Sennrich (2019) demonstrated that subtracting the mean contributes virtually nothing to training stability. RMSNorm simply scales by the root mean square, discarding mean subtraction and bias additions.",
      hint: "Is subtracting the mean necessary if scaling by root-mean-square gives all the stability?"
    },
    {
      id: '03-h3',
      difficulty: 'hard',
      question: "Why did modern Transformer architectures switch from Post-LN (original Attention is All You Need) to Pre-LN?",
      options: [
        { text: "Post-LN placed normalization on the residual path, causing vanishing/exploding gradients at depth without fragile warmups; Pre-LN preserves an unhindered identity highway.", correct: true },
        { text: "Pre-LN requires half the parameters of Post-LN.", correct: false },
        { text: "Post-LN cannot process text longer than 128 tokens.", correct: false },
        { text: "Pre-LN converts floats to ints.", correct: false }
      ],
      explanation: "Bullseye! In Pre-LN, x_{l+1} = x_l + F(Norm(x_l)). The identity connection x_l remains completely pristine, enabling training of 100+ layer deep models without delicate learning rate warmups.",
      hint: "Look at whether the residual connection + x has a normalization operation applied to it or not."
    },
    {
      id: '03-p1',
      difficulty: 'pro',
      question: "How does Spectral Normalization (Miyato et al. 2018) mathematically guarantee Lipschitz continuity in deep networks?",
      options: [
        { text: "By dividing each weight matrix W by its largest singular value σ(W), calculated via power iteration, ensuring the Lipschitz constant ≤ 1.", correct: true },
        { text: "By enforcing all weights to lie between -1 and +1.", correct: false },
        { text: "By setting diagonal entries of weight matrices to zero.", correct: false },
        { text: "By running an FFT after every forward pass.", correct: false }
      ],
      explanation: "Outstanding! The matrix operator norm ||W||_2 is its maximum singular value σ(W). Dividing W by σ(W) guarantees ||W / σ(W)||_2 = 1, bounding gradient amplification and preventing discriminator explosion in GANs and flow models.",
      hint: "What is the maximum stretching factor of a linear matrix transformation?"
    },
    {
      id: '03-p2',
      difficulty: 'pro',
      question: "Why does standard L2 regularization fail to act as genuine weight decay when combined with adaptive gradient optimizers like Adam?",
      options: [
        { text: "Because Adam uses momentum.", correct: false },
        { text: "In Adam, the L2 gradient term λ·w is divided by √(v_t); parameters with large historical gradients receive virtually zero decay, while inactive parameters are over-decayed.", correct: true },
        { text: "L2 regularization only works on convex loss functions.", correct: false },
        { text: "Adam sets all weights to zero under L2.", correct: false }
      ],
      explanation: "Precisely! When you add λ·w to the gradient, Adam's denominator √(v_t) scales it down for high-frequency features. True weight decay must subtract η·λ·w directly outside the adaptive denominator, as solved by AdamW.",
      hint: "The gradient is divided by the moving average of squared gradients."
    }
  ],

  // ==========================================
  // MODULE 04: Spatial Vision & Convolutions
  // ==========================================
  '04': [
    {
      id: '04-b1',
      difficulty: 'basic',
      question: "What is the intuitive mechanism of a 2D Convolutional layer?",
      options: [
        { text: "A small filter (kernel) slides across an image, computing dot products to detect local spatial patterns like edges and textures.", correct: true },
        { text: "It compresses the entire image into a single scalar value.", correct: false },
        { text: "It shuffles all the pixels randomly to prevent overfitting.", correct: false },
        { text: "It converts RGB colors to grayscale.", correct: false }
      ],
      explanation: "Spot on! Think of a 3x3 stencil scanning across a canvas. Wherever the underlying image pixels match the stencil pattern (e.g. a vertical edge), the convolution produces a high activation.",
      hint: "A sliding window pattern detector that scans the image."
    },
    {
      id: '04-b2',
      difficulty: 'basic',
      question: "What is 'Translation Invariance' in computer vision?",
      options: [
        { text: "Translating image captions into other languages.", correct: false },
        { text: "The ability of a vision model to recognize an object (like a cat) regardless of where it appears in the frame.", correct: true },
        { text: "Translating Python vision code into CUDA code.", correct: false },
        { text: "Rotating an image by 180 degrees.", correct: false }
      ],
      explanation: "Correct! Because CNN filters share weights across every pixel position, a cat ear detector filter will trigger whether the cat is in the top-left, center, or bottom-right of the image.",
      hint: "A cat is still a cat whether it sits on the left or right of the photo."
    },
    {
      id: '04-b3',
      difficulty: 'basic',
      question: "What is the main role of Max Pooling in classical vision architectures?",
      options: [
        { text: "Downsampling spatial dimensions, reducing compute while retaining the most salient feature activations.", correct: true },
        { text: "Increasing image resolution by 4x.", correct: false },
        { text: "Normalizing RGB color channels.", correct: false },
        { text: "Adding random Gaussian noise to pixels.", correct: false }
      ],
      explanation: "Exactly! Taking the maximum value in every 2x2 grid cuts width and height in half, dramatically reducing compute while preserving the strongest detected features.",
      hint: "Picking the biggest value in a 2x2 window to downscale the image."
    },
    {
      id: '04-h1',
      difficulty: 'hard',
      question: "Why does stacking two 3x3 convolutional layers (as popularized by VGG-16) outperform a single 5x5 convolution?",
      options: [
        { text: "3x3 convolutions run only on CPU, saving GPU power.", correct: false },
        { text: "Both span an identical 5x5 receptive field, but two 3x3 convs use 18·C² parameters instead of 25·C² (28% fewer parameters) while injecting two non-linear activations.", correct: true },
        { text: "5x5 convolutions cannot detect vertical edges.", correct: false },
        { text: "Two 3x3 convolutions eliminate all padding requirements.", correct: false }
      ],
      explanation: "Exactly! Two consecutive 3x3 filters have a receptive field of 3 + (3-1) = 5. You get the exact same spatial coverage with 28% fewer weights and double the non-linear decision capacity.",
      hint: "3 + (3 - 1) = 5. Compare 2 x (3x3) = 18 vs 1 x (5x5) = 25."
    },
    {
      id: '04-h2',
      difficulty: 'hard',
      question: "In ResNet (He et al. 2015), how does the residual identity connection H(x) = F(x) + x mathematically conquer the deep degradation problem?",
      options: [
        { text: "It doubles the resolution of input images.", correct: false },
        { text: "The backward gradient contains an unhindered +1 identity term (∂L/∂x = ∂L/∂H · [I + ∂F/∂x]), allowing error signals to flow back directly across hundreds of layers with zero attenuation.", correct: true },
        { text: "It converts all floating-point numbers to integers.", correct: false },
        { text: "It bypasses GPU memory limits.", correct: false }
      ],
      explanation: "Masterful! Because of the addition +x, the gradient always carries a clean identity factor +1. Even if the residual block F(x) has small weights, the gradient never vanishes!",
      hint: "Differentiate F(x) + x with respect to x. What is dx/dx?"
    },
    {
      id: '04-h3',
      difficulty: 'hard',
      question: "What is the formula for the output spatial dimension W_out given input width W, filter size K, padding P, and stride S?",
      options: [
        { text: "W_out = ⌊(W - K + 2P) / S⌋ + 1", correct: true },
        { text: "W_out = (W · K) / (P + S)", correct: false },
        { text: "W_out = W - K + S", correct: false },
        { text: "W_out = ⌊(W + P) / (K · S)⌋", correct: false }
      ],
      explanation: "Bullseye! Padding adds 2P pixels total. Subtracting kernel width K leaves the remaining sliding span. Dividing by stride S and adding 1 gives the exact number of valid window placements.",
      hint: "Effective width is W + 2P; sliding a window of size K in strides of S."
    },
    {
      id: '04-p1',
      difficulty: 'pro',
      question: "How does the im2col algorithm transform 2D spatial convolutions into high-performance General Matrix Multiply (GEMM)?",
      options: [
        { text: "It unrolls every receptive field patch into columns of a large matrix and flattens kernels into rows, converting sliding convolutions into a single dense BLAS GEMM.", correct: true },
        { text: "It converts images to black-and-white bitmasks.", correct: false },
        { text: "It skips 75% of pixel multiplications using hashing.", correct: false },
        { text: "It replaces tensor operations with SQLite queries.", correct: false }
      ],
      explanation: "Brilliant! Hardware accelerators (NVIDIA Tensor Cores) are heavily optimized for dense GEMM matrix multiplication. im2col replicates image patches so that all convolutions execute as one blistering C = A · B matrix multiply.",
      hint: "Flattening receptive field patches into columns of an expanded matrix."
    },
    {
      id: '04-p2',
      difficulty: 'pro',
      question: "What key architectural changes allowed ConvNeXt (Liu et al. 2022) to modernize pure CNNs to match Vision Transformers (ViTs)?",
      options: [
        { text: "Inverted bottleneck, 7x7 depthwise convolutions, replacing BatchNorm with LayerNorm, fewer activation layers (GELU), and separate patchifying downsamplers.", correct: true },
        { text: "Adding multi-head self-attention inside every convolutional block.", correct: false },
        { text: "Removing all convolutional layers and using pure MLPs.", correct: false },
        { text: "Increasing batch sizes to 1,000,000.", correct: false }
      ],
      explanation: "Spot on! Liu et al. systematically adopted Transformer design patterns (inverted bottleneck, depthwise 7x7 separable filters, LayerNorm instead of BatchNorm, GELU) to prove CNNs remain competitive with ViTs without attention.",
      hint: "Adopting Transformer structural patterns while keeping convolutional inductive bias."
    }
  ],

  // ==========================================
  // MODULE 05: Sequential Dynamics & Memory
  // ==========================================
  '05': [
    {
      id: '05-b1',
      difficulty: 'basic',
      question: "Why are Recurrent Neural Networks (RNNs) suited for sequential data like sentences or time series?",
      options: [
        { text: "They maintain an internal hidden state (h_t) that passes from step to step, acting as working memory of past context.", correct: true },
        { text: "They run 1000x faster than feedforward networks.", correct: false },
        { text: "They can only process one single word per document.", correct: false },
        { text: "They do not require any training data.", correct: false }
      ],
      explanation: "Correct! At step t, an RNN combines current input x_t with previous memory h_{t-1}. This recurrent loop allows information from earlier tokens to persist across time.",
      hint: "Reading a book word-by-word while keeping what you just read in your head."
    },
    {
      id: '05-b2',
      difficulty: 'basic',
      question: "What happens when an early vanilla RNN tries to learn dependencies across a 100-word sentence?",
      options: [
        { text: "Gradients vanish exponentially during backpropagation through time, causing the network to forget early words.", correct: true },
        { text: "The network translates the text into French.", correct: false },
        { text: "The weights automatically lock to prevent memory leaks.", correct: false },
        { text: "The GPU runs out of hard drive space.", correct: false }
      ],
      explanation: "Spot on! Multiplying by weight matrix W_hh across 100 steps causes gradients to decay exponentially (e.g. 0.9^100 ≈ 0.00002). The model becomes virtually blind to words from the beginning of the sentence.",
      hint: "Multiply a number slightly smaller than 1.0 by itself 100 times."
    },
    {
      id: '05-b3',
      difficulty: 'basic',
      question: "What is the intuitive role of the Forget Gate (f_t) in an LSTM neuron?",
      options: [
        { text: "It decides what percentage of existing cell state memory to erase or retain (using a sigmoid 0 to 1 value).", correct: true },
        { text: "It deletes the dataset after training finishes.", correct: false },
        { text: "It resets the model's weights to zero randomly.", correct: false },
        { text: "It shuts down the computer when loss converges.", correct: false }
      ],
      explanation: "Right on! If a sentence shifts topics (e.g., from 'The boy...' to 'The girls...'), the forget gate squashes old singular subject information to near 0 so new plural subject context can be stored.",
      hint: "A dimmer switch between 0.0 (dump all) and 1.0 (keep everything)."
    },
    {
      id: '05-h1',
      difficulty: 'hard',
      question: "Why does the LSTM Constant Error Carousel (CEC) prevent the vanishing gradient catastrophe that crippled vanilla RNNs?",
      options: [
        { text: "It uses larger learning rates.", correct: false },
        { text: "The cell state updates additively (c_t = f_t · c_{t-1} + i_t · c̃_t), meaning the gradient flow ∂c_t / ∂c_{t-1} = f_t (approx 1.0 when forget gate is open), bypassing exponential matrix decay.", correct: true },
        { text: "It skips all matrix multiplications.", correct: false },
        { text: "LSTMs only process the first 5 words of any sentence.", correct: false }
      ],
      explanation: "Bullseye! In vanilla RNNs, backpropagating through time repeatedly multiplies by W_hh, causing exponential decay (e.g. 0.85^50 ≈ 0.0003). In an LSTM, the cell state is a linear highway where error signals travel undiminished.",
      hint: "Additive update paths bypass repeated matrix multiplication chains."
    },
    {
      id: '05-h2',
      difficulty: 'hard',
      question: "What critical architectural bottleneck did Bahdanau Additive Attention (2014) resolve in Seq2Seq models?",
      options: [
        { text: "It eliminated the need for training data.", correct: false },
        { text: "Traditional Seq2Seq compressed an entire 50-word sentence into a single fixed-length vector h_T; Bahdanau attention allows the decoder to dynamically query and spotlight all source words.", correct: true },
        { text: "It allowed recurrent networks to run without activations.", correct: false },
        { text: "It translated text into binary ASCII directly.", correct: false }
      ],
      explanation: "Outstanding! Compressing a complex sentence into a single vector created a memory bottleneck. Bahdanau attention acts as an associative searchlight, letting the decoder focus on relevant source words at every decoding step.",
      hint: "Squeezing an entire encyclopedia into a single short tweet."
    },
    {
      id: '05-h3',
      difficulty: 'hard',
      question: "How does the Gated Recurrent Unit (GRU, Cho et al. 2014) streamline the LSTM architecture?",
      options: [
        { text: "It merges cell and hidden state into one, and couples forget and input gates into a single update gate (z_t), reducing parameter count.", correct: true },
        { text: "It replaces gates with convolutional filters.", correct: false },
        { text: "It removes all non-linear activation functions.", correct: false },
        { text: "It can only process numbers, not text.", correct: false }
      ],
      explanation: "Precisely! GRU combines c_t and h_t into a single hidden vector and uses update gate z_t where (1 - z_t) dictates forgetting and z_t dictates new writing, cutting down parameters and speeding execution.",
      hint: "Two gates instead of three, with no separate cell state conveyor belt."
    },
    {
      id: '05-p1',
      difficulty: 'pro',
      question: "In autoregressive RNN training with Teacher Forcing, what is the phenomenon of 'Exposure Bias'?",
      options: [
        { text: "Overheating of GPU cores during long sequence backprop.", correct: false },
        { text: "During training the model always sees ground-truth previous tokens, but during inference it feeds its own predictions, leading to compounding error divergence on mistakes.", correct: true },
        { text: "Privacy leaks revealing training data to web users.", correct: false },
        { text: "A bug where dropout is not disabled during eval mode.", correct: false }
      ],
      explanation: "Brilliant! Bengio et al. (2015) identified Exposure Bias: because the model is trained with training wheels (true targets fed at every step), a single mistake at generation time leads into unseen state space where errors compound catastrophically.",
      hint: "What happens to a student guided by the teacher's hand on every stroke when tested alone?"
    },
    {
      id: '05-p2',
      difficulty: 'pro',
      question: "How do modern Selective State Space Models (Mamba, Gu & Dao 2023) achieve linear O(T) sequence scaling while enabling parallel GPU training?",
      options: [
        { text: "By unrolling linear time-invariant recurrent dynamics into a global 1D convolution for training, while operating as an O(1) state-space recurrence during token generation.", correct: true },
        { text: "By discarding 90% of tokens in the context window.", correct: false },
        { text: "By replacing PyTorch with pure C assembly.", correct: false },
        { text: "By training without any gradient descent.", correct: false }
      ],
      explanation: "Masterful! Gu & Dao formulated SSMs so that the recurrent equation can be computed in parallel across an entire sequence during training using parallel associative scan, yet yields O(1) memory per step during inference!",
      hint: "Parallel scan algorithms allow linear recurrence to run like a convolution across time."
    }
  ],

  // ==========================================
  // MODULE 06: Transformers & Self-Attention
  // ==========================================
  '06': [
    {
      id: '06-b1',
      difficulty: 'basic',
      question: "In plain English, what does the 'Attention' mechanism allow tokens in a sentence to do?",
      options: [
        { text: "Look at and blend relevant information from every other token in the sequence simultaneously based on contextual relevance.", correct: true },
        { text: "Translate text directly into morse code.", correct: false },
        { text: "Delete punctuation marks from the input text.", correct: false },
        { text: "Check grammar rules against a dictionary.", correct: false }
      ],
      explanation: "Spot on! In 'The bank of the river', the token 'bank' pays high attention to 'river', allowing the model to know it refers to land by water rather than a financial vault.",
      hint: "Dynamic spotlighting: every word can look at all other words to decipher meaning."
    },
    {
      id: '06-b2',
      difficulty: 'basic',
      question: "In Self-Attention, what do Query (Q), Key (K), and Value (V) represent intuitively?",
      options: [
        { text: "Query is what you're looking for, Key is what tokens advertise, and Value is the actual payload content retrieved.", correct: true },
        { text: "Three cryptographic encryption keys used for security.", correct: false },
        { text: "The three color channels (Red, Green, Blue) of token embeddings.", correct: false },
        { text: "Quantization, Kernel, and Vector settings.", correct: false }
      ],
      explanation: "Exactly! Like a search engine: your search text is the Query. It matches against video titles (Keys). Once a match is scored, you watch the actual video content (Value).",
      hint: "Think of YouTube search: search query matches video title keys to stream the video values."
    },
    {
      id: '06-b3',
      difficulty: 'basic',
      question: "Why do Transformers require Positional Embeddings?",
      options: [
        { text: "Because self-attention is permutation-invariant (order-agnostic); without position encoding, 'cat ate mouse' and 'mouse ate cat' look identical.", correct: true },
        { text: "To compress token vectors into 8-bit integers.", correct: false },
        { text: "To prevent GPU out-of-memory errors.", correct: false },
        { text: "To make attention matrices symmetric.", correct: false }
      ],
      explanation: "Right on! Attention treats inputs as an unordered set of tokens. Adding or injecting positional markers tells the model which token came first, second, or hundredth.",
      hint: "Without position tags, word order is completely lost."
    },
    {
      id: '06-h1',
      difficulty: 'hard',
      question: "Why must the dot product Q · K^T be scaled by 1 / sqrt(d_k) in Scaled Dot-Product Attention (Vaswani et al. 2017)?",
      options: [
        { text: "To make the matrix square.", correct: false },
        { text: "For large dimensions d_k, the dot product variance scales to d_k, pushing the Softmax function into flat saturation zones with near-zero gradients.", correct: true },
        { text: "Because square roots run faster on tensor cores.", correct: false },
        { text: "To prevent memory leaks in PyTorch.", correct: false }
      ],
      explanation: "Precisely! If Q and K have mean 0 and variance 1, their dot product has variance d_k. When d_k = 128, values can reach ±30, causing Softmax to output 1.0 and 0.0 with virtually zero gradient flow. Dividing by sqrt(d_k) restores unit variance.",
      hint: "Summing d_k independent random product terms multiplies variance by d_k."
    },
    {
      id: '06-h2',
      difficulty: 'hard',
      question: "What is the key advantage of Grouped-Query Attention (GQA, Ainslie et al. 2023) used in modern LLMs like LLaMA 3?",
      options: [
        { text: "It reduces the KV Cache memory footprint during inference by having multiple query heads share a smaller number of key/value heads.", correct: true },
        { text: "It doubles the parameter count of the model.", correct: false },
        { text: "It eliminates the need for feed-forward MLP layers.", correct: false },
        { text: "It replaces Softmax with ReLU.", correct: false }
      ],
      explanation: "Outstanding! During autoregressive decoding, storing past Keys and Values in GPU VRAM (KV Cache) is the primary throughput bottleneck. GQA groups e.g. 8 Q heads per 1 KV head, slashing KV cache memory by 8x with minimal loss in model capability.",
      hint: "Sharing key and value heads across multiple query heads to save VRAM during generation."
    },
    {
      id: '06-h3',
      difficulty: 'hard',
      question: "What is the time and memory complexity of standard full Self-Attention with sequence length N and embedding dimension D?",
      options: [
        { text: "O(N² · D) time and O(N²) attention matrix memory.", correct: true },
        { text: "O(N · D) time and O(N) memory.", correct: false },
        { text: "O(N³) time and O(D²) memory.", correct: false },
        { text: "O(log N) time and O(1) memory.", correct: false }
      ],
      explanation: "Bullseye! Calculating the Q · K^T matrix requires comparing every token to all N tokens, producing an N × N score matrix that scales quadratically with sequence length.",
      hint: "Every token attends to every token, forming an N-by-N attention grid."
    },
    {
      id: '06-p1',
      difficulty: 'pro',
      question: "How does Rotary Position Embedding (RoPE, Su et al. 2021) mathematically encode relative token distance without absolute position lookup tables?",
      options: [
        { text: "By adding sinusoidal waves directly to the word embeddings.", correct: false },
        { text: "By rotating query and key vector pairs in 2D coordinate planes by angle m·θ; the dot product <R_m q, R_n k> naturally collapses into a function depending strictly on relative distance (n - m).", correct: true },
        { text: "By measuring the character count of each word.", correct: false },
        { text: "By sorting all tokens alphabetically.", correct: false }
      ],
      explanation: "Brilliant! RoPE uses complex rotation matrices: R_m^T R_n = R_{n-m}. This guarantees that the attention dot product depends only on how far apart two tokens are, not where they sit in absolute sequence space!",
      hint: "Rotation in the complex plane: e^(i m θ) * e^(-i n θ) = e^(i (m-n) θ)."
    },
    {
      id: '06-p2',
      difficulty: 'pro',
      question: "In FlashAttention (Dao et al. 2022), how is the attention matrix computed without ever materializing the massive N × N attention score matrix in GPU HBM?",
      options: [
        { text: "By tiling inputs into fast on-chip SRAM blocks, computing softmax incrementally via online softmax normalization, and fusing the backward pass.", correct: true },
        { text: "By approximating attention with random Fourier features.", correct: false },
        { text: "By converting all floating point numbers to 1-bit booleans.", correct: false },
        { text: "By pruning 90% of attention weights randomly.", correct: false }
      ],
      explanation: "Masterful! Tri Dao realized that moving data between slow GPU High Bandwidth Memory (HBM) and fast SRAM was the true bottleneck. By tiling Q, K, and V and computing the online softmax running max and sum, FlashAttention avoids writing the N×N matrix to memory entirely!",
      hint: "Online softmax algorithm allows computing exact softmax in chunks inside SRAM."
    }
  ],

  // ==========================================
  // MODULE 07: Generative Models & Diffusion
  // ==========================================
  '07': [
    {
      id: '07-b1',
      difficulty: 'basic',
      question: "What is the fundamental intuition behind Diffusion generative models (like Stable Diffusion)?",
      options: [
        { text: "Gradually add noise to an image until it turns into pure static, then train a neural network to reverse the noise step-by-step to generate new images.", correct: true },
        { text: "Copy and paste random pixels from Google Images.", correct: false },
        { text: "Compress photos into zip files.", correct: false },
        { text: "Search a database for the closest photograph and display it.", correct: false }
      ],
      explanation: "Spot on! Think of dropping food coloring into a glass of water until thoroughly dispersed (forward process), and then training a miraculous AI wand that can pull the dye particles back into sharp, beautiful patterns (reverse process).",
      hint: "Adding noise forward, learning to denoise backward."
    },
    {
      id: '07-b2',
      difficulty: 'basic',
      question: "Why do modern diffusion models (Latent Diffusion) operate in a compressed 'latent space' rather than raw pixel space?",
      options: [
        { text: "A VAE compresses high-res pixels into a compact feature representation (e.g. 8x smaller), making diffusion 64x faster and memory-efficient.", correct: true },
        { text: "Pixel space cannot represent color.", correct: false },
        { text: "Diffusion only works on grayscale images.", correct: false },
        { text: "To prevent users from viewing intermediate steps.", correct: false }
      ],
      explanation: "Correct! Rombach et al. (2022) realized high-frequency pixel details are computationally wasteful for the core generative process. Compressing 512x512 pixels to 64x64 latent codes saves massive compute while retaining perceptual quality.",
      hint: "Generating in a compressed 64x64 feature space instead of 512x512 raw pixels."
    },
    {
      id: '07-b3',
      difficulty: 'basic',
      question: "What is Classifier-Free Guidance (CFG) in text-to-image models?",
      options: [
        { text: "A technique that balances prompt adherence vs visual diversity by combining conditional and unconditional predictions.", correct: true },
        { text: "A way to train image generators without using any GPUs.", correct: false },
        { text: "An automatic copyright detection filter.", correct: false },
        { text: "A feature that forces all generated images to be square.", correct: false }
      ],
      explanation: "Exactly! CFG scales the prompt signal: ε_guided = ε_uncond + s · (ε_text - ε_uncond). Increasing the scale s forces the generator to adhere tightly to the prompt description.",
      hint: "Amplifying the text direction vector relative to the unconditioned baseline."
    },
    {
      id: '07-h1',
      difficulty: 'hard',
      question: "Why did modern generative AI (Flux.1, Stable Diffusion 3, AlphaFold 3) transition from DDPM Diffusion to Optimal Transport Flow Matching?",
      options: [
        { text: "Flow Matching models require no GPUs and can train on standard CPUs.", correct: false },
        { text: "Flow Matching generates straight, deterministic probability trajectories via Optimal Transport, enabling 10x faster ODE generation in 10-20 steps instead of noisy Brownian SDE curves.", correct: true },
        { text: "Diffusion models cannot generate 2D or 3D images.", correct: false },
        { text: "Flow Matching completely eliminates all neural network weights.", correct: false }
      ],
      explanation: "Spot on! Diffusion models generate curved, stochastic paths driven by Brownian motion, which requires 50-1000 Langevin steps. Flow Matching straightens trajectories so standard Runge-Kutta ODE solvers can generate samples in just 10-20 steps!",
      hint: "Straight probability paths allow ODE solvers to take giant, straight steps."
    },
    {
      id: '07-h2',
      difficulty: 'hard',
      question: "In DDPM forward noise corruption q(x_t | x_0), what allows sampling at arbitrary timestep t in a single closed-form step without running steps 1...t sequentially?",
      options: [
        { text: "Linear algebra matrix inversion.", correct: false },
        { text: "The sum of independent Gaussian random variables is analytically another Gaussian, parameterized directly by cumulative signal retention ᾱ_t = ∏(1 - β_s).", correct: true },
        { text: "Backpropagation through time.", correct: false },
        { text: "Rejection sampling with Monte Carlo rollouts.", correct: false }
      ],
      explanation: "Exactly! Because the convolution of Gaussians is Gaussian, the cumulative variance across t steps collapses analytically to x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε.",
      hint: "Convolving independent normal distributions produces another normal distribution in closed form."
    },
    {
      id: '07-h3',
      difficulty: 'hard',
      question: "What is Tweedie's Formula and why is it central to empirical Bayes denoising in diffusion models?",
      options: [
        { text: "It expresses the posterior mean E[x_0 | x_t] directly in terms of the observation x_t and the score function ∇_x log p(x_t).", correct: true },
        { text: "It calculates the optimal learning rate for Adam.", correct: false },
        { text: "It compresses JPEG images by 50%.", correct: false },
        { text: "It guarantees that loss is strictly convex.", correct: false }
      ],
      explanation: "Bullseye! Tweedie's formula reveals that the best estimate of the clean data x_0 given noisy measurement x_t is x_0 ≈ (x_t + σ² ∇_x log p(x_t)). This unifies score matching directly with denoising!",
      hint: "Estimating the clean signal directly from the score of the marginal density."
    },
    {
      id: '07-p1',
      difficulty: 'pro',
      question: "Under Song et al. (2020) Score-Based Generative Modeling, how are DDPM and continuous-time SDEs unified?",
      options: [
        { text: "Forward diffusion corresponds to an Ito SDE, and reversing it corresponds to a reverse-time SDE or deterministic Probability Flow ODE whose drift is governed by score ∇_x log p_t(x).", correct: true },
        { text: "They proved diffusion models cannot generate high-fidelity images.", correct: false },
        { text: "By showing that diffusion and GANs have identical loss functions.", correct: false },
        { text: "By removing time dependence completely.", correct: false }
      ],
      explanation: "Masterful! Yang Song proved that predicting noise ε_θ in DDPM is mathematically identical to estimating the score function ∇_x log p_t(x). The deterministic Probability Flow ODE allows exact likelihood computation via continuous normalizing flows!",
      hint: "Reversing a stochastic differential equation requires only the score of the marginal distribution."
    },
    {
      id: '07-p2',
      difficulty: 'pro',
      question: "What is the training objective of Conditional Flow Matching (Lipman et al. 2023) with target vector field u_t(x | x_1)?",
      options: [
        { text: "E_{t, x_0, x_1} || v_θ(x_t, t) - u_t(x_t | x_1) ||², directly regressing model velocity onto constant vector fields connecting noise to data.", correct: true },
        { text: "Minimizing the Wasserstein distance via adversarial discriminator min-max game.", correct: false },
        { text: "Computing exact second-order Hessian matrix inversions.", correct: false },
        { text: "Maximizing the mutual information between RGB channels.", correct: false }
      ],
      explanation: "Brilliant! Conditional Flow Matching regresses a neural network vector field v_θ directly onto the straight conditional velocity u_t = x_1 - (1 - σ_min) x_0 without needing expensive marginal drift simulation!",
      hint: "Simple L2 regression between predicted velocity and the straight-line target displacement."
    }
  ],

  // ==========================================
  // MODULE 08: Modern Architectures & RLHF
  // ==========================================
  '08': [
    {
      id: '08-b1',
      difficulty: 'basic',
      question: "What is a 'Sparse Mixture of Experts' (MoE) model (like Mixtral or DeepSeek-V3)?",
      options: [
        { text: "An architecture where different sub-networks ('experts') specialize in different domains, and a router activates only a few experts per token.", correct: true },
        { text: "A team of human programmers who manually edit weights.", correct: false },
        { text: "A model that can only run on quantum processors.", correct: false },
        { text: "A single tiny network trained on 100 different datasets sequentially.", correct: false }
      ],
      explanation: "Correct! Instead of firing all 500 billion parameters on every word, an MoE router activates only 2 out of 64 experts per token. You get the knowledge capacity of a giant model with the inference speed of a small one!",
      hint: "A triage nurse directing patients to cardiology or orthopedics instead of all doctors seeing every patient."
    },
    {
      id: '08-b2',
      difficulty: 'basic',
      question: "What is the primary purpose of RLHF (Reinforcement Learning from Human Feedback)?",
      options: [
        { text: "Aligning raw autocomplete text models with human intentions to make them helpful, harmless, and conversational.", correct: true },
        { text: "Teaching neural networks how to play chess only.", correct: false },
        { text: "Translating code into machine binary.", correct: false },
        { text: "Reducing the electricity consumption of GPUs.", correct: false }
      ],
      explanation: "Spot on! Raw base models just predict what internet word comes next (often rambling or repeating toxic tropes). RLHF steers the model to behave as an honest, helpful assistant.",
      hint: "Turning an untamed autocomplete engine into an attentive assistant."
    },
    {
      id: '08-b3',
      difficulty: 'basic',
      question: "Why do reasoning models (like DeepSeek-R1 or OpenAI o1) perform so much better on complex math and coding?",
      options: [
        { text: "They generate extended 'thinking tokens' (Chain of Thought), testing hypotheses and self-correcting mistakes before answering.", correct: true },
        { text: "They look up answers directly on Google during inference.", correct: false },
        { text: "They have 1000x higher learning rates.", correct: false },
        { text: "They bypass all neural network layers.", correct: false }
      ],
      explanation: "Exactly! Instead of guessing the immediate next word for a tough question, reasoning models 'think out loud' across hundreds of tokens, backtracking when they hit logical dead ends.",
      hint: "Working out intermediate scratchpad calculations before blurting out the final number."
    },
    {
      id: '08-h1',
      difficulty: 'hard',
      question: "In Sparse Mixture of Experts (MoE), what devastating pathology occurs if the auxiliary load balancing loss α · N · ∑(f_i · P_i) is omitted?",
      options: [
        { text: "The model runs out of memory immediately.", correct: false },
        { text: "Router Collapse: a self-reinforcing feedback loop routes nearly all tokens to 1 or 2 favorite experts while other experts starve and receive zero gradients.", correct: true },
        { text: "All weights instantly become NaN.", correct: false },
        { text: "The model switches automatically from PyTorch to TensorFlow.", correct: false }
      ],
      explanation: "Bullseye! Without load balancing, the router quickly picks early 'favorite' experts that gain better gradients, attracting even more tokens in a vicious cycle until 80%+ of experts are completely idle.",
      hint: "Early lucky experts get better gradients and hog all the traffic."
    },
    {
      id: '08-h2',
      difficulty: 'hard',
      question: "What is the primary breakthrough of DeepSeek-R1's Group Relative Policy Optimization (GRPO) over standard PPO RLHF?",
      options: [
        { text: "It uses larger learning rates.", correct: false },
        { text: "It completely eliminates the memory-heavy Critic/Value neural network by sampling groups of candidate completions and calculating relative advantages within the group.", correct: true },
        { text: "It trains without any prompts or data.", correct: false },
        { text: "It only works on chess engines.", correct: false }
      ],
      explanation: "Precisely! PPO requires maintaining a Critic network the same size as the Policy LLM (doubling GPU VRAM). GRPO samples G rollouts and normalizes advantages directly as A_i = (r_i - mean(r)) / (std(r) + eps), saving enormous compute!",
      hint: "Evaluating advantages relative to group peers instead of training a separate value critic network."
    },
    {
      id: '08-h3',
      difficulty: 'hard',
      question: "How does Direct Preference Optimization (DPO, Rafailov et al. 2023) simplify alignment compared to traditional PPO RLHF?",
      options: [
        { text: "It derives an exact closed-form mapping between the reward function and optimal policy, optimizing preferences with a simple binary cross-entropy loss without explicit RL loops.", correct: true },
        { text: "It eliminates the need for human preference pairs.", correct: false },
        { text: "It trains purely through supervised next-token prediction on Wikipedia.", correct: false },
        { text: "It only works on computer vision models.", correct: false }
      ],
      explanation: "Outstanding! Rafailov et al. showed that the Bradley-Terry preference model can be reparameterized directly in terms of the language model probabilities: r(x,y) = β log(π_θ(y|x) / π_ref(y|x)), discarding the need for an explicit reward model or complex PPO actor-critic loop.",
      hint: "Reparameterizing the reward function directly using the policy's log-likelihood ratios."
    },
    {
      id: '08-p1',
      difficulty: 'pro',
      question: "In DeepSeek-V3/R1, how does Multi-Head Latent Attention (MLA) compress the KV Cache by over 90% while maintaining expressivity?",
      options: [
        { text: "It low-rank compresses Keys and Values into a single latent vector c_t^{KV} before caching, absorbing projection matrices into weights during query dot products.", correct: true },
        { text: "It quantizes all KV values into 1-bit booleans.", correct: false },
        { text: "It removes self-attention from all intermediate layers.", correct: false },
        { text: "It caches only the first 10 tokens of every document.", correct: false }
      ],
      explanation: "Masterful! Traditional GQA caches high-dimensional vectors for every head. MLA projects K and V down to a tiny 512-dim latent space. During generation, only this compressed latent vector is cached, slashing memory while retaining full multi-head expressivity through matrix absorption!",
      hint: "Low-rank projection down to a compact latent vector combined with matrix absorption."
    },
    {
      id: '08-p2',
      difficulty: 'pro',
      question: "What did the emergence of reasoning in DeepSeek-R1-Zero demonstrate about large-scale Reinforcement Learning on cold-start models?",
      options: [
        { text: "Reasoning behaviors (self-reflection, backtracking, alternative exploration) spontaneously emerge through pure rule-based RL search optimization without human demonstration data.", correct: true },
        { text: "Supervised fine-tuning on human reasoning steps is strictly necessary.", correct: false },
        { text: "Models cannot learn math without human teachers writing step-by-step guides.", correct: false },
        { text: "Reinforcement learning always degrades conversational fluency.", correct: false }
      ],
      explanation: "Brilliant! R1-Zero was rewarded purely for getting math/code compiler checks right without any initial human demonstration data. As training progressed, the model spontaneously invented 'Wait, let me double check...' internal verification, proving search creates reasoning!",
      hint: "The famous 'Aha moment' where the policy learned to verify its own derivations purely for reward."
    }
  ],

  // ==========================================
  // MODULE 09: Mechanistic Interpretability
  // ==========================================
  '09': [
    {
      id: '09-b1',
      difficulty: 'basic',
      question: "What is the mission of 'Mechanistic Interpretability' in AI safety research?",
      options: [
        { text: "To reverse-engineer the internal weights and activations of neural networks into human-understandable circuits and algorithms.", correct: true },
        { text: "To make neural networks compute answers faster on GPUs.", correct: false },
        { text: "To write user manuals for AI chatbots.", correct: false },
        { text: "To replace artificial intelligence with mechanical clockwork gears.", correct: false }
      ],
      explanation: "Spot on! Rather than treating deep networks as mysterious black boxes, mechanistic interpretability treats them like compiled computer programs, decompiling neural circuits to understand how decisions are formed.",
      hint: "Decompiling the black box into readable algorithmic circuits."
    },
    {
      id: '09-b2',
      difficulty: 'basic',
      question: "What is a 'polysemantic neuron' in deep learning?",
      options: [
        { text: "A single neuron that activates for multiple completely unrelated concepts (e.g. academic citations, cats, and Arabic text).", correct: true },
        { text: "A neuron that speaks multiple human languages fluently.", correct: false },
        { text: "A neuron that has more than 100 incoming connections.", correct: false },
        { text: "A dead neuron that never fires.", correct: false }
      ],
      explanation: "Correct! 'Poly' = many, 'semantic' = meaning. Because models have more concepts to represent than physical neurons, they compress features into overlapping directions, making individual neurons look like scrambled nonsense.",
      hint: "One neuron pulling double or triple duty for unrelated topics."
    },
    {
      id: '09-b3',
      difficulty: 'basic',
      question: "What does an 'Induction Head' do in a Transformer?",
      options: [
        { text: "It recognizes repeating patterns like [A][B] ... [A] and predicts that [B] is about to follow.", correct: true },
        { text: "It charges the GPU battery using wireless induction.", correct: false },
        { text: "It compresses token embeddings to 4-bit precision.", correct: false },
        { text: "It prevents models from generating repetitive sentences.", correct: false }
      ],
      explanation: "Right on! If a document mentions 'Harry Potter ... Harry', an induction head remembers that 'Potter' followed 'Harry' earlier and completes the name. It is the primary micro-engine of in-context few-shot learning!",
      hint: "Pattern completion: [A][B] ... [A] -> [B]."
    },
    {
      id: '09-h1',
      difficulty: 'hard',
      question: "In Anthropic's Transformer Circuits framework, how does the 2-layer Induction Head circuit perform in-context pattern copying [A][B] ... [A] -> [B]?",
      options: [
        { text: "Through a hardcoded Python regex in the tokenizer.", correct: false },
        { text: "Layer 0 previous-token head transfers token A's identity into token B's residual stream; Layer 1 induction head queries for 'preceded by A', attends to B, and the OV circuit boosts B's logit.", correct: true },
        { text: "By storing all previous tokens in an external SQL database.", correct: false },
        { text: "By using recurrent feedback loops between layers.", correct: false }
      ],
      explanation: "Brilliant! This microscopic 2-layer circuit is the fundamental engine of in-context learning in modern Transformers, emerging during the sudden macroscopic 'induction bump' phase change.",
      hint: "A Previous-Token Head in Layer 0 composes with an Attention Head in Layer 1."
    },
    {
      id: '09-h2',
      difficulty: 'hard',
      question: "Why do Sparse Autoencoders (SAEs) enforce a strict unit norm constraint ||W_dec, i||_2 = 1.0 on decoder dictionary columns?",
      options: [
        { text: "To prevent GPU floating point overflow.", correct: false },
        { text: "To prevent the network from trivially shrinking activation features f_i to cheat the L1 sparsity loss while blowing up decoder weights to compensate.", correct: true },
        { text: "To make the matrix symmetric.", correct: false },
        { text: "It is required by the Adam optimizer.", correct: false }
      ],
      explanation: "Exact! If decoder weights were unconstrained, the model could scale W_dec by 1000x and scale feature activations by 1/1000x, achieving near-zero L1 loss without actual sparsity. Unit norm columns enforce genuine sparse dictionary learning!",
      hint: "Without fixed column norms, activations could shrink to zero to bypass the L1 penalty."
    },
    {
      id: '09-h3',
      difficulty: 'hard',
      question: "What is the mathematical definition of 'Superposition' (Elhage et al. 2022) in neural representations?",
      options: [
        { text: "Representing more almost-orthogonal features than the physical dimension of the activation space by exploiting high-dimensional geometry and feature sparsity.", correct: true },
        { text: "Running two models simultaneously on dual GPUs.", correct: false },
        { text: "Adding two images together pixel by pixel.", correct: false },
        { text: "A quantum physics effect that only occurs in optical neural networks.", correct: false }
      ],
      explanation: "Bullseye! The Johnson-Lindenstrauss lemma implies high-dimensional vector spaces can accommodate an exponential number of almost-orthogonal vectors. If features are sparse (rarely fire at the same time), interference is minimal!",
      hint: "Fitting 5000 sparse concepts into 512 dimensions using almost-orthogonal vector directions."
    },
    {
      id: '09-p1',
      difficulty: 'pro',
      question: "In modern Sparse Autoencoders (SAEs), what is 'shrinkage bias' caused by L1 regularization, and how does Top-K or JumpReLU resolve it?",
      options: [
        { text: "L1 penalty constantly pulls true feature magnitudes toward zero; Top-K SAEs activate only the top-K latents without an L1 penalty, preserving unbiased feature amplitudes.", correct: true },
        { text: "L1 causes model weights to become negative; Top-K takes absolute values.", correct: false },
        { text: "Shrinkage bias causes GPU memory leaks during training.", correct: false },
        { text: "JumpReLU doubles the number of dictionary features.", correct: false }
      ],
      explanation: "Masterful! OpenAI (Gao et al. 2024) and DeepMind (Rajamanoharan et al. 2024) discovered L1 regularization acts like a constant drag force on active features, under-reconstructing true activations. Top-K SAEs enforce hard sparsity without any shrinkage bias!",
      hint: "L1 loss subtracts a constant gradient penalty from active features, shrinking their true scale."
    },
    {
      id: '09-p2',
      difficulty: 'pro',
      question: "According to mathematical mechanistic interpretability, how do the QK (Query-Key) circuit and the OV (Output-Value) circuit strictly differ in function?",
      options: [
        { text: "The QK circuit (W_Q^T W_K) determines WHERE to route attention (the N×N score pattern), while the OV circuit (W_O W_V) determines WHAT information is read and written into the residual stream.", correct: true },
        { text: "QK is for English text; OV is for code.", correct: false },
        { text: "QK computes gradients; OV computes activations.", correct: false },
        { text: "They perform identical mathematical operations on the same tensors.", correct: false }
      ],
      explanation: "Brilliant! Elhage et al. showed that an attention head decomposes into two independent bilinear operators: QK determines the routing graph (which source token gets attended to), while OV determines the linear feature transformation applied to the attended token's representation!",
      hint: "One decides who talks to whom; the other decides what message is transmitted."
    }
  ]
};
