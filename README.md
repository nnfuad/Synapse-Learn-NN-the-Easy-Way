# 🧠 Project "Synapse": The Graphical Neural Network Explainer
### *From Historical Genesis (1943) & Zero to State-of-the-Art AI Research (2026)*

Welcome to **Project Synapse**, an interactive, graphical, dual-tier pedagogical platform designed to take any student from zero foundational knowledge to the frontier of deep learning and state-of-the-art research.

Every single module is built around **The Pedagogical Triad**:
1. **The Intuitive Physical Analogy**: Grounding abstract mathematics into tangible, sensory metaphors.
2. **The 60 FPS Graphical Engine**: Interactive visualizers, live vector fields, 2D decision hyperplanes, loss landscapes, and space-warping canvases.
3. **Dual-Tier Runnable Code**:
   - **Tier 1 (Scratch)**: Zero-dependency pure Python and NumPy arithmetic exposing every calculation.
   - **Tier 2 (Production SOTA)**: Modern, idiomatic PyTorch 2.x and distributed AI architecture.

---

## 🚀 Quick Start: Launching the Interactive Web Explainer

The platform includes a high-class, responsive, dark-mode interactive web application:

```bash
cd web
npm install
npm run dev
```

Open your browser to: **`http://localhost:5173`**

### Live Interactive Features in Module 00:
- **Historical Odyssey (1943 → 2026)**: An interactive timeline detailing the 80-year evolution of neural computing (McCulloch-Pitts, Rosenblatt, Minsky-Papert XOR barrier, Rumelhart Backprop, AlexNet, Transformers, DeepSeek Reasoning).
- **1943 McCulloch-Pitts Logic Simulator**: Toggle biological excitatory inputs and emergency inhibitory vetoes; watch action potentials spike in real-time.
- **1957 Perceptron 2D Playground**: A 60 FPS canvas with live decision boundary line $w_1 x_1 + w_2 x_2 + b = 0$, clickable points, real-time normal vector arrow, learning rate adjustments, and auto-training.
- **1986 XOR Space Warp Visualizer**: Drag a slider to witness how adding a non-linear hidden layer twists and folds the 2D coordinate space, transforming the impossible XOR problem into clean linear separation.
- **Code Laboratory**: Live side-by-side code viewer for NumPy scratch vs PyTorch production code.
- **Concept Quiz**: Interactive knowledge check with instant feedback and explanations.

---

## 🐍 Python & PyTorch Laboratory

All Python scripts are self-contained, heavily documented, and runnable directly:

```bash
# 1. Install dependencies
pip install -r python/requirements.txt

# 2. Run Part 1: 1943 McCulloch-Pitts Logic Gates
python3 python/module00_history_and_perceptron/mcculloch_pitts.py

# 3. Run Part 2: 1957 Rosenblatt Perceptron Learning Rule
python3 python/module00_history_and_perceptron/rosenblatt_perceptron.py

# 4. Run Part 3: The 1969 XOR Crisis & First AI Winter
python3 python/module00_history_and_perceptron/xor_crisis.py

# 5. Run Part 4: The 1986 Multilayer Perceptron (MLP) Resolution
python3 python/module00_history_and_perceptron/mlp_solution_pytorch.py

# 6. Run Module 00: Automated pytest verification suite
pytest python/module00_history_and_perceptron/test_module00.py

# 7. Run Module 01: Computational Graphs, Autograd & Reverse Chain Rule
python3 python/module01_computational_graphs_autograd/autograd_engine.py
python3 python/module01_computational_graphs_autograd/pytorch_autograd_comparison.py
pytest python/module01_computational_graphs_autograd/test_module01.py

# 8. Run Module 02: Loss Geometry, Optimization & SOTA Optimizers
python3 python/module02_optimization_and_loss_landscapes/optimizers_scratch.py
python3 python/module02_optimization_and_loss_landscapes/loss_landscapes.py
python3 python/module02_optimization_and_loss_landscapes/pytorch_optimizers.py
pytest python/module02_optimization_and_loss_landscapes/test_module02.py

# 9. Run Module 05: Sequential Dynamics & Attention Spark
python3 python/module05_sequential_processing_and_attention/lstm_scratch.py
python3 python/module05_sequential_processing_and_attention/bahdanau_attention_pytorch.py
pytest python/module05_sequential_processing_and_attention/test_module05.py

# 10. Run Module 06: The Modern Transformer Revolution & Attention Mechanics
python3 python/module06_transformers_and_attention/attention_scratch.py
python3 python/module06_transformers_and_attention/rope_and_kv_cache.py
python3 python/module06_transformers_and_attention/transformer_block_pytorch.py
pytest python/module06_transformers_and_attention/test_module06.py

# 11. Run the full unified test suite across all modules (79 passing tests)
pytest -v

# 12. Run Module 07: Generative Paradigms (DDPM & Optimal Transport Flow Matching)
python3 python/module07_generative_diffusion_flow/diffusion_ddpm_scratch.py
python3 python/module07_generative_diffusion_flow/flow_matching_toy.py
pytest python/module07_generative_diffusion_flow/test_module07.py

# 13. Run Module 08: Modern LLMs, Alignment & Reasoning (MoE, DPO, GRPO)
python3 python/module08_llm_reasoning_and_alignment/moe_routing_scratch.py
python3 python/module08_llm_reasoning_and_alignment/dpo_and_grpo_loss.py
pytest python/module08_llm_reasoning_and_alignment/test_module08.py

# 14. Run Module 09: Mechanistic Interpretability & Frontiers (SAEs & Induction Heads)
python3 python/module09_mechanistic_interpretability/sparse_autoencoder_sae.py
python3 python/module09_mechanistic_interpretability/circuit_induction_heads.py
pytest python/module09_mechanistic_interpretability/test_module09.py
```

---

## 📓 Companion Jupyter Notebooks

For interactive exploratory data science and plotting:
```bash
jupyter notebook notebooks/00_genesis_and_perceptrons.ipynb
jupyter notebook notebooks/01_computational_graphs_and_autograd.ipynb
jupyter notebook notebooks/02_optimization_and_loss_landscapes.ipynb
jupyter notebook notebooks/05_sequential_dynamics_and_attention.ipynb
jupyter notebook notebooks/06_transformers_and_attention_mechanics.ipynb
jupyter notebook notebooks/07_diffusion_and_flow_matching.ipynb
jupyter notebook notebooks/08_reasoning_and_alignment.ipynb
jupyter notebook notebooks/09_mechanistic_interpretability.ipynb
```

---

## 🗺️ Master Curriculum Roadmap (Zero to SOTA)

| Stage | Module Name | Era | Core Physical Analogy | Visualizer |
| :---: | :--- | :---: | :--- | :--- |
| **00** | **The Genesis & The Atomic Neuron** *(Completed)* | 1943–1969 | The Audio Mixer Knob & Threshold Switch | 2D Decision Hyperplane + XOR Space Warp |
| **01** | **Computational Graphs & Autograd** *(Completed)* | 1970–1986 | Water pipes with reverse pressure gauges | Dynamic DAG forward/backward flow |
| **02** | **Optimization & The Terrain of Loss** *(Completed)* | 1950–2017 | Skiing down a foggy mountain ravine | 3D Loss landscape with rolling marbles |
| **03** | **Generalization & Normalization** | 2014–2019 | Factory assembly line quality control | Live distribution histograms & dropout flicker |
| **04** | **Spatial Vision & Convolutions (CNNs)** | 1998–2015 | Stenciled flashlight scanning canvas | Interactive 2D kernel slider & receptive field |
| **05** | **Sequential Dynamics & Attention Spark** *(Completed)* | 1997–2014 | Telephone game with shrinking notepad | Unrolled temporal memory tape & gradient fade |
| **06** | **The Modern Transformer Revolution** *(Completed)* | 2017–2024 | Diplomats trading Query, Key & Value envelopes | Interactive QKV Matrix tiles & attention beams |
| **07** | **Generative Paradigms (Diffusion & Flow)** *(Completed)* | 2020–2024 | Ink dispersing in water played in reverse | Denoising trajectory & vector field simulator |
| **08** | **LLMs, Alignment & Reasoning Engines** *(Completed)* | 2022–2026 | Socratic debate coach scoring rollouts | KV-Cache memory blocks & MoE routing |
| **09** | **Mechanistic Interpretability & Frontiers** *(Completed)* | 2024–2026 | MRI / Brain X-Ray of an artificial network | Induction head circuits & Sparse Autoencoders |
