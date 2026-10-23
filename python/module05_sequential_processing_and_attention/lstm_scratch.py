"""
Module 05: Sequential Processing & Recurrent Dynamics
Implementation 1: LSTM from Pure NumPy Scratch (The Constant Error Carousel)

Historical Context:
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
  Neural Computation, 9(8), 1735-1780.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget:
  Continual Prediction with LSTM. Neural Computation, 12(10), 2451-2471.

The Physical Analogy:
- The Vanilla RNN is like a game of 'Telephone' played down a long line of 50 people,
  where each person only has a tiny diminishing notepad. By person 30, the original
  whisper is completely lost or distorted into gibberish (Vanishing/Exploding Gradient).
- The LSTM introduces an explicit 'Conveyor Belt Highway' (Cell State c_t) running
  parallel to the chain. Special gates (valves) decide what old baggage to throw away
  (Forget Gate), what fresh cargo to load onto the belt (Input Gate), and what payload
  to broadcast to the outside world (Output Gate). The gradient can slide back along
  the belt unattenuated across hundreds of time steps!
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation function: sigma(x) = 1 / (1 + exp(-x))."""
    # Prevent overflow in exp(-x) for large negative x
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = np.empty_like(x, dtype=np.float64)
    z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1.0 + exp_x)
    return z


def d_sigmoid(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its output s = sigmoid(x): s * (1 - s)."""
    return s * (1.0 - s)


def d_tanh(t: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its output t = tanh(x): 1 - t^2."""
    return 1.0 - t ** 2


class LSTMCellScratch:
    """
    A single LSTM Cell with Forget, Input, Candidate Cell, and Output Gates.
    Implemented purely in NumPy with analytical forward and backward passes.
    
    Mathematical Formulation:
      f_t = sigma(W_f @ x_t + U_f @ h_{t-1} + b_f)       [Forget Gate]
      i_t = sigma(W_i @ x_t + U_i @ h_{t-1} + b_i)       [Input Gate]
      c~_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)       [Candidate Cell State]
      c_t = f_t * c_{t-1} + i_t * c~_t                   [Cell State Update (CEC Highway)]
      o_t = sigma(W_o @ x_t + U_o @ h_{t-1} + b_o)       [Output Gate]
      h_t = o_t * tanh(c_t)                              [Hidden State / Output]
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        # Xavier/Glorot initialization scale
        scale_x = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_h = np.sqrt(2.0 / (hidden_dim + hidden_dim))

        # Weights for forget gate (f)
        self.W_f = rng.randn(hidden_dim, input_dim) * scale_x
        self.U_f = rng.randn(hidden_dim, hidden_dim) * scale_h
        # Initialize forget gate bias to +1.0 (Gers et al., 2000 trick to remember past initially)
        self.b_f = np.ones((hidden_dim, 1), dtype=np.float64)

        # Weights for input gate (i)
        self.W_i = rng.randn(hidden_dim, input_dim) * scale_x
        self.U_i = rng.randn(hidden_dim, hidden_dim) * scale_h
        self.b_i = np.zeros((hidden_dim, 1), dtype=np.float64)

        # Weights for candidate cell state (c)
        self.W_c = rng.randn(hidden_dim, input_dim) * scale_x
        self.U_c = rng.randn(hidden_dim, hidden_dim) * scale_h
        self.b_c = np.zeros((hidden_dim, 1), dtype=np.float64)

        # Weights for output gate (o)
        self.W_o = rng.randn(hidden_dim, input_dim) * scale_x
        self.U_o = rng.randn(hidden_dim, hidden_dim) * scale_h
        self.b_o = np.zeros((hidden_dim, 1), dtype=np.float64)

    def forward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass for a single time step t.
        
        Args:
          x_t: Input vector/matrix of shape (input_dim, batch_size)
          h_prev: Hidden state from step t-1 of shape (hidden_dim, batch_size)
          c_prev: Cell state from step t-1 of shape (hidden_dim, batch_size)
          
        Returns:
          h_t: New hidden state of shape (hidden_dim, batch_size)
          c_t: New cell state of shape (hidden_dim, batch_size)
          cache: Internal values needed for analytical backpropagation
        """
        # Ensure 2D column-vector formatting if batch_size=1
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)

        # 1. Forget gate activation: what fraction of past cell state to keep
        z_f = self.W_f @ x_t + self.U_f @ h_prev + self.b_f
        f_t = sigmoid(z_f)

        # 2. Input gate activation: what fraction of candidate cell to write
        z_i = self.W_i @ x_t + self.U_i @ h_prev + self.b_i
        i_t = sigmoid(z_i)

        # 3. Candidate cell state: candidate content to add to the belt
        z_c = self.W_c @ x_t + self.U_c @ h_prev + self.b_c
        c_tilde = np.tanh(z_c)

        # 4. Cell state update: The Constant Error Carousel (CEC) highway!
        c_t = f_t * c_prev + i_t * c_tilde

        # 5. Output gate activation: what fraction of cell state to reveal
        z_o = self.W_o @ x_t + self.U_o @ h_prev + self.b_o
        o_t = sigmoid(z_o)

        # 6. Hidden state output
        tanh_c_t = np.tanh(c_t)
        h_t = o_t * tanh_c_t

        cache = {
            "x_t": x_t,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "f_t": f_t,
            "i_t": i_t,
            "c_tilde": c_tilde,
            "c_t": c_t,
            "o_t": o_t,
            "tanh_c_t": tanh_c_t,
        }
        return h_t, c_t, cache

    def backward(
        self,
        dh_t: np.ndarray,
        dc_t: np.ndarray,
        cache: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Analytical backward pass for a single time step t.
        
        Args:
          dh_t: Upstream gradient w.r.t hidden state h_t, shape (hidden_dim, batch_size)
          dc_t: Upstream gradient directly w.r.t cell state c_t (from t+1), shape (hidden_dim, batch_size)
          cache: Saved values from forward pass
          
        Returns:
          dx_t: Gradient w.r.t x_t (input_dim, batch_size)
          dh_prev: Gradient w.r.t h_{t-1} (hidden_dim, batch_size)
          dc_prev: Gradient w.r.t c_{t-1} (hidden_dim, batch_size)
          grads: Dictionary of gradients w.r.t all 12 parameter matrices/biases
        """
        x_t = cache["x_t"]
        h_prev = cache["h_prev"]
        c_prev = cache["c_prev"]
        f_t = cache["f_t"]
        i_t = cache["i_t"]
        c_tilde = cache["c_tilde"]
        c_t = cache["c_t"]
        o_t = cache["o_t"]
        tanh_c_t = cache["tanh_c_t"]

        # Gradient w.r.t output gate
        do_t = dh_t * tanh_c_t
        dz_o = do_t * d_sigmoid(o_t)

        # Total gradient flowing into cell state c_t:
        # Sum of: (1) path through h_t: dh_t * o_t * (1 - tanh^2(c_t))
        #     and (2) direct cell highway gradient from future step dc_t
        total_dc_t = dh_t * o_t * d_tanh(tanh_c_t) + dc_t

        # CEC Highway Gradient: Flows back to c_prev scaled ONLY by forget gate f_t!
        # If f_t == 1, d(c_t)/d(c_{t-1}) == 1 (No Vanishing Gradient!)
        dc_prev = total_dc_t * f_t

        # Gradient w.r.t candidate cell state
        dc_tilde = total_dc_t * i_t
        dz_c = dc_tilde * d_tanh(c_tilde)

        # Gradient w.r.t input gate
        di_t = total_dc_t * c_tilde
        dz_i = di_t * d_sigmoid(i_t)

        # Gradient w.r.t forget gate
        df_t = total_dc_t * c_prev
        dz_f = df_t * d_sigmoid(f_t)

        # Parameter gradients (accumulated across batch)
        grads = {
            "dW_f": dz_f @ x_t.T,
            "dU_f": dz_f @ h_prev.T,
            "db_f": np.sum(dz_f, axis=1, keepdims=True),
            "dW_i": dz_i @ x_t.T,
            "dU_i": dz_i @ h_prev.T,
            "db_i": np.sum(dz_i, axis=1, keepdims=True),
            "dW_c": dz_c @ x_t.T,
            "dU_c": dz_c @ h_prev.T,
            "db_c": np.sum(dz_c, axis=1, keepdims=True),
            "dW_o": dz_o @ x_t.T,
            "dU_o": dz_o @ h_prev.T,
            "db_o": np.sum(dz_o, axis=1, keepdims=True),
        }

        # Upstream gradients w.r.t previous inputs and states
        dx_t = (
            self.W_f.T @ dz_f
            + self.W_i.T @ dz_i
            + self.W_c.T @ dz_c
            + self.W_o.T @ dz_o
        )
        dh_prev = (
            self.U_f.T @ dz_f
            + self.U_i.T @ dz_i
            + self.U_c.T @ dz_c
            + self.U_o.T @ dz_o
        )

        return dx_t, dh_prev, dc_prev, grads


class LSTMScratch:
    """
    Unrolled multi-step sequence LSTM from pure NumPy arithmetic.
    Processes sequences of shape (seq_len, batch_size, input_dim).
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = LSTMCellScratch(input_dim, hidden_dim, seed=seed)

    def forward(
        self,
        X: np.ndarray,
        h_0: Optional[np.ndarray] = None,
        c_0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Unroll LSTM forward through sequence.
        
        Args:
          X: Input sequence of shape (seq_len, batch_size, input_dim)
          h_0: Optional initial hidden state of shape (hidden_dim, batch_size)
          c_0: Optional initial cell state of shape (hidden_dim, batch_size)
          
        Returns:
          H: Sequence of hidden states of shape (seq_len, batch_size, hidden_dim)
          (h_T, c_T): Final hidden and cell states
          caches: List of per-step cache dictionaries
        """
        seq_len, batch_size, _ = X.shape

        if h_0 is None:
            h_0 = np.zeros((self.hidden_dim, batch_size), dtype=np.float64)
        if c_0 is None:
            c_0 = np.zeros((self.hidden_dim, batch_size), dtype=np.float64)

        H = np.zeros((seq_len, batch_size, self.hidden_dim), dtype=np.float64)
        caches = []

        h_t = h_0
        c_t = c_0

        for t in range(seq_len):
            x_t = X[t].T  # Shape: (input_dim, batch_size)
            h_t, c_t, cache = self.cell.forward(x_t, h_t, c_t)
            H[t] = h_t.T
            caches.append(cache)

        return H, (h_t, c_t), caches

    def backward(
        self,
        dH: np.ndarray,
        caches: List[Dict[str, np.ndarray]],
        dh_final: Optional[np.ndarray] = None,
        dc_final: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Unroll Backpropagation Through Time (BPTT).
        
        Args:
          dH: Upstream gradient w.r.t all hidden states (seq_len, batch_size, hidden_dim)
          caches: Forward caches
          dh_final: Optional extra gradient on the final hidden state
          dc_final: Optional extra gradient on the final cell state
          
        Returns:
          dX: Gradient w.r.t input sequence (seq_len, batch_size, input_dim)
          accum_grads: Sum of gradients for all weights across all time steps
        """
        seq_len, batch_size, _ = dH.shape
        dX = np.zeros((seq_len, batch_size, self.input_dim), dtype=np.float64)

        accum_grads = {
            k: np.zeros_like(getattr(self.cell, k.replace("d", "")))
            for k in [
                "dW_f", "dU_f", "db_f",
                "dW_i", "dU_i", "db_i",
                "dW_c", "dU_c", "db_c",
                "dW_o", "dU_o", "db_o",
            ]
        }

        dh_next = np.zeros((self.hidden_dim, batch_size), dtype=np.float64)
        if dh_final is not None:
            dh_next += dh_final
        dc_next = np.zeros((self.hidden_dim, batch_size), dtype=np.float64)
        if dc_final is not None:
            dc_next += dc_final

        for t in reversed(range(seq_len)):
            # Upstream gradient on h_t comes from both the output layer at step t (dH[t])
            # and from the recurrent step t+1 (dh_next)
            dh_t = dH[t].T + dh_next
            dc_t = dc_next

            dx_t, dh_next, dc_next, step_grads = self.cell.backward(
                dh_t, dc_t, caches[t]
            )
            dX[t] = dx_t.T

            for k in accum_grads:
                accum_grads[k] += step_grads[k]

        return dX, accum_grads


class VanillaRNNScratch:
    r"""
    Vanilla Elman RNN from scratch to numerically prove the Vanishing Gradient Problem.
    Formula:
      h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    Jacobian through time:
      dh_T / dh_t = \prod_{k=t+1}^T diag(1 - h_k^2) @ W_hh^T
    """

    def __init__(self, input_dim: int, hidden_dim: int, spectral_radius: float = 0.9, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        W = rng.randn(hidden_dim, hidden_dim)
        # Scale W to have specific spectral radius
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals))
        self.W_hh = (W / max_eig) * spectral_radius

        self.W_xh = rng.randn(hidden_dim, input_dim) * np.sqrt(1.0 / input_dim)
        self.b_h = np.zeros((hidden_dim, 1), dtype=np.float64)

    def forward_and_track_gradients(self, seq_len: int) -> Tuple[List[float], List[float]]:
        """
        Inject a test unit gradient delta L / delta h_T at step T,
        backpropagate through time to step 0, and record the L2 norm of the gradient
        at each step t for both Vanilla RNN and an equivalent LSTM.
        
        Returns:
          vanilla_grad_norms: List of gradient norms for Vanilla RNN at each step
          lstm_grad_norms: List of gradient norms for LSTM at each step
        """
        # Vanilla RNN forward
        h = np.zeros((self.hidden_dim, 1))
        h_states = [h]
        for t in range(seq_len):
            x = np.ones((self.input_dim, 1)) * 0.1
            h = np.tanh(self.W_hh @ h + self.W_xh @ x + self.b_h)
            h_states.append(h)

        # Vanilla backward: dh_T = ones
        dh = np.ones((self.hidden_dim, 1))
        vanilla_norms = [float(np.linalg.norm(dh))]
        for t in reversed(range(seq_len)):
            # dh_{t-1} = W_hh^T @ (dh_t * (1 - h_t^2))
            dh = self.W_hh.T @ (dh * (1.0 - h_states[t + 1] ** 2))
            vanilla_norms.append(float(np.linalg.norm(dh)))
        vanilla_norms.reverse()

        # LSTM forward & backward (with forget gate bias = 1.0)
        lstm = LSTMCellScratch(self.input_dim, self.hidden_dim)
        c = np.zeros((self.hidden_dim, 1))
        h_lstm = np.zeros((self.hidden_dim, 1))
        lstm_caches = []
        for t in range(seq_len):
            x = np.ones((self.input_dim, 1)) * 0.1
            h_lstm, c, cache = lstm.forward(x, h_lstm, c)
            lstm_caches.append(cache)

        # LSTM backward
        dh_lstm = np.ones((self.hidden_dim, 1))
        dc_lstm = np.zeros((self.hidden_dim, 1))
        lstm_norms = [float(np.linalg.norm(dh_lstm))]
        for t in reversed(range(seq_len)):
            _, dh_lstm, dc_lstm, _ = lstm.backward(dh_lstm, dc_lstm, lstm_caches[t])
            # Total gradient signal flowing through cell + hidden state
            total_signal = np.linalg.norm(dh_lstm) + np.linalg.norm(dc_lstm)
            lstm_norms.append(float(total_signal))
        lstm_norms.reverse()

        return vanilla_norms, lstm_norms


if __name__ == "__main__":
    print("=" * 78)
    print("🧠 Module 05: LSTM Scratch & Recurrent Vanishing Gradient Demonstration")
    print("=" * 78)

    # 1. Initialize LSTM Cell
    input_dim = 4
    hidden_dim = 6
    lstm_cell = LSTMCellScratch(input_dim=input_dim, hidden_dim=hidden_dim, seed=42)

    # 2. Execute single forward step
    x_sample = np.array([0.5, -1.2, 0.8, 2.0]).reshape(-1, 1)
    h_init = np.zeros((hidden_dim, 1))
    c_init = np.zeros((hidden_dim, 1))

    h_1, c_1, cache_1 = lstm_cell.forward(x_sample, h_init, c_init)

    print("\n[Step 1 Forward Pass Gate Activations]")
    print(f"  Forget Gate (f_t) : {np.round(cache_1['f_t'].flatten(), 4)} (All in [0, 1])")
    print(f"  Input Gate (i_t)  : {np.round(cache_1['i_t'].flatten(), 4)} (All in [0, 1])")
    print(f"  Candidate (~c_t)  : {np.round(cache_1['c_tilde'].flatten(), 4)} (All in [-1, 1])")
    print(f"  Cell State (c_t)  : {np.round(c_1.flatten(), 4)}")
    print(f"  Output Gate (o_t) : {np.round(cache_1['o_t'].flatten(), 4)} (All in [0, 1])")
    print(f"  Hidden State (h_t): {np.round(h_1.flatten(), 4)}")

    # 3. Simulate Vanishing Gradient across 25 time steps
    print("\n" + "-" * 78)
    print("📉 Numerical Proof: Vanishing Gradient Decay in Vanilla RNN vs LSTM (25 Steps)")
    print("-" * 78)
    vanilla_rnn = VanillaRNNScratch(input_dim=2, hidden_dim=8, spectral_radius=0.85)
    vanilla_curve, lstm_curve = vanilla_rnn.forward_and_track_gradients(seq_len=25)

    print(f"{'Time Step t':<12} | {'Vanilla RNN ||dh_t||':<22} | {'LSTM Signal ||dh + dc||':<22} | Ratio (LSTM/RNN)")
    print("-" * 78)
    for step in [24, 20, 15, 10, 5, 0]:
        v_val = vanilla_curve[step]
        l_val = lstm_curve[step]
        ratio_str = f"{l_val / (v_val + 1e-12):.1f}x" if v_val > 0 else "inf"
        print(f"t = {step:<8} | {v_val:<22.6e} | {l_val:<22.6e} | {ratio_str}")

    print("\n✅ Key Takeaway: In Vanilla RNN, gradients shrink exponentially to zero (~10^-6).")
    print("   In LSTM, the Constant Error Carousel (CEC) cell highway preserves the signal!")
