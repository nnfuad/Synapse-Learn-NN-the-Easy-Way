"""
Module 05: Sequential Processing & Recurrent Dynamics
Implementation 2: Bahdanau Additive Attention & Seq2Seq in PyTorch

Historical Context:
- Bahdanau, D., Cho, K., & Bengio, Y. (2014/2015).
  Neural Machine Translation by Jointly Learning to Align and Translate.
  International Conference on Learning Representations (ICLR 2015).
- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder
  for Statistical Machine Translation. EMNLP 2014.

The Physical Analogy:
- Pre-attention Seq2Seq models were forced to summarize an entire encyclopedic
  paragraph into a single fixed-length vector (The 'Diminishing Notepad' crisis).
  For sentences longer than 20 words, translation quality collapsed.
- Bahdanau Attention functions as a steerable mental searchlight. When translating
  the word 'chat' into 'cat', the decoder shines its attention beam directly onto
  the source token representations, bypassing the fixed-length memory bottleneck!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class BahdanauAttention(nn.Module):
    r"""
    Bahdanau Additive Attention Module.
    
    Mathematical Formulation:
      Score:       e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j + b_a)
      Weights:     \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
      Context:     c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
      
    Where:
      - s_{i-1}: Decoder hidden state at step i-1 (Query), shape: (batch_size, dec_hidden_dim)
      - h_j:     Encoder hidden state at position j (Key/Value), shape: (batch_size, enc_seq_len, enc_hidden_dim)
      - \alpha:  Attention alignment weights, shape: (batch_size, enc_seq_len)
      - c_i:     Dynamic context vector, shape: (batch_size, enc_hidden_dim)
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim

        # Projection for decoder query: W_a * s_{i-1}
        self.W_a = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        # Projection for encoder keys: U_a * h_j
        self.U_a = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        # Additive attention alignment vector: v_a^\top
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention context vector and alignment weights.
        
        Args:
          decoder_hidden: (batch_size, dec_hidden_dim)
          encoder_outputs: (batch_size, enc_seq_len, enc_hidden_dim)
          mask: Optional binary mask of shape (batch_size, enc_seq_len) where 0 indicates padding
          
        Returns:
          context: (batch_size, enc_hidden_dim)
          attn_weights: (batch_size, enc_seq_len)
        """
        # enc_seq_len = encoder_outputs.shape[1]
        
        # 1. Project decoder hidden state: (batch_size, 1, attn_dim)
        query = self.W_a(decoder_hidden).unsqueeze(1)
        
        # 2. Project all encoder outputs: (batch_size, enc_seq_len, attn_dim)
        keys = self.U_a(encoder_outputs)
        
        # 3. Additive combination + tanh non-linearity: (batch_size, enc_seq_len, attn_dim)
        energy = torch.tanh(query + keys)
        
        # 4. Score alignment via inner product with v_a: (batch_size, enc_seq_len)
        scores = self.v_a(energy).squeeze(-1)
        
        # 5. Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            
        # 6. Softmax across encoder sequence length -> probabilities summing to 1.0
        attn_weights = F.softmax(scores, dim=-1)
        
        # 7. Weighted sum of encoder hidden representations: (batch_size, enc_hidden_dim)
        # (batch_size, 1, enc_seq_len) @ (batch_size, enc_seq_len, enc_hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class Seq2SeqEncoder(nn.Module):
    """
    Bidirectional GRU Encoder that processes input token sequences
    into rich contextual representations.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        padding_idx: int = 0
    ):
        super().__init__()
        assert enc_hidden_dim % 2 == 0, "enc_hidden_dim must be divisible by 2 for bidirectional GRU"
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            embed_dim,
            enc_hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        # Linear layer to combine forward and backward final hidden states for decoder init
        self.fc_hidden = nn.Linear(enc_hidden_dim, dec_hidden_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          src: Token IDs of shape (batch_size, src_seq_len)
          
        Returns:
          encoder_outputs: (batch_size, src_seq_len, enc_hidden_dim)
          decoder_init_hidden: (batch_size, dec_hidden_dim)
        """
        embedded = self.embedding(src)  # (batch_size, src_len, embed_dim)
        outputs, hidden = self.gru(embedded)  # hidden: (2, batch_size, enc_hidden_dim // 2)

        # Concatenate forward and backward hidden states from the final step
        combined_hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        dec_init = torch.tanh(self.fc_hidden(combined_hidden))

        return outputs, dec_init


class Seq2SeqDecoder(nn.Module):
    """
    Bahdanau-Attentive GRU Decoder.
    At each step:
      1. Inspects previous decoder state and attends to encoder memory.
      2. Concatenates context vector with current embedded input token.
      3. Updates decoder state via GRUCell.
      4. Projects combined state to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        padding_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim)
        # GRUCell takes [embedded_token; context_vector]
        self.gru_cell = nn.GRUCell(embed_dim + enc_hidden_dim, dec_hidden_dim)
        # Final output projection
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim + embed_dim, vocab_size)

    def forward_step(
        self,
        token_input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A single decoding step.
        
        Args:
          token_input: (batch_size,) current input token IDs
          hidden: (batch_size, dec_hidden_dim) previous decoder state
          encoder_outputs: (batch_size, enc_seq_len, enc_hidden_dim)
          mask: Optional attention mask
          
        Returns:
          logits: (batch_size, vocab_size)
          next_hidden: (batch_size, dec_hidden_dim)
          attn_weights: (batch_size, enc_seq_len)
        """
        embedded = self.embedding(token_input)  # (batch_size, embed_dim)
        
        # 1. Compute dynamic attention spotlight
        context, attn_weights = self.attention(hidden, encoder_outputs, mask=mask)
        
        # 2. Update decoder state using GRUCell fed with [token; context]
        gru_input = torch.cat([embedded, context], dim=1)
        next_hidden = self.gru_cell(gru_input, hidden)
        
        # 3. Project [next_hidden; context; embedded] to vocabulary
        out_features = torch.cat([next_hidden, context, embedded], dim=1)
        logits = self.fc_out(out_features)
        
        return logits, next_hidden, attn_weights


class Seq2SeqModel(nn.Module):
    """
    Complete Encoder-Decoder Seq2Seq architecture with Bahdanau Attention.
    """

    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full Seq2Seq forward pass for training.
        
        Args:
          src: Source token IDs (batch_size, src_len)
          trg: Target token IDs (batch_size, trg_len)
          teacher_forcing_ratio: Probability of using ground-truth target token
          
        Returns:
          outputs: Logits for all target steps (batch_size, trg_len, vocab_size)
          attentions: Attention matrices (batch_size, trg_len, src_len)
        """
        batch_size, trg_len = trg.shape
        src_len = src.shape[1]

        encoder_outputs, hidden = self.encoder(src)

        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size, device=src.device)
        attentions = torch.zeros(batch_size, trg_len, src_len, device=src.device)

        # First input token is the first token of target (usually <BOS>)
        current_input = trg[:, 0]

        for t in range(1, trg_len):
            logits, hidden, attn_weights = self.decoder.forward_step(
                current_input, hidden, encoder_outputs
            )
            outputs[:, t] = logits
            attentions[:, t] = attn_weights

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            current_input = trg[:, t] if teacher_force else top1

        return outputs, attentions

    @torch.no_grad()
    def get_attention_matrix(
        self,
        src_tokens: torch.Tensor,
        max_len: int = 20
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Greedy inference decoding that extracts the full 2D Attention Alignment Heatmap.
        
        Returns:
          pred_tokens: List of generated token IDs
          alignment_matrix: Tensor of shape (pred_len, src_len)
        """
        self.eval()
        batch_size = src_tokens.shape[0]
        assert batch_size == 1, "Inference alignment heatmap extractor expects batch_size=1"
        src_len = src_tokens.shape[1]

        encoder_outputs, hidden = self.encoder(src_tokens)

        current_token = torch.tensor([self.bos_idx], device=src_tokens.device)
        pred_tokens = []
        attn_list = []

        for _ in range(max_len):
            logits, hidden, attn_weights = self.decoder.forward_step(
                current_token, hidden, encoder_outputs
            )
            top1 = logits.argmax(1).item()
            attn_list.append(attn_weights.squeeze(0).cpu())
            pred_tokens.append(top1)

            if top1 == self.eos_idx:
                break
            current_token = torch.tensor([top1], device=src_tokens.device)

        if len(attn_list) == 0:
            return pred_tokens, torch.zeros(0, src_len)

        alignment_matrix = torch.stack(attn_list, dim=0)  # (pred_len, src_len)
        return pred_tokens, alignment_matrix


if __name__ == "__main__":
    print("=" * 78)
    print("🔦 Module 05: Bahdanau Attention & Seq2Seq Alignment Demonstration")
    print("=" * 78)

    torch.manual_seed(42)

    # Configuration
    vocab_size = 50
    embed_dim = 32
    enc_hidden = 64
    dec_hidden = 64
    attn_dim = 32

    # Instantiate modules
    encoder = Seq2SeqEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        enc_hidden_dim=enc_hidden,
        dec_hidden_dim=dec_hidden
    )
    decoder = Seq2SeqDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        enc_hidden_dim=enc_hidden,
        dec_hidden_dim=dec_hidden
    )
    model = Seq2SeqModel(encoder, decoder)

    # Mock sequence: batch_size=2, src_len=6, trg_len=5
    src_mock = torch.tensor([[4, 8, 15, 16, 23, 2], [10, 11, 12, 13, 2, 0]])
    trg_mock = torch.tensor([[1, 23, 16, 15, 2], [1, 13, 12, 11, 2]])

    outputs, attentions = model(src_mock, trg_mock, teacher_forcing_ratio=0.0)

    print(f"\n[Model Forward Pass Shapes]")
    print(f"  Source Input Shape    : {src_mock.shape} (batch, src_len)")
    print(f"  Target Input Shape    : {trg_mock.shape} (batch, trg_len)")
    print(f"  Decoder Output Logits : {outputs.shape} (batch, trg_len, vocab_size)")
    print(f"  Attention Matrix Shape: {attentions.shape} (batch, trg_len, src_len)")

    # Verify probability distribution property: Sum of attention weights along src_len == 1.0
    attn_sums = attentions[:, 1:].sum(dim=-1)
    print(f"\n[Attention Weight Verification]")
    print(f"  Attention sums across encoder sequence: {attn_sums.mean().item():.6f} (Must equal 1.000000)")

    # Extract 2D alignment matrix for single sequence
    single_src = torch.tensor([[5, 12, 9, 7, 2]])
    preds, align_mat = model.get_attention_matrix(single_src, max_len=6)
    print(f"\n[Greedy Generated Sequence Alignment Matrix: {align_mat.shape}]")
    print("ASCII Attention Heatmap Preview (Decoder Tokens ↓ vs Encoder Tokens →):")
    src_labels = ["tok5", "tok12", "tok9", "tok7", "<eos>"]
    header = f"{'Dec Step':<10} | " + " | ".join([f"{l:>6}" for l in src_labels])
    print(header)
    print("-" * len(header))
    for t_step, row in enumerate(align_mat):
        row_str = " | ".join([f"{val.item():.3f}" for val in row])
        print(f"Step {t_step:<5} | {row_str}")

    print("\n✅ Verification Complete: Bahdanau additive attention dynamically focuses")
    print("   encoder states into context vectors, overcoming the fixed-length bottleneck!")
