"""
Module 05: Sequential Processing & Recurrent Dynamics
Project Synapse: The Graphical Neural Network Explainer

Historical Era: 1997 - 2014
- Hochreiter & Schmidhuber (1997): Long Short-Term Memory (LSTM)
- Gers et al. (2000): Forget Gate addition
- Cho et al. (2014): Gated Recurrent Unit (GRU) & Seq2Seq
- Bahdanau et al. (2014): Neural Machine Translation by Jointly Learning to Align and Translate (Additive Attention)
"""

from .lstm_scratch import LSTMCellScratch, LSTMScratch, VanillaRNNScratch
from .bahdanau_attention_pytorch import (
    BahdanauAttention,
    Seq2SeqEncoder,
    Seq2SeqDecoder,
    Seq2SeqModel,
)

__all__ = [
    "LSTMCellScratch",
    "LSTMScratch",
    "VanillaRNNScratch",
    "BahdanauAttention",
    "Seq2SeqEncoder",
    "Seq2SeqDecoder",
    "Seq2SeqModel",
]
