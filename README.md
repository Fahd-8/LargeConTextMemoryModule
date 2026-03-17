# Large Context Memory Module (LMM)

Implementation of Google's Titans LMM (Large-context Memory Module) 
— a trainable MLP that acts as external memory for Transformers, 
enabling persistent key→value recall beyond standard context limits.

## What Makes This Different

Standard Transformers forget. Context windows have hard limits.
This module gives a Transformer external memory that persists —
trained to store and retrieve facts across any context length.

## Architecture

- **Trainable MLP** acts as external memory bank
- **Frozen embeddings** + learnable memory network with weight decay
- **Key→value recall** — persistent facts retrieved on demand
- **100% recall** on interleaved facts benchmark
- Pure PyTorch — under 100 lines of core logic

## How It Works
```
Input → Frozen Embeddings → Memory MLP (trainable)
→ Key→Value Lookup → Retrieved Context
→ Transformer + Memory Context → Output
```

## Key Results

- 100% recall on interleaved facts
- Persistent memory beyond context window limits
- Lightweight — full implementation under 100 lines
- Inspired by Google's Titans research paper (2024)

## Setup
```bash
git clone https://github.com/Fahd-8/LargeConTextMemoryModule
cd LargeConTextMemoryModule
pip install torch numpy jupyter
jupyter notebook "Titan Variance LMM (Long-Term Memory Module).ipynb"
```

## Reference

Inspired by: Titans: Learning to Memorize at Test Time (Google, 2024)

---
Built by [Fahad Zaman](https://github.com/Fahd-8) — AI Engineer
