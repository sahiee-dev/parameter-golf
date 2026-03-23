# XSA + Value Residual + Gated Attention + EMA + AdamW TTT — sahiee-dev

## Base
Built on: 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04 by thwu1
Base score: 1.1428 val_bpb
records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50

## Novel contributions

### 1. BigramHash(10240) — restored to full size
Retains thwu1's original BigramHash at 10240 buckets (dim=128, projected to 512).
Ablation confirmed: bigram=10240 beats bigram=4096+TrigramHash by 0.35 loss units
at 1000 steps on T4. TrigramHash added noise at this scale — removed.

### 2. XSA — Exclusive Self Attention (last 4 layers)
Removes self-value bias from attention output via orthogonal projection.
GQA-aware implementation from PR #287, adapted for (B,T,H,D) layout.
Zero parameter cost. Applied to final 4 layers only.

### 3. EMA — Exponential Moving Average of weights
Maintains shadow model with decay=0.9999 updated every training step.
Final val_bpb evaluation uses EMA weights instead of raw model weights.
EMA coexists with SWA: SWA averages warmdown checkpoints, EMA tracks
the full training trajectory. Zero artifact cost — EMA weights not stored.
Consistent with technique in PR #338 (current best open PR, 1.1254 bpb).

### 4. Value Residual (all layers)
Each Block adds a learned scalar (lambda_v, init=0) times the raw token embedding (v0)
to its output. v0 is captured before SmearGate/RMSNorm and threaded through all blocks.
Zero-init means no-op at step 0; model learns whether and how much direct input access
helps per layer. Source: PR #487, #490 (confirmed ~-0.015 bpb on frontier PRs).

### 5. Gated Attention
Learned per-layer scalar gate on attention output. init=1 means no-op at step 0.
Model learns to scale attention contribution per layer independently.
Addresses over-smoothing — complements Value Residual.
Cost: 1 parameter per layer (10 params total). Zero artifact overhead.

### 6. AdamW TTT replacing SGD TTT
Test-time training uses AdamW (lr=0.001, betas=(0.9,0.999)) instead of SGD.
AdamW's per-parameter adaptive moments retain learning signal across sequences
better than SGD with momentum. Fixed lr — no cosine scheduling needed.
Consistent with PR #490 implementation (1.0891 bpb).

### Evaluated and dropped
QAT: confirmed negative (PR #360) — 8% throughput penalty within 600s budget.
TrigramHash: ablation showed negative result — bigram=10240 alone beats
bigram=4096+trigram=20480 by 0.35 val_loss at 1000 steps (T4 ablation, seed=42).
11th layer: does not fit within 16MB given current budget (~0.91MB needed, ~0.36MB available).

## Architecture
Identical to thwu1 base:
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA), ReLU²
- MLP 3x expansion, SmearGate, BigramHash(10240)
- OrthoInit, U-Net skips, Muon WD=0.04, SWA start_frac=0.4
- Sliding window eval stride=64, zstd-22, ~15.64MB artifact

## Ablation table (T4, 1000 steps, seed=42)
| Variant | loss | result |
|---------|------|--------|
| bigram=10240, no trigram (V2) | 5.4379 | WINNER |
| bigram=8192 + trigram=8192 dim=16 (V4) | 5.6956 | |
| bigram=4096 + trigram=20480 dim=32 (V3) | 5.7924 | was our submission |
| bigram=4096, no trigram (V1) | 5.8414 | |

## Final H100 ablation
| Variant | val_bpb | delta |
|---------|---------|-------|
| thwu1 base | 1.1428 | — |
| + XSA | pending | pending |
| + EMA | pending | pending |
| + Value Residual | pending | pending |
| + Gated Attention | pending | pending |
| + AdamW TTT | pending | pending |
| + all five (ours) | pending | pending |

## Status
Code complete. Syntax OK. All smoke tests passing.
Awaiting H100 compute credits (RunPod) for 3-seed validation run.
val_bpb and ablation table will be updated before marking ready for review.
Training logs (seed 42, 1337, 2024) to be added after H100 run.
