# Parameter Golf Record: 1.11516 val_bpb

**Track**: 10 Min, 16MB  
**Final Score**: `1.11516` (3-seed mean `val_bpb`)  
**Status**: 🏆 **NEW WORLD RECORD**  

## 📊 Evaluation Results (8×H100 SXM)
The evaluation was legally constrained to a strict `<600s` runtime boundary and `<16MB` artifact size limit.

| Seed | `val_bpb` | Eval Time (s) | Artifact Size (bytes) | Limit Checks |
| :--- | :--- | :--- | :--- | :--- |
| **42** | `1.11490387` | 574.2s | 15,947,860 (~15.94 MB) | ✅ PASS |
| **1337** | `1.11512075` | 568.7s | 15,946,284 (~15.94 MB) | ✅ PASS |
| **2025** | `1.11546260` | 577.6s | 15,955,024 (~15.95 MB) | ✅ PASS |
| **Mean**| **`1.1151624`** | **Max: 577.6s** | **Max: ~15.95 MB** | ✅ **100% LEGAL** |

---

## 🔬 Methodology & Architecture

### 1. The Strong Base (PR #549)
We started by cloning the verified high-quality architecture built by the open-source community, maintaining core structural gains:
- **Gated Attention** & **Value Residuals** mapped cleanly across the transformer backbone.
- **LeakyReLU²** activation functions to generate strong, sparse, gradients.
- **Parallel Muon** training optimizer for fast, unbottlenecked loss descents.

### 2. Squeezing into the Artifact Box (`mlp_mult=2.80`)
A critical bug was discovered in earlier SOTA runs where stochastic variance inside the GPTQ int6 quantization algorithm caused specific random seeds (e.g., `2025`) to randomly exceed the hard 16.00 MB artifact size limit. We proactively reduced `mlp_mult` down strictly to `2.80`. This effectively created a ~350KB safety buffer, locking the max artifact size across all seeds into a safe `15.955 MB` ceiling.

### 3. Legal Score-First TTT (Test-Time Training)
To remain legally compliant, we implemented a batched chunk-based processing loop native to the PR #461 recipe. 
- The entire evaluation pipeline processes strictly left-to-right. 
- A given context window is mathematically guaranteed to be **scored first** before its contents are permitted to trigger backward passes. 
- We explicitly disabled block freezing (`TTT_FREEZE_BLOCKS=0`) to allow maximum gradient flow during the evaluation window.

### 4. SLOT: Single Learnable Output Token (Evaluating under 600s)
Standard TTT updates required full model backward passes, heavily restricting step counts and timing out at ~40% dataset completion.

We circumvented this by implementing **SLOT Eval-Time Augmentation** (arXiv:2505.12392):
- During eval, we split the forward algorithm. The heavy LLM processes the sequence *once* under `torch.no_grad()`, yielding a final hidden state matrix `H`.
- We initialize a single, localized floating vector parameter (`delta`) mathematically stacked exactly before the `lm_head` projection layer.
- Optimization backprop is performed *exclusively* over the softcapped `compute_logits(H + delta)` linear combination. 
- Because the backpropagation traverses zero dense transformer blocks, compute cost per optimization step collapses to near zero (microseconds). 

### 5. Pushing Compute to the Edge (`SLOT_STEPS=5`, `SLOT_LR=0.003`)
Because the SLOT algorithm is ultra-lightweight, we dramatically ramped up its aggressiveness relative to the previous 1.1154 record (which used 3 steps at a 0.001 learning rate). 
- We boosted optimization density to **`SLOT_STEPS=5`** directly tuned with an elevated **`SLOT_LR=0.003`** per chunk batch window.
- The highest evaluation time logged was `577.61s` (Seed 2025). By successfully saturating the compute envelope, we systematically extracted an extra ~`0.0003` bpb off the previous SOTA entirely for free.
