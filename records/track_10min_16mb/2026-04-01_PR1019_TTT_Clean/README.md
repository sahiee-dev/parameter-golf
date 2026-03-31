# PR #1019 Base + Legal TTT (Score-First SGD)

## Architecture
Built on PR #1019 (AR Self-Gen Full Hessian GPTQ + XSA-all 11L + BigramHash 3072x112).
Added legal score-first TTT (SGD, lr=0.002, 3 epochs, chunk_tokens=32768, freeze_blocks=0).
No SLOT. No n-gram caches.

## TTT Legality
Score-first: each chunk is evaluated under torch.inference_mode() before any weight update.
Chunk N is scored, then model adapts on chunk N for future chunks.
No multi-epoch pre-eval training. No future token access.

## Results (pending)
| Seed | val_bpb | val_loss | eval_time |
|------|---------|----------|-----------|
| 314  | TBD     | TBD      | TBD       |
| 42   | TBD     | TBD      | TBD       |
| 999  | TBD     | TBD      | TBD       |
| mean | TBD     | TBD      | TBD       |

## Run Command
```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=314 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-04-01_PR1019_TTT_Clean/train_gpt.py \
2>&1 | tee seed314_ttt.log
```
