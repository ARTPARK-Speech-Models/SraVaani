# SraVaani Fine-tuning

Fine-tune the [SraVaani](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) speech recognition checkpoint on custom Indian-language audio using [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

The base model is a **Hybrid RNN-T + CTC BPE** FastConformer pre-trained on 63 Indian languages. This repo provides scripts to download the checkpoint and run fine-tuning with PyTorch Lightning.

---

## Folder Structure

```
sravaani_finetune/
├── finetune.py                # Fine-tuning training script
├── SraVaani-nemo-checkpoint.nemo   # Downloaded checkpoint (not in git, ~1.7 GB)
└── data/                      # Not in git — populate manually (see Data Format below)
    ├── train/
    │   ├── shard_0000.tar     # Tarred FLAC audio for training
    │   └── shard_0000.json    # JSONL manifest for training
    └── val/
        ├── shard_0000.tar     # Tarred FLAC audio for validation
        └── shard_0000.json    # JSONL manifest for validation
```

> `*.nemo` files and the `data/` directory are excluded from git (see `.gitignore`).

---

## Setup

```bash
pip install "nemo_toolkit[asr,cu13]" pytorch-lightning omegaconf
```

Note the `cu13` extra — match it to your CUDA driver version (`nvidia-smi`, top-right "CUDA Version" field; use `cu12` for CUDA 12). Plain `nemo_toolkit[asr]` is **not enough**: it installs generic `numba` but not the `numba-cuda` GPU kernel backend that NeMo's RNNT/TDT loss needs, so the first training step fails with a numba `Signature mismatch` error.

A CUDA-capable NVIDIA GPU is required for training. At least **16 GB VRAM** is recommended with the default `batch_size=4` — If you hit an OOM, lower `BATCH_SIZE` in `finetune.py` further.

---

## Step 1 — Download the .nemo Checkpoint

Download `SraVaani-nemo-checkpoint.nemo` (~1.7 GB) from Google Drive and place it in the root of this repo:

**[Download SraVaani-nemo-checkpoint.nemo](https://drive.google.com/drive/folders/1QdMG0R3oGgOThm7uqYeMnt8zGOSx467x)**


Don't have access? [Request access here](https://script.google.com/macros/s/AKfycbyAC0vGcGSbu55LNKOn_hY3TsPy8CyF1nasLJULWMrzCdksBJG33bUfLbiKee_m_aKH-w/exec) — access will be shared ASAP.


The checkpoint filename must match the value of `NEMO_CHECKPOINT` in [finetune.py](finetune.py).

---

## Step 2 — Prepare Training Data

Populate the `data/` directory before training. Each split (`train/`, `val/`) needs a pair of files:

| File | Contents |
|------|----------|
| `shard_NNNN.tar` | Uncompressed tar archive containing audio files |
| `shard_NNNN.json` | JSONL manifest — one JSON object per line, in the same order as the tar entries |

### Audio spec
- Format: FLAC or WAV (16-bit PCM)
- Sample rate: 16 kHz
- Channels: Mono

### Manifest format (JSONL)

Each line is a standalone JSON object referencing one audio file:

```json
{"audio_filepath": "train_00000.flac", "text": "ಹಾಗು <pause> ಇಬ್ಬರು ಕುರ್ಚಿಯ ಮೇಲೆ ಕುಳಿತಿದ್ದಾರೆ", "duration": 3.98}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | string | Filename **inside** the tar (e.g. `train_00000.flac`) |
| `text` | string | Transcript in the target language |
| `duration` | float | Audio duration in seconds |

### Updating data paths

If your shards are named differently or split across multiple files, update the path constants at the top of [finetune.py](finetune.py):

```python
TRAIN_SHARDS    = "data/train/shard_0000.tar"
TRAIN_MANIFESTS = "data/train/shard_0000.json"
VAL_SHARDS      = "data/val/shard_0000.tar"
VAL_MANIFESTS   = "data/val/shard_0000.json"
```

For multiple shards use brace expansion: `"data/train/shard_{0000..0009}.tar"`.

---

## Step 3 — Run Fine-tuning

```bash
python finetune.py
```

### What the script does

1. Loads the pre-trained model from `SraVaani-nemo-checkpoint.nemo`
2. **Freezes the encoder** — only the decoder and joint network weights are updated (remove lines 20–21 in `finetune.py` for full end-to-end fine-tuning)
3. Configures training and validation data loaders from the tarred shards
4. Sets up an **AdamW** optimizer with **CosineAnnealing** LR schedule
5. Trains for 30 epochs on 1 GPU with gradient accumulation (effective batch size = 32)
6. Saves the top-3 checkpoints (by `val_wer`) to `experiments/sra-vaani-finetune/`
7. Exports the final model to `sra-vaani-finetuned.nemo`

### Key hyperparameters

| Parameter | Default | Where to change |
|-----------|---------|-----------------|
| Learning rate | `1e-4` | `model.cfg.optim.lr` in `finetune.py` |
| Batch size | `8` | `model.cfg.train_ds.batch_size` |
| Grad accumulation | `4` (eff. batch = 32) | `accumulate_grad_batches` in Trainer |
| Max epochs | `30` | `max_epochs` in Trainer |
| Warmup steps | `500` | `model.cfg.optim.sched.warmup_steps` |
| Encoder frozen | Yes | Remove lines 20–21 to unfreeze |

### Outputs

```
experiments/
└── sra-vaani-finetune/
    ├── checkpoints/        # Top-3 .ckpt files (best val_wer)
    └── tb_logs/            # TensorBoard event files

sra-vaani-finetuned.nemo    # Final exported NeMo model
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir experiments/sra-vaani-finetune/tb_logs
```

---

## Optional: Resume from a Raw `.ckpt` Instead of the Released `.nemo`

If you have a raw PyTorch Lightning checkpoint from the original SraVaani training run (e.g. `SraVaani-Hybrid-TDT-CTC-BPE.ckpt`, an intermediate epoch saved with full training state) rather than the released `.nemo`, use [resume_finetune_from_ckpt.py](resume_finetune_from_ckpt.py) instead of `finetune.py`:

**[Download SraVaani-Hybrid-TDT-CTC-BPE.ckpt](https://drive.google.com/drive/folders/1QdMG0R3oGgOThm7uqYeMnt8zGOSx467x)** — same folder as the `.nemo` checkpoint above. Don't have access? [Request access here](https://script.google.com/macros/s/AKfycbyAC0vGcGSbu55LNKOn_hY3TsPy8CyF1nasLJULWMrzCdksBJG33bUfLbiKee_m_aKH-w/exec) — access will be shared ASAP.

```bash
python resume_finetune_from_ckpt.py
```

A `.nemo` file is weights-only. A `.ckpt` like this also carries the **optimizer state** (AdamW per-parameter momentum/variance buffers), **LR scheduler state**, and the **global step counter** — so this script does a true resume via PyTorch Lightning's `trainer.fit(model, ckpt_path=...)`, rather than just a weights-only warm start.

This only works because the model is configured to structurally match what the checkpoint was saved with:

- **The encoder is left unfrozen.** The checkpoint's optimizer state covers a single param group of all 654 trainable tensors; freezing the encoder would drop most of those params from the optimizer and break the state restore.
- **Optimizer/scheduler match the checkpoint's**: AdamW with the same betas, and **NoamAnnealing** (not `finetune.py`'s CosineAnnealing) — so their `load_state_dict()` calls succeed.

Two practical consequences worth knowing before running it:

1. **Heavier on memory than `finetune.py`.** An unfrozen encoder means encoder gradients and doubled AdamW state for it too, on top of the checkpoint itself being ~3x the size of the `.nemo` (weights + optimizer + scheduler state). The script defaults to a smaller `batch_size=2` (vs. `4`) accordingly — lower it further if you hit an OOM.
2. **The restored learning rate is already near its floor.** This checkpoint is deep enough into its original schedule that the LR has decayed to `min_lr=1e-5` — so resuming continues training at a small, roughly constant LR rather than through a fresh warmup/decay cycle.

Output: `sra-vaani-finetuned-resumed.nemo`, with its own experiment directory under `experiments/sra-vaani-resume-finetune/`.
