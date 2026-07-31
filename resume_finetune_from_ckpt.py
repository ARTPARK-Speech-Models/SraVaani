"""Resume fine-tuning on Wancho from a raw PyTorch Lightning .ckpt, restoring
model weights AND optimizer/scheduler state (not just weights).

Unlike finetune_from_ckpt.py (which only transfers weights and starts a fresh
optimizer with the encoder frozen), this uses PyTorch Lightning's native
`trainer.fit(model, ckpt_path=...)` resume path, which restores the full
training state in one coherent step: model weights, AdamW optimizer state
(including per-parameter momentum/variance buffers), LR scheduler state, and
the global step counter.

This only works if the model we hand to the trainer has the SAME optimizer
structure the checkpoint was saved with:
  - a single param group covering every trainable parameter (654 tensors) —
    which means the encoder must stay UNFROZEN here, unlike finetune.py
  - the same optimizer (AdamW) and scheduler (NoamAnnealing) classes, so
    their load_state_dict() calls succeed
Both were verified against this checkpoint before writing this script.

Practical implication of restoring the exact scheduler state: this checkpoint
is 371k steps into a NoamAnnealing schedule that's already decayed to its
min_lr=1e-5 floor, so effectively this continues training at a small, ~constant
LR rather than through another warmup/decay cycle.
"""

import torch
import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

BASE_NEMO_CHECKPOINT = "SraVaani-nemo-checkpoint.nemo"
RESUME_CKPT = "SraVaani-Hybrid-TDT-CTC-BPE.ckpt"

TRAIN_SHARDS    = "data/wancho/train/shard_0000.tar"
TRAIN_MANIFESTS = "data/wancho/train/shard_0000.json"
VAL_SHARDS      = "data/wancho/validation/shard_0000.tar"
VAL_MANIFESTS   = "data/wancho/validation/shard_0000.json"

ADDITIONAL_EPOCHS = 2       # epochs of *new* (Wancho) data to train beyond the checkpoint
# Lower than finetune.py's batch_size=4 — the encoder is unfrozen here, so
# there's far more to hold in memory (encoder activations + doubled AdamW
# state for its parameters) than the decoder-only fine-tune.
BATCH_SIZE  = 2
NUM_WORKERS = 1
ACCUMULATE  = 16            # effective batch size = BATCH_SIZE * ACCUMULATE = 32, same as finetune.py

with open(TRAIN_MANIFESTS, encoding="utf-8") as _f:
    _n = sum(1 for line in _f if line.strip())
ADDITIONAL_STEPS = (-(-_n // BATCH_SIZE) // ACCUMULATE) * ADDITIONAL_EPOCHS

# Read just the step counter out of the checkpoint (mmap keeps this cheap —
# doesn't materialize the 5.4GB of weights/optimizer state just to read one int).
_ckpt_meta = torch.load(RESUME_CKPT, map_location="cpu", mmap=True, weights_only=False)
CKPT_GLOBAL_STEP = _ckpt_meta["global_step"]
CKPT_EPOCH = _ckpt_meta["epoch"]
del _ckpt_meta
print(f"Resuming from global_step={CKPT_GLOBAL_STEP}, epoch={CKPT_EPOCH} "
      f"-> training {ADDITIONAL_STEPS} further steps ({ADDITIONAL_EPOCHS} epochs of Wancho data)")

TARGET_MAX_STEPS = CKPT_GLOBAL_STEP + ADDITIONAL_STEPS

# Build the model from the released .nemo so tokenizer/cfg paths resolve
# correctly (the .ckpt's own cfg points at the original training machine's
# paths). Its weights get overwritten by the ckpt_path resume below anyway.
model = EncDecHybridRNNTCTCBPEModel.restore_from(BASE_NEMO_CHECKPOINT)
# Deliberately NOT calling model.encoder.freeze() here — see module docstring.

with open_dict(model.cfg):
    # Training data — tarred
    model.cfg.train_ds.manifest_filepath = TRAIN_MANIFESTS
    model.cfg.train_ds.tarred_audio_filepaths = TRAIN_SHARDS
    model.cfg.train_ds.is_tarred = True
    model.cfg.train_ds.shuffle_n = 2048   # in-memory shuffle buffer across shards
    model.cfg.train_ds.batch_size = BATCH_SIZE
    model.cfg.train_ds.num_workers = NUM_WORKERS
    model.cfg.train_ds.pin_memory = True

    # Validation data — tarred
    model.cfg.validation_ds.manifest_filepath = VAL_MANIFESTS
    model.cfg.validation_ds.tarred_audio_filepaths = VAL_SHARDS
    model.cfg.validation_ds.is_tarred = True
    model.cfg.validation_ds.batch_size = BATCH_SIZE
    model.cfg.validation_ds.num_workers = NUM_WORKERS

    # Optimizer/scheduler — structurally matching what RESUME_CKPT was saved
    # with (AdamW, NoamAnnealing) so ckpt_path resume can load their state
    # correctly. The exact values here get overwritten by the restored state
    # anyway; they just need to be the right shape/class.
    model.cfg.optim.name = "adamw"
    model.cfg.optim.lr = 1e-5
    model.cfg.optim.betas = [0.9, 0.98]
    model.cfg.optim.weight_decay = 1e-3
    model.cfg.optim.sched = OmegaConf.create({
        "name":         "NoamAnnealing",
        "d_model":      1024,
        "warmup_steps": 10000,
        "min_lr":       1e-05,
    })

model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)
model.setup_optimization(model.cfg.optim)


class MetricsLogger(pl.Callback):
    """Prints train loss every step and val WER at every eval step."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            print(f"[step {trainer.global_step}] train_loss={loss.item():.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        wer = trainer.callback_metrics.get("val_wer")
        loss = trainer.callback_metrics.get("val_loss")
        if wer is not None:
            msg = f"[step {trainer.global_step}] val_wer={wer.item():.4f}"
            if loss is not None:
                msg += f" val_loss={loss.item():.4f}"
            print(msg)


trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=-1,               # unbounded — epoch counting from the original
                                  # 63-language run is meaningless against Wancho
    max_steps=TARGET_MAX_STEPS,  # governs stopping instead
    accumulate_grad_batches=ACCUMULATE,
    gradient_clip_val=1.0,
    val_check_interval=1.0,
    log_every_n_steps=10,
    logger=False,
    enable_checkpointing=False,
    callbacks=[MetricsLogger()],
)

exp_manager(trainer, {
    "exp_dir": "experiments/",
    "name": "sra-vaani-resume-finetune",
    "create_tensorboard_logger": True,
    "create_checkpoint_callback": True,
    "checkpoint_callback_params": {
        "monitor": "val_wer",
        "mode": "min",
        "save_top_k": 3,
    },
})

# ckpt_path resume restores model weights, optimizer state, scheduler state,
# and the global step counter all together — this is what actually makes it
# a "resume" rather than a weights-only warm start.
trainer.fit(model, ckpt_path=RESUME_CKPT)

model.save_to("sra-vaani-finetuned-resumed.nemo")
