import types
import torch
import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

NEMO_CHECKPOINT = "SraVaani-nemo-checkpoint.nemo"

# Tarred shard paths — use brace expansion for multiple shards.
# Each shard_NNNN.tar must have a paired shard_NNNN.json manifest alongside it.
TRAIN_SHARDS    = "data/train/shard_0000.tar"
TRAIN_MANIFESTS = "data/train/shard_0000.json"
VAL_SHARDS      = "data/validation/shard_0000.tar"
VAL_MANIFESTS   = "data/validation/shard_0000.json"

MAX_EPOCHS = 2
BATCH_SIZE  = 4
NUM_WORKERS = 1
ACCUMULATE  = 8
with open(TRAIN_MANIFESTS, encoding="utf-8") as _f:
    _n = sum(1 for line in _f if line.strip())
MAX_STEPS = (-(-_n // BATCH_SIZE) // ACCUMULATE) * MAX_EPOCHS

model = EncDecHybridRNNTCTCBPEModel.restore_from(NEMO_CHECKPOINT)

# Optionally freeze the encoder and only fine-tune the decoder heads.
# Good starting point if your domain shift is small or you have limited data.
# Remove these two lines to fine-tune end-to-end.
model.encoder.freeze()
print("Encoder frozen — only decoder/joint weights will be updated.")

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

    # Optimizer — keep LR small to avoid destroying pretrained weights
    model.cfg.optim.lr = 1e-4
    model.cfg.optim.name = "adamw"
    model.cfg.optim.weight_decay = 1e-3
    model.cfg.optim.sched = OmegaConf.create({
        "name":         "CosineAnnealing",
        "warmup_steps": 10,
        "max_steps":    MAX_STEPS,
        "min_lr":       1e-6,
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
    max_epochs=MAX_EPOCHS,
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
    "name": "sra-vaani-finetune",
    "create_tensorboard_logger": True,
    "create_checkpoint_callback": True,
    "checkpoint_callback_params": {
        "monitor": "val_wer",
        "mode": "min",
        "save_top_k": 3,
    },
})

trainer.fit(model)

model.save_to("sra-vaani-finetuned.nemo")
