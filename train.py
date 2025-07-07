import logging
import time

from utils import seed_everything
seed_everything(1143)
import torch
torch.set_float32_matmul_precision('medium')
from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model import AUREXA_SE_AVSE, AVSE4LightningModule
from dataset import AVSE4DataModule
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        filename="model-{epoch:02d}-{val_loss:.3f}",
        save_top_k=2,
        save_last=True,
        every_n_epochs=1
    )
    callbacks = [checkpoint_callback]
    
    datamodule = AVSE4DataModule(data_root=cfg.data.root, batch_size=cfg.data.batch_size,
                                 audio_norm=cfg.data.audio_norm, rgb=cfg.data.rgb,
                                 num_channels=cfg.data.num_channels)
    
    model = AUREXA_SE_AVSE(
        audio_encoder_dim=256,
        video_encoder_dim=512,
        cross_attn_heads=8,
        cross_attn_layers=2,
        squeezeformer_blocks=4,
        squeezeformer_heads=8,
        output_audio_len=48000,
        frontend_ckpt_path=cfg.trainer.frontend_ckpt_path
    )
    
    lightning_model = AVSE4LightningModule(model, lr=cfg.trainer.lr)
    
    trainer = Trainer(default_root_dir=cfg.trainer.log_dir,
                      callbacks=callbacks, deterministic=cfg.trainer.deterministic,
                      log_every_n_steps=cfg.trainer.log_every_n_steps,
                      fast_dev_run=cfg.trainer.fast_dev_run, devices=cfg.trainer.gpus,
                      accelerator=cfg.trainer.accelerator,
                      precision=cfg.trainer.precision, strategy=cfg.trainer.strategy,
                      max_epochs=cfg.trainer.max_epochs,
                      accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                      detect_anomaly=cfg.trainer.detect_anomaly,
                      limit_train_batches=cfg.trainer.limit_train_batches,
                      limit_val_batches=cfg.trainer.limit_val_batches,
                      num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
                      gradient_clip_val=cfg.trainer.gradient_clip_val,
                      gradient_clip_algorithm="norm",
                      profiler=cfg.trainer.profiler
                      )
    
    start = time.time()
    trainer.fit(lightning_model, datamodule, ckpt_path=cfg.trainer.ckpt_path)
    log.info(f"Training completed in {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    main()