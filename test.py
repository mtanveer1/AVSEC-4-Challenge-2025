import os
from os.path import isfile, join
from os import makedirs

import soundfile as sf
import torch
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import numpy as np

from dataset import AVSE4DataModule
from model import AVSE4LightningModule
from model import AUREXA_SE

SAMPLE_RATE = 16000

def enhance(model, data, device):
    model.eval()
    with torch.no_grad():
        noisy_audio = data['noisy_audio'].unsqueeze(0).to(device)
        vis_feat = data['vis_feat'].unsqueeze(0).to(device)
        estimated_audio = model(noisy_audio, vis_feat)
        if isinstance(estimated_audio, (tuple, list)):
            estimated_audio = estimated_audio[0]
        estimated_audio = estimated_audio.squeeze().cpu().numpy()
        return None, None, estimated_audio

def enhance_full_audio(model, data, device, chunk_size=48000, video_fps=25, frames_per_chunk=75):

    noisy_audio = data['noisy_audio'].to(device)
    vis_feat = data['vis_feat'].to(device)

    if noisy_audio.dim() == 2 and noisy_audio.shape[0] > 1:
        noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
    noisy_audio = noisy_audio.squeeze(0)  # [N]

    total_len = noisy_audio.shape[0]
    total_frames = vis_feat.shape[0]
    outputs = []

    samples_per_frame = total_len / total_frames

    # Padding
    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        chunk = noisy_audio[start:end]
        
        if chunk.shape[0] < chunk_size:
            pad = torch.zeros(chunk_size - chunk.shape[0], device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=0)
        chunk = chunk.unsqueeze(0).unsqueeze(0)

        frame_start = int(start / samples_per_frame)
        frame_end = int(end / samples_per_frame)
        video_chunk = vis_feat[frame_start:frame_end]
        
        if video_chunk.shape[0] < frames_per_chunk:
            pad_shape = (frames_per_chunk - video_chunk.shape[0],) + video_chunk.shape[1:]
            video_chunk = torch.cat([video_chunk, torch.zeros(pad_shape, device=video_chunk.device, dtype=video_chunk.dtype)], dim=0)
        video_chunk = video_chunk.unsqueeze(0)

        out = model(chunk, video_chunk)
        out = out.squeeze().cpu().numpy()
        outputs.append(out[:end-start])
        
    enhanced_audio = np.concatenate(outputs)
    return None, None, enhanced_audio

@hydra.main(config_path="conf", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main evaluation function
    """
    model_uid = cfg.get("model_uid", "default_model")
    enhanced_root = join(cfg.save_dir, model_uid)
    makedirs(cfg.save_dir, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)
    
    datamodule = AVSE4DataModule(data_root=cfg.data.root, batch_size=1, rgb=cfg.data.rgb,
                                 num_channels=cfg.data.num_channels, audio_norm=cfg.data.audio_norm)
    
    if cfg.data.dev_set and cfg.data.eval_set:
        raise RuntimeError("Select either dev set or test set")
    elif cfg.data.dev_set:
        dataset = datamodule.dev_dataset
    elif cfg.data.eval_set:
        dataset = datamodule.eval_dataset
    else:
        raise RuntimeError("Select one of dev set and test set")
    
    try:
        model = AUREXA_SE(
            audio_encoder_dim=256,
            video_encoder_dim=512,
            cross_attn_heads=8,
            cross_attn_layers=2,
            squeezeformer_blocks=4,
            squeezeformer_heads=8,
            output_audio_len=48000,
            frontend_ckpt_path=None
        )
        lightning_model = AVSE4LightningModule.load_from_checkpoint(cfg.ckpt_path, model=model)
        print("Model loaded successfully")
    except Exception as e:
        raise FileNotFoundError(f"Cannot load model weights: {cfg.ckpt_path}\n{e}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    lightning_model.to(device)
    lightning_model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Enhancing audio"):
            data = dataset[i]
            scene = data['scene'] if 'scene' in data else str(i)
            filename = f"{scene}.wav"
            enhanced_path = join(enhanced_root, filename)
            
            if not isfile(enhanced_path):
                _, _, estimated_audio = enhance_full_audio(lightning_model.model, data, device)
                
                os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
                
                if not np.issubdtype(estimated_audio.dtype, np.floating):
                    estimated_audio = estimated_audio.astype(np.float32)
                    
                sf.write(enhanced_path, estimated_audio, samplerate=SAMPLE_RATE)

if __name__ == '__main__':
    main()
