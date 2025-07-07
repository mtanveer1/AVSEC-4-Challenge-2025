import os
from os.path import isfile, join
import logging
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from pytorch_lightning import LightningDataModule
import torchaudio
from torch.utils.data import Dataset


MAX_FRAMES = 75
MAX_AUDIO_LEN = 48000
SEED = 1143
SAMPLING_RATE = 16000
FRAMES_PER_SECOND = 25

def subsample_list(inp_list: List, sample_rate: float) -> List:
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]

class AVSE4Dataset(Dataset):
    def __init__(self, scenes_root, shuffle=False, seed=SEED, subsample=1,
                 clipped_batch=False, test_set=False, rgb=False,
                 audio_norm=False, num_channels=1):
        super().__init__()

        assert num_channels in [1, 2], "Number of channels must be 1 or 2"
        assert os.path.isdir(scenes_root), f"Scenes root {scenes_root} not found"
        
        self.num_channels = num_channels
        self.mono = num_channels == 1
        self.img_size = 112
        self.audio_norm = audio_norm
        self.test_set = test_set
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.rgb = rgb
        
        self.files_list = self.build_files_list()
        if shuffle:
            random.seed(seed)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
            
        logging.info(f"Found {len(self.files_list)} utterances")

    def build_files_list(self) -> List[Tuple[str, str, str, str, str]]:
        if isinstance(self.scenes_root, list):
            return [file for root in self.scenes_root for file in self.get_files_list(root)]
        return self.get_files_list(self.scenes_root)

    def get_files_list(self, scenes_root: str) -> List[Tuple[str, str, str, str, str]]:
        files_list = []
        
        if self.test_set:

            for file in os.listdir(scenes_root):
                if file.endswith("_mono_mix.wav"):
                    scene_name = file.replace("_mono_mix.wav", "")
                    mix_path = join(scenes_root, file)
                    video_path = join(scenes_root, f"{scene_name}_silent.mp4")
                    

                    if os.path.isfile(mix_path) and os.path.isfile(video_path):

                        dummy_target = join(scenes_root, f"{scene_name}_target_anechoic.wav")
                        dummy_interferer = join(scenes_root, f"{scene_name}_interferer.wav")
                        dummy_mix = join(scenes_root, f"{scene_name}_mix.wav")
                        
                        files = (
                            dummy_target,
                            dummy_interferer,
                            mix_path,
                            video_path,
                            dummy_mix,
                        )
                        files_list.append(files)
        else:
            for file in os.listdir(scenes_root):
                if file.endswith("_target_anechoic.wav"):
                    files = (
                        join(scenes_root, file),
                        join(scenes_root, file.replace("target_anechoic", "interferer")),
                        join(scenes_root, file.replace("target_anechoic", "mono_mix")),
                        join(scenes_root, file.replace("target_anechoic.wav", "silent.mp4")),
                        join(scenes_root, file.replace("target_anechoic", "mix")),
                    )

                    if all(isfile(f) for f in files if not f.endswith("_interferer.wav")):
                        files_list.append(files)
        
        return files_list

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, idx):
        clean_path, _, mix_path, video_path, _ = self.files_list[idx]

        video_tensor = self._load_video(video_path)
        noisy_audio, _ = torchaudio.load(mix_path)

        if self.test_set:
            clean_audio = torch.zeros_like(noisy_audio)

        else:
            clean_audio, _ = torchaudio.load(clean_path)

        if self.mono:
            if noisy_audio.dim() == 2:
                noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
            elif noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0)
        else:
            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0).repeat(2, 1)

        if self.mono:
            if clean_audio.dim() == 2:
                clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
            elif clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0)
        else:
            if clean_audio.dim() == 1:
                clean_audio = clean_audio.unsqueeze(0).repeat(2, 1)

        if self.audio_norm:
            noisy_audio = noisy_audio / (noisy_audio.abs().max() + 1e-6)
            clean_audio = clean_audio / (clean_audio.abs().max() + 1e-6)

        video_fps = 25
        audio_sr = 16000
        n_video_frames = video_tensor.shape[0] if self.rgb else video_tensor.shape[1]
        video_duration = n_video_frames / video_fps
        audio_duration = noisy_audio.shape[1] / audio_sr

        def pad_or_truncate_audio(audio, max_len=48000):
            c, n = audio.shape
            if n < max_len:
                pad = torch.zeros((c, max_len - n), dtype=audio.dtype)
                audio = torch.cat([audio, pad], dim=1)
            else:
                audio = audio[:, :max_len]
            return audio

        def pad_or_truncate_video(video, max_frames=75):
            T = video.shape[0]
            if T < max_frames:
                pad_shape = (max_frames - T,) + video.shape[1:]
                pad = torch.zeros(pad_shape, dtype=video.dtype)
                video = torch.cat([video, pad], dim=0)
            else:
                video = video[:max_frames]
            return video

        if self.clipped_batch:
            noisy_audio = pad_or_truncate_audio(noisy_audio)
            clean_audio = pad_or_truncate_audio(clean_audio)
            video_tensor = pad_or_truncate_video(video_tensor)
            duration = 3
            n_video_frames = int(duration * video_fps)
            num_audio_samples = int(duration * audio_sr)
        else:
            duration = audio_duration
            n_video_frames = video_tensor.shape[0]
            num_audio_samples = noisy_audio.shape[1]

        if self.rgb:
            video_tensor = video_tensor.transpose(1, 2).transpose(2, 3)
        else:
            video_tensor = video_tensor.transpose(0, 1).transpose(1, 2)

        scene = os.path.basename(mix_path).split('_')[0]
        
        return {
            "noisy_audio": noisy_audio,
            "target": clean_audio,
            "vis_feat": video_tensor,
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "duration": duration,
            "video_fps": video_fps,
            "audio_sr": audio_sr,
            "video_path": video_path,
            "audio_path": mix_path,
            "scene": scene,
            "n_video_frames": n_video_frames,
            "num_audio_samples": num_audio_samples,
        }

    def _load_video(self, video_path: str) -> torch.Tensor:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        frames = self._process_frames(frames)
        
        if not self.rgb:
            frames = frames[np.newaxis, ...]
        else:
            frames = frames.transpose(0, 3, 1, 2)
            
        return torch.from_numpy(frames).float()

    def _process_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.array([frame[56:-56, 56:-56, :] for frame in frames])
        
        if not self.rgb:
            bg_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]).astype(np.float32)
        else:
            bg_frames = frames.astype(np.float32)
            
        bg_frames /= 255.0

        if len(bg_frames) < MAX_FRAMES:
            pad_shape = (MAX_FRAMES - len(bg_frames), self.img_size, self.img_size, 3) if self.rgb else (
                        MAX_FRAMES - len(bg_frames), self.img_size, self.img_size)
            bg_frames = np.concatenate((bg_frames, np.zeros(pad_shape, dtype=bg_frames.dtype)), axis=0)

        return bg_frames

class AVSE4DataModule(LightningDataModule):
    def __init__(self, data_root, batch_size=16, audio_norm=False, rgb=True, num_channels=1):
        super().__init__()
        self.batch_size = batch_size
        
        self.train_dataset_batch = AVSE4Dataset(
            join(data_root, "train/scenes"), 
            rgb=rgb, shuffle=True, num_channels=num_channels, 
            clipped_batch=True, audio_norm=audio_norm
        )
        
        self.dev_dataset_batch = AVSE4Dataset(
            join(data_root, "dev/scenes"), 
            rgb=rgb, num_channels=num_channels, 
            clipped_batch=True, audio_norm=audio_norm
        )
        
        self.dev_dataset = AVSE4Dataset(
            join(data_root, "dev/scenes"), 
            clipped_batch=True, rgb=rgb, num_channels=num_channels,
            audio_norm=audio_norm
        )
        
        self.eval_dataset = AVSE4Dataset(
            join(data_root, "dev/scenes"), 
            clipped_batch=False, rgb=rgb, num_channels=num_channels,
            audio_norm=audio_norm, test_set=True
        )

    def train_dataloader(self):
        assert len(self.train_dataset_batch) > 0, "No training data found"
        return torch.utils.data.DataLoader(
            self.train_dataset_batch, 
            batch_size=self.batch_size, 
            num_workers=2, pin_memory=True, 
            persistent_workers=True, shuffle=True
        )

    def val_dataloader(self):
        assert len(self.dev_dataset_batch) > 0, "No validation data found"
        return torch.utils.data.DataLoader(
            self.dev_dataset_batch, 
            batch_size=self.batch_size, 
            num_workers=2, pin_memory=True, 
            persistent_workers=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset, 
            batch_size=self.batch_size, 
            num_workers=2
        ) 
