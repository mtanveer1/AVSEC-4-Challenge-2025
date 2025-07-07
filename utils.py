import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.swinv2 import Swinv2Model, Swinv2Config
from pytorch_lightning import LightningModule
from math import sqrt
import torchaudio

EPS = np.finfo(float).eps
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# VISUAL PROCESSING MODULES

class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=112, patch_size=24, in_chans=3, embed_dim=128, depths=[2, 2, 6, 2],
                 num_heads=[4, 8, 16, 32], window_size=7, drop_path_rate=0.1, checkpoint_path=None):
        super().__init__()
        config = Swinv2Config(
            image_size=img_size, patch_size=patch_size, num_channels=in_chans, embed_dim=embed_dim,
            depths=depths, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate
        )
        self.model = Swinv2Model(config)
        self.projection = nn.Linear(self.model.config.hidden_size, 512)
        self.max_frames = 75
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'module.' in k} if any('module.' in k for k in state_dict.keys()) else state_dict
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict, strict=False)
            self.model.eval()

    def forward(self, x):

        if self.training:
            self.model.train()
        else:
            self.model.eval()
        B, C, H, W = x.shape
        outputs = self.model(x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.projection(pooled_output)


# AUDIO ENCODER MODULES

class UNetAudioEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_layers=4, feature_dim=256):
        super().__init__()
        self.downs = nn.ModuleList()
        channels = in_channels
        self.skip_channels = []
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.downs.append(
                nn.Sequential(
                    nn.Conv1d(channels, out_channels, 4, stride=2, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            self.skip_channels.append(out_channels)
            channels = out_channels
        self.final_proj = nn.Conv1d(channels, feature_dim, 1)

    def forward(self, x, return_skips=False):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.final_proj(x)
        x = x.transpose(1, 2)
        if return_skips:
            skips_out = [s.transpose(1, 2) for s in skips]
            return x, skips_out, self.skip_channels
        return x


# CROSS-MODAL FUSION MODULES

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention mechanism for audio-visual fusion.
    Enables mutual attention between audio and video features to capture cross-modal dependencies.
    """
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.layers = layers
        self.audio_to_video = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(layers)])
        self.video_to_audio = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers * 2)])

    def forward(self, audio, video):
        """
        Bidirectional cross-attention between audio and video to return fused features.
        """
        if audio.size(1) != video.size(1):
            video = F.interpolate(video.transpose(1, 2), size=audio.size(1), mode='linear', align_corners=False).transpose(1, 2)
        for i in range(self.layers):
            a2v, _ = self.audio_to_video[i](audio, video, video)
            v2a, _ = self.video_to_audio[i](video, audio, audio)
            audio = audio + self.norms[i * 2](v2a)
            video = video + self.norms[i * 2 + 1](a2v)
        return audio, video


# TEMPORAL MODELING MODULES

class SqueezeformerBlock(nn.Module):
    def __init__(self, dim, heads, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.conv = nn.Conv1d(dim, dim, 31, padding=15, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim), nn.Dropout(0.1)
        )
        if downsample:
            self.downsample_layer = nn.Conv1d(dim, dim, 2, stride=2)

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = x + self.ffn(self.norm2(x))
        if self.downsample:
            x = self.downsample_layer(x.transpose(1, 2)).transpose(1, 2)
        return x

class Squeezeformer(nn.Module):
    def __init__(self, dim, heads, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(dim, heads, downsample=(i == num_blocks - 1)) for i in range(num_blocks)
        ])

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        return x


# DECODER MODULES

class ImprovedAudioDecoder(nn.Module):
    def __init__(self, input_dim=256, output_len=48000, skip_channels=None, hidden_dims=[128, 64, 32, 16]):
        super().__init__()
        self.output_len = output_len
        self.upsample_factors = [2, 2, 2, 2, 2, 2, 2]

        dims = [input_dim] + hidden_dims + [1]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer = nn.Sequential(
                nn.Upsample(scale_factor=self.upsample_factors[i], mode='nearest'),
                nn.Conv1d(dims[i], dims[i+1], kernel_size=5, padding=2),
                nn.BatchNorm1d(dims[i+1]) if i < len(dims) - 2 else nn.Identity(),
                nn.LeakyReLU(0.2) if i < len(dims) - 2 else nn.Tanh()
            )
            self.layers.append(layer)

        if skip_channels is None:
            skip_channels = [input_dim] * (len(dims) - 2)
        self.skip_projections = nn.ModuleList([
            nn.Conv1d(skip_channels[i], dims[i+1], 1) for i in range(len(dims) - 2)
        ])

    def forward(self, x, skip_connections=None):
        x = x.transpose(1, 2)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if skip_connections is not None and i < len(skip_connections):
                skip = skip_connections[i].transpose(1, 2)
                skip = F.interpolate(skip, size=x.shape[-1], mode='nearest')
                if i < len(self.skip_projections):
                    skip = self.skip_projections[i](skip)
                    x = x + skip

            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        if x.shape[-1] != self.output_len:
            x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=False)

        return x


# UTILITY FUNCTIONS
def seed_everything(seed: int):
    """
    Set random seeds for reproducibility across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
