import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from utils import SwinTransformerV2, UNetAudioEncoder, ImprovedAudioDecoder, BidirectionalCrossAttention, Squeezeformer
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR

pesq_available = False
try:
    from torchmetrics.audio import PerceptualEvaluationSpeechQuality as PESQ
    from torchmetrics.audio import ShortTimeObjectiveIntelligibility as STOI
    pesq_available = True
except ImportError:
    pass

def check_nan_inf(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")
        return True
    return False

class AUREXA_SE(nn.Module):
    """
    AVSE model using bidirectional cross-attention
    """
    def __init__(self, audio_encoder_dim=256, video_encoder_dim=512, cross_attn_heads=8, cross_attn_layers=2,
                 squeezeformer_blocks=4, squeezeformer_heads=8, output_audio_len=48000, frontend_ckpt_path=None):
        super().__init__()
        self.device = 'cpu'
        self.output_audio_len = output_audio_len
        
        # Audio processing pipeline
        self.audio_encoder = UNetAudioEncoder(in_channels=1, base_channels=32, num_layers=4, feature_dim=audio_encoder_dim)
        self._skip_channels = self.audio_encoder.skip_channels  # For decoder skip connections
        
        # Video processing pipeline
        self.video_encoder = SwinTransformerV2(
            img_size=112, patch_size=4, in_chans=3, embed_dim=128,
            depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
            window_size=7, drop_path_rate=0.1, checkpoint_path=frontend_ckpt_path
        )
        self.video_proj = nn.Linear(video_encoder_dim, audio_encoder_dim)
        
        # Cross-modal fusion and temporal modeling
        self.cross_attention = BidirectionalCrossAttention(audio_encoder_dim, cross_attn_heads, cross_attn_layers)
        self.temporal_model = Squeezeformer(audio_encoder_dim, squeezeformer_heads, squeezeformer_blocks)
        
        self.linear_decoder = ImprovedAudioDecoder(audio_encoder_dim, output_audio_len, skip_channels=self._skip_channels)
        
        self.temporal_norm = nn.LayerNorm(audio_encoder_dim)
        self.audio_norm = nn.LayerNorm(audio_encoder_dim)
        self.video_norm = nn.LayerNorm(audio_encoder_dim)
        self.fusion_norm = nn.LayerNorm(audio_encoder_dim)

    def forward(self, noisy_audio, vis_feat):
        if check_nan_inf(noisy_audio, "noisy_audio"):
            raise ValueError("NaN/Inf in input noisy_audio")
        
        if noisy_audio.dim() == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        elif noisy_audio.dim() == 3 and noisy_audio.shape[1] != 1:
            noisy_audio = noisy_audio.mean(dim=1, keepdim=True)
        
        noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
        noisy_audio[torch.isnan(noisy_audio)] = 0
        noisy_audio[torch.isinf(noisy_audio)] = 0
        
        audio_features, audio_skips, _ = self.audio_encoder(noisy_audio, return_skips=True)
        audio_features = self.audio_norm(audio_features)
        audio_features = torch.clamp(audio_features, -10.0, 10.0)
        audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if check_nan_inf(audio_features, "audio_features"):
            raise ValueError("NaN/Inf in audio features")
        
        if vis_feat.shape[-1] == 3:
            vis_feat = vis_feat.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError("vis_feat must have 3 channels in last dim")
        
        vis_feat = torch.clamp(vis_feat, 0.0, 1.0)
        vis_feat[torch.isnan(vis_feat)] = 0
        vis_feat[torch.isinf(vis_feat)] = 0
        
        B, C, T, H, W = vis_feat.shape
        
        vis_feat_batch = vis_feat.reshape(B * T, C, H, W)
        video_embs_batch = self.video_encoder(vis_feat_batch)
        video_embs_batch = torch.clamp(video_embs_batch, -10.0, 10.0)
        video_embs_batch = torch.nan_to_num(video_embs_batch, nan=0.0, posinf=10.0, neginf=-10.0)
        video_features = video_embs_batch.reshape(B, T, -1)
        
        video_features = self.video_proj(video_features)
        video_features = self.video_norm(video_features)
        video_features = torch.clamp(video_features, -10.0, 10.0)
        video_features = torch.nan_to_num(video_features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if torch.rand(1).item() < 0.01:
            if check_nan_inf(video_features, "video_features"):
                raise ValueError("NaN/Inf in video features")
        
        audio_fused, video_fused = self.cross_attention(audio_features, video_features)
        fused = (audio_fused + video_fused) / 2
        fused = torch.clamp(fused, -10.0, 10.0)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=10.0, neginf=-10.0)
        fused = self.fusion_norm(fused)
        fused = torch.clamp(fused, -10.0, 10.0)
        fused[torch.isnan(fused)] = 0
        fused[torch.isinf(fused)] = 0
        
        if torch.rand(1).item() < 0.01:
            if check_nan_inf(fused, "fused_features"):
                raise ValueError("NaN/Inf in fused features")
        
        temporal_features = self.temporal_model(fused)
        temporal_features = torch.clamp(temporal_features, -10.0, 10.0)
        temporal_features = torch.nan_to_num(temporal_features, nan=0.0, posinf=10.0, neginf=-10.0)

        temporal_features = self.temporal_norm(temporal_features)

        output = self.linear_decoder(temporal_features, skip_connections=audio_skips)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        output = torch.clamp(output, -1.0, 1.0)
        return output

# Backward compatibility alias for existing checkpoints
ChatterboxSwinAVSE = AUREXA_SE

class AVSE4LightningModule(LightningModule):
    """
    PyTorch Lightning module for training the AVSE model.
    """
    def __init__(self, model, lr=0.0001, loss_fn=F.mse_loss):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model'])
        
        self.sdr_metric = SISDR()
        if pesq_available:
            self.pesq_metric = PESQ(16000, 'wb')
            self.stoi_metric = STOI(16000, False)
        else:
            self.pesq_metric = None
            self.stoi_metric = None

    def forward(self, noisy_audio, vis_feat):
        return self.model(noisy_audio, vis_feat)
    
    def training_step(self, batch, batch_idx):
        if torch.isnan(batch['noisy_audio']).any():
            print(f"WARNING: NaN in noisy_audio at batch {batch_idx}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        
        if torch.isnan(batch['vis_feat']).any():
            print(f"WARNING: NaN in vis_feat at batch {batch_idx}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        
        output = self(batch['noisy_audio'], batch['vis_feat'])
        target = batch['target']
        
        if target.dim() == 3:
            target = target.mean(dim=1)
        
        target = torch.clamp(target, -1.0, 1.0)
        
        if output.dim() == 3:
            output = output.squeeze(1)
        
        if torch.isnan(output).any():
            print(f"WARNING: NaN in model output at batch {batch_idx}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        
        loss = self.loss_fn(output, target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected at batch {batch_idx}")
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch['noisy_audio'], batch['vis_feat'])
        target = batch['target']
        
        if target.dim() == 3:
            target = target.mean(dim=1)
        target = torch.clamp(target, -1.0, 1.0)
        
        if output.dim() == 3:
            output = output.squeeze(1)
            
        loss = self.loss_fn(output, target)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf validation loss detected at batch {batch_idx}")
            loss = torch.tensor(1.0, device=self.device)
        self.log('val_loss', loss)
        
        sdr = self.sdr_metric(output, target)
        self.log('val_sdr', sdr, prog_bar=True)
        
        if pesq_available and self.pesq_metric is not None and self.stoi_metric is not None:
            pesq = self.pesq_metric(output, target)
            stoi = self.stoi_metric(output, target)
            self.log('val_pesq', pesq, prog_bar=True)
            self.log('val_stoi', stoi, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-8)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=3, factor=0.5
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler] 
