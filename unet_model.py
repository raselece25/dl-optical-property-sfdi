"""
unet_model.py
=============
U-Net and CNN architectures for deep learning-based optical property
extraction from SFDI (Spatial Frequency Domain Imaging) data.

Both networks take multi-channel SFDI inputs (one channel per spatial
frequency and/or wavelength) and output two-channel optical property maps:
    Channel 0 → μa  (absorption coefficient, mm⁻¹)
    Channel 1 → μs' (reduced scattering coefficient, mm⁻¹)

Reference:
    Tonge et al., Biomed Opt Express 12 (2021)
    Aguénounon et al., Biomed Opt Express 11 (2020)
    Ahmmed et al., bioRxiv 2026

Author : Rasel Ahmmed, PhD Candidate
         Stony Brook University – Biomedical Engineering
Email  : rasel.ahmmed@stonybrook.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ── Building Blocks ────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (standard encoder/decoder block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two sequential ConvBnRelu blocks (standard U-Net cell)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch,  out_ch, dropout=dropout),
            ConvBnRelu(out_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with optional projection shortcut.
    Used in the residual U-Net variant.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


# ── U-Net ──────────────────────────────────────────────────────────────────────

class SFDIUNet(nn.Module):
    """
    U-Net for pixel-wise SFDI optical property regression.

    Architecture:
        Encoder: 4 downsampling levels (max-pool)
        Bottleneck: double-conv
        Decoder: 4 upsampling levels (bilinear + double-conv)
        Head: 1×1 Conv → output channels (2: μa, μs')

    Args:
        in_channels  : Number of input SFDI channels (e.g. 2 freqs × 2 phases = 4)
        out_channels : 2 (μa and μs' maps)
        base_features: Feature maps in first encoder level (doubles each level)
        dropout      : Dropout probability in encoder/decoder
    """

    def __init__(self,
                 in_channels:   int = 4,
                 out_channels:  int = 2,
                 base_features: int = 32,
                 dropout:       float = 0.1):
        super().__init__()

        f = base_features
        self.enc1 = DoubleConv(in_channels, f,   dropout=dropout)
        self.enc2 = DoubleConv(f,           f*2, dropout=dropout)
        self.enc3 = DoubleConv(f*2,         f*4, dropout=dropout)
        self.enc4 = DoubleConv(f*4,         f*8, dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f*8, f*16, dropout=dropout)

        self.up4   = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.dec4  = DoubleConv(f*16, f*8, dropout=dropout)

        self.up3   = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.dec3  = DoubleConv(f*8, f*4, dropout=dropout)

        self.up2   = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.dec2  = DoubleConv(f*4, f*2, dropout=dropout)

        self.up1   = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec1  = DoubleConv(f*2, f, dropout=dropout)

        self.head  = nn.Conv2d(f, out_channels, 1)

        # Softplus ensures strictly positive output (optical properties ≥ 0)
        self.activation = nn.Softplus()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.activation(self.head(d1))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Lightweight CNN (fast inference) ──────────────────────────────────────────

class SFDILightCNN(nn.Module):
    """
    Lightweight fully-convolutional CNN for real-time SFDI inference.

    Pixel receptive field: 33×33 (5 layers, kernel 3).
    ~10× fewer parameters than U-Net – suitable for bedside/laparoscopy use.

    Args:
        in_channels : Number of SFDI input channels
        out_channels: 2 (μa and μs')
        n_filters   : Width of hidden layers
        n_layers    : Number of intermediate convolutional layers
    """

    def __init__(self,
                 in_channels:  int = 4,
                 out_channels: int = 2,
                 n_filters:    int = 64,
                 n_layers:     int = 5):
        super().__init__()

        layers: List[nn.Module] = [ConvBnRelu(in_channels, n_filters)]
        for _ in range(n_layers - 2):
            layers.append(ResidualBlock(n_filters, n_filters))
        layers.append(nn.Conv2d(n_filters, out_channels, 1))

        self.net        = nn.Sequential(*layers)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Attention U-Net variant ────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al. 2018) for skip connections.
    Helps the network focus on tissue regions of interest.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


class AttentionSFDIUNet(SFDIUNet):
    """
    U-Net with attention gates on all skip connections.
    Inherits encoder/decoder from SFDIUNet; replaces plain skip connections.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 2,
                 base_features: int = 32, dropout: float = 0.1):
        super().__init__(in_channels, out_channels, base_features, dropout)
        f = base_features
        self.att4 = AttentionGate(F_g=f*8,  F_l=f*8,  F_int=f*4)
        self.att3 = AttentionGate(F_g=f*4,  F_l=f*4,  F_int=f*2)
        self.att2 = AttentionGate(F_g=f*2,  F_l=f*2,  F_int=f)
        self.att1 = AttentionGate(F_g=f,    F_l=f,    F_int=f//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, e4)], dim=1))

        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))

        return self.activation(self.head(d1))


# ── Model Factory ──────────────────────────────────────────────────────────────

def build_model(arch: str = 'unet', **kwargs) -> nn.Module:
    """
    Convenience factory.

    Args:
        arch: 'unet' | 'attention_unet' | 'cnn'
        **kwargs: passed to the model constructor
    """
    arch = arch.lower()
    registry = {
        'unet':           SFDIUNet,
        'attention_unet': AttentionSFDIUNet,
        'cnn':            SFDILightCNN,
    }
    if arch not in registry:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(registry)}")
    model = registry[arch](**kwargs)
    print(f"[build_model] {arch.upper()} | params: {model.n_params:,}")
    return model


# ── Quick Smoke Test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    B, C, H, W = 2, 4, 128, 128   # batch=2, 4 SFDI input channels, 128×128

    for arch in ('unet', 'attention_unet', 'cnn'):
        model  = build_model(arch, in_channels=C, out_channels=2).to(device)
        x      = torch.randn(B, C, H, W, device=device)
        out    = model(x)
        assert out.shape == (B, 2, H, W), f"Output shape mismatch: {out.shape}"
        print(f"  {arch}: input {tuple(x.shape)} → output {tuple(out.shape)}  ✓")
        assert (out >= 0).all(), "Negative optical property values!"

    print("\nAll model smoke tests passed.")
