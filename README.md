# Deep Learning for SFDI Optical Property Extraction

**Stony Brook University – Biomedical Engineering | NIH R01-Funded**

CNN and U-Net architectures that replace slow iterative fitting in Spatial Frequency Domain Imaging (SFDI), achieving real-time pixel-wise maps of tissue absorption (μa) and reduced scattering (μs′).

## Motivation

Classical SFDI optical property extraction relies on iterative look-up-table (LUT) or nonlinear fitting, which takes minutes per frame. This project trains neural networks directly on (R_AC, R_DC) → (μa, μs′) to achieve inference at video rate (>25 fps on a modern GPU).

## Models

| Architecture | Parameters | Speed (128×128) |
|---|---|---|
| `SFDIUNet` | ~7.7M | ~15 ms/frame |
| `AttentionSFDIUNet` | ~8.5M | ~20 ms/frame |
| `SFDILightCNN` | ~0.7M | ~4 ms/frame |

## File Structure

| File | Description |
|------|-------------|
| `unet_model.py` | U-Net, Attention U-Net, lightweight CNN architectures |
| `dataset.py` | Simulated & real SFDI dataset loaders |
| `train.py` | Full training loop with cosine LR, TensorBoard, checkpointing |

## Quick Start

```bash
pip install torch numpy scipy matplotlib

# Train on synthetic data (no real dataset needed):
python train.py --arch unet --data simulated --epochs 100 --batch 8

# Train on real experimental data:
python train.py --arch attention_unet --data /path/to/sfdi_dataset --cuda

# Smoke test all architectures:
python unet_model.py
```

## Input / Output Format

```
Input  X: [batch, 2×n_freqs, H, W]   # (R_AC, R_DC) stacked per frequency
Output y: [batch, 2, H, W]           # channel 0 = μa, channel 1 = μs'  (mm⁻¹)
```

## Loss Function

Combined MAE + relative MAE on log-scale values (scale-invariant across orders of magnitude):
```
L = 0.5 · MAE(log(pred), log(target)) + 0.5 · RelMAE(log(pred), log(target))
```

## Publications

- **Ahmmed et al.** *Depth-Sensitive Optical Property Characterization Using Multi-Frequency Laparoscopic SFDI*, bioRxiv 2026
- Tonge et al., Biomed Opt Express 12 (2021) – foundational DL-SFDI
- Aguénounon et al., Biomed Opt Express 11 (2020)

## Author

Rasel Ahmmed | rasel.ahmmed@stonybrook.edu | [Portfolio](https://raselece25.github.io)
