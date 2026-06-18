"""
dataset.py
==========
Dataset classes for SFDI deep learning optical property extraction.

Two modes:
  1. SFDISimulatedDataset – on-the-fly Monte Carlo / diffusion-based simulation
     for pre-training without real data.
  2. SFDIDataset – loads experimentally acquired SFDI images paired with
     ground-truth optical property maps (from iterative fitting).

Dataset tensor format:
  Input  X : [2*n_freqs, H, W]  –  (I_AC, I_DC) stacked per frequency
  Target y : [2, H, W]          –  (mu_a, mu_s_prime) in mm^-1

Author : Rasel Ahmmed | rasel.ahmmed@stonybrook.edu
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ── Physics helper (duplicated here so dataset.py is self-contained) ───────────

def _r_eff(n: float) -> float:
    return -1.440 * n**-2 + 0.710 * n**-1 + 0.668 + 0.0636 * n


def _diffuse_reflectance(mu_a: np.ndarray, mu_s_prime: np.ndarray,
                          fx: float, n: float = 1.40):
    """Vectorised Cuccia 2009 diffuse reflectance model."""
    mu_t  = mu_a + mu_s_prime
    mu_e  = np.sqrt(3.0 * mu_a * mu_t)
    mu_ef = np.sqrt(mu_e**2 + (2 * np.pi * fx)**2)
    A     = (1.0 - _r_eff(n)) / (2.0 * (1.0 + _r_eff(n)))
    R_dc  = A / (1.0 + (2.0 * A * mu_e)  / (3.0 * mu_t))
    R_ac  = A / (1.0 + (2.0 * A * mu_ef) / (3.0 * mu_t))
    return R_ac, R_dc


# ── Simulated Dataset ──────────────────────────────────────────────────────────

class SFDISimulatedDataset(Dataset):
    """
    Synthetic SFDI dataset generated on-the-fly using the diffusion model.

    Each sample is a spatially heterogeneous tissue phantom with smoothly
    varying optical properties (Gaussian random field).

    Args:
        n_samples    : Number of synthetic images to generate
        image_size   : Spatial resolution (H = W)
        n_freqs      : Number of spatial frequencies
        freq_list    : List of spatial frequencies in mm^-1 (None → default)
        noise_level  : Gaussian noise standard deviation added to reflectance
        mu_a_range   : (min, max) absorption [mm^-1]
        mu_sp_range  : (min, max) reduced scattering [mm^-1]
        n_tissue     : Tissue refractive index
        seed         : Random seed for reproducibility
    """

    def __init__(self,
                 n_samples:    int   = 5000,
                 image_size:   int   = 128,
                 n_freqs:      int   = 2,
                 freq_list:    Optional[List[float]] = None,
                 noise_level:  float = 0.005,
                 mu_a_range:   Tuple[float, float] = (1e-3, 0.1),
                 mu_sp_range:  Tuple[float, float] = (0.5, 3.0),
                 n_tissue:     float = 1.40,
                 seed:         int   = 42):
        self.n_samples   = n_samples
        self.image_size  = image_size
        self.n_freqs     = n_freqs
        self.noise_level = noise_level
        self.mu_a_range  = mu_a_range
        self.mu_sp_range = mu_sp_range
        self.n_tissue    = n_tissue

        if freq_list is None:
            self.freqs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25][:n_freqs]
        else:
            self.freqs = freq_list[:n_freqs]

        self.rng = np.random.default_rng(seed)
        logger.info(f"SFDISimulatedDataset: {n_samples} samples, "
                    f"size={image_size}×{image_size}, freqs={self.freqs}")

    def _gaussian_random_field(self, correlation_length: float = 20.0) -> np.ndarray:
        """
        Generate a smooth 2D random field via Gaussian smoothing of white noise.
        Used to create spatially varying optical property maps.
        """
        from scipy.ndimage import gaussian_filter
        noise = self.rng.standard_normal((self.image_size, self.image_size))
        return gaussian_filter(noise, sigma=correlation_length)

    def _generate_op_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spatially heterogeneous mu_a and mu_s' maps."""
        field1 = self._gaussian_random_field(15.0)
        field2 = self._gaussian_random_field(12.0)

        # Normalize to [0, 1] then rescale to physiological range
        def norm_scale(f, lo, hi):
            f_n = (f - f.min()) / (f.ptp() + 1e-12)
            return lo + f_n * (hi - lo)

        mu_a       = norm_scale(field1, *self.mu_a_range)
        mu_s_prime = norm_scale(field2, *self.mu_sp_range)
        return mu_a, mu_s_prime

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.rng = np.random.default_rng(idx)   # deterministic per index

        mu_a, mu_s_prime = self._generate_op_maps()

        channels = []
        for fx in self.freqs:
            R_ac, R_dc = _diffuse_reflectance(mu_a, mu_s_prime, fx, self.n_tissue)
            noise = self.rng.standard_normal(R_ac.shape) * self.noise_level
            channels.append(R_ac + noise)
            channels.append(R_dc + noise)

        X = np.stack(channels, axis=0).astype(np.float32)     # [2*n_freqs, H, W]
        y = np.stack([mu_a, mu_s_prime], axis=0).astype(np.float32)  # [2, H, W]

        return torch.from_numpy(X), torch.from_numpy(y)


# ── Real Experimental Dataset ──────────────────────────────────────────────────

class SFDIDataset(Dataset):
    """
    Load experimentally acquired SFDI data.

    Expected directory layout:
        root/
          sample_0001/
            input_fx0.npy      # calibrated R_AC at frequency 0
            input_dc0.npy      # calibrated R_DC at frequency 0
            input_fx1.npy
            input_dc1.npy
            ...
            mu_a.npy           # ground-truth absorption map
            mu_s_prime.npy     # ground-truth reduced scattering map
          sample_0002/
            ...

    All .npy files should be float32 arrays of shape [H, W].

    Args:
        root       : Path to dataset root directory
        image_size : Resize images to this size (None → no resize)
        augment    : Apply random horizontal/vertical flips (training only)
    """

    def __init__(self, root: str,
                 image_size: Optional[int] = 128,
                 augment: bool = False,
                 n_freqs: int = 2):
        self.root       = Path(root)
        self.image_size = image_size
        self.augment    = augment
        self.n_freqs    = n_freqs

        self.samples = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not self.samples:
            raise FileNotFoundError(f"No sample directories found in {root}")
        logger.info(f"SFDIDataset: {len(self.samples)} samples from {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npy(self, path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if self.image_size is not None and arr.shape != (self.image_size, self.image_size):
            from scipy.ndimage import zoom
            scale = self.image_size / arr.shape[0]
            arr = zoom(arr, scale, order=1)
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.samples[idx]

        channels = []
        for k in range(self.n_freqs):
            ac = self._load_npy(sample_dir / f'input_fx{k}.npy')
            dc = self._load_npy(sample_dir / f'input_dc{k}.npy')
            channels.extend([ac, dc])

        mu_a       = self._load_npy(sample_dir / 'mu_a.npy')
        mu_s_prime = self._load_npy(sample_dir / 'mu_s_prime.npy')

        X = np.stack(channels, axis=0)                           # [2*n_freqs, H, W]
        y = np.stack([mu_a, mu_s_prime], axis=0)                 # [2, H, W]

        if self.augment:
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=-1).copy()   # horizontal flip
                y = np.flip(y, axis=-1).copy()
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=-2).copy()   # vertical flip
                y = np.flip(y, axis=-2).copy()

        return torch.from_numpy(X), torch.from_numpy(y)


# ── Dataset Stats ──────────────────────────────────────────────────────────────

def compute_dataset_stats(dataset: Dataset, n_samples: int = 500):
    """
    Compute per-channel mean and std over a subset of the dataset.
    Useful for input normalisation.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=True, num_workers=0)
    all_X, all_y = [], []
    for i, (X, y) in enumerate(loader):
        all_X.append(X)
        all_y.append(y)
        if i * 32 >= n_samples:
            break

    X_cat = torch.cat(all_X, dim=0)   # [N, C, H, W]
    y_cat = torch.cat(all_y, dim=0)

    stats = {
        'input_mean': X_cat.mean(dim=(0, 2, 3)).numpy(),
        'input_std':  X_cat.std(dim=(0, 2, 3)).numpy(),
        'mu_a_mean':  y_cat[:, 0].mean().item(),
        'mu_a_std':   y_cat[:, 0].std().item(),
        'mu_sp_mean': y_cat[:, 1].mean().item(),
        'mu_sp_std':  y_cat[:, 1].std().item(),
    }
    return stats


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing SFDISimulatedDataset …")
    ds = SFDISimulatedDataset(n_samples=100, image_size=64, n_freqs=2)
    X, y = ds[0]
    print(f"  X shape: {tuple(X.shape)}, dtype: {X.dtype}")
    print(f"  y shape: {tuple(y.shape)}, dtype: {y.dtype}")
    print(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  mu_a range: [{y[0].min():.4f}, {y[0].max():.4f}]")
    print(f"  mu_s' range: [{y[1].min():.4f}, {y[1].max():.4f}]")

    stats = compute_dataset_stats(ds, n_samples=200)
    print(f"\nDataset stats (input mean): {stats['input_mean']}")
    print("SFDISimulatedDataset OK ✓")
