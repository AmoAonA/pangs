"""Minimal, dependency-light I/O and metric helpers for the PanSplat release.

Vendored (not imported from the experimental repo) so the release is
self-contained. Covers: channel-first coercion, TIFF loading of an
MS / PAN (/ GT) pair, and reference metrics (PSNR / SAM) used only when a
ground-truth HRMS is supplied for evaluation.
"""
from __future__ import annotations

import numpy as np
import tifffile


def as_chw(x: np.ndarray) -> np.ndarray:
    """Coerce a 2D/3D array to channel-first [C, H, W] float32."""
    x = np.squeeze(np.asarray(x))
    if x.ndim == 2:
        return x[None].astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {x.shape}")
    if x.shape[0] <= 16:
        return x.astype(np.float32)
    if x.shape[-1] <= 16:
        return np.transpose(x, (2, 0, 1)).astype(np.float32)
    raise ValueError(f"Cannot infer channel axis for shape {x.shape}")


def load_tif(path: str) -> np.ndarray:
    """Load a TIFF as channel-first float32 (no normalization)."""
    return as_chw(tifffile.imread(path).astype(np.float32))


def load_pair(ms_path: str, pan_path: str, gt_path: str | None = None):
    """Load an MS + PAN (+ optional GT) TIFF triple, radiometrically
    normalized to [0, 1] by their shared max. Returns (ms, pan, gt|None).
    """
    ms = load_tif(ms_path)
    pan = load_tif(pan_path)
    gt = load_tif(gt_path) if gt_path else None
    hi = max(
        float(ms.max()),
        float(pan.max()),
        float(gt.max()) if gt is not None else 0.0,
    )
    if hi > 1.0:
        ms = ms / hi
        pan = pan / hi
        if gt is not None:
            gt = gt / hi
    return ms, pan, gt


def psnr_global(fused: np.ndarray, gt: np.ndarray) -> float:
    """Full-channel PSNR (single MSE over all bands). This is the
    UCL/literature-consistent convention used in our benchmark table.
    """
    f = np.clip(fused, 0.0, 1.0).astype(np.float64)
    g = gt.astype(np.float64)
    mse = float(np.mean((f - g) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def psnr_per_band(fused: np.ndarray, gt: np.ndarray) -> float:
    """Per-band-averaged PSNR (common in pansharpening papers; ~1.5-2 dB
    higher than the global convention).
    """
    f = np.clip(fused, 0.0, 1.0).astype(np.float64)
    g = gt.astype(np.float64)
    vals = [
        10.0 * np.log10(1.0 / max(float(np.mean((f[b] - g[b]) ** 2)), 1e-12))
        for b in range(g.shape[0])
    ]
    return float(np.mean(vals))


def sam_deg(fused: np.ndarray, gt: np.ndarray) -> float:
    """Spectral Angle Mapper in degrees (lower = better color fidelity)."""
    f = np.clip(fused, 0.0, 1.0).astype(np.float64)
    g = gt.astype(np.float64)
    num = (f * g).sum(0)
    den = np.sqrt((f ** 2).sum(0)) * np.sqrt((g ** 2).sum(0)) + 1e-8
    return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))).mean())
