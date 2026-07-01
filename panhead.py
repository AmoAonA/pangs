#!/usr/bin/env python3
"""Zero-shot local-PAN head — the core contribution.

Given a Gaussian-baseline fused output (`direct_s1.tif`) and its rendered PAN
carrier (`pan_rendered_s1.tif`), a small convolutional head reads the LOCAL PAN
texture and predicts a per-band high-frequency correction to the (smooth)
spectral coefficient:

    coeff = smooth_LF(baseline)  +  head(PAN)
    HRMS  = coeff * carrier

The head is fit PER IMAGE, from scratch, supervised ONLY by low-resolution
consistency (MTF-downsampled HRMS must match the input LRMS). No ground-truth
HRMS is ever used for training; a GT, if provided, is used only to report
metrics. Because the head is a purely local operator on PAN, it can be applied
to PAN rendered at any output scale (arbitrary-scale friendly).

Recipe (validated over WV3/WV2/QB): 3-layer / width-64 conv head, PAN-only
input, ~3500 iters, fixed-iteration stop (no GT peak-picking). Regularization,
bigger receptive fields, multi-scale input, and PAN spectral-consistency were
all tested and did NOT help, so they are intentionally omitted here.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from psutils import genMTF
import io_utils as io


def build_mtf(ratio: int, sensor: str, channels: int, device):
    """Sensor MTF blur + stride-down operator matching the LR degradation."""
    h = genMTF(ratio, sensor, channels)
    kernel = torch.from_numpy(np.float32(h)).permute(2, 0, 1).unsqueeze(1).to(device)
    pad = (kernel.shape[-1] - 1) // 2

    def degrade(x):
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(x, kernel, groups=channels)[:, :, 2::ratio, 2::ratio]

    return degrade


class PanHead(nn.Module):
    """Small local conv head: PAN -> per-band coeff HF correction."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 64, layers: int = 3):
        super().__init__()
        mods, ch = [], in_ch
        for _ in range(layers):
            mods += [nn.Conv2d(ch, width, 3, padding=1), nn.ReLU()]
            ch = width
        self.body = nn.Sequential(*mods)
        self.out = nn.Conv2d(width, out_ch, 3, padding=1)
        with torch.no_grad():  # start as a no-op (coeff == baseline LF)
            self.out.weight.mul_(0.01)
            self.out.bias.zero_()

    def forward(self, x):
        return self.out(self.body(x))


def _fit_one(lf, carrier, ms, pan_in, mtf, C, iters, lam, layers, width, seed, device):
    """Fit one head; return final fused HRMS [C, H, W] numpy in [0, 1]."""
    torch.manual_seed(seed)
    lf_t = torch.from_numpy(lf)[None].to(device)
    carrier_t = torch.from_numpy(carrier)[None].to(device)
    ms_t = torch.from_numpy(ms)[None].to(device)
    feat = torch.from_numpy(pan_in)[None].to(device)

    net = PanHead(feat.shape[1], C, width=width, layers=layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(iters):
        opt.zero_grad()
        hrms = (lf_t + net(feat)) * carrier_t
        loss = lam * F.mse_loss(mtf(hrms), ms_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        fused = ((lf_t + net(feat)) * carrier_t).clamp(0, 1)[0].cpu().numpy()
    return fused


def run_panhead(
    ms: np.ndarray,
    pan: np.ndarray,
    carrier: np.ndarray,
    baseline_fused: np.ndarray,
    sensor: str = "WV3",
    ratio: int = 4,
    iters: int = 3500,
    lam: float = 100.0,
    layers: int = 3,
    width: int = 64,
    seeds: int = 1,
    device: str = "cuda:0",
) -> np.ndarray:
    """Fit the local-PAN head and return the refined fused HRMS [C, H, W].

    ms      : LRMS [C, h, w] in [0, 1]
    pan     : PAN  [1, H, W] in [0, 1] (input high-res panchromatic)
    carrier : rendered-PAN carrier [C, H, W] (baseline `pan_rendered_s1`)
    baseline_fused : baseline fused HRMS [C, H, W] (baseline `direct_s1`)
    seeds   : average this many independently-fit heads (>=1; 1 by default)
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    C = ms.shape[0]
    carrier = np.clip(carrier, 0, 1).astype(np.float32)
    coeff = np.clip(baseline_fused / np.maximum(carrier, 1e-3), 0, 8).astype(np.float32)
    lf = np.stack([gaussian_filter(coeff[i], 1.0) for i in range(C)]).astype(np.float32)
    pan_in = pan[:1].astype(np.float32)
    mtf = build_mtf(ratio, sensor.upper(), C, dev)

    acc = None
    for s in range(max(1, seeds)):
        fused = _fit_one(lf, carrier, ms, pan_in, mtf, C, iters, lam, layers, width, s, dev)
        acc = fused if acc is None else acc + fused
    return np.clip(acc / max(1, seeds), 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Zero-shot local-PAN head refinement")
    ap.add_argument("--run_dir", required=True,
                    help="dir with direct_s1.tif + pan_rendered_s1.tif (baseline outputs)")
    ap.add_argument("--ms", required=True, help="LRMS .tif (same patch as baseline)")
    ap.add_argument("--pan", required=True, help="PAN .tif")
    ap.add_argument("--gt", default="", help="optional GT HRMS .tif (evaluation only)")
    ap.add_argument("--sensor", default="WV3", choices=["WV3", "WV2", "QB", "GF2"])
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3500)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_fused", default="", help="output .tif path for refined HRMS")
    args = ap.parse_args()

    ms, pan, gt = io.load_pair(args.ms, args.pan, args.gt or None)
    carrier = io.load_tif(os.path.join(args.run_dir, "pan_rendered_s1.tif"))
    baseline = io.load_tif(os.path.join(args.run_dir, "direct_s1.tif"))

    fused = run_panhead(
        ms, pan, carrier, baseline,
        sensor=args.sensor, ratio=args.ratio, iters=args.iters,
        seeds=args.seeds, device=args.device,
    )

    if gt is not None:
        base_psnr = io.psnr_global(np.clip(baseline, 0, 1), gt)
        print(f"baseline  PSNR {base_psnr:.3f}  SAM {io.sam_deg(baseline, gt):.3f}")
        print(f"PAN-head  PSNR {io.psnr_global(fused, gt):.3f}  SAM {io.sam_deg(fused, gt):.3f}")

    if args.save_fused:
        os.makedirs(os.path.dirname(args.save_fused) or ".", exist_ok=True)
        tifffile.imwrite(args.save_fused, fused.astype(np.float32))
        print("saved fused ->", args.save_fused)


if __name__ == "__main__":
    main()
