#!/usr/bin/env python3
"""PanSplat — one-command zero-shot, arbitrary-scale pansharpening.

Pipeline (per image, from scratch, no pretrained weights, no ground truth):

    1. Gaussian baseline   : fit 2D Gaussians to PAN geometry and a smooth
       spectral coefficient (Stage 1 + Stage 2), rendering a fused HRMS and its
       PAN carrier at each requested output scale.
    2. Local-PAN head      : refine the coefficient's high frequency from local
       PAN texture, supervised only by low-resolution consistency.

Usage:
    python run_pansplat.py MS.tif PAN.tif --sensor WV3 --scales 1,2,4
    python run_pansplat.py MS.tif PAN.tif --gt GT.tif        # also prints PSNR/SAM

Output: for each scale s, `<out>/fused_s<scale>.tif` (the refined HRMS).
Scale 1.0 is native resolution; scales > 1 render beyond PAN resolution with no
retraining — the representation's continuous-field advantage.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tifffile

import eval_arbitrary_scale as eas
import io_utils as io
from panhead import run_panhead


BENCHMARK_BASELINE_OVERRIDES = [
    "--stage2_densify_score_mode", "pan_hf",
    "--stage2_densify_shape_mode", "default",
    "--stage2_loss_curriculum", "True",
    "--stage2_loss_switch_step", "1000",
    "--stage2_loss_switch_ramp_steps", "0",
    "--stage2_loss_lr_low_weight", "100.0",
    "--stage2_loss_lr_low_sigma", "1.0",
    "--stage2_loss_lr_low_type", "mse",
    "--stage2_loss_gate_aux", "True",
    "--stage2_loss_direct_scale_after", "0.05",
]


def _tag(scale: float) -> str:
    return f"s{scale:.4g}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser(description="Zero-shot arbitrary-scale pansharpening")
    ap.add_argument("ms", help="LRMS GeoTIFF")
    ap.add_argument("pan", help="PAN GeoTIFF")
    ap.add_argument("--gt", default="", help="optional GT HRMS .tif (evaluation only)")
    ap.add_argument("--sensor", default="WV3", choices=["WV3", "WV2", "QB", "GF2"])
    ap.add_argument("--scales", default="1", help="comma list, e.g. 1,2,4")
    ap.add_argument("--iters", type=int, default=3500, help="PAN-head iterations")
    ap.add_argument("--seeds", type=int, default=1, help="PAN-head seed-average count")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cfg", default="cfgs/ucl_s1cap15k_s2panhf15k_newgeo_fixedgain_tv_fast.yaml")
    ap.add_argument("--out", default="pansplat_out")
    ap.add_argument(
        "--plain_baseline",
        action="store_true",
        help="disable benchmark baseline overrides; mainly for ablation/debugging",
    )
    args = ap.parse_args()

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)

    # 1) Gaussian baseline via the arbitrary-scale driver (writes direct_/pan_rendered_ per scale)
    base_ns = argparse.Namespace(
        dataset=args.sensor.lower(), mode="rr", data_dir="", patch_index=1,
        data_root=os.path.dirname(os.path.abspath(args.ms)) or ".",
        input_path=os.path.basename(args.ms), ref_path=os.path.basename(args.pan),
        gt_path=(os.path.abspath(args.gt) if args.gt else ""),
        cfg=args.cfg, device=args.device, model_module="model",
        scales=",".join(str(s) for s in scales), output_dir=args.out,
        self_consistency=False, save_visuals=False, visual_tile_size=256,
        chroma_safe_projection="off", chroma_safe_topk=0.10,
        chroma_safe_global_weight=0.25, chroma_safe_bright_weight=0.75,
        chroma_safe_intensity_source="pred", chroma_safe_intensity_mult=1.05,
        chroma_safe_intensity_margin=0.02, chroma_safe_eps=1e-3, keep_temp=False,
    )
    print("[1/2] Fitting Gaussian baseline ...")
    cfg_overrides = [] if args.plain_baseline else BENCHMARK_BASELINE_OVERRIDES
    eas.run(base_ns, cfg_overrides=cfg_overrides)
    run_dir = os.path.join(args.out, f"{args.sensor.lower()}_rr_p001")

    # inputs for the head (normalized identically to the baseline loader)
    ms, pan, gt = io.load_pair(args.ms, args.pan, args.gt or None)

    # 2) Local-PAN head refinement at each scale
    print("[2/2] Fitting zero-shot local-PAN head ...")
    for s in scales:
        tag = _tag(s)
        carrier = io.load_tif(os.path.join(run_dir, f"pan_rendered_{tag}.tif"))
        baseline = io.load_tif(os.path.join(run_dir, f"direct_{tag}.tif"))
        fused = run_panhead(
            ms, pan, carrier, baseline,
            sensor=args.sensor, ratio=4, iters=args.iters,
            seeds=args.seeds, device=args.device,
        )
        out_path = os.path.join(args.out, f"fused_{tag}.tif")
        tifffile.imwrite(out_path, fused.astype(np.float32))
        msg = f"  scale {s:g}: {out_path}"
        if gt is not None and abs(s - 1.0) < 1e-6:
            msg += (f"  |  baseline {io.psnr_global(np.clip(baseline,0,1), gt):.2f} dB"
                    f" -> PAN-head {io.psnr_global(fused, gt):.2f} dB"
                    f"  (SAM {io.sam_deg(fused, gt):.2f})")
        print(msg)


if __name__ == "__main__":
    main()
