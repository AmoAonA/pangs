# PanSplat — Zero-Shot, Arbitrary-Scale Pansharpening via 2D Gaussian Splatting

Fuse a low-resolution multispectral (MS) image with a high-resolution
panchromatic (PAN) image into a high-resolution multispectral (HRMS) image —
**per image, from scratch, with no pretrained weights, no cross-image training,
and no ground truth**. Because the fused image is represented as a continuous
field of 2D Gaussians plus a local operator on PAN, it can be rendered at **any
output scale** (including beyond native PAN resolution) without retraining.

## Method

Two zero-shot stages, both fit on a single MS/PAN pair:

1. **Gaussian baseline** — fit 2D Gaussians to PAN geometry (Stage 1) and a
   smooth spectral coefficient (Stage 2). Produces `HRMS = coeff · rendered_PAN`,
   supervised by low-resolution (MTF) consistency with the input MS.
2. **Local-PAN head** — a small convolutional head reads the *local PAN texture*
   and predicts the coefficient's high-frequency correction that the smooth
   Gaussian field cannot represent:

   ```
   coeff = smooth_LF(baseline) + head(PAN)
   HRMS  = coeff · carrier
   ```

   The head is fit per image, supervised **only** by low-resolution consistency
   (no ground truth). As a purely local operator on PAN, it applies at any
   rendered scale.

## Results (reduced-resolution, full-channel PSNR / SAM, 20 patches each)

| Method | WV3 PSNR | WV2 PSNR | QB PSNR |
|---|---|---|---|
| MTF-GLP-FS (best traditional) | 30.59 | 28.17 | 31.53 |
| Gaussian baseline | 30.78 | 29.16 | 31.79 |
| **PanSplat (ours)** | **32.40** | **29.74** | **33.22** |

Ours beats the strongest traditional method by ~1.6–1.8 dB and improves SAM
(color fidelity), while remaining zero-shot, teacher-free, and arbitrary-scale.

## Install

Requires a CUDA-capable GPU (the 2D rasterizer is a custom CUDA extension).

```bash
pip install -r requirements.txt
```

The custom 2D Gaussian rasterizer in `gsplat/` is **JIT-compiled on first
import** via PyTorch's `cpp_extension` — no separate build step is needed, just
a working CUDA toolchain (`nvcc` on PATH) matching your PyTorch build. The first
run therefore takes an extra ~1–2 minutes to compile; subsequent runs use the
cached extension.

Tested with PyTorch >= 2.0 and CUDA 12.x.

> If you prefer an explicit install: `cd gsplat && pip install -e . --no-build-isolation`.
> Do **not** ship a prebuilt `gsplat/gsplat/csrc.so` — a `.so` compiled against a
> different Python/PyTorch causes `undefined symbol` errors; delete it and let the
> JIT path rebuild.

## Run

One command on your own MS/PAN GeoTIFF pair:

```bash
python run_pansplat.py MS.tif PAN.tif --sensor WV3 --scales 1,2,4
```

Try the bundled sample (WV3 patch, with ground truth for evaluation):

```bash
python run_pansplat.py sample_data/ms.tif sample_data/pan.tif \
    --sensor WV3 --gt sample_data/gt.tif
```

This prints `baseline -> PAN-head` PSNR/SAM and writes `pansplat_out/fused_s1.tif`
(and `fused_s2.tif`, `fused_s4.tif`, ... for extra scales).

By default, `run_pansplat.py` uses the same Gaussian-baseline settings as the
reported benchmark (`stage2_loss_curriculum=True`, low-frequency LR consistency,
and fixed Stage-1 PAN carrier). Use `--plain_baseline` only for ablations.

### Options

- `--sensor {WV3,WV2,QB,GF2}` — selects the sensor MTF (must match your data).
- `--scales 1,2,4` — output scales; `1` = native, `>1` = beyond PAN resolution.
- `--gt GT.tif` — optional ground truth; enables PSNR/SAM reporting only.
- `--seeds N` — average N independently fit heads (`+~0.15 dB`, default 1).
- `--iters` — PAN-head iterations (default 3500).
- `--device cuda:0`.
- `--plain_baseline` — disable the benchmark baseline overrides for debugging.

## Repository layout

- `run_pansplat.py` — one-command driver (baseline → PAN-head → fused HRMS).
- `panhead.py` — the zero-shot local-PAN head (core contribution); reusable
  `run_panhead(...)`.
- `model.py` — 2D Gaussian pansharpening model with `render_ucl_at_scale`.
- `eval_arbitrary_scale.py` — arbitrary-scale baseline driver used by the above.
- `gsplat/` — custom CUDA 2D Gaussian rasterizer (build with `pip install -e .`).
- `psutils.py` — sensor MTF (`genMTF`) for WV3/WV2/QB/GF2.
- `io_utils.py` — TIFF I/O and PSNR/SAM metrics.
- `cfgs/ucl_s1cap15k_s2panhf15k_newgeo_fixedgain_tv_fast.yaml` — recommended config.
- `sample_data/` — a WV3 MS/PAN/GT triple to test the install.

## Data format

Any MS/PAN GeoTIFF pair: MS is `[H, W, C]` or `[C, H, W]`, PAN is `[H', W']`
(4× the MS resolution). Values are radiometrically normalized to `[0, 1]`
internally. No specific dataset is required.

## Notes

- Ground truth is used **only** for reporting metrics, never for fitting.
- Metrics use full-channel PSNR (single MSE over all bands), consistent with the
  UCL reference implementation. Per-band-averaged PSNR (common in some papers)
  runs ~1.5–2 dB higher.
