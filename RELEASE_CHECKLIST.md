# Release Checklist

This package is the cleaned release candidate generated from the verified
`release_fixed_20260701` tree.

## Verified Locally

```bash
python -m py_compile model.py eval_arbitrary_scale.py run_pansplat.py panhead.py
```

## Verified On A100

WV3 RR patch 012, fixed release entry point:

```text
baseline: PSNR 30.5581, SAM 6.6004, ERGAS 4.6415, Q2n 0.8258
PAN-head: PSNR 31.6434, SAM 5.7897, ERGAS 4.0798, Q2n 0.8661
```

This confirms the public `run_pansplat.py` entry point matches the benchmark
baseline behavior and no longer uses the broken reduced `release/model.py`.

## Full Benchmark Source

The reported 20-patch RR means are from the benchmark workflow:

```text
Gaussian baseline + local PAN-head, 3500 head iterations, seed=1
```

Expected approximate means:

```text
WV3: 32.35 PSNR
WV2: 29.77 PSNR
QB : 33.20 PSNR
```

Before tagging a final GitHub release, rerun the full 20-patch benchmark if the
model, config, PAN-head, or dependency versions change.
