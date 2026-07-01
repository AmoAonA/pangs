"""
Arbitrary-scale rendering probe for PanSplat UCL 2DGS.

Fits one per-image model once, then renders the continuous Gaussian field at
multiple output scales relative to the native PAN grid. Scale 1.0 is the normal
baseline output; scales > 1 render beyond PAN resolution without retraining.
"""

import argparse
import importlib
import json
import math
import os
import shutil
import tempfile
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image, ImageDraw, ImageFont

from utils.misc_utils import load_cfg
from utils.pansharpening_metrics import full_ref_metrics


def _get_log_dir(args):
    s = f"num-{args.num_gaussians:d}"
    s += (
        f"_inv-scale-{args.init_scale:.1f}"
        if not args.disable_inverse_scale
        else f"_scale-{args.init_scale:.1f}"
    )
    if not args.quantize:
        args.pos_bits = args.scale_bits = args.rot_bits = args.feat_bits = 32
    s += f"_bits-{args.pos_bits}-{args.scale_bits}-{args.rot_bits}-{args.feat_bits}"
    if not args.disable_topk_norm:
        s += f"_top-{args.topk}"
    s += f"_{args.init_mode[0]}-{args.init_random_ratio:.1f}"
    lcfg = (
        f"l1-{args.l1_loss_ratio:.1f}_"
        f"l2-{args.l2_loss_ratio:.1f}_"
        f"ssim-{args.ssim_loss_ratio:.1f}"
    )
    folder = f"{s}_{lcfg}"
    if not args.disable_lr_schedule:
        folder += f"_decay-{args.max_decay_times}-{args.decay_ratio:.1f}"
    if not args.disable_prog_optim:
        folder += "_prog"
    return f"{args.log_root}/{args.exp_name}/{folder}"


def _parse_scales(spec):
    scales = []
    for item in str(spec).split(","):
        item = item.strip()
        if item:
            value = float(item)
            if value <= 0:
                raise ValueError(f"Scale must be positive, got {value}")
            scales.append(value)
    if not scales:
        raise ValueError("At least one scale is required")
    return scales


def _as_chw(array):
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {array.shape}")
    if array.shape[0] <= 16:
        return array
    return np.transpose(array, (2, 0, 1))


def _write_tif(path, chw):
    arr = chw.detach().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    if arr.shape[0] == 1:
        arr = arr[0]
    else:
        arr = np.transpose(arr, (1, 2, 0))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(path, arr)


def _to_numpy_chw(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    return _as_chw(np.asarray(image, dtype=np.float32))


def _rgb_indices(dataset, channels):
    dataset = str(dataset).lower()
    if channels >= 8 or dataset in ("wv2", "wv3"):
        return [4, 2, 1]
    if channels >= 3:
        return [2, 1, 0]
    return None


def _stretch01(image, lo=None, hi=None, percentiles=(1.0, 99.0)):
    image = np.asarray(image, dtype=np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(image, percentiles)
    denom = np.maximum(np.asarray(hi, dtype=np.float32) - np.asarray(lo, dtype=np.float32), 1.0e-6)
    return np.clip((image - lo) / denom, 0.0, 1.0)


def _rgb_image(chw, dataset, ref_chw=None):
    chw = _to_numpy_chw(chw)
    bands = _rgb_indices(dataset, chw.shape[0])
    if bands is None:
        gray = chw[0]
        if ref_chw is not None:
            ref = _to_numpy_chw(ref_chw)[0]
            lo, hi = np.percentile(ref, [1.0, 99.0])
        else:
            lo = hi = None
        rgb = np.repeat(_stretch01(gray, lo, hi)[..., None], 3, axis=2)
    else:
        rgb = np.stack([chw[i] for i in bands], axis=2)
        if ref_chw is not None:
            ref = _to_numpy_chw(ref_chw)
            ref_rgb = np.stack([ref[i] for i in bands], axis=2)
            lo = np.percentile(ref_rgb.reshape(-1, 3), 1.0, axis=0)
            hi = np.percentile(ref_rgb.reshape(-1, 3), 99.0, axis=0)
        else:
            lo = hi = None
        rgb = _stretch01(rgb, lo, hi)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _gray_image(chw, ref_chw=None):
    chw = _to_numpy_chw(chw)
    gray = chw[0] if chw.shape[0] == 1 else chw.mean(axis=0)
    if ref_chw is not None:
        ref = _to_numpy_chw(ref_chw)
        ref = ref[0] if ref.shape[0] == 1 else ref.mean(axis=0)
        lo, hi = np.percentile(ref, [1.0, 99.0])
    else:
        lo = hi = None
    out = _stretch01(gray, lo, hi)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _save_png(path, array):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(array).save(path)


def _labeled_tile(array, label, size=256, label_h=26):
    image = Image.fromarray(array)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    tile = Image.new("RGB", (size, size + label_h), "white")
    x = (size - image.width) // 2
    y = label_h + (size - image.height) // 2
    tile.paste(image, (x, y))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((6, 7), label[:42], fill=(0, 0, 0), font=font)
    return tile


def _save_contact_sheet(path, items, columns=4, tile_size=256):
    if not items:
        return
    tiles = [_labeled_tile(array, label, size=tile_size) for label, array in items]
    columns = max(1, int(columns))
    rows = int(math.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + 26)), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_size
        y = (idx // columns) * (tile_size + 26)
        sheet.paste(tile, (x, y))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path)


def _load_h5_patch(data_dir, dataset, mode, patch_index):
    suffix = "rr" if mode == "rr" else "full"
    h5_path = os.path.join(data_dir, f"{dataset}_{suffix}.h5")
    with h5py.File(h5_path, "r") as f:
        n = int(f["ms"].shape[0])
        idx = int(patch_index) - 1
        if idx < 0 or idx >= n:
            raise ValueError(f"patch_index {patch_index} out of range 1..{n}")
        ms = np.asarray(f["ms"][idx], dtype=np.float32)
        pan = np.asarray(f["pan"][idx], dtype=np.float32)
        gt = np.asarray(f["gt"][idx], dtype=np.float32) if "gt" in f else None

    ms = _as_chw(ms)
    pan = _as_chw(pan)
    gt = _as_chw(gt) if gt is not None else None
    if gt is not None and mode == "rr":
        radiometric_max = float(np.max(gt))
    else:
        radiometric_max = max(float(np.max(ms)), float(np.max(pan)))
    radiometric_max = max(radiometric_max, 1.0e-12)
    ms = ms / radiometric_max
    pan = pan / radiometric_max
    if gt is not None:
        gt = gt / radiometric_max
    return h5_path, torch.from_numpy(ms), torch.from_numpy(pan), (
        torch.from_numpy(gt) if gt is not None else None
    )


def _load_tif_pair(data_root, input_path, ref_path, gt_path=""):
    ms = tifffile.imread(os.path.join(data_root, input_path)).astype(np.float32)
    pan = tifffile.imread(os.path.join(data_root, ref_path)).astype(np.float32)
    gt = (
        tifffile.imread(os.path.join(data_root, gt_path)).astype(np.float32)
        if gt_path
        else None
    )
    ms = _as_chw(ms)
    pan = _as_chw(pan)
    gt = _as_chw(gt) if gt is not None else None
    radiometric_max = max(
        float(np.max(ms)),
        float(np.max(pan)),
        float(np.max(gt)) if gt is not None else 0.0,
    )
    if radiometric_max > 1.0:
        ms = ms / radiometric_max
        pan = pan / radiometric_max
        if gt is not None:
            gt = gt / radiometric_max
    return "tif_pair", torch.from_numpy(ms), torch.from_numpy(pan), (
        torch.from_numpy(gt) if gt is not None else None
    )


def _prepare_temp_inputs(ms_chw, pan_chw, gt_chw, prefix):
    tmp = tempfile.mkdtemp(prefix=prefix)
    ms_path = os.path.join(tmp, "ms.tif")
    pan_path = os.path.join(tmp, "pan.tif")
    tifffile.imwrite(ms_path, ms_chw.numpy().transpose(1, 2, 0).astype(np.float32))
    tifffile.imwrite(pan_path, pan_chw.numpy()[0].astype(np.float32))
    if gt_chw is not None:
        tifffile.imwrite(
            os.path.join(tmp, "gt_hr.tif"),
            gt_chw.numpy().transpose(1, 2, 0).astype(np.float32),
        )
    return tmp


def _apply_eval_overrides(args, device, tmp, dataset, mode, patch_index, output_dir):
    args.device = device
    args.data_root = tmp
    args.input_path = "ms.tif"
    args.ref_path = "pan.tif"
    args.exp_name = f"arbitrary_{mode}_{dataset}/p{int(patch_index):03d}"
    args.log_root = os.path.join(output_dir, "_fit_logs")
    args.eval = False
    args.quantize = False
    args.gamma = 1.0
    args.sensor = dataset.upper()
    if not hasattr(args, "seed"):
        args.seed = 0
    args.log_dir = _get_log_dir(args)


def _clamp_initial_ratio(args, ms_chw):
    ms_h, ms_w = ms_chw.shape[-2], ms_chw.shape[-1]
    initial_n = math.ceil(float(args.initial_ratio) * int(args.num_gaussians))
    if initial_n <= ms_h * ms_w:
        return
    init_mode = str(getattr(args, "init_mode", "gradient")).lower()
    bypass = init_mode in (
        "pan_gradient",
        "pan_saliency",
        "saliency",
        "pan_sg_alloc",
        "sg_alloc",
        "pan_entropy_alloc",
        "entropy_alloc",
        "pan_highpass",
        "pan_highpass_stratified",
        "hrms_trad_gradient",
        "trad_hr_gradient",
        "mtf_glp_gradient",
        "ms_lr_plus_pan_hf",
        "pan_uniform",
    ) or bool(getattr(args, "stage1_scaffold_expand", False))
    if not bypass:
        args.initial_ratio = (ms_h * ms_w) / float(args.num_gaussians)


def _render_at_scale(model, scale):
    if hasattr(model, "render_ucl_at_scale"):
        return model.render_ucl_at_scale(render_scale_ratio=scale)
    raise RuntimeError(
        "Arbitrary-scale rendering requires model.render_ucl_at_scale(); "
        "use --model_module model_base or add that method to the selected model."
    )


def _tensor_delta(a, b):
    diff = (a.detach().cpu() - b.detach().cpu()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "bit_exact": bool(float(diff.max().item()) == 0.0),
    }


def _full_ref_summary(pred, gt, device):
    if gt is None or pred.shape != gt.shape:
        return None
    metrics = full_ref_metrics(
        pred.unsqueeze(0).to(device),
        gt.unsqueeze(0).to(device),
        data_range=(0.0, 1.0),
    )
    out = {}
    for key, value in metrics.items():
        out[key] = float(value.item()) if torch.is_tensor(value) else float(value)
    return out


def _pan_weights(channels, device, dtype):
    if channels == 4:
        weights = torch.tensor([0.1, 0.3, 0.3, 0.3], device=device, dtype=dtype)
    else:
        weights = torch.ones(channels, device=device, dtype=dtype)
    return weights / weights.sum().clamp_min(1.0e-12)


def _pan_estimate_chw(image):
    weights = _pan_weights(image.shape[0], image.device, image.dtype)
    return (image * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)


def _chroma_safe_project(direct, ms, pan, args):
    mode = str(getattr(args, "chroma_safe_projection", "off")).lower()
    if mode in ("", "off", "none", "false", "0"):
        return direct, None

    eps = float(getattr(args, "chroma_safe_eps", 1.0e-3))
    target = F.interpolate(
        ms.unsqueeze(0),
        size=direct.shape[-2:],
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)
    pan_up = F.interpolate(
        pan.unsqueeze(0),
        size=direct.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)

    pred_i = _pan_estimate_chw(direct).clamp_min(eps)
    target_i = _pan_estimate_chw(target).clamp_min(eps)
    pred_chroma = direct / pred_i
    target_chroma = target / target_i

    if mode == "pan_lms_global":
        global_w = 1.0
        bright_w = 0.0
        intensity_source = "pan"
    elif mode == "pan_lms_top":
        global_w = 0.0
        bright_w = 1.0
        intensity_source = "pan"
    else:
        global_w = float(getattr(args, "chroma_safe_global_weight", 0.25))
        bright_w = float(getattr(args, "chroma_safe_bright_weight", 0.75))
        intensity_source = str(getattr(args, "chroma_safe_intensity_source", "pred")).lower()

    topk = float(getattr(args, "chroma_safe_topk", 0.10))
    topk = min(max(topk, 1.0e-4), 1.0)
    q0 = torch.quantile(pan_up.flatten(), 1.0 - topk)
    q1 = torch.quantile(pan_up.flatten(), 0.995)
    ramp = ((pan_up - q0) / (q1 - q0).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    chroma_w = (global_w + bright_w * ramp).clamp(0.0, 1.0)
    chroma = (1.0 - chroma_w) * pred_chroma + chroma_w * target_chroma

    intensity_mult = float(getattr(args, "chroma_safe_intensity_mult", 1.05))
    intensity_margin = float(getattr(args, "chroma_safe_intensity_margin", 0.02))
    upper = intensity_mult * pan_up + intensity_margin
    if intensity_source == "pan":
        safe_i = torch.minimum(pan_up, upper)
    elif intensity_source == "blend":
        safe_i = 0.5 * torch.minimum(pred_i, upper) + 0.5 * pan_up
    else:
        safe_i = torch.minimum(pred_i, upper)

    projected = (chroma * safe_i).clamp(0.0, 1.0)
    meta = {
        "mode": mode,
        "topk": topk,
        "global_weight": float(global_w),
        "bright_weight": float(bright_w),
        "intensity_source": intensity_source,
        "intensity_mult": intensity_mult,
        "intensity_margin": intensity_margin,
    }
    return projected, meta


def _save_visual_outputs(
    out_dir,
    dataset,
    ms,
    pan,
    gt,
    baseline,
    records,
    direct_by_scale,
    bicubic_by_scale,
    pan_by_scale,
    tile_size=256,
):
    visual_dir = os.path.join(out_dir, "visuals")
    os.makedirs(visual_dir, exist_ok=True)

    rgb_ref = gt if gt is not None else baseline
    items = []
    visual_paths = {"visual_dir": visual_dir}

    if gt is not None:
        gt_rgb = _rgb_image(gt, dataset, ref_chw=rgb_ref)
        path = os.path.join(visual_dir, "gt_rgb.png")
        _save_png(path, gt_rgb)
        visual_paths["gt_rgb"] = path
        items.append(("GT RGB", gt_rgb))

    ms_up = F.interpolate(
        ms.unsqueeze(0),
        size=baseline.shape[-2:],
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)
    ms_rgb = _rgb_image(ms_up, dataset, ref_chw=rgb_ref)
    path = os.path.join(visual_dir, "lrms_bicubic_rgb.png")
    _save_png(path, ms_rgb)
    visual_paths["lrms_bicubic_rgb"] = path
    items.append(("LRMS bicubic RGB", ms_rgb))

    pan_gray = _gray_image(pan)
    path = os.path.join(visual_dir, "pan_input_gray.png")
    _save_png(path, pan_gray)
    visual_paths["pan_input_gray"] = path
    items.append(("PAN input gray", pan_gray))

    baseline_rgb = _rgb_image(baseline, dataset, ref_chw=rgb_ref)
    path = os.path.join(visual_dir, "baseline_s1_rgb.png")
    _save_png(path, baseline_rgb)
    visual_paths["baseline_s1_rgb"] = path
    items.append(("Baseline s=1 RGB", baseline_rgb))

    for rec in records:
        scale = float(rec["scale"])
        tag = f"s{scale:.4g}".replace(".", "p")

        direct = direct_by_scale[scale]
        direct_rgb = _rgb_image(direct, dataset, ref_chw=rgb_ref)
        path = os.path.join(visual_dir, f"direct_{tag}_rgb.png")
        _save_png(path, direct_rgb)
        rec["direct_rgb_png"] = path
        items.append((f"Direct s={scale:g}", direct_rgb))

        bicubic = bicubic_by_scale[scale]
        bicubic_rgb = _rgb_image(bicubic, dataset, ref_chw=rgb_ref)
        path = os.path.join(visual_dir, f"bicubic_from_s1_{tag}_rgb.png")
        _save_png(path, bicubic_rgb)
        rec["bicubic_from_s1_rgb_png"] = path
        if abs(scale - 1.0) > 1.0e-12:
            items.append((f"Bicubic s={scale:g}", bicubic_rgb))

        pan_rendered = pan_by_scale[scale]
        pan_rendered_gray = _gray_image(pan_rendered, ref_chw=pan)
        path = os.path.join(visual_dir, f"pan_rendered_{tag}_gray.png")
        _save_png(path, pan_rendered_gray)
        rec["pan_gray_png"] = path
        if scale == max(pan_by_scale):
            items.append((f"PAN render s={scale:g}", pan_rendered_gray))

        if gt is not None and direct.shape == gt.shape:
            err = (direct - gt).abs().mean(dim=0, keepdim=True)
            err_gray = _gray_image(err)
            path = os.path.join(visual_dir, f"error_{tag}_gray.png")
            _save_png(path, err_gray)
            rec["error_gray_png"] = path
            if abs(scale - 1.0) <= 1.0e-12:
                items.append(("Abs error s=1", err_gray))

    contact_path = os.path.join(visual_dir, "contact_sheet.png")
    _save_contact_sheet(contact_path, items, columns=4, tile_size=tile_size)
    visual_paths["contact_sheet"] = contact_path
    return visual_paths


def run(args, cfg_overrides):
    scales = _parse_scales(args.scales)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_path and args.ref_path:
        source, ms, pan, gt = _load_tif_pair(
            args.data_root,
            args.input_path,
            args.ref_path,
            gt_path=args.gt_path,
        )
        dataset = args.dataset
        mode = args.mode
    else:
        source, ms, pan, gt = _load_h5_patch(
            args.data_dir,
            args.dataset,
            args.mode,
            args.patch_index,
        )
        dataset = args.dataset
        mode = args.mode

    run_label = f"{dataset}_{mode}_p{int(args.patch_index):03d}"
    out_dir = os.path.join(args.output_dir, run_label)
    os.makedirs(out_dir, exist_ok=True)

    tmp = _prepare_temp_inputs(ms, pan, gt, prefix=f"as_{run_label}_")
    module = importlib.import_module(args.model_module)
    Model = getattr(module, "GaussianSplatting2D")
    captured = {}
    orig_save = Model._save_pansharpened

    def _capture_save(self, image, stage_tag, h, w):
        if stage_tag in ("stage2_hr", "stage3"):
            captured[stage_tag] = image.detach().cpu().clone()
        return orig_save(self, image, stage_tag, h, w)

    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path=args.cfg, parser=parser)
    model_args = parser.parse_args(cfg_overrides)
    _apply_eval_overrides(
        model_args,
        args.device,
        tmp,
        dataset,
        mode,
        args.patch_index,
        args.output_dir,
    )
    _clamp_initial_ratio(model_args, ms)

    Model._save_pansharpened = _capture_save
    try:
        fit_t0 = perf_counter()
        model = Model(model_args)
        model.optimize_three_stage_complete()
        fit_time = perf_counter() - fit_t0
    finally:
        Model._save_pansharpened = orig_save
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)

    baseline = captured.get("stage3", captured.get("stage2_hr"))
    if baseline is None:
        raise RuntimeError("Model did not save stage2_hr or stage3 output")
    _write_tif(os.path.join(out_dir, "baseline_stage3_s1p00.tif"), baseline)

    records = []
    direct_by_scale = {}
    bicubic_by_scale = {}
    pan_by_scale = {}

    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )
    gt_device = gt.to(device) if gt is not None else None

    for scale in scales:
        tag = f"s{scale:.4g}".replace(".", "p")
        render_t0 = perf_counter()
        with torch.no_grad():
            direct, ratio_hr, product_hr, pan_rendered, _ = _render_at_scale(model, scale)
            direct = direct.detach().cpu().clamp(0.0, 1.0)
            pan_rendered = pan_rendered.detach().cpu().clamp(0.0, 1.0)
            direct, chroma_projection_meta = _chroma_safe_project(direct, ms, pan, args)
        render_time = perf_counter() - render_t0

        direct_by_scale[scale] = direct
        pan_by_scale[scale] = pan_rendered

        direct_path = os.path.join(out_dir, f"direct_{tag}.tif")
        pan_path = os.path.join(out_dir, f"pan_rendered_{tag}.tif")
        bicubic_path = os.path.join(out_dir, f"bicubic_from_s1_{tag}.tif")
        _write_tif(direct_path, direct)
        _write_tif(pan_path, pan_rendered)

        bicubic = F.interpolate(
            baseline.unsqueeze(0),
            size=direct.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        bicubic_by_scale[scale] = bicubic
        _write_tif(bicubic_path, bicubic)

        rec = {
            "scale": float(scale),
            "height": int(direct.shape[-2]),
            "width": int(direct.shape[-1]),
            "direct_path": direct_path,
            "pan_path": pan_path,
            "bicubic_from_s1_path": bicubic_path,
            "render_time_sec": float(render_time),
            "direct_vs_bicubic_from_s1": _tensor_delta(direct, bicubic),
        }
        if chroma_projection_meta is not None:
            rec["chroma_safe_projection"] = chroma_projection_meta
        if abs(scale - 1.0) <= 1.0e-12:
            rec["direct_vs_saved_baseline"] = _tensor_delta(direct, baseline)
            rec["full_ref"] = (
                _full_ref_summary(direct, gt_device, args.device)
                if gt_device is not None
                else None
            )
        records.append(rec)

    self_consistency = []
    if args.self_consistency:
        sorted_scales = sorted(direct_by_scale)
        for hi in sorted_scales:
            for lo in sorted_scales:
                if hi <= lo:
                    continue
                down = F.interpolate(
                    direct_by_scale[hi].unsqueeze(0),
                    size=direct_by_scale[lo].shape[-2:],
                    mode="area",
                ).squeeze(0)
                row = {
                    "from_scale": float(hi),
                    "to_scale": float(lo),
                    "area_down_vs_direct": _tensor_delta(down, direct_by_scale[lo]),
                }
                self_consistency.append(row)

    visual_paths = None
    if args.save_visuals:
        visual_paths = _save_visual_outputs(
            out_dir,
            dataset,
            ms,
            pan,
            gt,
            baseline,
            records,
            direct_by_scale,
            bicubic_by_scale,
            pan_by_scale,
            tile_size=args.visual_tile_size,
        )

    summary = {
        "source": source,
        "dataset": dataset,
        "mode": mode,
        "patch_index_1based": int(args.patch_index),
        "cfg": args.cfg,
        "cfg_overrides": cfg_overrides,
        "model_module": args.model_module,
        "fit_time_sec": float(fit_time),
        "native_pan_shape": [int(pan.shape[-2]), int(pan.shape[-1])],
        "native_ms_shape": [int(ms.shape[-2]), int(ms.shape[-1])],
        "scale_semantics": "render scale is relative to the native PAN grid; 1.0 is the baseline PAN resolution",
        "scale_rendering_note": (
            "Gaussian render scales are multiplied by render_scale_ratio so "
            "continuous footprints stay size-consistent on the output grid."
        ),
        "renders": records,
        "self_consistency": self_consistency,
        "visuals": visual_paths,
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved arbitrary-scale outputs: {out_dir}")
    print(f"Saved metrics: {metrics_path}")
    if visual_paths is not None:
        print(f"Saved visuals: {visual_paths['contact_sheet']}")
    s1 = next((r for r in records if abs(r["scale"] - 1.0) <= 1.0e-12), None)
    if s1 is not None:
        delta = s1.get("direct_vs_saved_baseline", {})
        print(
            "s=1 sanity: "
            f"max_abs={delta.get('max_abs', float('nan')):.8g} "
            f"mean_abs={delta.get('mean_abs', float('nan')):.8g} "
            f"bit_exact={delta.get('bit_exact', False)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wv3", "wv2", "qb", "gf2"], default="wv3")
    parser.add_argument("--mode", choices=["rr", "fr"], default="rr")
    parser.add_argument("--data_dir", default="datasets/pancollection_raw")
    parser.add_argument("--patch_index", type=int, default=1, help="1-based HDF5 patch index")
    parser.add_argument("--data_root", default="media", help="Used with --input_path/--ref_path")
    parser.add_argument("--input_path", default="", help="Optional MS TIF path relative to --data_root")
    parser.add_argument("--ref_path", default="", help="Optional PAN TIF path relative to --data_root")
    parser.add_argument("--gt_path", default="", help="Optional GT TIF path relative to --data_root")
    parser.add_argument("--cfg", default="cfgs/ucl_s1cap15k_s2panhf15k_newgeo_fixedgain_tv_fast.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_module", default="model")
    parser.add_argument("--scales", default="1,1.5,2,2.5,3,3.7,4,6,8")
    parser.add_argument("--output_dir", default="arbitrary_scale_outputs")
    parser.add_argument("--no_self_consistency", dest="self_consistency", action="store_false")
    parser.add_argument("--no_visuals", dest="save_visuals", action="store_false")
    parser.add_argument("--visual_tile_size", type=int, default=256)
    parser.add_argument(
        "--chroma_safe_projection",
        default="off",
        choices=["off", "pred_lms", "pan_lms_top", "pan_lms_global"],
        help=(
            "Optional render-time chroma safety projection. "
            "pan_lms_global uses PAN intensity with bicubic LRMS chroma."
        ),
    )
    parser.add_argument("--chroma_safe_topk", type=float, default=0.10)
    parser.add_argument("--chroma_safe_global_weight", type=float, default=0.25)
    parser.add_argument("--chroma_safe_bright_weight", type=float, default=0.75)
    parser.add_argument(
        "--chroma_safe_intensity_source",
        default="pred",
        choices=["pred", "pan", "blend"],
    )
    parser.add_argument("--chroma_safe_intensity_mult", type=float, default=1.05)
    parser.add_argument("--chroma_safe_intensity_margin", type=float, default=0.02)
    parser.add_argument("--chroma_safe_eps", type=float, default=1.0e-3)
    parser.add_argument("--keep_temp", action="store_true")
    parser.set_defaults(self_consistency=True, save_visuals=True)
    args, cfg_overrides = parser.parse_known_args()
    run(args, cfg_overrides)


if __name__ == "__main__":
    main()
