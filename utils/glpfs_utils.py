import torch
import torch.nn.functional as F

from utils.frequency_utils import gaussian_blur_chw


def build_pan_hat(pan_hr, ms_up, eps=1.0e-8):
    """
    Band-wise histogram matched PAN.
    pan_hr: [1,H,W], ms_up: [C,H,W].
    """
    pan_mu = pan_hr.mean()
    pan_std = pan_hr.std(unbiased=False).clamp_min(eps)
    ms_mu = ms_up.mean(dim=(-2, -1), keepdim=True)
    ms_std = ms_up.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    return (pan_hr.repeat(ms_up.shape[0], 1, 1) - pan_mu) * (ms_std / pan_std) + ms_mu


def build_base_image(ms_lr, pan_hr, mode="glpfs", ratio=4, eps=1.0e-8):
    """
    Build bicubic/GLP-FS style base image.
    Returns x_base, ms_up, pan_hat, gain_fs.
    """
    h, w = pan_hr.shape[-2:]
    ms_up = F.interpolate(
        ms_lr.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)
    pan_hat = build_pan_hat(pan_hr, ms_up, eps=eps).clamp(0.0, 1.0)
    mode = str(mode).lower()
    if mode == "bicubic":
        return ms_up, ms_up, pan_hat, torch.ones(ms_lr.shape[0], device=ms_lr.device, dtype=ms_lr.dtype)
    if mode != "glpfs":
        raise ValueError(f"Unsupported base mode: {mode}")

    pan_lp = gaussian_blur_chw(pan_hr, max(1.0, float(ratio) * 0.5))
    pan_hp = pan_hr - pan_lp
    p = pan_hr.flatten()
    pl = pan_lp.flatten()
    pl_center = pl - pl.mean()
    denom = (pl_center * (p - p.mean())).mean().clamp_min(eps)
    gains = []
    out = []
    for band in range(ms_up.shape[0]):
        m = ms_up[band].flatten()
        cov = ((m - m.mean()) * (p - p.mean())).mean()
        gain = cov / denom
        gains.append(gain)
        out.append((ms_up[band:band + 1] + gain * pan_hp).clamp(0.0, 1.0))
    gain_fs = torch.stack(gains).to(dtype=ms_lr.dtype, device=ms_lr.device)
    x_base = torch.cat(out, dim=0)
    return x_base, ms_up, pan_hat, gain_fs
