import math

import torch
import torch.nn.functional as F


def gaussian_blur_chw(x, sigma):
    """
    Depth-wise Gaussian blur for [C,H,W] or [1,H,W] tensors.
    """
    sigma = float(sigma)
    if sigma <= 1.0e-8:
        return x
    squeeze = x.dim() == 3
    xb = x.unsqueeze(0) if squeeze else x
    channels = xb.shape[1]
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    coords = torch.arange(ksize, dtype=xb.dtype, device=xb.device) - ksize // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-12)
    kernel = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
    kernel = kernel.expand(channels, 1, ksize, ksize)
    pad = ksize // 2
    out = F.conv2d(
        F.pad(xb, (pad, pad, pad, pad), mode="reflect"),
        kernel,
        groups=channels,
    )
    return out.squeeze(0) if squeeze else out


def decompose_freq(x, sigma_mid=2.0, sigma_low=4.0):
    """
    Split [C,H,W] image into low/mid/high bands.
    """
    blur_mid = gaussian_blur_chw(x, sigma_mid)
    low = gaussian_blur_chw(x, sigma_low)
    mid = blur_mid - low
    high = x - blur_mid
    return low, mid, high


def normalize01(x, eps=1.0e-8):
    x = x - x.min()
    return (x / (x.max() + eps)).clamp(0.0, 1.0)


def degrade_mtf_fallback(x, out_h, out_w, sigma_scale=0.5):
    """
    Fallback MTF-like degradation: Gaussian low-pass then area resize.
    """
    ratio_h = x.shape[-2] / float(out_h)
    ratio_w = x.shape[-1] / float(out_w)
    ratio = max(1.0, 0.5 * (ratio_h + ratio_w))
    blurred = gaussian_blur_chw(x, max(1.0e-6, sigma_scale * ratio))
    return F.interpolate(blurred.unsqueeze(0), size=(out_h, out_w), mode="area").squeeze(0)


def corr_loss(a, b, eps=1.0e-8):
    a = a - a.mean()
    b = b - b.mean()
    return 1.0 - (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False) + eps)


def sample_image_at_xy(img_chw, xy):
    """
    Bilinear sample CHW image at normalized xy coordinates in [0,1].
    Returns [N,C].
    """
    samples = F.grid_sample(
        img_chw.unsqueeze(0),
        xy[None, None, ...] * 2.0 - 1.0,
        align_corners=False,
        mode="bilinear",
    )
    return samples[0, :, 0, :].permute(1, 0).contiguous()


def build_pan_gate(pan_hr, xy, sigma_mid=2.0, sigma_low=4.0):
    _, p_mid, p_high = decompose_freq(pan_hr, sigma_mid=sigma_mid, sigma_low=sigma_low)
    gate_map = normalize01(p_high.abs() + 0.5 * p_mid.abs())
    return sample_image_at_xy(gate_map, xy).clamp(0.0, 1.0)
