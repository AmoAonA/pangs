import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchmetrics.image as TM_IMAGE
except Exception:
    TM_IMAGE = None

try:
    import torchmetrics.functional as TM_FUNC
except Exception:
    TM_FUNC = None

_TM_CLASS_MAP = {
    "structural_similarity_index_measure": "StructuralSimilarityIndexMeasure",
    "root_mean_squared_error_using_sliding_window": "RootMeanSquaredErrorUsingSlidingWindow",
    "error_relative_global_dimensionless_synthesis": "ErrorRelativeGlobalDimensionlessSynthesis",
    "spectral_angle_mapper": "SpectralAngleMapper",
    "relative_average_spectral_error": "RelativeAverageSpectralError",
    "peak_signal_noise_ratio": "PeakSignalNoiseRatio",
    "universal_image_quality_index": "UniversalImageQualityIndex",
}


def _tm_metric(name, *args, **kwargs):
    device = None
    for item in args:
        if torch.is_tensor(item):
            device = item.device
            break
    if TM_IMAGE is not None:
        if hasattr(TM_IMAGE, name):
            return getattr(TM_IMAGE, name)(*args, **kwargs)
        cls_name = _TM_CLASS_MAP.get(name)
        if cls_name is not None and hasattr(TM_IMAGE, cls_name):
            init_kwargs = {}
            if name in ("structural_similarity_index_measure", "peak_signal_noise_ratio"):
                if "data_range" in kwargs:
                    init_kwargs["data_range"] = kwargs["data_range"]
            metric = getattr(TM_IMAGE, cls_name)(**init_kwargs)
            if device is not None:
                metric = metric.to(device)
            return metric(*args)
    if TM_FUNC is not None and hasattr(TM_FUNC, name):
        return getattr(TM_FUNC, name)(*args, **kwargs)
    raise AttributeError(f"torchmetrics does not provide '{name}' in image or functional modules.")


class PansharpeningMetrics:
    def __init__(self, device="cuda"):
        self.device = device

    def _q_index(self, img1, img2, block_size=32):
        """
        计算 Q-index (UIQI) - 基于滑动窗口 (Unfold)
        这是最慢的部分，但也是最标准的部分。
        """
        n, c, h, w = img1.shape
        kernel = block_size
        stride = max(1, block_size // 2)
        unfold = torch.nn.Unfold(kernel_size=kernel, stride=stride)

        patches1 = unfold(img1).view(n, c, kernel * kernel, -1)
        patches2 = unfold(img2).view(n, c, kernel * kernel, -1)

        mean1 = patches1.mean(dim=2)
        mean2 = patches2.mean(dim=2)

        var1 = patches1.var(dim=2, unbiased=False)
        var2 = patches2.var(dim=2, unbiased=False)
        cov12 = (patches1 * patches2).mean(dim=2) - mean1 * mean2

        q_map = (4 * mean1 * mean2 * cov12) / (
            (mean1 ** 2 + mean2 ** 2) * (var1 + var2) + 1e-8
        )

        return q_map.mean(dim=(1, 2))

    def qnr(self, pan, ms, fused, fused_lr_override=None):
        """
        计算 QNR
        pan: [B, 1, H, W]
        ms:  [B, C, h, w] (低分)
        fused: [B, C, H, W] (高分)
        """
        # For multi-scale pipelines, allow using direct LR rasterization output instead of HR downsample.
        if fused_lr_override is None:
            fused_lr = F.interpolate(fused, size=ms.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        else:
            fused_lr = fused_lr_override.clamp(0, 1)
        ms_clamped = ms.clamp(0, 1)

        pan_lr = F.interpolate(pan, size=ms.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        fused_clamped = fused.clamp(0, 1)
        pan_clamped = pan.clamp(0, 1)

        b, c, _, _ = fused.shape

        d_lambda = torch.zeros(b, device=self.device)
        count = 0
        for i in range(c):
            for j in range(i + 1, c):
                q_fused = self._q_index(fused_lr[:, i:i + 1], fused_lr[:, j:j + 1])
                q_ms = self._q_index(ms_clamped[:, i:i + 1], ms_clamped[:, j:j + 1])
                d_lambda += torch.abs(q_fused - q_ms)
                count += 1
        d_lambda /= max(count, 1)

        d_s = torch.zeros(b, device=self.device)
        for i in range(c):
            q_high = self._q_index(fused_clamped[:, i:i + 1], pan_clamped)
            q_low = self._q_index(ms_clamped[:, i:i + 1], pan_lr)
            d_s += torch.abs(q_high - q_low)
        d_s /= c

        score = (1 - d_lambda) * (1 - d_s)
        return score.mean()


def _sobel_gradient_magnitude(img):
    """Compute Sobel gradient magnitude for a single-channel [B,1,H,W] tensor.
    Matches DLPan-Toolbox SCC.m: applies Sobel to interior (crop 1px border),
    returns sqrt(sobel_x^2 + sobel_y^2)."""
    # Sobel kernels (MATLAB fspecial('sobel'))
    sobel_y = img.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    sobel_x = img.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    # Crop 1px border first (like MATLAB I_F(2:end-1,2:end-1,:))
    inner = img[:, :, 1:-1, 1:-1]
    gy = F.conv2d(inner, sobel_y, padding=1)
    gx = F.conv2d(inner, sobel_x, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2)


def scc(pan, fused):
    """
    Spatial Correlation Coefficient (SCC) — DLPan-Toolbox standard.
    Uses Sobel gradient magnitude + global normalized dot product (no mean subtraction).
    Reference: Vivone et al., IEEE TGRS 2015 / IEEE GRSM 2020.
    pan:   [B, 1, H, W]
    fused: [B, C, H, W]
    """
    b, c, h, w = fused.shape
    pan_grad = _sobel_gradient_magnitude(pan)  # [B, 1, H-2, W-2]
    cc_sum = pan.new_tensor(0.0)
    for i in range(c):
        band_grad = _sobel_gradient_magnitude(fused[:, i:i + 1, :, :])
        # Global normalized correlation (no mean subtraction)
        num = (band_grad * pan_grad).sum()
        den = torch.sqrt(band_grad.pow(2).sum() * pan_grad.pow(2).sum()) + 1e-8
        cc_sum = cc_sum + num / den
    return cc_sum / c


def scc_rr(gt, fused, return_perband=False):
    """
    SCC for reduced-resolution evaluation — DLPan-Toolbox standard.
    Correlates Sobel gradient magnitudes of each fused band against GT band.
    gt:    [B, C, H, W]
    fused: [B, C, H, W]
    """
    b, c, h, w = fused.shape
    cc_sum = gt.new_tensor(0.0)
    per_band = []
    for i in range(c):
        g_grad = _sobel_gradient_magnitude(gt[:, i:i + 1])
        f_grad = _sobel_gradient_magnitude(fused[:, i:i + 1])
        num = (g_grad * f_grad).sum()
        den = torch.sqrt(g_grad.pow(2).sum() * f_grad.pow(2).sum()) + 1e-8
        band_cc = num / den
        per_band.append(float(band_cc))
        cc_sum = cc_sum + band_cc
    if return_perband:
        return cc_sum / c, per_band
    return cc_sum / c


def D_lambda_khan(lr_ms, fused, ratio=4):
    """
    Khan's spectral distortion D_lambda^K for HQNR.
    Approximates MTF with Gaussian low-pass, decimates fused image,
    then computes 1 - mean_Q(fused_lr, ms).
    lr_ms: [B, C, h, w] (original LR MS)
    fused: [B, C, H, W] (pansharpened HR)
    ratio: spatial resolution ratio (HR/LR)
    """
    b, c, h_lr, w_lr = lr_ms.shape
    # Gaussian MTF approximation: sigma ~ 0.39 * ratio (matched to Nyquist)
    sigma = 0.39 * ratio
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    # Build 1D Gaussian kernel
    x = torch.arange(ksize, dtype=fused.dtype, device=fused.device) - ksize // 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    # Separable 2D filter per channel
    gauss_2d = gauss_1d.view(1, 1, -1, 1) * gauss_1d.view(1, 1, 1, -1)
    gauss_2d = gauss_2d.expand(c, 1, -1, -1)
    pad = ksize // 2
    fused_lp = F.conv2d(fused, gauss_2d, padding=pad, groups=c)
    # Decimate to LR resolution
    fused_lr = F.interpolate(fused_lp, size=(h_lr, w_lr), mode="area")
    # Mean Q-index across bands
    q_sum = fused.new_tensor(0.0)
    for i in range(c):
        q_sum = q_sum + QIndex_torch(fused_lr[:, i, :, :], lr_ms[:, i, :, :])
    q_mean = q_sum / c
    return 1.0 - q_mean


def hqnr(lr_ms, pan, fused, ratio=4):
    """
    Hybrid QNR (HQNR) = (1 - D_lambda^K) * (1 - D_s).
    Uses Khan's MTF-based spectral distortion + standard QNR spatial distortion.
    """
    d_lk = D_lambda_khan(lr_ms, fused, ratio=ratio)
    d_s = D_s_torch(lr_ms, pan, fused)
    return (1.0 - d_lk) * (1.0 - d_s), d_lk, d_s


def mean_uiqi(fused, gt, block_size=32, return_diagnostic=False):
    """
    Per-band UIQI averaged over bands and non-overlapping blocks.

    This is useful as a diagnostic, but it is not the standard Q2^n index used
    by pansharpening benchmarks. Use Q2n() for paper/reporting numbers.

    UIQI = structure × luminance × contrast, where:
        structure  = cov(f,g) / (σf·σg)          (Pearson correlation)
        luminance  = 2·μf·μg / (μf² + μg²)
        contrast   = 2·σf·σg / (σf² + σg²)

    Perfect reconstruction → Q2^n = 1.  Random noise → Q2^n ≈ 0.

    fused: [B, C, H, W] in [0,1]
    gt:    [B, C, H, W] in [0,1]
    """
    b, c, h, w = fused.shape
    stride = block_size
    unfold = torch.nn.Unfold(kernel_size=block_size, stride=stride)

    n_px = block_size * block_size
    uf = unfold(fused).view(b, c, n_px, -1)  # [B, C, n_px, L]
    ug = unfold(gt).view(b, c, n_px, -1)

    mean_f = uf.mean(dim=2)  # [B, C, L]
    mean_g = ug.mean(dim=2)

    eps = 1e-8
    q_sum = fused.new_tensor(0.0)

    band_q = []
    band_struct = []
    band_lum = []
    band_con = []

    for k in range(c):
        fk = uf[:, k]          # [B, n_px, L]
        gk = ug[:, k]
        mfk = mean_f[:, k]     # [B, L]
        mgk = mean_g[:, k]

        fk_c = fk - mfk.unsqueeze(1)   # [B, n_px, L]
        gk_c = gk - mgk.unsqueeze(1)

        cov_fg = (fk_c * gk_c).mean(dim=1)   # [B, L]
        var_f  = (fk_c ** 2).mean(dim=1)
        var_g  = (gk_c ** 2).mean(dim=1)

        sigma_f = torch.sqrt(var_f.clamp(min=eps))
        sigma_g = torch.sqrt(var_g.clamp(min=eps))

        # Three-component decomposition
        structure  = cov_fg / (sigma_f * sigma_g + eps)           # Pearson r
        luminance  = 2.0 * mfk * mgk / (mfk ** 2 + mgk ** 2 + eps)
        contrast   = 2.0 * sigma_f * sigma_g / (var_f + var_g + eps)

        q_k = (structure * luminance * contrast).mean()
        q_sum = q_sum + q_k

        band_q.append(float(q_k))
        band_struct.append(float(structure.mean()))
        band_lum.append(float(luminance.mean()))
        band_con.append(float(contrast.mean()))

    q_avg = q_sum / c

    if not return_diagnostic:
        return q_avg

    # Cross-band joint term: Frobenius distance between inter-band correlation
    # matrices of fused and GT (measures spectral structure preservation).
    # corr_matrix[i,j] = Pearson(band_i, band_j) computed on flattened pixels.
    f_flat = fused[0].reshape(c, -1)  # [C, H*W]
    g_flat = gt[0].reshape(c, -1)
    f_centered = f_flat - f_flat.mean(dim=1, keepdim=True)
    g_centered = g_flat - g_flat.mean(dim=1, keepdim=True)
    f_norm = f_centered / (f_centered.norm(dim=1, keepdim=True) + eps)
    g_norm = g_centered / (g_centered.norm(dim=1, keepdim=True) + eps)
    corr_f = f_norm @ f_norm.t()  # [C, C]
    corr_g = g_norm @ g_norm.t()  # [C, C]
    # Off-diagonal difference (cross-band distortion)
    mask = 1.0 - torch.eye(c, device=fused.device)
    cross_band_dist = float(((corr_f - corr_g) ** 2 * mask).sum().sqrt() / mask.sum())

    diag = {
        "band_q":       band_q,
        "band_struct":  band_struct,
        "band_lum":     band_lum,
        "band_con":     band_con,
        "cross_band_corr_dist": cross_band_dist,
    }
    return q_avg, diag


def _next_power_of_two(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pad_spatial_for_q2n(x, block_size, shift):
    b, c, h, w = x.shape
    steps_h = max(1, math.ceil(h / shift))
    steps_w = max(1, math.ceil(w / shift))
    out_h = (steps_h - 1) * shift + block_size
    out_w = (steps_w - 1) * shift + block_size
    pad_h = out_h - h
    pad_w = out_w - w
    if pad_h == 0 and pad_w == 0:
        return x, steps_h, steps_w

    out = x.new_empty((b, c, out_h, out_w))
    out[:, :, :h, :w] = x
    if pad_w > 0:
        cols = torch.arange(w - 1, w - pad_w - 1, -1, device=x.device).clamp(min=0)
        out[:, :, :h, w:] = x.index_select(-1, cols)
    if pad_h > 0:
        rows = torch.arange(h - 1, h - pad_h - 1, -1, device=x.device).clamp(min=0)
        out[:, :, h:, :] = out[:, :, :h, :].index_select(-2, rows)
    return out, steps_h, steps_w


def _pad_channels_to_power_of_two(x):
    b, c, h, w = x.shape
    c2 = _next_power_of_two(c)
    if c2 == c:
        return x, c
    pad = x.new_zeros((b, c2 - c, h, w))
    return torch.cat([x, pad], dim=1), c


def _q2n_onion_mult(x, y):
    n = x.shape[-1]
    if n == 1:
        return x * y

    half = n // 2
    a = x[..., :half]
    b = x[..., half:]
    b = torch.cat([b[..., :1], -b[..., 1:]], dim=-1)
    c = y[..., :half]
    d = y[..., half:]
    d = torch.cat([d[..., :1], -d[..., 1:]], dim=-1)

    if n == 2:
        return torch.cat([a * c - d * b, a * d + c * b], dim=-1)

    b_conj = torch.cat([b[..., :1], -b[..., 1:]], dim=-1)
    a_conj = torch.cat([a[..., :1], -a[..., 1:]], dim=-1)
    left = _q2n_onion_mult(a, c) - _q2n_onion_mult(d, b_conj)
    right = _q2n_onion_mult(a_conj, d) + _q2n_onion_mult(c, b)
    return torch.cat([left, right], dim=-1)


def Q2n(fused, gt, block_size=32, shift=None, return_diagnostic=False):
    """
    Standard Q2^n for multispectral full-reference evaluation.

    This follows the Garzelli-Nencini hypercomplex extension of UQI and the
    Vivone et al. pansharpening toolbox protocol: pad spectral channels to the
    next power of two, evaluate local hypercomplex correlation on spatial
    blocks, and average the magnitude of the local Q2^n vectors.

    fused: [B, C, H, W]
    gt:    [B, C, H, W]
    """
    if fused.shape != gt.shape:
        raise ValueError(f"Q2n expects fused and gt with same shape, got {fused.shape} and {gt.shape}")
    if fused.ndim != 4:
        raise ValueError(f"Q2n expects BCHW tensors, got {fused.ndim}D input")
    if block_size < 2:
        raise ValueError("Q2n block_size must be >= 2")

    shift = block_size if shift is None else shift
    dtype = torch.float64 if fused.dtype in (torch.float16, torch.bfloat16, torch.float32) else fused.dtype
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-12
    ref = gt.to(dtype=dtype)
    fus = fused.to(dtype=dtype)

    ref, original_channels = _pad_channels_to_power_of_two(ref)
    fus, _ = _pad_channels_to_power_of_two(fus)
    ref, steps_h, steps_w = _pad_spatial_for_q2n(ref, block_size, shift)
    fus, _, _ = _pad_spatial_for_q2n(fus, block_size, shift)

    b, c, _, _ = ref.shape
    n_px = block_size * block_size
    unfold = torch.nn.Unfold(kernel_size=block_size, stride=shift)
    ref_blocks = unfold(ref).transpose(1, 2).reshape(b, steps_h * steps_w, c, n_px)
    fus_blocks = unfold(fus).transpose(1, 2).reshape(b, steps_h * steps_w, c, n_px)
    ref_blocks = ref_blocks.permute(0, 1, 3, 2).reshape(-1, n_px, c)
    fus_blocks = fus_blocks.permute(0, 1, 3, 2).reshape(-1, n_px, c)

    mean_ref = ref_blocks.mean(dim=1, keepdim=True)
    std_ref = ref_blocks.std(dim=1, unbiased=True, keepdim=True).clamp_min(eps)
    dat1 = (ref_blocks - mean_ref) / std_ref + 1.0

    fus_norm = (fus_blocks - mean_ref) / std_ref + 1.0
    sign = fus_norm.new_ones((1, 1, c))
    if c > 1:
        sign[..., 1:] = -1.0
    dat2 = fus_norm * sign

    m1 = dat1.mean(dim=1)
    m2 = dat2.mean(dim=1)
    mod_q1m = torch.linalg.vector_norm(m1, dim=-1)
    mod_q2m = torch.linalg.vector_norm(m2, dim=-1)
    mod_q1_sq_mean = dat1.square().sum(dim=-1).mean(dim=1)
    mod_q2_sq_mean = dat2.square().sum(dim=-1).mean(dim=1)

    sample_factor = n_px / (n_px - 1.0)
    term4 = mod_q1m.square() + mod_q2m.square()
    mean_bias = 2.0 * mod_q1m * mod_q2m / term4.clamp_min(eps)
    term3 = sample_factor * (
        mod_q1_sq_mean + mod_q2_sq_mean - mod_q1m.square() - mod_q2m.square()
    )

    q = dat1.new_zeros((dat1.shape[0], c))
    valid = term3.abs() > eps
    if valid.any():
        qu = _q2n_onion_mult(dat1[valid], dat2[valid])
        qm = _q2n_onion_mult(m1[valid], m2[valid])
        qv = sample_factor * qu.mean(dim=1)
        q_valid = (qv - sample_factor * qm) * mean_bias[valid].unsqueeze(-1) * (2.0 / term3[valid]).unsqueeze(-1)
        q[valid] = q_valid
    if (~valid).any():
        q[~valid, -1] = mean_bias[~valid]

    q_map = torch.linalg.vector_norm(q, dim=-1).reshape(b, steps_h, steps_w)
    q2n = q_map.mean()

    if not return_diagnostic:
        return q2n.to(dtype=fused.dtype)

    uiqi_avg, uiqi_diag = mean_uiqi(fused, gt, block_size=block_size, return_diagnostic=True)
    diag = {
        "block_size": block_size,
        "shift": shift,
        "original_channels": original_channels,
        "padded_channels": c,
        "num_blocks": int(q_map.numel()),
        "q2n_map_mean": float(q_map.mean().detach().cpu()),
        "q2n_map_min": float(q_map.min().detach().cpu()),
        "q2n_map_max": float(q_map.max().detach().cpu()),
        "mean_uiqi": float(uiqi_avg.detach().cpu()),
        "mean_uiqi_diag": uiqi_diag,
    }
    return q2n.to(dtype=fused.dtype), diag


def cross_correlation(pred, gt):
    n_spectral = pred.shape[1]
    pred_reshaped = pred.reshape(n_spectral, -1)
    gt_reshaped = gt.reshape(n_spectral, -1)
    mean_fuse = torch.mean(pred_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(gt_reshaped, 1).unsqueeze(1)
    cc = torch.sum((pred_reshaped - mean_fuse) * (gt_reshaped - mean_ref), 1) / (
        torch.sqrt(torch.sum((pred_reshaped - mean_fuse) ** 2, 1) *
                   torch.sum((gt_reshaped - mean_ref) ** 2, 1))
    )
    return torch.mean(cc)


def QIndex_torch(a, b, eps=1e-8):
    e_a = torch.mean(a, dim=(1, 2))
    e_a2 = torch.mean(a * a, dim=(1, 2))
    e_b = torch.mean(b, dim=(1, 2))
    e_b2 = torch.mean(b * b, dim=(1, 2))
    e_ab = torch.mean(a * b, dim=(1, 2))
    var_a = e_a2 - e_a * e_a
    var_b = e_b2 - e_b * e_b
    cov_ab = e_ab - e_a * e_b
    return torch.mean(4 * cov_ab * e_a * e_b / ((var_a + var_b) * (e_a ** 2 + e_b ** 2) + eps))


def D_lambda_torch(l_ms, ps):
    l = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(l):
        for j in range(l):
            if j != i:
                sum += torch.abs(
                    QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) -
                    QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :])
                )
    return sum / l / (l - 1)


def D_s_torch(l_ms, pan, ps):
    l_pan = F.interpolate(pan, (l_ms.shape[-2], l_ms.shape[-1]), mode="bicubic", align_corners=True)
    l = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(l):
        sum += torch.abs(
            QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) -
            QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :])
        )
    return sum / l


def psnr_ms(fused, ms, max_value=1.0, mode="bicubic", align_corners=True):
    fused_lr = F.interpolate(fused, size=ms.shape[-2:], mode=mode, align_corners=align_corners)
    mse = torch.mean((fused_lr - ms) ** 2)
    if mse.item() <= 1e-7:
        return fused.new_tensor(float("inf"))
    return 20 * torch.log10(fused.new_tensor(max_value) / torch.sqrt(mse))


def SAM_numpy(fused_hwc, gt_hwc):
    """
    SAM (Spectral Angle Mapper) — PanCollection standard implementation.
    fused_hwc, gt_hwc: numpy arrays [H, W, C], any data range.
    Returns SAM in degrees (scalar).
    """
    fused = fused_hwc.reshape(-1, fused_hwc.shape[-1]).astype(np.float64)
    gt = gt_hwc.reshape(-1, gt_hwc.shape[-1]).astype(np.float64)
    dot = np.sum(fused * gt, axis=1)
    norm_f = np.linalg.norm(fused, axis=1)
    norm_g = np.linalg.norm(gt, axis=1)
    denom = norm_f * norm_g
    valid = denom > 1e-10
    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.mean(angles) * 180.0 / np.pi)


def ERGAS_numpy(fused_hwc, gt_hwc, ratio=4):
    """
    ERGAS — PanCollection / SSDiff standard implementation.
    fused_hwc, gt_hwc: numpy arrays [H, W, C], any data range.
    ratio: spatial resolution ratio (default 4).
    Returns ERGAS (scalar, lower is better).
    """
    fused = fused_hwc.astype(np.float64)
    gt = gt_hwc.astype(np.float64)
    n_bands = gt.shape[-1]
    ergas_sum = 0.0
    for k in range(n_bands):
        err = fused[..., k] - gt[..., k]
        rmse_k = np.sqrt(np.mean(err ** 2))
        mu_k = np.mean(gt[..., k])
        if abs(mu_k) < 1e-10:
            continue
        ergas_sum += (rmse_k / mu_k) ** 2
    return float((100.0 / ratio) * np.sqrt(ergas_sum / n_bands))


def full_ref_metrics(pred, gt, data_range=(0.0, 1.0)):
    # PanCollection-compatible SAM/ERGAS using numpy (matches SSDiff evaluation)
    pred_np = pred[0].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    gt_np = gt[0].detach().cpu().numpy().transpose(1, 2, 0)
    sam_deg = SAM_numpy(pred_np, gt_np)
    ergas_val = ERGAS_numpy(pred_np, gt_np, ratio=4)

    sam_rad = _tm_metric("spectral_angle_mapper", pred, gt)
    ergas_tm = _tm_metric("error_relative_global_dimensionless_synthesis", pred, gt)
    return {
        "MAE": F.l1_loss(pred, gt),
        "SSIM": _tm_metric("structural_similarity_index_measure", pred, gt, data_range=data_range),
        "RMSE": _tm_metric("root_mean_squared_error_using_sliding_window", pred, gt),
        "ERGAS": torch.tensor(ergas_val),
        "SAM": sam_rad,
        "SAM_deg": torch.tensor(sam_deg),
        "ERGAS_tm": ergas_tm,
        "SAM_tm_deg": sam_rad * (180.0 / math.pi),
        "RASE": _tm_metric("relative_average_spectral_error", pred, gt),
        "PSNR": _tm_metric("peak_signal_noise_ratio", pred, gt, data_range=data_range),
        "UQI": _tm_metric("universal_image_quality_index", pred, gt),
        "CC": cross_correlation(pred, gt),
        "Q2n": Q2n(pred, gt),
    }


def no_ref_metrics(lr_ms, pan, fused, fused_lr_override=None, ratio=4):
    d_lambda = D_lambda_torch(lr_ms, fused)
    d_s = D_s_torch(lr_ms, pan, fused)
    qnr_impl = PansharpeningMetrics(device=fused.device)
    qnr_val = qnr_impl.qnr(pan, lr_ms, fused, fused_lr_override=fused_lr_override)
    hqnr_val, d_lambda_k, _ = hqnr(lr_ms, pan, fused, ratio=ratio)
    scc_val = scc(pan, fused)
    return {
        "D_lambda": d_lambda,
        "D_s": d_s,
        "QNR": qnr_val,
        "D_lambda_K": d_lambda_k,
        "HQNR": hqnr_val,
        "SCC": scc_val,
    }


def sweep_no_ref_metrics(lr_ms, pan, fused, params_list):
    results = []
    for params in params_list:
        metrics = no_ref_metrics(lr_ms, pan, fused)
        ms_psnr = psnr_ms(fused, lr_ms, **params)
        result = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
        result["MS_PSNR"] = ms_psnr.item() if torch.is_tensor(ms_psnr) else ms_psnr
        result.update(params)
        results.append(result)
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    fake_pan = torch.rand(1, 1, 256, 256)
    fake_ms = torch.rand(1, 4, 64, 64)
    fake_fused = torch.rand(1, 4, 256, 256)

    print("=== No-Reference Metrics ===")
    nr = no_ref_metrics(fake_ms, fake_pan, fake_fused, ratio=4)
    for k, v in nr.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n=== Sweep ===")
    params_list = [
        {"mode": "bicubic", "align_corners": True},
        {"mode": "bilinear", "align_corners": False},
    ]
    results = sweep_no_ref_metrics(fake_ms, fake_pan, fake_fused, params_list)
    for item in results:
        print(item)
