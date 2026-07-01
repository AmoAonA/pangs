"""
PanSplat: Pansharpening via 2D Gaussian Splatting
==================================================
Three-stage optimization pipeline:
  Stage 1 - Fit Gaussians to PAN (geometry)
  Stage 2 - Learn spectral coefficients G using G*P fuse mode (MTF-GLP target)
  Stage 3 - Detail injection via auxiliary Gaussians (low-rank residual)

Iterative refinement: Stage 2 can be repeated for multiple rounds. Stage 3 can
then be run once as a separate detail-injection pass so it does not feed back
into the Stage 2 target loop.
"""

import logging
import json
import math
import os
import sys
import warnings
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim

_LOCAL_GSPLAT_PATH = os.path.join(os.path.dirname(__file__), "gsplat")
if os.path.isdir(os.path.join(_LOCAL_GSPLAT_PATH, "gsplat")):
    sys.path.insert(0, _LOCAL_GSPLAT_PATH)
    if "gsplat" in sys.modules and getattr(sys.modules["gsplat"], "__file__", None) is None:
        del sys.modules["gsplat"]

from gsplat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from torchvision.transforms.functional import gaussian_blur

from utils import full_ref_metrics, no_ref_metrics
from utils.image_utils import (
    compute_image_gradients,
    get_grid,
    load_images,
    save_image,
    separate_image_channels,
)
from utils.misc_utils import clean_dir, save_cfg, set_random_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")


class _PSDipResBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class _PSDipFusionNet(nn.Module):
    """Small FusionNet used only for PSDIP-style G initialization."""

    def __init__(self, spectral_num, channels=32):
        super().__init__()
        self.spectral_num = spectral_num
        self.conv1 = nn.Conv2d(spectral_num, channels, kernel_size=3, stride=1, padding=1)
        self.backbone = nn.Sequential(
            _PSDipResBlock(channels),
            _PSDipResBlock(channels),
            _PSDipResBlock(channels),
            _PSDipResBlock(channels),
        )
        self.conv3 = nn.Conv2d(channels, spectral_num, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x, pan):
        pan_concat = pan.repeat(1, self.spectral_num, 1, 1)
        feat = self.relu(self.conv1(pan_concat - x))
        feat = self.backbone(feat)
        return self.relu(self.conv3(feat))


class _Stage2RenderDecoder(nn.Module):
    """Decode rendered Gaussian latent maps into spectral coefficients.

    Rendering latent features before decoding is more expressive than decoding
    each Gaussian independently because the nonlinearity sees the blended local
    feature field at each pixel.
    """

    def __init__(self, latent_dim, output_dim, hidden=32, residual_scale=1.0):
        super().__init__()
        self.output_dim = output_dim
        self.residual_scale = float(residual_scale)
        hidden = max(int(hidden), output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim + output_dim + 1, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, output_dim, kernel_size=1),
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.net[0].bias, 0.0)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=1.0e-4)
        nn.init.constant_(self.net[2].bias, 0.0)

    def forward(self, latent, base_image, pan_image):
        squeeze = latent.dim() == 3
        if squeeze:
            latent = latent.unsqueeze(0)
        if base_image.dim() == 3:
            base_image = base_image.unsqueeze(0)
        if pan_image.dim() == 3:
            pan_image = pan_image.unsqueeze(0)
        if base_image.shape[-2:] != latent.shape[-2:]:
            base_image = F.interpolate(
                base_image, size=latent.shape[-2:], mode="bicubic",
                align_corners=False,
            )
        if pan_image.shape[-2:] != latent.shape[-2:]:
            pan_image = F.interpolate(
                pan_image, size=latent.shape[-2:], mode="bicubic",
                align_corners=False,
            )
        coeff = latent[:, : self.output_dim]
        residual = self.net(torch.cat([latent, base_image, pan_image], dim=1))
        out = coeff + self.residual_scale * residual
        return out.squeeze(0) if squeeze else out


class _Stage2HyperSpectralDecoder(nn.Module):
    """HyperGS-style shared spectral decoder for rendered latent Gaussian fields.

    Unlike the lightweight residual decoder above, this module treats the
    rendered latent field as the primary representation and maps it directly to
    spectral bands. The first output_dim channels provide an identity skip so
    target-based initialization still gives a stable starting point.
    """

    def __init__(self, latent_dim, output_dim, hidden=64, residual_scale=1.0):
        super().__init__()
        self.output_dim = output_dim
        self.residual_scale = float(residual_scale)
        hidden = max(int(hidden), latent_dim, output_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(latent_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head = nn.Conv2d(hidden, output_dim, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.head.weight, mean=0.0, std=1.0e-4)

    def forward(self, latent):
        squeeze = latent.dim() == 3
        if squeeze:
            latent = latent.unsqueeze(0)
        skip = latent[:, : self.output_dim]
        feat = self.stem(latent)
        feat = feat * self.gate(feat)
        out = skip + self.residual_scale * self.head(feat)
        return out.squeeze(0) if squeeze else out


class _Stage2GammaNet(nn.Module):
    """UCL-style pixel MLP that predicts a PAN-conditioned coefficient image."""

    def __init__(self, channels, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, channels),
        )

    def forward(self, image):
        squeeze = image.dim() == 3
        if squeeze:
            image = image.unsqueeze(0)
        out = self.net(image.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2).contiguous()
        return out.squeeze(0) if squeeze else out


class _Stage2RatioINR(nn.Module):
    """Pixel MLP over Gaussian-aggregated features and continuous coordinates."""

    def __init__(self, latent_dim, output_dim, hidden=64):
        super().__init__()
        hidden = max(int(hidden), 16)
        mid = max(hidden // 2, output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(int(latent_dim) + 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, int(output_dim), kernel_size=1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, latent):
        squeeze = latent.dim() == 3
        if squeeze:
            latent = latent.unsqueeze(0)
        _, _, h, w = latent.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, dtype=latent.dtype, device=latent.device),
            torch.linspace(-1.0, 1.0, w, dtype=latent.dtype, device=latent.device),
            indexing="ij",
        )
        coord = torch.stack([x, y], dim=0).unsqueeze(0).expand(latent.shape[0], -1, -1, -1)
        out = self.net(torch.cat([coord, latent], dim=1))
        return out.squeeze(0) if squeeze else out


class _Stage2RatioContextINR(nn.Module):
    """1x1-conv coordinate INR for additive HF ratio residual.

    Inputs per pixel: (x, y) plus 4*n_freq Fourier features for the same,
    plus n_aux PAN-context channels (PAN, |grad PAN|, local std at sigmas).
    Outputs an (output_dim, H, W) tensor; last conv is zero-init so the
    residual starts at zero. Pure 1x1 convs => arbitrary-scale-compatible."""

    def __init__(self, n_aux, output_dim, n_freq=6, hidden=64, n_layers=3):
        super().__init__()
        self.n_freq = int(n_freq)
        in_ch = 2 + 4 * self.n_freq + int(n_aux)
        layers = []
        d = in_ch
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Conv2d(d, int(hidden), kernel_size=1))
            layers.append(nn.GELU())
            d = int(hidden)
        last = nn.Conv2d(d, int(output_dim), kernel_size=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, coord_aux):
        return self.net(coord_aux)


class _Stage2PanStatMLP(nn.Module):
    """Per-Gaussian feature decoder: PAN-context features at the Gaussian center
    -> additive ratio coefficient. Last layer is zero-init so the residual
    starts at zero and the optimizer eases it in."""

    def __init__(self, n_feat, output_dim, hidden=32, n_layers=2):
        super().__init__()
        layers = []
        d = int(n_feat)
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(d, int(hidden)))
            layers.append(nn.GELU())
            d = int(hidden)
        last = nn.Linear(d, int(output_dim))
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Stage2M2PWeights(nn.Module):
    """Per-channel alpha vector in R^C for the per-channel PAN HF match loss
    Sigma_c || HP(R_c * PAN) - alpha_c * HP(PAN) ||^2.

    Each alpha_c learns the magnitude with which the HF of band c follows the
    HF of PAN. Init at 1.0 (channels see PAN HF unscaled by default).
    """

    def __init__(self, n_channels, init_value=1.0):
        super().__init__()
        init = torch.full((int(n_channels),), float(init_value))
        self.alpha = nn.Parameter(init)

    def forward(self):
        return self.alpha


class _Stage2SpectralBasisB(nn.Module):
    """Low-rank cross-band basis B in R^{R x C}. Used to lift per-Gaussian
    rank-R latent coefficients to per-channel coefficients before
    rasterization, coupling the C bands through a learned R-dim subspace.
    Default init: uniform 1/R so initial lifted coefficient = original
    constant target across channels.
    """

    def __init__(self, R, C, init="uniform"):
        super().__init__()
        R = int(R); C = int(C)
        if init == "uniform":
            B = torch.full((R, C), 1.0 / float(R))
        else:
            # truncated identity-ish: first R rows of I_C / sqrt(C)
            B = torch.zeros(R, C)
            for r in range(min(R, C)):
                B[r, r] = 1.0
            B = B + (1.0 / float(R)) * (1.0 - B)  # mostly identity, smoothed
        self.B = nn.Parameter(B)

    def forward(self):
        return self.B


class _Stage2P2MGaussianColors(nn.Module):
    """Per-Gaussian free MS colors for the P2M (PAN-geometry -> MS) head.

    Holds an [N_gauss, C] tensor rasterized through the same Gaussian
    geometry (xy/scale/rot) as the main UCL ratio. Output is the rendered
    MS image directly (NOT multiplied by PAN), supervised by an MS_LR
    observation loss + LF cross-consistency to the main path.
    """

    def __init__(self, n_gauss, n_channels, init_std=1.0e-3):
        super().__init__()
        init = torch.randn(int(n_gauss), int(n_channels)) * float(init_std)
        self.colors = nn.Parameter(init)

    def forward(self):
        return self.colors


class _Stage2ScaffoldGaussianDecoder(nn.Module):
    """Scaffold-GS-style decoder from PAN anchors to Stage2 child Gaussians.

    Fixed geometry: children are placed uniformly around each parent anchor.
    MLP learns only spectral ratio, not geometry offsets.
    """

    def __init__(
        self,
        output_dim,
        children=4,
        hidden=64,
        offset_scale=0.015,
        child_scale_mult=1.0,
        ratio_base=1.0,
        ratio_amp=1.0,
        latent_dim=8,
        residual_mode=False,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_children = max(1, int(children))
        self.offset_scale = float(offset_scale)
        self.child_scale_mult = float(child_scale_mult)
        self.ratio_amp = float(ratio_amp)
        self.latent_dim = max(0, int(latent_dim))
        self.residual_mode = bool(residual_mode)
        # ratio_base can be scalar or per-channel vector
        if isinstance(ratio_base, (list, tuple, torch.Tensor)):
            rb = torch.as_tensor(ratio_base, dtype=torch.float32)
            if rb.dim() == 1:
                rb = rb[None, None, :]  # [1, 1, C] for broadcasting
            self.register_buffer("_ratio_base", rb)
            self.ratio_base = 0.0
        else:
            self.ratio_base = float(ratio_base)

        hidden = max(int(hidden), 32)
        self.net = nn.Sequential(
            nn.Linear(6 + self.latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_children * self.output_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.net[-1].weight, 0.0, 0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def set_ratio_base(self, value):
        """Update ratio_base (e.g. from global LRMS/PAN statistics)."""
        if isinstance(value, (list, tuple)):
            value = torch.tensor(value, dtype=torch.float32)
        if isinstance(value, torch.Tensor):
            if value.dim() == 1:
                value = value[None, None, :]  # [1, 1, C]
            self.register_buffer("_ratio_base", value)
        else:
            self.ratio_base = float(value)

    def forward(self, feat_pan, xy, rot, scale, anchor_latent=None, ratio_bias=None):
        scale_safe = scale.clamp_min(1.0e-6)
        anchor_inputs = [
            feat_pan,
            xy * 2.0 - 1.0,
            rot,
            torch.log(scale_safe),
        ]
        if self.latent_dim > 0:
            if anchor_latent is None:
                anchor_latent = feat_pan.new_zeros(feat_pan.shape[0], self.latent_dim)
            anchor_inputs.append(anchor_latent)
        anchor = torch.cat(anchor_inputs, dim=1)
        # MLP is a decoder; per-anchor latent/bias carry image-specific state.
        raw = self.net(anchor).view(anchor.shape[0], self.num_children, self.output_dim)

        base_xy = xy[:, None, :]
        base_rot = rot[:, None, :].expand(-1, self.num_children, -1)
        base_scale = scale[:, None, :].expand(-1, self.num_children, -1) * self.child_scale_mult
        base_pan = feat_pan[:, None, :].expand(-1, self.num_children, -1) / float(self.num_children)

        # Fixed geometry: children uniformly distributed around parent
        device = xy.device
        angles = torch.linspace(
            0, 2 * math.pi, self.num_children + 1, device=device
        )[:-1]
        offsets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        offsets = offsets * self.offset_scale  # [K, 2]
        child_xy = (base_xy + offsets[None, :, :]).clamp(0.0, 1.0)

        child_scale = base_scale.clamp(1.0e-3, 10.0)
        child_rot = base_rot
        child_pan = base_pan

        ratio_base = getattr(self, "_ratio_base", self.ratio_base)
        ratio_residual = self.ratio_amp * torch.tanh(raw)
        if ratio_bias is not None:
            if ratio_bias.dim() == 2:
                ratio_bias = ratio_bias[:, None, :]
            ratio_residual = ratio_residual + ratio_bias
        if self.residual_mode:
            feat_ratio = ratio_residual
        else:
            feat_ratio = (ratio_base + ratio_residual).clamp_min(0.0)

        return (
            feat_ratio.reshape(-1, self.output_dim),
            child_pan.reshape(-1, 1),
            child_xy.reshape(-1, 2),
            child_rot.reshape(-1, 1),
            child_scale.reshape(-1, 2),
        )


class GaussianSplatting2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        set_random_seed(seed=args.seed)
        self.device = args.device
        self.dtype = torch.float32

        self._init_logging(args)
        self._init_target(args)
        self._init_gaussians(args)
        self._init_loss(args)
        self._init_optimization(args)
        self._init_pos_scale_feat(args)

    # -- Logging ---------------------------------------------------------------

    def _init_logging(self, args):
        self.log_dir = args.log_dir
        self.log_level = args.log_level
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.train_dir = os.path.join(self.log_dir, "train")
        self.save_image_format = args.save_image_format
        self.save_image_steps = args.save_image_steps
        self.eval_steps = args.eval_steps

        clean_dir(path=self.log_dir)
        os.makedirs(self.log_dir, exist_ok=False)
        os.makedirs(self.ckpt_dir, exist_ok=False)
        os.makedirs(self.train_dir, exist_ok=False)
        self._gen_logger(args)
        save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)

    def _gen_logger(self, args):
        log_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(level=log_level)
        self.worklog = logging.getLogger("PanSplat")
        self.worklog.propagate = False
        datefmt = "%Y/%m/%d %H:%M:%S"
        file_handler = logging.FileHandler(
            f"{self.log_dir}/log_train.txt", mode="a", encoding="utf8"
        )
        file_handler.setFormatter(
            logging.Formatter(fmt="[{asctime}] {message}", datefmt=datefmt, style="{")
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                fmt="\x1b[32m[{asctime}] \x1b[0m{message}", datefmt=datefmt, style="{"
            )
        )
        self.worklog.handlers = [file_handler, console_handler]
        self.worklog.info(
            f"Start optimizing {args.num_gaussians:d} Gaussians for '{args.input_path}'"
        )

    # -- Data loading ----------------------------------------------------------

    def _init_target(self, args):
        self.gamma = args.gamma
        self.block_h, self.block_w = 16, 16
        self._load_target_images(
            path=os.path.join(args.data_root, args.input_path)
        )
        self._load_ref_images(
            path=os.path.join(args.data_root, args.ref_path)
        )
        self.num_pixels = self.img_h * self.img_w
        self.image_ratio = self.ref_images.shape[-1] / self.gt_images.shape[-1]

        # Pre-build PSDip-style MTF blur+down for UCL loss
        if bool(getattr(args, "stage2_ucl_ratio", False)):
            self._build_ucl_mtf_degrade()

        path = f"{self.log_dir}/gt_res-{self.img_h:d}x{self.img_w:d}"
        self._separate_and_save_images(
            images=self.gt_images, channels=self.input_channels, path=path
        )
        path2 = f"{self.log_dir}/pan_res-{self.img_h*4:d}x{self.img_w*4:d}"
        self._separate_and_save_images(
            images=self.ref_images, channels=[1], path=path2
        )

    def _build_ucl_mtf_degrade(self):
        """Build PSDip-style MTF blur + stride-down for UCL LR_MS supervision."""
        import sys as _sys
        _sys.path.append("/data/ms/PSDip")
        from psutils import genMTF
        from functools import partial
        from torchvision.transforms import Compose

        C = sum(self.input_channels)
        ds = int(round(self.image_ratio))
        sensor = str(getattr(self.args, "sensor", "QB")).upper()
        h_mtf = genMTF(ds, sensor, C)
        K = torch.from_numpy(np.float32(h_mtf)).permute(2, 0, 1).unsqueeze(1).to(self.device)
        pd = int((K.shape[-1] - 1) / 2)

        def _mypad(x, pd=pd):
            s1, s2, s3, s4 = x.shape
            y = torch.zeros((s1, s2, s3 + 2*pd, s4 + 2*pd), device=x.device, dtype=x.dtype)
            y[:, :, :pd, :pd]            = x[:, :, :1,  :1]
            y[:, :, :pd, pd:pd+s4]       = x[:, :, :1,  :]
            y[:, :, :pd, pd+s4:]         = x[:, :, :1,  -1:]
            y[:, :, pd:pd+s3, :pd]       = x[:, :, :,   :1]
            y[:, :, pd:pd+s3, pd:pd+s4]  = x
            y[:, :, pd:pd+s3, pd+s4:]    = x[:, :, :,   -1:]
            y[:, :, pd+s3:, :pd]         = x[:, :, -1:, :1]
            y[:, :, pd+s3:, pd:pd+s4]    = x[:, :, -1:, :]
            y[:, :, pd+s3:, pd+s4:]      = x[:, :, -1:, -1:]
            return y

        self._ucl_blur = Compose([_mypad, partial(F.conv2d, weight=K, groups=C)])
        self._ucl_down = lambda x: x[:, :, 2::ds, 2::ds]

    def _load_target_images(self, path):
        (
            self.gt_images,
            self.input_channels,
            self.image_fnames,
            self.bit_depths,
        ) = load_images(load_path=path, downsample_ratio=None, gamma=self.gamma)
        self.gt_images = torch.from_numpy(self.gt_images).to(
            dtype=self.dtype, device=self.device
        )
        self.img_h, self.img_w = self.gt_images.shape[1:]
        self.tile_bounds = (
            (self.img_w + self.block_w - 1) // self.block_w,
            (self.img_h + self.block_h - 1) // self.block_h,
            1,
        )

    def _load_ref_images(self, path):
        self.ref_images, _, _, _ = load_images(
            load_path=path, downsample_ratio=None, gamma=self.gamma
        )
        self.ref_images = torch.from_numpy(self.ref_images).to(
            dtype=self.dtype, device=self.device
        )

    def _separate_and_save_images(self, images, channels, path):
        images_sep = separate_image_channels(
            images=images, input_channels=channels
        )
        for idx, image in enumerate(images_sep, 1):
            suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
            save_image(
                image, f"{path}{suffix}.{self.save_image_format}", gamma=self.gamma
            )

    # -- Gaussian initialization -----------------------------------------------

    def _init_gaussians(self, args):
        self.num_gaussians = args.num_gaussians
        self.aux_num_gaussians = args.num_gaussians
        self.total_num_gaussians = args.num_gaussians
        self._densify_round_count = 0
        self.disable_prog_optim = args.disable_prog_optim
        if not self.disable_prog_optim:
            self.initial_ratio = args.initial_ratio
            self.add_times = args.add_times
            self.add_steps = args.add_steps
            self.num_gaussians = math.ceil(
                self.initial_ratio * self.total_num_gaussians
            )
            self.aux_num_gaussians = self.num_gaussians
            self.max_add_num = math.ceil(
                float(self.total_num_gaussians - self.num_gaussians)
                / self.add_times
            )
            min_steps = self.add_steps * self.add_times + args.post_min_steps
            if args.max_steps < min_steps:
                args.max_steps = min_steps

        self.init_scale = args.init_scale
        self.disable_topk_norm = args.disable_topk_norm
        self.disable_inverse_scale = args.disable_inverse_scale
        self.disable_color_init = args.disable_color_init
        self.output_dim = sum(self.input_channels)
        self.feat_dim = self.output_dim
        self.stage2_feat_dim = max(
            self.output_dim, int(getattr(args, "stage2_feat_dim", self.output_dim))
        )
        self.stage2_decode_mode = str(
            getattr(args, "stage2_decode_mode", "direct")
        ).lower()
        self.skip_unused_ms_x = bool(getattr(args, "stage2_ucl_ratio", False)) and bool(
            getattr(args, "stage2_ucl_skip_unused_ms_x", True)
        )

        self.xy = nn.Parameter(
            torch.rand(self.num_gaussians, 2, dtype=self.dtype, device=self.device)
        )
        self.scale = nn.Parameter(
            torch.ones(self.num_gaussians, 2, dtype=self.dtype, device=self.device)
        )
        self.rot = nn.Parameter(
            torch.zeros(self.num_gaussians, 1, dtype=self.dtype, device=self.device)
        )
        self.feat_pan = nn.Parameter(
            torch.rand(self.num_gaussians, 1, dtype=self.dtype, device=self.device)
        )
        if self.skip_unused_ms_x:
            self.feat_ms = None
            self.feat_x = None
            self.feat_ms_detail = None
        else:
            self.feat_ms = nn.Parameter(
                torch.rand(
                    self.num_gaussians, self.stage2_feat_dim,
                    dtype=self.dtype, device=self.device
                )
            )
            self.feat_x = nn.Parameter(
                torch.rand(
                    self.num_gaussians, self.output_dim,
                    dtype=self.dtype, device=self.device
                )
            )
            self.feat_ms_detail = nn.Parameter(
                torch.zeros(
                    self.num_gaussians, self.output_dim,
                    dtype=self.dtype, device=self.device,
                )
            )
        self.stage2_decoder = None
        self.stage2_scaffold_decoder = None
        self.stage2_scaffold_latent = None
        self.stage2_scaffold_ratio_bias = None
        self.stage2_scaffold_base_ratio = None
        self.stage2_gamma_net = None
        self.stage2_ratio_inr = None
        self.stage2_ratio_latent = None
        self.stage2_ratio_base = None
        self.stage2_ucl_output_ratio_base = None
        self.stage2_pixel_residual = None
        self.stage2_grid_residual = None
        self.stage2_pan_detail_coeff = None
        self.stage2_pan_gap_gain_coeff = None
        self.stage2_lig_gap_coeff = None
        self.stage2_local_ratio_coeff = None
        self.stage2_pan_stat_mlp = None
        self.stage2_ratio_context_inr = None
        self.stage2_m2p_weights = None
        self.stage2_p2m_colors = None
        self.stage2_spectral_basis_B = None
        self.stage2_local_ratio_basis = None
        self.stage2_local_ratio_edge_src = None
        self.stage2_local_ratio_edge_dst = None
        self.stage2_ucl_teacher = None
        self.feat_residual = None
        self.stage2_residual_basis_B = None
        self.stage2_ratio_cluster_scalar = None
        self.stage2_ratio_cluster_gain = None
        self.stage2_ratio_cluster_assign = None
        self.feat_ratio_dx = None
        self.feat_ratio_dy = None
        self.feat_ratio_band_mask = None
        self.stage2_pan_descriptor = None
        self.stage2_pan_descriptor_names = None
        self.stage2_pan_descriptor_weight = None
        if self._stage2_uses_post_decoder_config():
            decoder_hidden = int(getattr(args, "stage2_decoder_hidden", 32))
            decoder_residual_scale = float(
                getattr(args, "stage2_decoder_residual_scale", 1.0)
            )
            if self.stage2_decode_mode in ("hypergs_decoder", "hypergs", "spectral_decoder"):
                self.stage2_decoder = _Stage2HyperSpectralDecoder(
                    self.stage2_feat_dim,
                    self.output_dim,
                    hidden=decoder_hidden,
                    residual_scale=decoder_residual_scale,
                ).to(self.device)
            else:
                self.stage2_decoder = _Stage2RenderDecoder(
                    self.stage2_feat_dim,
                    self.output_dim,
                    hidden=decoder_hidden,
                    residual_scale=decoder_residual_scale,
                ).to(self.device)

    def _init_aux_gaussians(self, args):
        n = getattr(args, "num_gaussians_aux", self.num_gaussians)
        self.num_gaussians_aux = n
        self.xy_aux = nn.Parameter(torch.rand(n, 2, device=self.device))
        self.scale_aux = nn.Parameter(torch.ones(n, 2, device=self.device))
        self.rot_aux = nn.Parameter(torch.zeros(n, 1, device=self.device))
        self.feat_aux = nn.Parameter(torch.zeros(n, 1, device=self.device))
        self.band_coeff = nn.Parameter(torch.ones(self.output_dim, device=self.device))
        self.feat_aux_res = nn.Parameter(
            torch.zeros(n, self.output_dim, device=self.device)
        )
        self.aux_num_gaussians = n

    # -- Loss functions --------------------------------------------------------

    def _init_loss(self, args):
        self.l1_loss_ratio = args.l1_loss_ratio
        self.ssim_loss_ratio = args.ssim_loss_ratio
        self.pan_grad_weight = float(getattr(args, "pan_grad_weight", 1.0))
        self.pan_hp_weight = float(getattr(args, "pan_hp_weight", 1.0))
        self.pan_ssim_weight = float(getattr(args, "pan_ssim_weight", 0.0))
        self.pan_highpass_ksize = int(getattr(args, "pan_highpass_ksize", 7))
        if self.pan_highpass_ksize % 2 == 0:
            self.pan_highpass_ksize += 1

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=self.dtype, device=self.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=self.dtype, device=self.device,
        ).view(1, 1, 3, 3)
        self.register_buffer("pan_sobel_x", sobel_x)
        self.register_buffer("pan_sobel_y", sobel_y)

        if self.output_dim == 4:
            pan_weight = torch.tensor(
                [0.1, 0.3, 0.3, 0.3], dtype=self.dtype, device=self.device
            )
        else:
            pan_weight = (
                torch.ones(self.output_dim, dtype=self.dtype, device=self.device)
                / max(self.output_dim, 1)
            )
        self.register_buffer("pan_channel_weight", pan_weight)

    def _pan_grad_mag(self, image):
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        grad_x = F.conv2d(image, self.pan_sobel_x, padding=1)
        grad_y = F.conv2d(image, self.pan_sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12).squeeze(0)

    def _pan_grad_xy(self, image):
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        grad_x = F.conv2d(image, self.pan_sobel_x, padding=1).squeeze(0)
        grad_y = F.conv2d(image, self.pan_sobel_y, padding=1).squeeze(0)
        return grad_x, grad_y

    def _highpass(self, image):
        if image.dim() == 2:
            image = image.unsqueeze(0)
        ks = [self.pan_highpass_ksize, self.pan_highpass_ksize]
        return image - gaussian_blur(image, ks)

    def _local_density_map(self, xy, pan_h, pan_w, sigma_px):
        n = int(xy.shape[0])
        if n <= 0 or sigma_px <= 0.0:
            return torch.ones(pan_h, pan_w, dtype=self.dtype, device=self.device)
        cols = torch.clamp(
            torch.round(xy[:, 0] * pan_w - 0.5).long(), 0, pan_w - 1
        )
        rows = torch.clamp(
            torch.round(xy[:, 1] * pan_h - 0.5).long(), 0, pan_h - 1
        )
        idx = rows * pan_w + cols
        flat = torch.zeros(pan_h * pan_w, dtype=self.dtype, device=self.device)
        flat.scatter_add_(0, idx, torch.ones(n, dtype=self.dtype, device=self.device))
        density = flat.view(1, 1, pan_h, pan_w)
        density = self._gaussian_lowpass_sigma(density, float(sigma_px)).view(pan_h, pan_w)
        return density.clamp_min(0.0)

    def _available_sampling_mask(self, pan_h, pan_w, xy=None, radius_px=-1):
        if xy is None or int(xy.shape[0]) <= 0 or int(radius_px) < 0:
            return np.ones(pan_h * pan_w, dtype=bool)
        cols = torch.clamp(
            torch.round(xy[:, 0] * pan_w - 0.5).long(), 0, pan_w - 1
        )
        rows = torch.clamp(
            torch.round(xy[:, 1] * pan_h - 0.5).long(), 0, pan_h - 1
        )
        occ = torch.zeros(pan_h, pan_w, dtype=torch.bool, device=self.device)
        occ[rows, cols] = True
        if int(radius_px) > 0:
            occ_f = occ.to(dtype=self.dtype).view(1, 1, pan_h, pan_w)
            k = 2 * int(radius_px) + 1
            occ = (
                F.max_pool2d(
                    occ_f, kernel_size=k, stride=1, padding=int(radius_px)
                )
                .view(pan_h, pan_w)
                .bool()
            )
        return (~occ.reshape(-1)).detach().cpu().numpy()

    def _compute_pan_loss(self, pan_pred, pan_ref):
        pred_hp = self._highpass(pan_pred)
        ref_hp = self._highpass(pan_ref)
        loss = self.pan_hp_weight * F.l1_loss(pred_hp, ref_hp)
        loss += self.pan_grad_weight * F.l1_loss(
            self._pan_grad_mag(pan_pred), self._pan_grad_mag(pan_ref)
        )
        if self.pan_ssim_weight > 0:
            loss += self.pan_ssim_weight * (
                1 - fused_ssim(pan_pred.unsqueeze(0), pan_ref.unsqueeze(0))
            )
        return loss

    def _to_pan_estimate(self, features):
        if features.shape[0] == 1:
            return features
        weights = self.pan_channel_weight[: features.shape[0]].clamp_min(0.0)
        if weights.sum().item() <= 0:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        return (features * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)

    # -- HR target construction ------------------------------------------------

    def _gaussian_blur_chw(self, image, sigma):
        sigma = float(sigma)
        if sigma <= 0:
            return image
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(
            -radius, radius + 1, dtype=image.dtype, device=image.device
        )
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-12)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        weight = kernel_2d.view(1, 1, *kernel_2d.shape).repeat(image.shape[0], 1, 1, 1)
        pad = radius
        return F.conv2d(
            F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="replicate"),
            weight,
            groups=image.shape[0],
        ).squeeze(0)

    def _build_mtf_glp_target_at(self, out_h, out_w, log=True):
        """Build an MTF-GLP target at a requested supervision resolution.

        If `stage2_vivone_full_scale_iters > 0`, iteratively refines the
        per-band injection coefficients g_k at full HR scale per Vivone
        2018 (TIP), rather than using only the reduced-scale estimate.
        """
        ms_lr = self.gt_images
        pan_hr = self.ref_images
        ms_h, ms_w = ms_lr.shape[-2], ms_lr.shape[-1]

        pan_lr = F.interpolate(
            pan_hr.unsqueeze(0), size=(ms_h, ms_w), mode="area"
        ).squeeze(0)
        p_flat = pan_lr[0].flatten()
        p_mean = p_flat.mean()
        p_var = p_flat.var().clamp(min=1e-8)
        gain_scale = self._current_stage2_mtf_gain_scale()
        gain_clip = float(getattr(self.args, "stage2_mtf_gain_clip", 1.0))
        g_list = []
        for k in range(ms_lr.shape[0]):
            m_flat = ms_lr[k].flatten()
            cov_k = ((p_flat - p_mean) * (m_flat - m_flat.mean())).mean()
            g_list.append((cov_k / p_var).clamp(-gain_clip, gain_clip))
        g = gain_scale * torch.stack(g_list).view(-1, 1, 1)

        pan_cur = F.interpolate(
            pan_hr.unsqueeze(0), size=(out_h, out_w), mode="area"
        ).squeeze(0)
        pan_base = F.interpolate(
            pan_lr.unsqueeze(0), size=(out_h, out_w),
            mode="bicubic", align_corners=False,
        ).squeeze(0)
        pan_hp = pan_cur - pan_base

        ms_base = F.interpolate(
            ms_lr.unsqueeze(0), size=(out_h, out_w),
            mode="bicubic", align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)

        # Vivone 2018 full-scale iterative refinement of g_k.
        # Iterates: g_k^(t+1) = cov(HRMS^(t)_k − mean, pan_hp − mean) / var(pan_hp)
        #           HRMS^(t+1) = ms_base + g_k^(t+1) · pan_hp
        # Converges geometrically; ~5 iterations is enough on QB.
        vivone_iters = int(getattr(self.args, "stage2_vivone_full_scale_iters", 0))
        g_rs_log = [float(g_list[k].item()) for k in range(len(g_list))]
        if vivone_iters > 0:
            pan_hp_flat = pan_hp[0].flatten()  # PAN HP is identical across channels
            ph_mean = pan_hp_flat.mean()
            ph_var = (pan_hp_flat - ph_mean).pow(2).mean().clamp_min(1e-8)
            g_iter = g.clone()
            for _ in range(vivone_iters):
                hrms_t = (ms_base + g_iter * pan_hp).clamp(0.0, 1.0)
                new_g_list = []
                for k in range(ms_lr.shape[0]):
                    m_flat = hrms_t[k].flatten()
                    cov_k = (
                        (pan_hp_flat - ph_mean) * (m_flat - m_flat.mean())
                    ).mean()
                    new_g_list.append((cov_k / ph_var).clamp(-gain_clip, gain_clip))
                g_iter = gain_scale * torch.stack(new_g_list).view(-1, 1, 1)
            g_vivone_log = [float(g_iter[k, 0, 0].item()) for k in range(ms_lr.shape[0])]
            g = g_iter
            if log:
                rs_str = ", ".join(f"b{k}={v:.3f}" for k, v in enumerate(g_rs_log))
                fs_str = ", ".join(f"b{k}={v:.3f}" for k, v in enumerate(g_vivone_log))
                self.worklog.info(
                    f"Vivone full-scale g_k ({vivone_iters} iters): "
                    f"RS=[{rs_str}] -> FS=[{fs_str}]"
                )

        target = (ms_base + g * pan_hp).clamp(0.0, 1.0)

        if log:
            g_str = ", ".join(
                f"b{k}={g[k, 0, 0].item():.3f}" for k in range(ms_lr.shape[0])
            )
            self.worklog.info(
                f"MTF-GLP injection coefficients: {g_str} "
                f"(target={out_h}x{out_w}, scale={gain_scale:.2f}, clip={gain_clip:.2f})"
            )
        return target

    def _build_sfim_target_at(self, out_h, out_w, log=True):
        """Build an SFIM-style multiplicative target at a requested resolution."""
        ms_lr = self.gt_images
        pan_hr = self.ref_images
        sigma = float(getattr(self.args, "stage2_sfim_sigma", 3.0))

        pan_cur = F.interpolate(
            pan_hr.unsqueeze(0), size=(out_h, out_w), mode="area"
        ).squeeze(0)
        ms_base = F.interpolate(
            ms_lr.unsqueeze(0), size=(out_h, out_w),
            mode="bicubic", align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)

        mean_ms = ms_lr.mean(dim=(-2, -1), keepdim=True)
        std_ms = ms_lr.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-8)
        mean_pan = pan_cur.mean(dim=(-2, -1), keepdim=True)
        std_pan = pan_cur.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-8)
        pan_matched = (std_ms / std_pan) * (
            pan_cur.repeat(ms_lr.shape[0], 1, 1) - mean_pan
        ) + mean_ms
        pan_lp = self._gaussian_blur_chw(pan_matched, sigma)
        eps = float(getattr(self.args, "stage2_sfim_eps", 1.0e-6))
        target = ms_base * (pan_matched + eps) / (pan_lp + eps)
        target = target.clamp(0.0, 1.0)

        if log:
            self.worklog.info(
                f"SFIM target: sigma={sigma:.2f}, target={out_h}x{out_w}"
            )
        return target

    def _build_external_stage2_target_at(self, out_h, out_w, log=True):
        path = str(getattr(self.args, "stage2_external_target_path", "")).strip()
        if not path:
            raise ValueError("stage2_external_target_path is required for external Stage2 targets")
        cache = getattr(self, "_stage2_external_target_cache", None)
        if cache is None or cache.get("path") != path:
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                key = str(getattr(self.args, "stage2_external_target_key", "fused"))
                if key not in arr:
                    key = arr.files[0]
                arr = arr[key]
            tensor = torch.as_tensor(arr, dtype=self.dtype, device=self.device)
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            if tensor.dim() == 3 and tensor.shape[-1] == self.output_dim:
                tensor = tensor.permute(2, 0, 1)
            if tensor.dim() != 3 or tensor.shape[0] != self.output_dim:
                raise ValueError(
                    "External Stage2 target must have shape [C,H,W] or [H,W,C]; "
                    f"got {tuple(tensor.shape)}"
                )
            cache = {"path": path, "target": tensor.clamp(0.0, 1.0)}
            self._stage2_external_target_cache = cache

        target = cache["target"]
        if target.shape[-2:] != (out_h, out_w):
            target = F.interpolate(
                target.unsqueeze(0), size=(out_h, out_w),
                mode="bicubic", align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)
        if log:
            self.worklog.info(
                f"External Stage2 target: {path} target={out_h}x{out_w}"
            )
        return target

    def _build_stage2_target_at(self, out_h, out_w, log=True):
        mode = str(getattr(self.args, "stage2_hr_target_mode", "mtf_glp")).lower()
        if mode in ("mtf_glp", "mtf-glp", "glp"):
            return self._build_mtf_glp_target_at(out_h, out_w, log=log)
        if mode in ("sfim",):
            return self._build_sfim_target_at(out_h, out_w, log=log)
        if mode in ("external_npy", "external", "npy"):
            return self._build_external_stage2_target_at(out_h, out_w, log=log)
        if mode in ("mtf_sfim_blend", "sfim_mtf_blend", "blend_mtf_sfim"):
            alpha = float(getattr(self.args, "stage2_target_blend_alpha", 0.5))
            alpha = min(1.0, max(0.0, alpha))
            mtf_target = self._build_mtf_glp_target_at(out_h, out_w, log=log)
            sfim_target = self._build_sfim_target_at(out_h, out_w, log=log)
            target = (alpha * mtf_target + (1.0 - alpha) * sfim_target).clamp(0.0, 1.0)
            if log:
                self.worklog.info(
                    f"Stage2 target blend: alpha_mtf={alpha:.2f}, target={out_h}x{out_w}"
                )
            return target
        raise ValueError(f"Unsupported stage2_hr_target_mode: {mode}")

    def _build_mtf_glp_target(self):
        """Build full-resolution MTF-GLP target: bicubic(LRMS) + g_k * PAN_hp."""
        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        return self._build_mtf_glp_target_at(pan_h, pan_w, log=True)

    def _build_stage2_target(self):
        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        return self._build_stage2_target_at(pan_h, pan_w, log=True)

    def _load_stage2_densify_gt_hr(self, fallback_hr):
        """Load the HR reference used only by Stage2 densification score.

        Args:
            fallback_hr: Tensor [C,H,W] used when the densify source is the
                default Stage2 target.

        Returns:
            Tensor [C,H,W] on the current device/dtype.
        """
        source = str(
            getattr(self.args, "stage2_densify_gt_hr_source", "stage2_target")
        ).lower()
        if source in ("stage2_target", "target", "pseudo", "mtf_glp", "mtf-glp"):
            return fallback_hr
        if source in ("rr_gt", "external", "gt", "hr_gt"):
            path = str(getattr(self.args, "stage2_densify_gt_hr_path", "")).strip()
            if not path:
                raise ValueError(
                    "stage2_densify_gt_hr_path is required when "
                    "stage2_densify_gt_hr_source=rr_gt"
                )
            cache = getattr(self, "_stage2_densify_gt_hr_cache", None)
            if cache is None or cache.get("path") != path:
                if not os.path.isabs(path):
                    path = os.path.join(self.args.data_root, path)
                gt_hr_np, _, _, _ = load_images(
                    load_path=path, downsample_ratio=None, gamma=self.gamma
                )
                gt_hr = torch.from_numpy(gt_hr_np).to(
                    dtype=self.dtype, device=self.device
                )
                cache = {"path": path, "target": gt_hr}
                self._stage2_densify_gt_hr_cache = cache
                self.worklog.info(f"Stage2 densify GT HR source: {path}")
            gt_hr = cache["target"]
            if gt_hr.shape[-2:] != fallback_hr.shape[-2:]:
                gt_hr = F.interpolate(
                    gt_hr.unsqueeze(0),
                    size=fallback_hr.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).clamp(0.0, 1.0)
            return gt_hr.to(dtype=self.dtype, device=self.device)
        raise ValueError(
            "stage2_densify_gt_hr_source must be stage2_target, rr_gt, or external"
        )

    def _stage2_debug_hr_trace_source_is_real_gt(self):
        source = str(
            getattr(self.args, "stage2_densify_gt_hr_source", "stage2_target")
        ).lower()
        return source in ("rr_gt", "external", "gt", "hr_gt")

    def _stage2_debug_hr_trace_enabled(self):
        return (
            bool(getattr(self.args, "stage2_debug_hr_trace", False))
            and self._stage2_debug_hr_trace_source_is_real_gt()
        )

    def _stage2_debug_hr_trace_path(self):
        path = str(getattr(self.args, "stage2_debug_hr_trace_path", "")).strip()
        if not path:
            path = os.path.join(self.log_dir, "stage2_debug_hr_trace.json")
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    def _flush_stage2_debug_hr_trace(self):
        records = getattr(self, "_stage2_debug_hr_trace", None)
        if not records:
            return
        best = max(records, key=lambda item: item["hr_psnr"])
        payload = {
            "source": str(
                getattr(self.args, "stage2_densify_gt_hr_source", "stage2_target")
            ),
            "path": self._stage2_debug_hr_trace_path(),
            "n_records": len(records),
            "best": best,
            "last": records[-1],
            "records": records,
        }
        out_path = self._stage2_debug_hr_trace_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _record_stage2_debug_hr_trace(
        self,
        tag,
        round_index,
        step,
        ms_hr,
        gt_hr_real,
        ms_lr=None,
    ):
        if not self._stage2_debug_hr_trace_enabled() or gt_hr_real is None:
            return
        with torch.no_grad():
            pred_hr = ms_hr.detach().clamp(0.0, 1.0)
            ref_hr = gt_hr_real.detach().to(
                device=pred_hr.device, dtype=pred_hr.dtype
            ).clamp(0.0, 1.0)
            if ref_hr.shape[-2:] != pred_hr.shape[-2:]:
                ref_hr = F.interpolate(
                    ref_hr.unsqueeze(0),
                    size=pred_hr.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).clamp(0.0, 1.0)
            hr_mse = F.mse_loss(pred_hr, ref_hr).clamp_min(1.0e-12)
            hr_psnr = -10.0 * torch.log10(hr_mse)
            entry = {
                "tag": str(tag),
                "round": int(round_index),
                "step": int(step),
                "global_step": int(getattr(self, "step", step)),
                "hr_psnr": float(hr_psnr.item()),
                "hr_mse": float(hr_mse.item()),
            }
            if ms_lr is not None:
                pred_lr = ms_lr.detach().clamp(0.0, 1.0)
                ref_lr = self.gt_images.detach().to(
                    device=pred_lr.device, dtype=pred_lr.dtype
                ).clamp(0.0, 1.0)
                lr_mse = F.mse_loss(pred_lr, ref_lr).clamp_min(1.0e-12)
                lr_psnr = -10.0 * torch.log10(lr_mse)
                entry["lr_psnr"] = float(lr_psnr.item())
                entry["lr_mse"] = float(lr_mse.item())
            records = getattr(self, "_stage2_debug_hr_trace", None)
            if records is None:
                records = []
                self._stage2_debug_hr_trace = records
            records.append(entry)
            self.worklog.info(
                "Stage2 HR_TRACE | "
                f"tag={entry['tag']} round={entry['round']} "
                f"step={entry['step']} global={entry['global_step']} "
                f"HR_PSNR={entry['hr_psnr']:.4f}"
                + (
                    f" LR_PSNR={entry['lr_psnr']:.4f}"
                    if "lr_psnr" in entry
                    else ""
                )
            )
            self._flush_stage2_debug_hr_trace()

    def _mtf_glp_gain_image(self):
        ms_lr = self.gt_images
        pan_hr = self.ref_images
        pan_h, pan_w = pan_hr.shape[-2:]
        pan_lr = F.interpolate(
            pan_hr.unsqueeze(0),
            size=ms_lr.shape[-2:],
            mode="area",
        ).squeeze(0)
        pan_lr_up = F.interpolate(
            pan_lr.unsqueeze(0),
            size=(pan_h, pan_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        pan_hp = pan_hr - pan_lr_up

        gain_scale = self._current_stage2_mtf_gain_scale()
        gain_clip = float(getattr(self.args, "stage2_mtf_gain_clip", 1.0))
        p_var = pan_hp.var(unbiased=False).clamp(min=1.0e-6)
        gains = []
        for k in range(ms_lr.shape[0]):
            ms_up_k = F.interpolate(
                ms_lr[k:k+1].unsqueeze(0),
                size=(pan_h, pan_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            cov_k = ((ms_up_k - ms_up_k.mean()) * (pan_hp - pan_hp.mean())).mean()
            gains.append((cov_k / p_var).clamp(-gain_clip, gain_clip))
        g = gain_scale * torch.stack(gains).view(-1, 1, 1)
        return g, pan_hp, pan_lr_up

    # -- Gaussian splatting core -----------------------------------------------

    def _stage2_fuse_mode(self):
        return str(getattr(self.args, "stage2_fuse_mode", "g_times_p")).lower()

    def _stage2_uses_post_decoder_config(self):
        return (
            getattr(self, "stage2_feat_dim", self.output_dim) > self.output_dim
            and getattr(self, "stage2_decode_mode", "direct")
            in (
                "post_render_mlp", "post_mlp", "render_mlp", "latent_render",
                "hypergs_decoder", "hypergs", "spectral_decoder",
            )
        )

    def _stage2_uses_post_decoder(self):
        return self._stage2_uses_post_decoder_config() and self.stage2_decoder is not None

    def _stage2_uses_hypergs_decoder(self):
        return self._stage2_uses_post_decoder() and self.stage2_decode_mode in (
            "hypergs_decoder", "hypergs", "spectral_decoder"
        )

    def _stage2_uses_dense_phat_multiply(self):
        return self._stage2_fuse_mode() in (
            "render_g_times_input_phat",
            "dense_g_times_input_phat",
            "render_g_times_phat_image",
            "dense_phat",
        )

    def _stage2_uses_glp_adaptive_gain(self):
        return self._stage2_fuse_mode() in (
            "glp_adaptive_gain",
            "glp_fs_gaussian_gain",
            "rr_conservative",
        )

    def _stage2_uses_pan_residual_gain(self):
        return self._stage2_fuse_mode() in (
            "base_plus_pan_hpf_gain",
            "base_plus_pan_hp_gain",
            "pan_residual_gain",
            "ms_base_pan_gain",
        )

    def _stage2_uses_kernel_affine_pan(self):
        return self._stage2_fuse_mode() in (
            "kernel_affine_pan",
            "gaussian_affine_pan",
            "feat_pan_affine",
            "pan_feat_affine",
        )

    def _stage2_ucl_product_mode(self):
        return str(getattr(self.args, "stage2_ucl_product_mode", "external")).lower()

    def _stage2_uses_ucl_scaffold_mlp(self):
        return bool(getattr(self.args, "stage2_ucl_scaffold_mlp", False))

    def _stage2_uses_ucl_local_ratio(self):
        return bool(getattr(self.args, "stage2_ucl_local_ratio", False))

    def _stage2_uses_ucl_inr(self):
        return bool(getattr(self.args, "stage2_ucl_implicit_field", False))

    def _stage2_uses_ucl_pixel_residual(self):
        return bool(getattr(self.args, "stage2_ucl_pixel_residual", False))

    def _stage2_uses_ucl_grid_residual(self):
        return bool(getattr(self.args, "stage2_ucl_grid_residual", False))

    def _stage2_uses_ucl_pan_detail(self):
        return bool(getattr(self.args, "stage2_ucl_pan_detail", False))

    def _stage2_uses_ucl_pan_gap_residual(self):
        return bool(getattr(self.args, "stage2_ucl_pan_gap_residual", False))

    def _stage2_uses_ucl_lig_gap_residual(self):
        return bool(getattr(self.args, "stage2_ucl_lig_gap_residual", False))

    def _select_ucl_pan_image(self, name, pan_rendered_s1):
        name = str(name).lower()
        if name in ("raw", "raw_pan", "input", "input_pan", "ref"):
            return self.ref_images
        if name in ("render", "rendered", "rendered_s1", "stage1", "s1"):
            return pan_rendered_s1
        raise ValueError(f"Unsupported UCL PAN source: {name}")

    def _render_stage1_pan_at(self, img_h, img_w, full_h, full_w, pan_rendered_s1=None):
        if (
            pan_rendered_s1 is not None
            and pan_rendered_s1.shape[-2:] == (img_h, img_w)
        ):
            return pan_rendered_s1
        cache = getattr(self, "_stage2_ucl_pan_render_cache", None)
        if cache is None:
            cache = {}
            self._stage2_ucl_pan_render_cache = cache
        key = (int(img_h), int(img_w), int(full_h), int(full_w))
        if key not in cache:
            render_scale = 0.5 * (
                float(img_h) / float(full_h) + float(img_w) / float(full_w)
            )
            tb = self._stage2_tile_bounds(img_h, img_w)
            xy = getattr(self, "_stage1_pan_xy", self.xy.detach())
            scale_param = getattr(self, "_stage1_pan_scale", self.scale.detach())
            rot = getattr(self, "_stage1_pan_rot", self.rot.detach())
            feat_pan = getattr(self, "_stage1_pan_feat", self.feat_pan.detach())
            with torch.no_grad():
                pan, _ = self._rasterize_features(
                    xy.detach(),
                    self._scale_param_to_render_scale(
                        scale_param.detach(), render_scale
                    ).detach(),
                    rot.detach(),
                    feat_pan.detach(),
                    img_h,
                    img_w,
                    tb,
                )
                cache[key] = pan.clamp(0.0, 1.0).detach()
        return cache[key]

    def _snapshot_stage1_pan_state(self):
        self._stage1_pan_xy = self.xy.detach().clone()
        self._stage1_pan_scale = self.scale.detach().clone()
        self._stage1_pan_rot = self.rot.detach().clone()
        self._stage1_pan_feat = self.feat_pan.detach().clone()

    def render_ucl_at_scale(self, render_scale_ratio=1.0):
        """Render the fitted UCL ratio field at a scale relative to native PAN."""
        pan_h, pan_w = self.ref_images.shape[-2:]
        ratio = float(render_scale_ratio)
        out_h = max(1, int(round(pan_h * ratio)))
        out_w = max(1, int(round(pan_w * ratio)))
        tb = self._stage2_tile_bounds(out_h, out_w)
        pan_rendered = self._render_stage1_pan_at(
            out_h, out_w, pan_h, pan_w, pan_rendered_s1=None
        )
        product_pan = self._select_ucl_pan_image_at(
            getattr(self.args, "stage2_ucl_product_pan", "rendered"),
            pan_rendered,
            out_h,
            out_w,
            pan_h,
            pan_w,
        )
        ms_hr, ratio_hr, product_hr, pan_new = self._render_ucl_current_output(
            out_h,
            out_w,
            tb,
            product_pan,
            pan_rendered,
            render_scale_ratio=ratio,
        )
        return ms_hr.clamp(0.0, 1.0), ratio_hr, product_hr, pan_rendered, pan_new

    def _select_ucl_pan_image_at(
        self, name, pan_rendered_s1, img_h, img_w, full_h, full_w
    ):
        name = str(name).lower()
        if name in ("raw", "raw_pan", "input", "input_pan", "ref"):
            pan = self.ref_images
            if pan.shape[-2:] == (img_h, img_w):
                return pan
            return F.interpolate(
                pan.unsqueeze(0), size=(img_h, img_w), mode="area"
            ).squeeze(0)
        if name in ("render", "rendered", "rendered_s1", "stage1", "s1"):
            return self._render_stage1_pan_at(
                img_h, img_w, full_h, full_w, pan_rendered_s1=pan_rendered_s1
            )
        raise ValueError(f"Unsupported UCL PAN source: {name}")

    def _stage2_ucl_product_pan(self, pan_rendered_s1):
        source = getattr(self.args, "stage2_ucl_product_pan", "rendered")
        if str(source).lower() in (
            "current",
            "current_render",
            "current_rendered",
            "rendered_current",
            "stage2_rendered",
        ):
            pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
            tb = self._stage2_tile_bounds(pan_h, pan_w)
            with torch.no_grad():
                pan, _ = self._rasterize_features(
                    self.xy.detach(),
                    self._get_scale(1.0).detach(),
                    self.rot.detach(),
                    self.feat_pan.detach(),
                    pan_h,
                    pan_w,
                    tb,
                )
            return pan.clamp(0.0, 1.0).detach()
        return self._select_ucl_pan_image(source, pan_rendered_s1)

    def _stage2_ucl_uses_current_product_pan(self):
        source = str(getattr(self.args, "stage2_ucl_product_pan", "rendered")).lower()
        return source in (
            "current",
            "current_render",
            "current_rendered",
            "rendered_current",
            "stage2_rendered",
        )

    def _stage2_ucl_gamma_input_pan(self, pan_rendered_s1):
        source = getattr(self.args, "stage2_ucl_gamma_input_pan", "rendered")
        return self._select_ucl_pan_image(source, pan_rendered_s1)

    def _stage2_ucl_gamma_denom_pan(self, pan_rendered_s1):
        source = getattr(self.args, "stage2_ucl_gamma_denom_pan", "rendered")
        return self._select_ucl_pan_image(source, pan_rendered_s1)

    def _scale_param_to_render_scale(self, scale, upsample_ratio=1.0):
        render_scale = scale
        if not self.disable_inverse_scale:
            render_scale = 1.0 / render_scale.clamp_min(1.0e-8)
        if upsample_ratio is not None:
            render_scale = float(upsample_ratio) * render_scale
        return render_scale

    def _expand_ucl_ratio_features(self, ratio_feat):
        mode = getattr(self, "_ucl_ratio_mode", "full")
        if mode == "shared":
            return ratio_feat * self.ratio_band_gain.view(1, -1)
        if mode == "lowrank":
            return ratio_feat @ self.ratio_band_mix
        return ratio_feat

    def _expand_ucl_ratio_image(self, ratio_raw):
        mode = getattr(self, "_ucl_ratio_mode", "full")
        if mode == "shared":
            return ratio_raw * self.ratio_band_gain.view(-1, 1, 1)
        if mode == "lowrank":
            return torch.einsum("khw,kc->chw", ratio_raw, self.ratio_band_mix)
        return ratio_raw

    def _render_ucl_additive_base(self, scale, pan_h, pan_w, tb_hr):
        feat_ms_lr = getattr(self, "feat_ms_lr", None)
        if feat_ms_lr is None:
            return None
        base_hr, _ = self._rasterize_features(
            self.xy.detach(), scale, self.rot.detach(),
            feat_ms_lr, pan_h, pan_w, tb_hr,
        )
        return base_hr

    def _rasterize_ucl_ratio_features(
        self, xy, scale, rot, feat, img_h, img_w, tile_bounds
    ):
        return self._rasterize_features(
            xy, scale, rot, feat, img_h, img_w, tile_bounds,
            topk_norm_override=self._stage2_ucl_render_topk_norm_override(),
        )

    def _stage2_ucl_render_topk_norm_override(self):
        """Return the Stage2 UCL rasterization normalization override.

        None keeps the global renderer default. True uses per-pixel TOP_K
        normalized averaging. False uses raw Gaussian summation. The default
        mode preserves the historical baseline.
        """
        mode = str(getattr(self.args, "stage2_ucl_render_norm_mode", "topk_norm")).lower()
        if mode in ("default", "global", "inherit"):
            return None
        if mode in ("topk", "top_k", "topk_norm", "top_k_norm", "normalized"):
            return True
        if mode in ("sum", "raw_sum", "accum", "accum_sum", "gaussianimage"):
            return False
        raise ValueError(f"Unsupported stage2_ucl_render_norm_mode: {mode}")

    def _render_ucl_product(
        self,
        xy,
        scale,
        rot,
        ratio_feat,
        pan_h,
        pan_w,
        tb_hr,
        pan_rendered_s1,
        pan_feat=None,
    ):
        """Render UCL ratio/product variants.

        Returns (ratio_image, product_image). The caller decides whether to add
        an additive base image and when to clamp the final HRMS image.
        """
        ratio_hr = self._render_ratio_poly_image(
            xy, scale, rot, ratio_feat, pan_h, pan_w, tb_hr
        )
        external = ratio_hr * pan_rendered_s1

        mode = self._stage2_ucl_product_mode()
        if mode in ("external", "pixel", "pixel_product", "render_then_multiply"):
            return ratio_hr, external

        ratio_feat_expanded = self._expand_ucl_ratio_features(ratio_feat)
        if mode in ("precomp", "precompensated", "pan_precomp", "precomp_internal"):
            pan_gauss_feat = self._sample_features_from_image(
                pan_rendered_s1, xy.detach()
            )
        elif pan_feat is not None:
            pan_gauss_feat = pan_feat.detach()
        else:
            pan_gauss_feat = self._sample_features_from_image(
                self.ref_images, xy.detach()
            )

        internal, _ = self._rasterize_features(
            xy, scale, rot, ratio_feat_expanded * pan_gauss_feat,
            pan_h, pan_w, tb_hr,
            topk_norm_override=self._stage2_ucl_render_topk_norm_override(),
        )

        if mode in ("internal", "render_product", "inside", "naive_internal"):
            return ratio_hr, internal
        if mode in ("precomp", "precompensated", "pan_precomp", "precomp_internal"):
            return ratio_hr, internal
        if mode in ("hfbypass", "hf_bypass", "highfreq_bypass", "highpass_bypass"):
            sigma = float(getattr(self.args, "stage2_ucl_hf_sigma", 2.0))
            alpha = float(getattr(self.args, "stage2_ucl_hf_alpha", 1.0))
            pan_high = pan_rendered_s1 - self._gaussian_lowpass_sigma(
                pan_rendered_s1, sigma
            )
            return ratio_hr, internal + alpha * ratio_hr * pan_high
        if mode in ("alpha", "blend", "alpha_blend", "diagnostic_blend"):
            alpha = float(getattr(self.args, "stage2_ucl_alpha", 0.5))
            alpha = max(0.0, min(1.0, alpha))
            return ratio_hr, (1.0 - alpha) * internal + alpha * external

        raise ValueError(f"Unsupported stage2_ucl_product_mode: {mode}")

    def _stage2_ratio_split_enabled(self):
        mode = str(getattr(self.args, "stage2_ratio_split_mode", "off")).lower()
        return mode in ("low_high", "base_new", "split", "base_plus_hf")

    def _stage2_ratio_cluster_enabled(self):
        mode = str(getattr(self.args, "stage2_ratio_cluster_mode", "off")).lower()
        return mode in ("tile_lr_ratio", "tile_ratio", "local_kmeans", "kmeans")

    def _stage2_ratio_band_mask_enabled(self):
        return bool(getattr(self.args, "stage2_ratio_band_mask", False))

    def _stage2_ratio_band_mask_scale(self, feat):
        if (
            not self._stage2_ratio_band_mask_enabled()
            or self.feat_ratio_band_mask is None
            or feat is None
            or int(self.feat_ratio_band_mask.shape[0]) != int(feat.shape[0])
            or int(self.feat_ratio_band_mask.shape[1]) != int(feat.shape[1])
        ):
            return None
        mode = str(getattr(self.args, "stage2_ratio_band_mask_mode", "sigmoid2")).lower()
        raw = self.feat_ratio_band_mask.to(dtype=feat.dtype, device=feat.device)
        if mode in ("sigmoid2", "sigmoid_x2", "sigmoid"):
            # Identity at raw=0, bounded in (0, 2). This lets a Gaussian
            # become band-wise weak/strong without changing the default output.
            return 2.0 * torch.sigmoid(raw)
        if mode in ("tanh", "signed"):
            gain = float(getattr(self.args, "stage2_ratio_band_mask_gain", 1.0))
            return 1.0 + gain * torch.tanh(raw)
        if mode in ("softplus", "positive"):
            return F.softplus(raw) / math.log(2.0)
        raise ValueError(f"Unsupported stage2_ratio_band_mask_mode: {mode}")

    def _stage2_apply_ratio_band_mask(self, feat):
        scale = self._stage2_ratio_band_mask_scale(feat)
        if scale is None:
            return feat
        return feat * scale

    def _stage2_init_ratio_band_mask(self, n, channels):
        self.feat_ratio_band_mask = None
        if not self._stage2_ratio_band_mask_enabled():
            return
        self.feat_ratio_band_mask = nn.Parameter(
            torch.zeros(n, channels, dtype=self.dtype, device=self.device)
        )
        self.worklog.info(
            f">>> Stage2 ratio band mask active "
            f"(N={n}, C={channels}, "
            f"mode={getattr(self.args, 'stage2_ratio_band_mask_mode', 'sigmoid2')}, "
            f"lr_mult={float(getattr(self.args, 'stage2_ratio_band_mask_lr_mult', 1.0)):.3g})"
        )

    def _stage2_ratio_band_mask_param_group(self, feat_lr):
        if (
            not self._stage2_ratio_band_mask_enabled()
            or self.feat_ratio_band_mask is None
        ):
            return None
        return {
            "params": self.feat_ratio_band_mask,
            "lr": feat_lr * float(getattr(self.args, "stage2_ratio_band_mask_lr_mult", 1.0)),
        }

    def _stage2_ratio_band_mask_regularizer_loss(self):
        weight = float(getattr(self.args, "stage2_ratio_band_mask_l2_weight", 0.0))
        if (
            weight <= 0.0
            or not self._stage2_ratio_band_mask_enabled()
            or self.feat_ratio_band_mask is None
        ):
            device = self.feat_ratio.device if self.feat_ratio is not None else self.device
            return torch.tensor(0.0, dtype=self.dtype, device=device)
        return weight * self.feat_ratio_band_mask.pow(2).mean()

    def _stage2_ratio_band_mask_stats(self):
        if (
            not self._stage2_ratio_band_mask_enabled()
            or self.feat_ratio_band_mask is None
            or self.feat_ratio is None
        ):
            return None
        with torch.no_grad():
            scale = self._stage2_ratio_band_mask_scale(self.feat_ratio.detach())
            if scale is None:
                return None
            return (
                float(scale.min().item()),
                float(scale.max().item()),
                float(scale.mean().item()),
            )

    def _stage2_pan_descriptor_binding_enabled(self):
        return bool(getattr(self.args, "stage2_pan_descriptor_binding", False))

    def _stage2_pan_descriptor_tokens(self):
        spec = str(
            getattr(
                self.args,
                "stage2_pan_descriptor_features",
                (
                    "bias,feat_pan,feat_pan_centered,neighbor_mean,"
                    "neighbor_diff,log_sx,log_sy,anisotropy,cos_rot,"
                    "sin_rot,density"
                ),
            )
        )
        return [tok.strip() for tok in spec.split(",") if tok.strip()]

    def _stage2_pan_descriptor_knn_stats(self, xy, feat_pan):
        n = int(xy.shape[0])
        if n < 2:
            z = feat_pan.new_zeros(n, 1)
            return z, z, z
        k = int(getattr(self.args, "stage2_pan_descriptor_k", 8))
        k = max(1, min(k, n - 1))
        h, w = self.ref_images.shape[-2:]
        xy_px = xy.detach() * xy.new_tensor([float(w), float(h)])
        chunk = int(getattr(self.args, "stage2_pan_descriptor_knn_chunk", 2048))
        chunk = max(256, chunk)
        idxs = []
        dists = []
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            d = torch.cdist(xy_px[start:end], xy_px)
            row = torch.arange(end - start, device=d.device)
            col = torch.arange(start, end, device=d.device)
            d[row, col] = float("inf")
            kd, ki = d.topk(k, dim=1, largest=False)
            idxs.append(ki)
            dists.append(kd)
        idx = torch.cat(idxs, dim=0)
        dist = torch.cat(dists, dim=0)
        nb = feat_pan[idx]
        nb_mean = nb.mean(dim=1)
        nb_diff = (feat_pan - nb_mean).abs()
        mean_dist = dist.mean(dim=1, keepdim=True) / float(max(h, w))
        density = 1.0 / mean_dist.clamp_min(1.0e-6)
        return nb_mean, nb_diff, density

    def _stage2_build_pan_descriptor(self):
        if not self._stage2_pan_descriptor_binding_enabled():
            self.stage2_pan_descriptor = None
            self.stage2_pan_descriptor_names = None
            return None
        xy = self.xy.detach()
        feat_pan = self.feat_pan.detach()
        if feat_pan.dim() == 1:
            feat_pan = feat_pan[:, None]
        if feat_pan.shape[1] > 1:
            feat_pan_scalar = feat_pan.mean(dim=1, keepdim=True)
        else:
            feat_pan_scalar = feat_pan
        scale = self._scale_param_to_render_scale(
            self.scale.detach(), 1.0
        ).clamp_min(1.0e-6)
        log_scale = torch.log(scale)
        rot = self.rot.detach()
        if rot.dim() == 1:
            rot = rot[:, None]
        nb_mean = nb_diff = density = None
        cols = []
        names = []
        for tok in self._stage2_pan_descriptor_tokens():
            if tok == "bias":
                cols.append(torch.ones_like(feat_pan_scalar))
                names.append(tok)
            elif tok == "feat_pan":
                cols.append(feat_pan_scalar)
                names.append(tok)
            elif tok in ("feat_pan_centered", "pan_centered"):
                cols.append(feat_pan_scalar - feat_pan_scalar.mean(dim=0, keepdim=True))
                names.append(tok)
            elif tok in ("neighbor_mean", "pan_neighbor_mean"):
                if nb_mean is None:
                    nb_mean, nb_diff, density = self._stage2_pan_descriptor_knn_stats(
                        xy, feat_pan_scalar
                    )
                cols.append(nb_mean)
                names.append(tok)
            elif tok in ("neighbor_diff", "pan_neighbor_diff", "local_contrast"):
                if nb_diff is None:
                    nb_mean, nb_diff, density = self._stage2_pan_descriptor_knn_stats(
                        xy, feat_pan_scalar
                    )
                cols.append(nb_diff)
                names.append(tok)
            elif tok == "density":
                if density is None:
                    nb_mean, nb_diff, density = self._stage2_pan_descriptor_knn_stats(
                        xy, feat_pan_scalar
                    )
                cols.append(density)
                names.append(tok)
            elif tok == "log_sx":
                cols.append(log_scale[:, 0:1])
                names.append(tok)
            elif tok == "log_sy":
                cols.append(log_scale[:, 1:2])
                names.append(tok)
            elif tok == "anisotropy":
                cols.append(log_scale[:, 0:1] - log_scale[:, 1:2])
                names.append(tok)
            elif tok in ("scale_area", "log_area"):
                cols.append(log_scale.sum(dim=1, keepdim=True))
                names.append(tok)
            elif tok in ("cos_rot", "cos"):
                cols.append(torch.cos(rot[:, 0:1]))
                names.append(tok)
            elif tok in ("sin_rot", "sin"):
                cols.append(torch.sin(rot[:, 0:1]))
                names.append(tok)
            elif tok == "x":
                cols.append(xy[:, 0:1])
                names.append(tok)
            elif tok == "y":
                cols.append(xy[:, 1:2])
                names.append(tok)
            else:
                raise ValueError(f"Unsupported stage2 PAN descriptor token: {tok}")
        if not cols:
            self.stage2_pan_descriptor = None
            self.stage2_pan_descriptor_names = None
            return None
        desc = torch.cat(cols, dim=1).to(dtype=self.dtype, device=self.device)
        norm = str(getattr(self.args, "stage2_pan_descriptor_norm", "standard")).lower()
        if norm in ("standard", "zscore", "z_score"):
            for j, name in enumerate(names):
                if name == "bias":
                    continue
                col = desc[:, j:j + 1]
                desc[:, j:j + 1] = (col - col.mean()) / col.std(unbiased=False).clamp_min(1.0e-6)
        elif norm not in ("none", "off", "false"):
            raise ValueError(f"Unsupported stage2_pan_descriptor_norm: {norm}")
        clamp = float(getattr(self.args, "stage2_pan_descriptor_clamp", 5.0))
        if clamp > 0:
            for j, name in enumerate(names):
                if name != "bias":
                    desc[:, j].clamp_(-clamp, clamp)
        self.stage2_pan_descriptor = desc.detach()
        self.stage2_pan_descriptor_names = names
        return self.stage2_pan_descriptor

    def _stage2_init_pan_descriptor_binding(self):
        self.stage2_pan_descriptor_weight = None
        desc = self._stage2_build_pan_descriptor()
        if desc is None or self.feat_ratio is None:
            return
        feature_dim = int(self.feat_ratio.shape[1])
        init_std = float(getattr(self.args, "stage2_pan_descriptor_init_std", 0.0))
        if init_std > 0:
            init = init_std * torch.randn(
                desc.shape[1], feature_dim, dtype=self.dtype, device=self.device
            )
        else:
            init = torch.zeros(
                desc.shape[1], feature_dim, dtype=self.dtype, device=self.device
            )
        self.stage2_pan_descriptor_weight = nn.Parameter(init)
        self.worklog.info(
            ">>> Stage2 PAN descriptor binding active: "
            f"D={desc.shape[1]}, feat_dim={feature_dim}, "
            f"amp={float(getattr(self.args, 'stage2_pan_descriptor_amp', 0.1)):.3g}, "
            f"features={','.join(self.stage2_pan_descriptor_names or [])}"
        )

    def _stage2_pan_descriptor_param_group(self, feat_lr):
        if (
            not self._stage2_pan_descriptor_binding_enabled()
            or self.stage2_pan_descriptor_weight is None
        ):
            return None
        return {
            "params": [self.stage2_pan_descriptor_weight],
            "lr": feat_lr * float(getattr(self.args, "stage2_pan_descriptor_lr_mult", 1.0)),
            "name": "stage2_pan_descriptor_weight",
        }

    def _stage2_apply_pan_descriptor_binding(self, feat):
        if (
            not self._stage2_pan_descriptor_binding_enabled()
            or feat is None
            or self.stage2_pan_descriptor is None
            or self.stage2_pan_descriptor_weight is None
        ):
            self._stage2_last_pan_descriptor_delta = None
            return feat
        desc = self.stage2_pan_descriptor.to(dtype=feat.dtype, device=feat.device)
        n_feat = int(feat.shape[0])
        if int(desc.shape[0]) < n_feat:
            pad = feat.new_zeros(n_feat - int(desc.shape[0]), int(desc.shape[1]))
            desc = torch.cat([desc, pad], dim=0)
        elif int(desc.shape[0]) > n_feat:
            desc = desc[:n_feat]
        raw = desc @ self.stage2_pan_descriptor_weight.to(
            dtype=feat.dtype, device=feat.device
        )
        amp = float(getattr(self.args, "stage2_pan_descriptor_amp", 0.1))
        mode = str(getattr(self.args, "stage2_pan_descriptor_delta_mode", "tanh")).lower()
        if mode in ("tanh", "bounded"):
            delta = amp * torch.tanh(raw)
        elif mode in ("linear", "raw"):
            delta = amp * raw
        else:
            raise ValueError(f"Unsupported stage2_pan_descriptor_delta_mode: {mode}")
        self._stage2_last_pan_descriptor_delta = delta
        return feat + delta

    def _stage2_pan_descriptor_regularizer_loss(self):
        if (
            not self._stage2_pan_descriptor_binding_enabled()
            or self.stage2_pan_descriptor_weight is None
        ):
            device = self.feat_ratio.device if self.feat_ratio is not None else self.device
            return torch.tensor(0.0, dtype=self.dtype, device=device)
        loss = self.stage2_pan_descriptor_weight.new_tensor(0.0)
        w_l2 = float(getattr(self.args, "stage2_pan_descriptor_weight_l2", 0.0))
        if w_l2 > 0:
            loss = loss + w_l2 * self.stage2_pan_descriptor_weight.pow(2).mean()
        delta = getattr(self, "_stage2_last_pan_descriptor_delta", None)
        d_l1 = float(getattr(self.args, "stage2_pan_descriptor_delta_l1", 0.0))
        if d_l1 > 0 and delta is not None:
            loss = loss + d_l1 * delta.abs().mean()
        return loss

    def _stage2_pan_descriptor_stats(self):
        delta = getattr(self, "_stage2_last_pan_descriptor_delta", None)
        if (
            not self._stage2_pan_descriptor_binding_enabled()
            or self.stage2_pan_descriptor_weight is None
            or delta is None
        ):
            return None
        with torch.no_grad():
            w = self.stage2_pan_descriptor_weight
            return {
                "delta_abs_mean": float(delta.abs().mean().item()),
                "delta_min": float(delta.min().item()),
                "delta_max": float(delta.max().item()),
                "weight_abs_mean": float(w.abs().mean().item()),
                "weight_max_abs": float(w.abs().max().item()),
            }

    def _stage2_effective_ratio_features(self):
        if (
            self._stage2_ratio_cluster_enabled()
            and self.stage2_ratio_cluster_scalar is not None
            and self.stage2_ratio_cluster_gain is not None
            and self.stage2_ratio_cluster_assign is not None
        ):
            base_n = int(getattr(self, "_stage2_base_gaussian_count", self.num_gaussians))
            base_n = max(0, min(base_n, int(self.feat_ratio.shape[0])))
            new_n = int(self.feat_ratio.shape[0]) - base_n
            if new_n > 0 and int(self.stage2_ratio_cluster_scalar.shape[0]) == new_n:
                assign = self.stage2_ratio_cluster_assign.to(
                    device=self.feat_ratio.device, dtype=torch.long
                )
                new_feat = (
                    self.stage2_ratio_cluster_scalar
                    * self.stage2_ratio_cluster_gain[assign]
                )
                feat = torch.cat([self.feat_ratio[:base_n], new_feat], dim=0)
            else:
                feat = self.feat_ratio
        else:
            feat = self.feat_ratio
        feat = self._stage2_apply_pan_descriptor_binding(feat)
        return self._stage2_apply_ratio_band_mask(feat)

    def _stage2_ratio_poly_enabled(self):
        return int(getattr(self.args, "stage2_ratio_poly_order", 0)) >= 1

    def _expand_ratio_poly_features(self, feat):
        return self._expand_ucl_ratio_features(feat)

    def _render_ratio_poly_image(
        self,
        xy,
        scale,
        rot,
        feat0,
        img_h,
        img_w,
        tile_bounds,
    ):
        if (
            not self._stage2_ratio_poly_enabled()
            or self.feat_ratio_dx is None
            or self.feat_ratio_dy is None
            or feat0.shape[0] != int(self.feat_ratio_dx.shape[0])
        ):
            ratio0_raw, _ = self._rasterize_ucl_ratio_features(
                xy, scale, rot, feat0, img_h, img_w, tile_bounds
            )
            ratio_hr = self._expand_ucl_ratio_image(ratio0_raw)
            return ratio_hr

        feat0_exp = self._expand_ratio_poly_features(feat0)
        feat_dx = self._expand_ratio_poly_features(self.feat_ratio_dx)
        feat_dy = self._expand_ratio_poly_features(self.feat_ratio_dy)
        intercept = (
            feat0_exp
            - feat_dx * xy[:, 0:1].detach()
            - feat_dy * xy[:, 1:2].detach()
        )
        joint_feat = torch.cat([intercept, feat_dx, feat_dy], dim=1)
        joint_img, _ = self._rasterize_features(
            xy,
            scale,
            rot,
            joint_feat,
            img_h,
            img_w,
            tile_bounds,
            topk_norm_override=self._stage2_ucl_render_topk_norm_override(),
        )
        channels = int(feat0_exp.shape[1])
        intercept_img = joint_img[:channels]
        dx_img = joint_img[channels:2 * channels]
        dy_img = joint_img[2 * channels:3 * channels]
        x_grid = torch.linspace(
            0.0, 1.0, img_w, dtype=joint_img.dtype, device=joint_img.device
        ).view(1, 1, img_w)
        y_grid = torch.linspace(
            0.0, 1.0, img_h, dtype=joint_img.dtype, device=joint_img.device
        ).view(1, img_h, 1)
        poly_scale = float(getattr(self.args, "stage2_ratio_poly_scale", 1.0))
        return intercept_img + poly_scale * (x_grid * dx_img + y_grid * dy_img)

    def _stage2_ratio_poly_regularizer_loss(self):
        weight = float(getattr(self.args, "stage2_ratio_poly_l2_weight", 0.0))
        if (
            weight <= 0
            or not self._stage2_ratio_poly_enabled()
            or self.feat_ratio_dx is None
            or self.feat_ratio_dy is None
        ):
            device = self.feat_ratio.device if self.feat_ratio is not None else self.device
            return torch.tensor(0.0, dtype=self.dtype, device=device)
        return weight * (
            self.feat_ratio_dx.pow(2).mean() + self.feat_ratio_dy.pow(2).mean()
        )

    def _stage2_ratio_poly_sobel_init(self, ratio_ref, xy):
        channels = int(ratio_ref.shape[0])
        ratio_ref_4d = ratio_ref.unsqueeze(0)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=ratio_ref.dtype,
            device=ratio_ref.device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = F.conv2d(
            F.pad(ratio_ref_4d, (1, 1, 1, 1), mode="reflect"),
            sobel_x.expand(channels, 1, 3, 3),
            groups=channels,
        ).squeeze(0)
        grad_y = F.conv2d(
            F.pad(ratio_ref_4d, (1, 1, 1, 1), mode="reflect"),
            sobel_y.expand(channels, 1, 3, 3),
            groups=channels,
        ).squeeze(0)
        scale = float(getattr(self.args, "stage2_ratio_poly_grad_scale", 0.01))
        init_dx = self._sample_features_from_image(grad_x, xy) * scale
        init_dy = self._sample_features_from_image(grad_y, xy) * scale
        return init_dx.to(self.dtype), init_dy.to(self.dtype)

    def _stage2_kmeans_small(self, feats, k, iters):
        n = int(feats.shape[0])
        if k <= 1 or n <= 1:
            return torch.zeros(n, dtype=torch.long, device=feats.device), feats.mean(dim=0, keepdim=True)
        k = min(int(k), n)
        # Deterministic farthest-ish initialization from sorted feature order.
        order = torch.argsort(feats[:, 0])
        idx = order[torch.linspace(0, n - 1, k, device=feats.device).round().long()]
        centers = feats[idx].clone()
        assign = torch.zeros(n, dtype=torch.long, device=feats.device)
        for _ in range(max(1, int(iters))):
            dist = torch.cdist(feats, centers).pow(2)
            assign = dist.argmin(dim=1)
            new_centers = []
            for ci in range(k):
                mask = assign == ci
                if bool(mask.any().item()):
                    new_centers.append(feats[mask].mean(dim=0))
                else:
                    new_centers.append(centers[ci])
            centers = torch.stack(new_centers, dim=0)
        return assign, centers

    def _stage2_lr_ratio_image(self, product_pan, out_size):
        eps = float(getattr(self.args, "stage2_ratio_cluster_eps", 1.0e-6))
        lr_h, lr_w = self.gt_images.shape[-2:]
        pan_lr = F.interpolate(
            product_pan.detach().unsqueeze(0),
            size=(lr_h, lr_w),
            mode="area",
        ).squeeze(0).clamp_min(eps)
        ratio_lr = self.gt_images.detach().to(
            dtype=product_pan.dtype, device=product_pan.device
        ) / pan_lr
        clamp_val = float(getattr(self.args, "stage2_ratio_cluster_ratio_clamp", 5.0))
        if clamp_val > 0:
            ratio_lr = ratio_lr.clamp(0.0, clamp_val)
        return F.interpolate(
            ratio_lr.unsqueeze(0),
            size=out_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    def _append_stage2_ratio_clusters(self, new_xy, new_fr, product_pan, pan_h, pan_w):
        if not self._stage2_ratio_cluster_enabled():
            return
        with torch.no_grad():
            tile = int(getattr(self.args, "stage2_ratio_cluster_tile", 32))
            tile = max(1, tile)
            target = int(getattr(self.args, "stage2_ratio_cluster_target_per_k", 64))
            target = max(1, target)
            max_k = int(getattr(self.args, "stage2_ratio_cluster_max_k", 4))
            max_k = max(1, max_k)
            iters = int(getattr(self.args, "stage2_ratio_cluster_iters", 8))
            ratio_img = self._stage2_lr_ratio_image(product_pan, (pan_h, pan_w))
            ratio_feat = self._sample_features_from_image(ratio_img, new_xy).detach()
            cols = (new_xy[:, 0].detach() * max(pan_w - 1, 1)).floor().long().clamp(0, pan_w - 1)
            rows = (new_xy[:, 1].detach() * max(pan_h - 1, 1)).floor().long().clamp(0, pan_h - 1)
            tile_x = cols // tile
            tile_y = rows // tile
            tile_id = tile_y * ((pan_w + tile - 1) // tile) + tile_x

            assignments = torch.empty(new_xy.shape[0], dtype=torch.long, device=self.device)
            gains = []
            offset = 0
            for tid in torch.unique(tile_id):
                mask = tile_id == tid
                idx = torch.nonzero(mask, as_tuple=False).flatten()
                count = int(idx.numel())
                k = min(max_k, max(1, math.ceil(count / target)))
                local_assign, centers = self._stage2_kmeans_small(ratio_feat[idx], k, iters)
                assignments[idx] = local_assign + offset
                gains.append(centers)
                offset += int(centers.shape[0])
            new_gain = torch.cat(gains, dim=0) if gains else new_fr.mean(dim=0, keepdim=True)
            new_scalar = torch.ones(new_xy.shape[0], 1, dtype=self.dtype, device=self.device)

            old_scalar = (
                self.stage2_ratio_cluster_scalar.detach()
                if self.stage2_ratio_cluster_scalar is not None
                else torch.empty(0, 1, dtype=self.dtype, device=self.device)
            )
            old_gain = (
                self.stage2_ratio_cluster_gain.detach()
                if self.stage2_ratio_cluster_gain is not None
                else torch.empty(0, new_fr.shape[1], dtype=self.dtype, device=self.device)
            )
            old_assign = (
                self.stage2_ratio_cluster_assign.detach()
                if self.stage2_ratio_cluster_assign is not None
                else torch.empty(0, dtype=torch.long, device=self.device)
            )
            assignments = assignments + int(old_gain.shape[0])
            self.stage2_ratio_cluster_scalar = nn.Parameter(torch.cat([old_scalar, new_scalar], dim=0))
            self.stage2_ratio_cluster_gain = nn.Parameter(
                torch.cat([old_gain, new_gain.to(dtype=self.dtype, device=self.device)], dim=0)
            )
            self.stage2_ratio_cluster_assign = torch.cat([old_assign, assignments], dim=0)
            self.worklog.info(
                "Stage2 ratio cluster append | "
                f"mode={getattr(self.args, 'stage2_ratio_cluster_mode', 'off')} "
                f"tile={tile} new_points={int(new_xy.shape[0])} "
                f"new_clusters={int(new_gain.shape[0])} "
                f"total_clusters={int(self.stage2_ratio_cluster_gain.shape[0])} "
                f"gain_mean={float(new_gain.mean().item()):.6f} "
                f"gain_std={float(new_gain.std().item()):.6f}"
            )

    def _render_stage2_ratio_split(
        self,
        sc_ratio,
        img_h,
        img_w,
        tile_bounds,
        product_pan,
    ):
        """Render base/new ratio fields separately for Stage2.

        Shapes:
            feat_ratio: [N, C]
            ratio_low, ratio_hf, ratio_hr: [C, H, W]
        """
        base_n = int(getattr(self, "_stage2_base_gaussian_count", self.num_gaussians))
        base_n = max(0, min(base_n, int(self.num_gaussians)))
        ratio_feat_eff = self._stage2_effective_ratio_features()
        total_n = int(ratio_feat_eff.shape[0])
        base_n = min(base_n, total_n)
        alpha = float(getattr(self.args, "stage2_ratio_hf_alpha", 0.1))

        if base_n > 0:
            low_raw, _ = self._rasterize_ucl_ratio_features(
                self.xy[:base_n].detach(),
                sc_ratio[:base_n],
                self.rot[:base_n].detach(),
                ratio_feat_eff[:base_n],
                img_h,
                img_w,
                tile_bounds,
            )
            ratio_low = self._expand_ucl_ratio_image(low_raw)
        else:
            ratio_low = torch.zeros(
                self.output_dim, img_h, img_w, dtype=self.dtype, device=self.device
            )

        if total_n > base_n:
            hf_raw, _ = self._rasterize_ucl_ratio_features(
                self.xy[base_n:total_n],
                sc_ratio[base_n:total_n],
                self.rot[base_n:total_n],
                ratio_feat_eff[base_n:total_n],
                img_h,
                img_w,
                tile_bounds,
            )
            ratio_hf = self._expand_ucl_ratio_image(hf_raw)
        else:
            ratio_hf = torch.zeros_like(ratio_low)

        ratio_hr = (ratio_low + alpha * ratio_hf).clamp_min(0.0)
        self._stage2_last_ratio_low = ratio_low
        self._stage2_last_ratio_hf = ratio_hf
        self._stage2_last_ratio_hf_alpha = alpha
        product_hr = self._stage2_ratio_split_product(ratio_hr, product_pan)
        return ratio_hr, product_hr

    def _stage2_ratio_split_product(self, ratio_hr, product_pan):
        if not self._stage2_ratio_split_enabled():
            return ratio_hr * product_pan
        mode = str(getattr(self.args, "stage2_ratio_split_pan_mode", "full")).lower()
        if mode in ("full", "pan_full", "legacy", "none", "off", ""):
            self._stage2_last_ratio_split_pan_stats = {"mode": "full"}
            return ratio_hr * product_pan
        if mode not in ("low_high", "lowhigh", "pan_low_high", "pan_lh"):
            raise ValueError(
                "stage2_ratio_split_pan_mode must be full or low_high"
            )

        ratio_low = getattr(self, "_stage2_last_ratio_low", None)
        ratio_hf = getattr(self, "_stage2_last_ratio_hf", None)
        if ratio_low is None or ratio_hf is None:
            self._stage2_last_ratio_split_pan_stats = {"mode": "fallback_full"}
            return ratio_hr * product_pan

        sigma = float(getattr(self.args, "stage2_ratio_split_pan_sigma", 2.0))
        pan_low = self._gaussian_lowpass_sigma(product_pan, sigma)
        pan_hf = product_pan - pan_low
        alpha = float(getattr(self, "_stage2_last_ratio_hf_alpha", 1.0))
        low_gain = float(getattr(self.args, "stage2_ratio_split_pan_low_gain", 1.0))
        hf_gain = float(getattr(self.args, "stage2_ratio_split_pan_hf_gain", 1.0))
        if bool(getattr(self.args, "stage2_ratio_split_component_nonnegative", True)):
            ratio_low = ratio_low.clamp_min(0.0)
            ratio_hf = ratio_hf.clamp_min(0.0)

        product_hr = low_gain * ratio_low * pan_low + hf_gain * alpha * ratio_hf * pan_hf
        self._stage2_last_ratio_split_pan_stats = {
            "mode": mode,
            "sigma": sigma,
            "low_gain": low_gain,
            "hf_gain": hf_gain,
            "pan_low_mean": float(pan_low.detach().mean().item()),
            "pan_hf_absmean": float(pan_hf.detach().abs().mean().item()),
            "ratio_low_mean": float(ratio_low.detach().mean().item()),
            "ratio_hf_mean": float(ratio_hf.detach().mean().item()),
        }
        return product_hr

    def _parse_stage2_local_ratio_basis(self):
        spec = str(
            getattr(self.args, "stage2_ucl_local_ratio_basis", "const,dx,dy,pan,grad")
        )
        names = [x.strip().lower() for x in spec.split(",") if x.strip()]
        if "const" not in names:
            names.insert(0, "const")
        allowed = {
            "const", "dx", "dy", "pan", "grad",
            "pan_low", "pan_high", "pan_high_norm", "local_mean", "local_std",
            "grad_x", "grad_y",
        }
        unknown = [name for name in names if name not in allowed]
        if unknown:
            raise ValueError(f"Unsupported local ratio basis: {unknown}")
        return names

    def _stage2_uses_spectral_basis(self):
        return int(getattr(self.args, "stage2_basis_rank", 0)) > 0

    def _make_stage2_spectral_basis(self, ratio_base=None):
        if self.stage2_spectral_basis_B is None:
            R = int(getattr(self.args, "stage2_basis_rank", 0))
            C = self.output_dim
            # Init B so that lifted_const_c = ratio_base[c] when coeff const
            # slot of rank-0 is 1.0 and other ranks are 0:
            #   B[0, c] = ratio_base[c]
            #   B[r>0, c] = small noise
            if ratio_base is None:
                ratio_base = torch.ones(C, dtype=self.dtype, device=self.device)
            ratio_base = ratio_base.to(dtype=self.dtype, device=self.device).view(-1)
            B = torch.zeros(R, C, dtype=self.dtype, device=self.device)
            B[0, :] = ratio_base
            if R > 1:
                B[1:, :] = 1.0e-3 * torch.randn(R - 1, C, dtype=self.dtype, device=self.device)
            mod = _Stage2SpectralBasisB(R=R, C=C, init="uniform")
            with torch.no_grad():
                mod.B.copy_(B)
            self.stage2_spectral_basis_B = mod.to(self.device).to(dtype=self.dtype)
        return self.stage2_spectral_basis_B

    def _make_stage2_local_ratio(self, ratio_base):
        basis = self._parse_stage2_local_ratio_basis()
        n_anchor = self.xy.shape[0]
        if self._stage2_uses_spectral_basis():
            R = int(getattr(self.args, "stage2_basis_rank", 0))
            # Build B with ratio_base in row 0 so lifted_const = ratio_base
            # when coeff const slot has rank-0 = 1, others = 0.
            self._make_stage2_spectral_basis(ratio_base=ratio_base)
            coeff = torch.zeros(
                n_anchor, R, len(basis),
                dtype=self.dtype, device=self.device,
            )
            const_idx = basis.index("const")
            coeff[:, 0, const_idx] = 1.0
        else:
            coeff = torch.zeros(
                n_anchor, self.output_dim, len(basis),
                dtype=self.dtype, device=self.device,
            )
            const_idx = basis.index("const")
            coeff[:, :, const_idx] = ratio_base.to(
                dtype=self.dtype, device=self.device
            ).view(1, -1)
        self.stage2_local_ratio_basis = basis
        self.stage2_local_ratio_coeff = nn.Parameter(coeff)
        self._build_stage2_local_ratio_graph()
        return self.stage2_local_ratio_coeff

    def _build_stage2_local_ratio_graph(self):
        k = int(getattr(self.args, "stage2_ucl_local_ratio_graph_k", 0))
        if k <= 0:
            self.stage2_local_ratio_edge_src = None
            self.stage2_local_ratio_edge_dst = None
            return
        from scipy.spatial import cKDTree
        xy_np = self.xy.detach().cpu().numpy()
        k = min(k, max(1, xy_np.shape[0] - 1))
        nbr = cKDTree(xy_np).query(xy_np, k=k + 1)[1][:, 1:]
        src = np.repeat(np.arange(xy_np.shape[0], dtype=np.int64), k)
        dst = nbr.reshape(-1).astype(np.int64)
        self.stage2_local_ratio_edge_src = torch.from_numpy(src).to(self.device)
        self.stage2_local_ratio_edge_dst = torch.from_numpy(dst).to(self.device)

    def _stage2_local_ratio_graph_loss(self):
        weight = float(getattr(self.args, "stage2_ucl_local_ratio_graph_weight", 0.0))
        if (
            weight <= 0
            or self.stage2_local_ratio_coeff is None
            or self.stage2_local_ratio_edge_src is None
        ):
            device = (
                self.stage2_local_ratio_coeff.device
                if self.stage2_local_ratio_coeff is not None else self.device
            )
            return torch.tensor(0.0, dtype=self.dtype, device=device)
        src = self.stage2_local_ratio_edge_src
        dst = self.stage2_local_ratio_edge_dst
        coeff = self.stage2_local_ratio_coeff
        basis = self.stage2_local_ratio_basis or self._parse_stage2_local_ratio_basis()
        spec = str(getattr(self.args, "stage2_ucl_local_ratio_graph_basis", ""))
        names = [item.strip().lower() for item in spec.split(",") if item.strip()]
        if names:
            idx = [basis.index(name) for name in names if name in basis]
            if not idx:
                return coeff.new_tensor(0.0)
            coeff = coeff[:, :, idx]
        diff = coeff[src] - coeff[dst]
        return weight * torch.pow(diff, 2).mean()

    def _stage2_feat_ratio_graph_mask(self):
        mode = str(getattr(self.args, "stage2_feat_ratio_graph_mask_mode", "off")).lower()
        if mode in ("off", "none", "false", "0", ""):
            return None
        if self.feat_ratio is None or self.xy is None:
            return None
        n = int(self.feat_ratio.shape[0])
        if mode in ("all", "full", "every"):
            return torch.ones(n, dtype=torch.bool, device=self.device)
        base = int(getattr(self, "_stage2_base_gaussian_count", n))
        if mode in ("new", "new_geo", "newgauss", "stage2_new"):
            mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            if base < n:
                mask[base:] = True
            return mask
        pan_mask = self._stage2_bright_mask_map(
            torch.pow(self.ref_images.detach().clamp(0.0, 1.0), 1.0 / self.gamma)
        )
        if pan_mask is None:
            return None
        bright = self._sample_features_from_image(pan_mask, self.xy.detach())
        bright = bright.squeeze(-1) > 0.5 if bright.dim() == 3 else bright > 0.5
        if mode in ("bright", "pan_bright", "bright_only"):
            return bright.to(device=self.device, dtype=torch.bool)
        if mode in ("bright_new", "new_bright", "bright_stage2"):
            idx_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            if base < n:
                idx_mask[base:] = True
            return bright.to(device=self.device, dtype=torch.bool) & idx_mask
        raise ValueError(
            "stage2_feat_ratio_graph_mask_mode must be off, all, new, bright, or bright_new"
        )

    def _build_stage2_feat_ratio_graph(self):
        k = int(getattr(self.args, "stage2_feat_ratio_graph_k", 0))
        if k <= 0 or self.feat_ratio is None:
            self.stage2_feat_ratio_edge_src = None
            self.stage2_feat_ratio_edge_dst = None
            return
        from scipy.spatial import cKDTree
        mask = self._stage2_feat_ratio_graph_mask()
        if mask is not None:
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() < 2:
                self.stage2_feat_ratio_edge_src = None
                self.stage2_feat_ratio_edge_dst = None
                return
            idx_np = idx.detach().cpu().numpy().astype(np.int64)
            xy_np = self.xy.detach().cpu().numpy()[idx_np]
        else:
            idx_np = np.arange(int(self.xy.shape[0]), dtype=np.int64)
            xy_np = self.xy.detach().cpu().numpy()
        k = min(k, max(1, xy_np.shape[0] - 1))
        nbr = cKDTree(xy_np).query(xy_np, k=k + 1)[1][:, 1:]
        src = np.repeat(idx_np, k)
        dst = idx_np[nbr.reshape(-1)]
        self.stage2_feat_ratio_edge_src = torch.from_numpy(src).to(self.device)
        self.stage2_feat_ratio_edge_dst = torch.from_numpy(dst).to(self.device)

    def _stage2_feat_ratio_graph_loss(self):
        weight = float(getattr(self.args, "stage2_feat_ratio_graph_weight", 0.0))
        if (
            weight <= 0
            or self.feat_ratio is None
            or self.stage2_feat_ratio_edge_src is None
            or self.stage2_feat_ratio_edge_dst is None
        ):
            device = self.feat_ratio.device if self.feat_ratio is not None else self.device
            return torch.tensor(0.0, dtype=self.dtype, device=device)
        src = self.stage2_feat_ratio_edge_src
        dst = self.stage2_feat_ratio_edge_dst
        diff = self.feat_ratio[src] - self.feat_ratio[dst]
        return weight * torch.pow(diff, 2).mean()

    def _stage2_local_ratio_basis_reg_loss(self):
        if self.stage2_local_ratio_coeff is None:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        l2_w = float(getattr(self.args, "stage2_ucl_local_ratio_l2_weight", 0.0))
        l1_w = float(getattr(self.args, "stage2_ucl_local_ratio_l1_weight", 0.0))
        if l2_w <= 0 and l1_w <= 0:
            return self.stage2_local_ratio_coeff.new_tensor(0.0)

        basis = self.stage2_local_ratio_basis or self._parse_stage2_local_ratio_basis()
        spec = str(getattr(self.args, "stage2_ucl_local_ratio_reg_basis", ""))
        names = [item.strip().lower() for item in spec.split(",") if item.strip()]
        if not names:
            idx = list(range(len(basis)))
        else:
            idx = [basis.index(name) for name in names if name in basis]
        if not idx:
            return self.stage2_local_ratio_coeff.new_tensor(0.0)

        coeff = self.stage2_local_ratio_coeff[:, :, idx]
        loss = coeff.new_tensor(0.0)
        if l2_w > 0:
            loss = loss + l2_w * coeff.pow(2).mean()
        if l1_w > 0:
            loss = loss + l1_w * coeff.abs().mean()
        return loss

    def _stage2_uses_pan_stat_residual(self):
        return bool(getattr(self.args, "stage2_ucl_pan_stat_residual", False))

    def _stage2_pan_stat_feature_tokens(self):
        spec = str(
            getattr(
                self.args,
                "stage2_ucl_pan_stat_features",
                "pan,grad,std_1.5,std_3.0",
            )
        )
        return [t.strip() for t in spec.split(",") if t.strip()]

    def _stage2_pan_stat_feature_maps(self, product_pan):
        """Returns a dict {token: (C,H,W) feature map} for current product_pan."""
        feats = {}
        pan = product_pan.detach()
        for tok in self._stage2_pan_stat_feature_tokens():
            if tok == "pan":
                feats[tok] = pan
            elif tok == "grad":
                feats[tok] = self._pan_grad_mag(pan).detach()
            elif tok.startswith("std_"):
                try:
                    sigma = float(tok[4:])
                except ValueError as exc:
                    raise ValueError(f"bad pan_stat feature spec: {tok}") from exc
                mean = self._gaussian_lowpass_sigma(pan, sigma)
                mean_sq = self._gaussian_lowpass_sigma(pan * pan, sigma)
                feats[tok] = (mean_sq - mean * mean).clamp_min(0.0).sqrt().detach()
            elif tok.startswith("low_"):
                sigma = float(tok[4:])
                feats[tok] = self._gaussian_lowpass_sigma(pan, sigma).detach()
            elif tok.startswith("high_"):
                sigma = float(tok[5:])
                feats[tok] = (pan - self._gaussian_lowpass_sigma(pan, sigma)).detach()
            else:
                raise ValueError(f"Unsupported pan_stat feature token: {tok}")
        return feats

    def _stage2_pan_stat_sample_at_centers(self, product_pan):
        """Bilinear-sample each pan-stat feature at every Gaussian center.
        Returns (N, sum_C) tensor where each column block corresponds to one
        feature token's channels."""
        fmaps = self._stage2_pan_stat_feature_maps(product_pan)
        if not fmaps:
            return self.xy.new_zeros(self.xy.shape[0], 0)
        # xy is (N, 2) in normalized [0,1] (x first, y second).
        grid = (self.xy.detach() * 2.0 - 1.0).reshape(1, 1, -1, 2)
        cols = []
        for tok in self._stage2_pan_stat_feature_tokens():
            fmap = fmaps[tok]  # (C,H,W)
            sampled = F.grid_sample(
                fmap.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (1, C, 1, N)
            cols.append(sampled.squeeze(0).squeeze(1).transpose(0, 1))  # (N, C)
        return torch.cat(cols, dim=1)  # (N, sum_C)

    def _make_stage2_pan_stat_mlp(self, n_feat):
        if self.stage2_pan_stat_mlp is None:
            hidden = int(getattr(self.args, "stage2_ucl_pan_stat_hidden", 32))
            n_layers = int(getattr(self.args, "stage2_ucl_pan_stat_layers", 2))
            self.stage2_pan_stat_mlp = _Stage2PanStatMLP(
                n_feat=n_feat,
                output_dim=self.output_dim,
                hidden=hidden,
                n_layers=n_layers,
            ).to(self.device)
            self.stage2_pan_stat_mlp = self.stage2_pan_stat_mlp.to(dtype=self.dtype)
        return self.stage2_pan_stat_mlp

    def _stage2_uses_inr_residual(self):
        return bool(getattr(self.args, "stage2_ucl_inr_residual", False))

    def _stage2_inr_aux_feature_tokens(self):
        spec = str(
            getattr(
                self.args,
                "stage2_ucl_inr_aux_features",
                "pan,grad,std_1.5,std_3.0",
            )
        )
        return [t.strip() for t in spec.split(",") if t.strip()]

    def _stage2_inr_assemble_input(self, pan_h, pan_w, product_pan):
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0.0, 1.0, pan_h, device=self.device, dtype=self.dtype),
            torch.linspace(0.0, 1.0, pan_w, device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        feats = [x_grid.unsqueeze(0), y_grid.unsqueeze(0)]
        n_freq = int(getattr(self.args, "stage2_ucl_inr_n_freq", 6))
        for k in range(n_freq):
            freq = (2.0 ** k) * 3.141592653589793
            feats.append(torch.sin(freq * x_grid).unsqueeze(0))
            feats.append(torch.cos(freq * x_grid).unsqueeze(0))
            feats.append(torch.sin(freq * y_grid).unsqueeze(0))
            feats.append(torch.cos(freq * y_grid).unsqueeze(0))
        pan = product_pan.detach()
        for tok in self._stage2_inr_aux_feature_tokens():
            if tok == "pan":
                feats.append(pan)
            elif tok == "grad":
                feats.append(self._pan_grad_mag(pan).detach())
            elif tok.startswith("std_"):
                sigma = float(tok[4:])
                mean = self._gaussian_lowpass_sigma(pan, sigma)
                mean_sq = self._gaussian_lowpass_sigma(pan * pan, sigma)
                feats.append(
                    (mean_sq - mean * mean).clamp_min(0.0).sqrt().detach()
                )
            elif tok.startswith("low_"):
                sigma = float(tok[4:])
                feats.append(self._gaussian_lowpass_sigma(pan, sigma).detach())
            elif tok.startswith("high_"):
                sigma = float(tok[5:])
                feats.append(
                    (pan - self._gaussian_lowpass_sigma(pan, sigma)).detach()
                )
            else:
                raise ValueError(f"Unsupported INR aux feature token: {tok}")
        return torch.cat(feats, dim=0).unsqueeze(0)  # (1, C_in, H, W)

    def _make_stage2_ratio_context_inr(self, n_aux):
        if self.stage2_ratio_context_inr is None:
            n_freq = int(getattr(self.args, "stage2_ucl_inr_n_freq", 6))
            hidden = int(getattr(self.args, "stage2_ucl_inr_hidden", 64))
            n_layers = int(getattr(self.args, "stage2_ucl_inr_layers", 3))
            self.stage2_ratio_context_inr = _Stage2RatioContextINR(
                n_aux=n_aux,
                output_dim=self.output_dim,
                n_freq=n_freq,
                hidden=hidden,
                n_layers=n_layers,
            ).to(self.device).to(dtype=self.dtype)
        return self.stage2_ratio_context_inr

    def _apply_stage2_inr_residual(self, ratio, pan_h, pan_w, product_pan):
        if not self._stage2_uses_inr_residual():
            return ratio
        inp = self._stage2_inr_assemble_input(pan_h, pan_w, product_pan)
        n_aux = len(self._stage2_inr_aux_feature_tokens())
        inr = self._make_stage2_ratio_context_inr(n_aux=n_aux)
        delta = inr(inp).squeeze(0)  # (output_dim, H, W)
        amp = float(getattr(self.args, "stage2_ucl_inr_amp", 0.05))
        return ratio + amp * torch.tanh(delta)

    def _apply_stage2_pan_stat_residual(
        self, ratio, xy, scale, rot, pan_h, pan_w, tb_hr, product_pan
    ):
        if not self._stage2_uses_pan_stat_residual():
            return ratio
        feats_g = self._stage2_pan_stat_sample_at_centers(product_pan)
        if feats_g.shape[1] == 0:
            return ratio
        mlp = self._make_stage2_pan_stat_mlp(n_feat=feats_g.shape[1])
        delta_c = mlp(feats_g)  # (N, output_dim)
        amp = float(getattr(self.args, "stage2_ucl_pan_stat_amp", 0.05))
        delta_c = amp * torch.tanh(delta_c)
        delta_ratio, _ = self._rasterize_ucl_ratio_features(
            xy, scale, rot, delta_c, pan_h, pan_w, tb_hr,
        )
        return ratio + delta_ratio

    def _stage2_uses_m2p(self):
        hf = float(getattr(self.args, "stage2_m2p_hf_weight", 0.0))
        master = float(getattr(self.args, "stage2_m2p_weight", 0.0))
        return (master > 0.0) or (hf > 0.0)

    def _make_stage2_m2p_weights(self):
        if self.stage2_m2p_weights is None:
            init_value = float(getattr(self.args, "stage2_m2p_alpha_init", 1.0))
            self.stage2_m2p_weights = _Stage2M2PWeights(
                n_channels=self.output_dim,
                init_value=init_value,
            ).to(self.device).to(dtype=self.dtype)
        return self.stage2_m2p_weights

    def _stage2_m2p_loss(self, ms_hr_pred, pan_hr):
        """Per-channel PAN HF match.

        L = Sigma_c || HP(R_c * PAN) - alpha_c * HP(PAN) ||^2

        ``ms_hr_pred`` is [C, H, W] = R * PAN (rendered HRMS, already the
        per-channel product). ``pan_hr`` is [1, H, W] or [H, W] (HR PAN).
        Returns scalar tensor; 0 if disabled.
        """
        if not self._stage2_uses_m2p():
            return ms_hr_pred.new_tensor(0.0)
        hf_w = float(getattr(self.args, "stage2_m2p_hf_weight", 0.0))
        master_w = float(getattr(self.args, "stage2_m2p_weight", 0.0))
        if master_w > 0.0 and hf_w == 0.0:
            hf_w = master_w
        if hf_w == 0.0:
            return ms_hr_pred.new_tensor(0.0)
        m2p_mod = self._make_stage2_m2p_weights()
        alpha = m2p_mod()  # [C]
        if pan_hr.dim() == 3:
            pan_target = pan_hr[0]
        else:
            pan_target = pan_hr
        pan_target = pan_target.detach()
        sigma = float(getattr(self.args, "stage2_m2p_hf_sigma", 1.5))
        # Single-channel PAN HF
        pan_lp = self._gaussian_lowpass_sigma(
            pan_target.unsqueeze(0).unsqueeze(0), sigma
        ).squeeze(0).squeeze(0)
        pan_hp = pan_target - pan_lp  # [H, W]
        # Per-channel HRMS HF: HP(ms_hr_pred) channel-wise via batched conv
        ms_lp = self._gaussian_lowpass_sigma(
            ms_hr_pred.unsqueeze(0), sigma
        ).squeeze(0)  # [C, H, W]
        ms_hp = ms_hr_pred - ms_lp  # [C, H, W]
        # target: alpha_c * HP(PAN) broadcast to [C, H, W]
        target_hp = alpha.view(-1, 1, 1) * pan_hp.unsqueeze(0)
        return hf_w * F.mse_loss(ms_hp, target_hp)

    def _stage2_uses_p2m(self):
        obs = float(getattr(self.args, "stage2_p2m_obs_weight", 0.0))
        cross = float(getattr(self.args, "stage2_p2m_weight", 0.0))
        hf = float(getattr(self.args, "stage2_p2m_hf_weight", 0.0))
        return (obs > 0.0) or (cross > 0.0) or (hf > 0.0)

    def _make_stage2_p2m_colors(self):
        if self.stage2_p2m_colors is None:
            init_std = float(getattr(self.args, "stage2_p2m_init_std", 1.0e-3))
            n_gauss = int(self.xy.shape[0])
            self.stage2_p2m_colors = _Stage2P2MGaussianColors(
                n_gauss=n_gauss,
                n_channels=self.output_dim,
                init_std=init_std,
            ).to(self.device).to(dtype=self.dtype)
        return self.stage2_p2m_colors

    def _render_stage2_p2m(self, pan_h, pan_w, tb_hr, render_scale_ratio=1.0):
        """Rasterize per-Gaussian P2M MS colors through main pipeline geometry.
        Returns [C, H, W]. Geometry params are detached so P2M gradients
        do not perturb Gaussian positions/scales."""
        p2m_mod = self._make_stage2_p2m_colors()
        feat = p2m_mod()  # [N, C]
        scale = self._get_scale(render_scale_ratio).detach()
        xy = self.xy.detach()
        rot = self.rot.detach()
        out, _ = self._rasterize_ucl_ratio_features(
            xy, scale, rot, feat, pan_h, pan_w, tb_hr,
        )
        return out  # [C, H, W]

    def _stage2_p2m_loss(self, p2m_out, main_hrms_pred):
        """P2M observation + cross-consistency.

        L = obs_w * ||MTF_GLP(p2m_out) - MS_LR||^2
          + cross_w * ||MTF_GLP(main_hrms_pred) - MTF_GLP(p2m_out)||^2

        ``p2m_out`` is [C, H, W] (rendered MS via PAN geometry).
        ``main_hrms_pred`` is [C, H, W] (UCL R*PAN main path).
        Returns scalar; 0 if disabled.
        """
        if not self._stage2_uses_p2m():
            return p2m_out.new_tensor(0.0)
        obs_w = float(getattr(self.args, "stage2_p2m_obs_weight", 0.0))
        cross_w = float(getattr(self.args, "stage2_p2m_weight", 0.0))
        hf_w = float(getattr(self.args, "stage2_p2m_hf_weight", 0.0))
        loss = p2m_out.new_tensor(0.0)
        if obs_w > 0.0 or cross_w > 0.0:
            p2m_lr = self._ucl_degrade_once(p2m_out)
        if obs_w > 0.0:
            loss = loss + obs_w * F.mse_loss(p2m_lr, self.gt_images)
        if cross_w > 0.0:
            main_lr = self._ucl_degrade_once(main_hrms_pred)
            loss = loss + cross_w * F.mse_loss(p2m_lr, main_lr)
        if hf_w > 0.0:
            sigma = float(getattr(self.args, "stage2_p2m_hf_sigma", 1.5))
            p2m_lp = self._gaussian_lowpass_sigma(
                p2m_out.unsqueeze(0), sigma
            ).squeeze(0)
            main_lp = self._gaussian_lowpass_sigma(
                main_hrms_pred.unsqueeze(0), sigma
            ).squeeze(0)
            p2m_hp = p2m_out - p2m_lp
            main_hp = main_hrms_pred - main_lp
            loss = loss + hf_w * F.mse_loss(p2m_hp, main_hp)
        return loss

    def _render_ucl_local_ratio(
        self, pan_h, pan_w, tb_hr, product_pan, render_scale_ratio=1.0
    ):
        if self.stage2_local_ratio_coeff is None:
            raise RuntimeError("stage2_local_ratio_coeff is not initialized")
        basis = self.stage2_local_ratio_basis or self._parse_stage2_local_ratio_basis()
        coeff = self.stage2_local_ratio_coeff
        if self._stage2_uses_spectral_basis():
            B = self._make_stage2_spectral_basis()()  # [R, C]
            coeff = torch.einsum("arn,rc->acn", coeff, B)
        scale = self._get_scale(render_scale_ratio).detach()
        xy = self.xy.detach()
        rot = self.rot.detach()
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0.0, 1.0, pan_h, device=self.device, dtype=self.dtype),
            torch.linspace(0.0, 1.0, pan_w, device=self.device, dtype=self.dtype),
            indexing="ij",
        )
        ratio = product_pan.new_zeros(self.output_dim, pan_h, pan_w)
        feature_maps = {}

        def _feature_map(name):
            if name in feature_maps:
                return feature_maps[name]
            if name == "pan":
                value = product_pan
            elif name in ("pan_low", "local_mean"):
                sigma = float(getattr(self.args, "stage2_ucl_local_pan_low_sigma", 2.0))
                value = self._gaussian_lowpass_sigma(product_pan, sigma)
            elif name == "pan_high":
                sigma = float(getattr(self.args, "stage2_ucl_local_pan_low_sigma", 2.0))
                value = product_pan - self._gaussian_lowpass_sigma(product_pan, sigma)
            elif name == "pan_high_norm":
                sigma = float(getattr(self.args, "stage2_ucl_local_pan_low_sigma", 2.0))
                high = product_pan - self._gaussian_lowpass_sigma(product_pan, sigma)
                denom = high.abs().amax().clamp_min(1.0e-6)
                value = (high / denom).clamp(-1.0, 1.0).detach()
            elif name == "local_std":
                sigma = float(getattr(self.args, "stage2_ucl_local_std_sigma", 2.0))
                mean = self._gaussian_lowpass_sigma(product_pan, sigma)
                mean_sq = self._gaussian_lowpass_sigma(product_pan * product_pan, sigma)
                value = (mean_sq - mean * mean).clamp_min(0.0).sqrt()
            elif name == "grad":
                value = self._pan_grad_mag(product_pan).detach()
            elif name in ("grad_x", "grad_y"):
                grad_x, grad_y = self._pan_grad_xy(product_pan)
                value = (grad_x if name == "grad_x" else grad_y).detach()
            else:
                raise ValueError(f"Unsupported local ratio feature map: {name}")
            feature_maps[name] = value
            return value

        for idx, name in enumerate(basis):
            c = coeff[:, :, idx]
            rendered, _ = self._rasterize_ucl_ratio_features(
                xy, scale, rot, c, pan_h, pan_w, tb_hr,
            )
            if name == "const":
                ratio = ratio + rendered
            elif name == "dx":
                mu = c * xy[:, 0:1]
                mu_rendered, _ = self._rasterize_ucl_ratio_features(
                    xy, scale, rot, mu, pan_h, pan_w, tb_hr,
                )
                ratio = ratio + x_grid.unsqueeze(0) * rendered - mu_rendered
            elif name == "dy":
                mu = c * xy[:, 1:2]
                mu_rendered, _ = self._rasterize_ucl_ratio_features(
                    xy, scale, rot, mu, pan_h, pan_w, tb_hr,
                )
                ratio = ratio + y_grid.unsqueeze(0) * rendered - mu_rendered
            else:
                ratio = ratio + _feature_map(name) * rendered
        ratio = self._apply_stage2_pan_stat_residual(
            ratio, xy, scale, rot, pan_h, pan_w, tb_hr, product_pan
        )
        ratio = self._apply_stage2_inr_residual(ratio, pan_h, pan_w, product_pan)
        ratio = ratio.clamp_min(0.0)
        return ratio, ratio * product_pan

    def _make_stage2_scaffold_decoder(self, ratio_base=None):
        if ratio_base is None:
            ratio_base = float(getattr(self.args, "stage2_scaffold_ratio_base", 1.0))
        use_base_ratio = bool(getattr(self.args, "stage2_scaffold_use_base_ratio", False))
        decoder_ratio_base = 0.0 if use_base_ratio else ratio_base
        decoder = _Stage2ScaffoldGaussianDecoder(
            self.output_dim,
            children=int(getattr(self.args, "stage2_scaffold_children", 4)),
            hidden=int(getattr(self.args, "stage2_scaffold_hidden", 64)),
            offset_scale=float(getattr(self.args, "stage2_scaffold_offset_scale", 0.015)),
            child_scale_mult=float(
                getattr(self.args, "stage2_scaffold_child_scale_mult", 1.0)
            ),
            ratio_base=decoder_ratio_base,
            ratio_amp=float(getattr(self.args, "stage2_scaffold_ratio_amp", 1.0)),
            latent_dim=int(getattr(self.args, "stage2_scaffold_latent_dim", 8)),
            residual_mode=use_base_ratio,
        ).to(self.device)
        self.stage2_scaffold_decoder = decoder
        n_anchor = self.xy.shape[0]
        latent_dim = int(getattr(self.args, "stage2_scaffold_latent_dim", 8))
        self.stage2_scaffold_latent = nn.Parameter(
            torch.zeros(n_anchor, max(0, latent_dim), dtype=self.dtype, device=self.device)
        )
        self.stage2_scaffold_ratio_bias = nn.Parameter(
            torch.zeros(n_anchor, self.output_dim, dtype=self.dtype, device=self.device)
        )
        if use_base_ratio:
            if isinstance(ratio_base, torch.Tensor):
                base = ratio_base.to(dtype=self.dtype, device=self.device).view(1, -1)
                base = base.expand(n_anchor, -1).contiguous()
            else:
                base = torch.full(
                    (n_anchor, self.output_dim),
                    float(ratio_base),
                    dtype=self.dtype,
                    device=self.device,
                )
            self.stage2_scaffold_base_ratio = nn.Parameter(base)
        return decoder

    def _stage2_scaffold_gaussians(self):
        if self.stage2_scaffold_decoder is None:
            self._make_stage2_scaffold_decoder()
        return self.stage2_scaffold_decoder(
            self.feat_pan.detach(),
            self.xy.detach(),
            self.rot.detach(),
            self.scale.detach(),
            self.stage2_scaffold_latent,
            self.stage2_scaffold_ratio_bias,
        )

    def _render_ucl_scaffold_mlp(
        self, pan_h, pan_w, tb_hr, product_pan=None, render_scale_ratio=1.0
    ):
        feat_ratio, feat_pan_new, xy_new, rot_new, scale_new = (
            self._stage2_scaffold_gaussians()
        )
        sc_new = self._scale_param_to_render_scale(scale_new, render_scale_ratio)
        pan_new, _ = self._rasterize_features(
            xy_new, sc_new, rot_new, feat_pan_new, pan_h, pan_w, tb_hr,
        )
        pan_new = pan_new.clamp(0.0, 1.0)
        if product_pan is None:
            product_pan = pan_new
        base_ratio = getattr(self, "stage2_scaffold_base_ratio", None)
        if base_ratio is not None:
            parent_scale = self._get_scale(render_scale_ratio).detach()
            base_ratio_hr, _ = self._rasterize_ucl_ratio_features(
                self.xy.detach(), parent_scale, self.rot.detach(),
                base_ratio, pan_h, pan_w, tb_hr,
            )
            residual_ratio_hr, _ = self._rasterize_ucl_ratio_features(
                xy_new, sc_new, rot_new, feat_ratio, pan_h, pan_w, tb_hr,
            )
            residual_scale = float(
                getattr(self.args, "stage2_scaffold_residual_scale", 0.1)
            )
            ratio_hr = (base_ratio_hr + residual_scale * residual_ratio_hr).clamp_min(0.0)
            product_hr = ratio_hr * product_pan
        else:
            ratio_hr, product_hr = self._render_ucl_product(
                xy_new, sc_new, rot_new, feat_ratio,
                pan_h, pan_w, tb_hr, product_pan, pan_feat=feat_pan_new,
            )
        return ratio_hr, product_hr, pan_new

    def _stage2_scaffold_pan_loss(self, pan_new):
        pan_w = float(getattr(self.args, "stage2_scaffold_pan_weight", 0.0))
        if pan_w <= 0:
            return pan_new.new_tensor(0.0)
        direct_w = float(getattr(self.args, "stage2_scaffold_pan_l1_weight", 1.0))
        loss = self._compute_pan_loss(pan_new, self.ref_images)
        if direct_w > 0:
            loss = loss + direct_w * F.l1_loss(pan_new, self.ref_images)
        return pan_w * loss

    def _make_stage2_gamma_net(self, channels):
        if self.stage2_gamma_net is None:
            hidden = int(getattr(self.args, "stage2_ucl_gamma_hidden", 32))
            self.stage2_gamma_net = _Stage2GammaNet(
                int(channels), hidden=hidden
            ).to(device=self.device, dtype=self.dtype)
        return self.stage2_gamma_net

    def _stage2_ucl_gamma_prior_loss(self, ratio_hr, gamma_input_pan, gamma_denom_pan):
        weight = float(getattr(self.args, "stage2_ucl_gamma_prior_weight", 0.0))
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        gamma_net = self._make_stage2_gamma_net(ratio_hr.shape[0])
        pan_in = gamma_input_pan.expand(ratio_hr.shape[0], -1, -1).unsqueeze(0)
        gamma_img = gamma_net(pan_in).squeeze(0)
        eps = float(getattr(self.args, "stage2_ucl_gamma_prior_eps", 1.0e-3))
        mode = str(getattr(self.args, "stage2_ucl_gamma_prior_mode", "ratio")).lower()
        if mode in ("direct", "modulation", "gain_pan", "gamma"):
            diff = gamma_img - ratio_hr
        else:
            gamma_ratio = gamma_img / gamma_denom_pan.clamp_min(eps)
            diff = gamma_ratio - ratio_hr
        tv = (
            torch.pow(diff[:, :, :-1] - diff[:, :, 1:], 2).mean()
            + torch.pow(diff[:, :-1, :] - diff[:, 1:, :], 2).mean()
        )
        return weight * tv

    def _stage2_ucl_fixed_pan_gain_tv_loss(self, ratio_hr, product_pan):
        weight = float(getattr(self.args, "stage2_ucl_fixed_pan_gain_tv_weight", 0.0))
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        weight = weight * float(self._stage2_ucl_tv_schedule_scale())
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        eps = float(getattr(self.args, "stage2_ucl_fixed_pan_gain_eps", 1.0e-6))
        lms_hr = self._bicubic_hr_target().to(
            dtype=ratio_hr.dtype, device=ratio_hr.device
        )
        if lms_hr.shape[-2:] != ratio_hr.shape[-2:]:
            lms_hr = F.interpolate(
                lms_hr.unsqueeze(0),
                size=ratio_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        pan = product_pan.to(dtype=ratio_hr.dtype, device=ratio_hr.device)
        if pan.shape[-2:] != ratio_hr.shape[-2:]:
            pan = F.interpolate(
                pan.unsqueeze(0),
                size=ratio_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        gain = lms_hr.mean(dim=(1, 2), keepdim=True) / pan.mean().clamp_min(eps)
        diff = gain * pan - ratio_hr
        return weight * self._stage2_ucl_plain_tv_loss(diff)

    def _load_stage2_ucl_teacher(self, target_hw):
        path = str(getattr(self.args, "stage2_ucl_teacher_path", "")).strip()
        if not path:
            return None
        if self.stage2_ucl_teacher is None:
            teacher, _, _, _ = load_images(
                load_path=path, downsample_ratio=None, gamma=self.gamma
            )
            teacher = torch.from_numpy(teacher).to(
                dtype=self.dtype, device=self.device
            )
            if teacher.shape[0] != self.output_dim:
                raise ValueError(
                    f"UCL teacher expects {self.output_dim} channels, "
                    f"got {teacher.shape[0]} from {path}"
                )
            self.stage2_ucl_teacher = teacher.clamp(0.0, 1.0)
        teacher = self.stage2_ucl_teacher
        if teacher.shape[-2:] != target_hw:
            teacher = F.interpolate(
                teacher.unsqueeze(0), size=target_hw, mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)
        return teacher

    def _stage2_ucl_teacher_loss(self, ratio_hr, product_hr, pan_rendered_s1):
        ratio_w = float(getattr(self.args, "stage2_ucl_teacher_ratio_weight", 0.0))
        product_w = float(getattr(self.args, "stage2_ucl_teacher_product_weight", 0.0))
        if ratio_w <= 0 and product_w <= 0:
            return ratio_hr.new_tensor(0.0)
        teacher = self._load_stage2_ucl_teacher(ratio_hr.shape[-2:])
        if teacher is None:
            return ratio_hr.new_tensor(0.0)
        loss = ratio_hr.new_tensor(0.0)
        if ratio_w > 0:
            denom_source = getattr(
                self.args, "stage2_ucl_teacher_ratio_denom_pan", "raw"
            )
            denom_pan = self._select_ucl_pan_image(denom_source, pan_rendered_s1)
            eps = float(getattr(self.args, "stage2_ucl_gamma_prior_eps", 1.0e-3))
            ratio_teacher = teacher / denom_pan.clamp_min(eps)
            ratio_clip = float(getattr(self.args, "stage2_ucl_teacher_ratio_clip", 10.0))
            if ratio_clip > 0:
                ratio_teacher = ratio_teacher.clamp(0.0, ratio_clip)
            loss = loss + ratio_w * F.mse_loss(ratio_hr, ratio_teacher)
        if product_w > 0:
            loss = loss + product_w * F.mse_loss(product_hr, teacher)
        return loss

    def _stage2_ucl_teacher_output_loss(self, ms_hr):
        full_w = float(
            getattr(self.args, "stage2_ucl_teacher_output_full_l1_weight", 0.0)
        )
        hf_w = float(
            getattr(self.args, "stage2_ucl_teacher_output_hf_l1_weight", 0.0)
        )
        if full_w <= 0.0 and hf_w <= 0.0:
            return ms_hr.new_tensor(0.0)

        teacher = self._load_stage2_ucl_teacher(ms_hr.shape[-2:])
        if teacher is None:
            return ms_hr.new_tensor(0.0)

        loss = ms_hr.new_tensor(0.0)
        if full_w > 0.0:
            loss = loss + full_w * F.l1_loss(ms_hr, teacher.detach())
        if hf_w > 0.0:
            sigma = float(
                getattr(self.args, "stage2_ucl_teacher_output_hf_sigma", 2.0)
            )
            pred_hp = ms_hr - self._gaussian_lowpass_sigma(ms_hr, sigma)
            teacher_hp = teacher - self._gaussian_lowpass_sigma(teacher, sigma)
            loss = loss + hf_w * F.l1_loss(pred_hp, teacher_hp.detach())
        return loss

    def _stage2_ucl_observed_ratio_loss(self, ratio_hr, product_pan):
        weight = float(getattr(self.args, "stage2_ucl_obs_ratio_weight", 0.0))
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        eps = float(getattr(self.args, "stage2_ucl_obs_ratio_eps", 1.0e-3))
        clip = float(getattr(self.args, "stage2_ucl_obs_ratio_clip", 4.0))
        mode = str(getattr(self.args, "stage2_ucl_obs_ratio_mode", "lr")).lower()
        loss_type = str(getattr(self.args, "stage2_ucl_obs_ratio_loss", "l1")).lower()

        pan_c = product_pan.expand(self.output_dim, -1, -1)
        pan_lr = self._ucl_degrade_once(pan_c).clamp_min(eps)
        target_lr = self.gt_images / pan_lr
        if clip > 0:
            target_lr = target_lr.clamp(0.0, clip)

        if mode in ("hr", "hr_bicubic", "bicubic", "upsample"):
            target = F.interpolate(
                target_lr.unsqueeze(0),
                size=ratio_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            if clip > 0:
                target = target.clamp(0.0, clip)
            pred = ratio_hr
        else:
            target = target_lr
            pred = self._ucl_degrade_once(ratio_hr)

        if loss_type in ("mse", "l2"):
            return weight * F.mse_loss(pred, target)
        return weight * F.l1_loss(pred, target)

    def _ucl_degrade_once(self, image):
        return self._ucl_down(self._ucl_blur(image.unsqueeze(0))).squeeze(0)

    def _make_stage2_ratio_inr(self, ratio_base):
        latent_dim = int(getattr(self.args, "stage2_ucl_inr_latent_dim", 8))
        hidden = int(getattr(self.args, "stage2_ucl_inr_hidden", 64))
        self.stage2_ratio_latent = nn.Parameter(
            torch.zeros(
                self.xy.shape[0],
                max(1, latent_dim),
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.stage2_ratio_inr = _Stage2RatioINR(
            max(1, latent_dim), self.output_dim, hidden=hidden
        ).to(device=self.device, dtype=self.dtype)
        self.stage2_ratio_base = ratio_base.to(
            dtype=self.dtype, device=self.device
        ).view(1, -1, 1, 1)
        return self.stage2_ratio_inr

    def _render_ucl_inr_ratio(
        self, img_h, img_w, tile_bounds, render_scale_ratio=1.0
    ):
        if self.stage2_ratio_inr is None or self.stage2_ratio_latent is None:
            raise RuntimeError("stage2_ratio_inr is not initialized")
        latent, _ = self._rasterize_features(
            self.xy.detach(),
            self._get_scale(render_scale_ratio).detach(),
            self.rot.detach(),
            self.stage2_ratio_latent,
            img_h,
            img_w,
            tile_bounds,
        )
        amp = float(getattr(self.args, "stage2_ucl_inr_ratio_amp", 1.0))
        delta = amp * torch.tanh(self.stage2_ratio_inr(latent))
        return (self.stage2_ratio_base.squeeze(0) + delta).clamp_min(0.0)

    def _make_stage2_pixel_residual(self, pan_h, pan_w):
        if self.stage2_pixel_residual is None:
            self.stage2_pixel_residual = nn.Parameter(
                torch.zeros(
                    self.output_dim,
                    int(pan_h),
                    int(pan_w),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return self.stage2_pixel_residual

    def _make_stage2_grid_residual(self, pan_h, pan_w):
        if self.stage2_grid_residual is None:
            down = max(1, int(getattr(self.args, "stage2_ucl_grid_down", 8)))
            gh = max(1, int(round(float(pan_h) / float(down))))
            gw = max(1, int(round(float(pan_w) / float(down))))
            self.stage2_grid_residual = nn.Parameter(
                torch.zeros(
                    self.output_dim,
                    gh,
                    gw,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return self.stage2_grid_residual

    def _ucl_pixel_residual_mask(self, product_pan):
        mode = str(getattr(self.args, "stage2_ucl_pixel_residual_mask", "grad")).lower()
        if mode in ("", "none", "off", "false"):
            return product_pan.new_ones(1, product_pan.shape[-2], product_pan.shape[-1])
        grad = self._pan_grad_mag(product_pan).detach()
        q = float(getattr(self.args, "stage2_ucl_pixel_residual_grad_quantile", 0.70))
        q = max(0.0, min(0.99, q))
        th = torch.quantile(grad.flatten(), q)
        soft = float(getattr(self.args, "stage2_ucl_pixel_residual_mask_softness", 0.05))
        if soft <= 0:
            return (grad >= th).to(dtype=product_pan.dtype)
        scale = (grad.std(unbiased=False) * soft).clamp_min(1.0e-6)
        return torch.sigmoid((grad - th) / scale)

    def _apply_ucl_dense_residuals(self, ratio_hr, product_pan):
        ratio = ratio_hr
        if self.stage2_pixel_residual is not None:
            residual = self.stage2_pixel_residual
            if residual.shape[-2:] != ratio.shape[-2:]:
                residual = F.interpolate(
                    residual.unsqueeze(0),
                    size=ratio.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            if bool(getattr(self.args, "stage2_ucl_pixel_residual_highpass", True)):
                sigma = float(getattr(self.args, "stage2_ucl_pixel_residual_sigma", 2.0))
                residual = residual - self._gaussian_lowpass_sigma(residual, sigma)
            amp = float(getattr(self.args, "stage2_ucl_pixel_residual_amp", 1.0))
            ratio = ratio + amp * self._ucl_pixel_residual_mask(product_pan) * residual

        if self.stage2_grid_residual is not None:
            grid = F.interpolate(
                self.stage2_grid_residual.unsqueeze(0),
                size=ratio.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            amp = float(getattr(self.args, "stage2_ucl_grid_residual_amp", 1.0))
            ratio = ratio + amp * grid

        return ratio.clamp_min(0.0)

    def _stage2_dense_residual_reg_loss(self):
        loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if self.stage2_pixel_residual is not None:
            res = self.stage2_pixel_residual
            l1_w = float(getattr(self.args, "stage2_ucl_pixel_residual_l1_weight", 0.0))
            l2_w = float(getattr(self.args, "stage2_ucl_pixel_residual_l2_weight", 0.0))
            tv_w = float(getattr(self.args, "stage2_ucl_pixel_residual_tv_weight", 0.0))
            if l1_w > 0:
                loss = loss + l1_w * res.abs().mean()
            if l2_w > 0:
                loss = loss + l2_w * res.pow(2).mean()
            if tv_w > 0:
                loss = loss + tv_w * self._tv_loss(res)
        if self.stage2_grid_residual is not None:
            grid = self.stage2_grid_residual
            l1_w = float(getattr(self.args, "stage2_ucl_grid_l1_weight", 0.0))
            l2_w = float(getattr(self.args, "stage2_ucl_grid_l2_weight", 0.0))
            tv_w = float(getattr(self.args, "stage2_ucl_grid_tv_weight", 0.0))
            if l1_w > 0:
                loss = loss + l1_w * grid.abs().mean()
            if l2_w > 0:
                loss = loss + l2_w * grid.pow(2).mean()
            if tv_w > 0:
                loss = loss + tv_w * self._tv_loss(grid)
        return loss

    def _make_stage2_pan_detail(self, ratio_base):
        if self.stage2_pan_detail_coeff is None:
            init_mode = str(
                getattr(self.args, "stage2_ucl_pan_detail_init", "zero")
            ).lower()
            coeff = torch.zeros(
                self.xy.shape[0],
                self.output_dim,
                dtype=self.dtype,
                device=self.device,
            )
            if init_mode in ("ratio_base", "base", "mean"):
                coeff.copy_(ratio_base.to(dtype=self.dtype, device=self.device).view(1, -1))
            self.stage2_pan_detail_coeff = nn.Parameter(coeff)
        return self.stage2_pan_detail_coeff

    def _render_ucl_pan_detail_gain(self, img_h, img_w, tile_bounds, render_scale_ratio):
        if self.stage2_pan_detail_coeff is None:
            return None
        gain, _ = self._rasterize_features(
            self.xy.detach(),
            self._get_scale(render_scale_ratio).detach(),
            self.rot.detach(),
            self.stage2_pan_detail_coeff,
            img_h,
            img_w,
            tile_bounds,
        )
        gain_clip = float(getattr(self.args, "stage2_ucl_pan_detail_gain_clip", 0.0))
        if gain_clip > 0:
            gain = gain.clamp(-gain_clip, gain_clip)
        gain_low_sigma = float(
            getattr(self.args, "stage2_ucl_pan_detail_gain_low_sigma", 0.0)
        )
        if gain_low_sigma > 0:
            gain = self._gaussian_lowpass_sigma(gain, gain_low_sigma)
        return gain

    def _pan_highpass_for_detail(self, product_pan):
        sigma = float(getattr(self.args, "stage2_ucl_pan_detail_sigma", 2.0))
        hp = product_pan - self._gaussian_lowpass_sigma(product_pan, sigma)
        if bool(getattr(self.args, "stage2_ucl_pan_detail_norm", False)):
            hp = hp / hp.abs().amax().clamp_min(1.0e-6)
        return hp.detach()

    def _apply_ucl_pan_detail(
        self, product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
    ):
        gain = self._render_ucl_pan_detail_gain(
            img_h, img_w, tile_bounds, render_scale_ratio
        )
        if gain is None:
            return product_hr
        amp = float(getattr(self.args, "stage2_ucl_pan_detail_amp", 1.0))
        pan_hp = self._pan_highpass_for_detail(product_pan)
        return product_hr + amp * gain * pan_hp

    def _raw_pan_at_size(self, img_h, img_w):
        raw_pan = self.ref_images
        if raw_pan.shape[-2:] == (img_h, img_w):
            return raw_pan
        return F.interpolate(
            raw_pan.unsqueeze(0),
            size=(img_h, img_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)

    def _pan_gap_residual_mask(self, raw_pan, product_pan):
        mode = str(getattr(self.args, "stage2_ucl_pan_gap_mask", "none")).lower()
        if mode in ("", "none", "off", "false"):
            return raw_pan.new_ones(1, raw_pan.shape[-2], raw_pan.shape[-1])
        source = raw_pan if mode in ("raw_grad", "raw_edge") else product_pan
        grad = self._pan_grad_mag(source).detach()
        q = float(getattr(self.args, "stage2_ucl_pan_gap_mask_quantile", 0.70))
        q = max(0.0, min(0.99, q))
        th = torch.quantile(grad.flatten(), q)
        soft = float(getattr(self.args, "stage2_ucl_pan_gap_mask_softness", 0.05))
        if soft <= 0:
            return (grad >= th).to(dtype=raw_pan.dtype)
        scale = (grad.std(unbiased=False) * soft).clamp_min(1.0e-6)
        return torch.sigmoid((grad - th) / scale)

    def _make_stage2_pan_gap_gain_coeff(self):
        if self.stage2_pan_gap_gain_coeff is None:
            self.stage2_pan_gap_gain_coeff = nn.Parameter(
                torch.zeros(
                    self.xy.shape[0],
                    self.output_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return self.stage2_pan_gap_gain_coeff

    def _render_ucl_pan_gap_gain(self, img_h, img_w, tile_bounds):
        if self.stage2_pan_gap_gain_coeff is None:
            return None
        raw_gain, _ = self._rasterize_features(
            self.xy.detach(),
            self._get_scale(1.0).detach(),
            self.rot.detach(),
            self.stage2_pan_gap_gain_coeff,
            img_h,
            img_w,
            tile_bounds,
        )
        gain_amp = float(getattr(self.args, "stage2_ucl_pan_gap_gain_amp", 0.1))
        gain = 1.0 + gain_amp * torch.tanh(raw_gain)
        gain_clip = float(getattr(self.args, "stage2_ucl_pan_gap_gain_clip", 0.0))
        if gain_clip > 0:
            gain = gain.clamp(1.0 - gain_clip, 1.0 + gain_clip)
        return gain

    def _apply_ucl_pan_gap_residual(self, product_hr, ratio_hr, product_pan, img_h, img_w):
        if not self._stage2_uses_ucl_pan_gap_residual():
            return product_hr
        raw_pan = self._raw_pan_at_size(img_h, img_w).detach()
        gap = raw_pan - product_pan.detach()
        sigmas_cfg = getattr(self.args, "stage2_ucl_pan_gap_sigmas", None)
        if sigmas_cfg is not None and len(list(sigmas_cfg)) > 0:
            # Laplacian-style multi-band decomposition of the gap.
            # With sigmas σ_1 < σ_2 < ... < σ_K (sorted ascending):
            #   band_0 = gap - LP(σ_1)               (finest HF)
            #   band_k = LP(σ_k) - LP(σ_{k+1})       for k=1..K-1
            sigmas = sorted(float(s) for s in sigmas_cfg)
            amps_cfg = getattr(self.args, "stage2_ucl_pan_gap_amps", None)
            if amps_cfg is None or len(list(amps_cfg)) == 0:
                amps = [1.0] * len(sigmas)
            else:
                amps = [float(a) for a in amps_cfg]
                if len(amps) != len(sigmas):
                    raise ValueError(
                        f"stage2_ucl_pan_gap_amps length {len(amps)} != "
                        f"stage2_ucl_pan_gap_sigmas length {len(sigmas)}"
                    )
            lps = [self._gaussian_lowpass_sigma(gap, s) for s in sigmas]
            gap_hf = amps[0] * (gap - lps[0])
            for k in range(len(sigmas) - 1):
                gap_hf = gap_hf + amps[k + 1] * (lps[k] - lps[k + 1])
        else:
            if bool(getattr(self.args, "stage2_ucl_pan_gap_highpass", True)):
                sigma = float(getattr(self.args, "stage2_ucl_pan_gap_sigma", 2.0))
                gap_hf = gap - self._gaussian_lowpass_sigma(gap, sigma)
            else:
                gap_hf = gap
        if bool(getattr(self.args, "stage2_ucl_pan_gap_norm", False)):
            gap_hf = gap_hf / gap_hf.abs().amax().clamp_min(1.0e-6)
        if bool(getattr(self.args, "stage2_ucl_pan_gap_detach_ratio", False)):
            ratio = ratio_hr.detach()
        else:
            ratio = ratio_hr
        amp = float(getattr(self.args, "stage2_ucl_pan_gap_amp", 1.0))
        start_frac = float(getattr(self.args, "stage2_ucl_pan_gap_start_frac", 0.0))
        ramp_frac = float(getattr(self.args, "stage2_ucl_pan_gap_ramp_frac", 0.0))
        if start_frac > 0 or ramp_frac > 0:
            step_idx = float(getattr(self, "_stage2_ucl_step_idx", 1))
            total_steps = max(1.0, float(getattr(self, "_stage2_ucl_total_steps", 1)))
            progress = step_idx / total_steps
            if progress <= start_frac:
                amp = 0.0
            elif ramp_frac > 0:
                amp = amp * max(0.0, min(1.0, (progress - start_frac) / ramp_frac))
        mask = self._pan_gap_residual_mask(raw_pan, product_pan)
        tile_bounds = self._stage2_tile_bounds(img_h, img_w)
        gain = self._render_ucl_pan_gap_gain(img_h, img_w, tile_bounds)
        if gain is not None:
            ratio = ratio * gain
        return product_hr + amp * mask * ratio * gap_hf

    def _stage2_pan_gap_gain_reg_loss(self):
        if self.stage2_pan_gap_gain_coeff is None:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        coeff = self.stage2_pan_gap_gain_coeff
        loss = coeff.new_tensor(0.0)
        l2_w = float(getattr(self.args, "stage2_ucl_pan_gap_gain_l2_weight", 0.0))
        l1_w = float(getattr(self.args, "stage2_ucl_pan_gap_gain_l1_weight", 0.0))
        tv_w = float(getattr(self.args, "stage2_ucl_pan_gap_gain_tv_weight", 0.0))
        if l2_w > 0:
            loss = loss + l2_w * coeff.pow(2).mean()
        if l1_w > 0:
            loss = loss + l1_w * coeff.abs().mean()
        if tv_w > 0:
            pan_h, pan_w = self.ref_images.shape[-2:]
            tb = self._stage2_tile_bounds(pan_h, pan_w)
            gain = self._render_ucl_pan_gap_gain(pan_h, pan_w, tb)
            loss = loss + tv_w * self._tv_loss(gain)
        return loss

    def _make_stage2_lig_gap_coeff(self):
        if self.stage2_lig_gap_coeff is None:
            self.stage2_lig_gap_coeff = nn.Parameter(
                torch.zeros(
                    self.xy.shape[0],
                    self.output_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return self.stage2_lig_gap_coeff

    def _render_ucl_lig_gap_gain(self, img_h, img_w, tile_bounds, render_scale_ratio):
        if self.stage2_lig_gap_coeff is None:
            return None
        raw_gain, _ = self._rasterize_features(
            self.xy.detach(),
            self._get_scale(render_scale_ratio).detach(),
            self.rot.detach(),
            self.stage2_lig_gap_coeff,
            img_h,
            img_w,
            tile_bounds,
        )
        gain_amp = float(getattr(self.args, "stage2_ucl_lig_gap_gain_amp", 0.25))
        return 1.0 + gain_amp * torch.tanh(raw_gain)

    def _apply_ucl_lig_gap_residual(
        self, product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
    ):
        if not self._stage2_uses_ucl_lig_gap_residual():
            return product_hr
        eps = float(getattr(self.args, "stage2_ucl_lig_gap_eps", 1.0e-3))
        raw_pan = self._raw_pan_at_size(img_h, img_w).detach()
        base_for_gap = (
            product_hr.detach()
            if bool(getattr(self.args, "stage2_ucl_lig_gap_detach_base", True))
            else product_hr
        )
        y_base = self._to_pan_estimate(base_for_gap)
        gap = raw_pan - y_base
        if bool(getattr(self.args, "stage2_ucl_lig_gap_highpass", True)):
            sigma = float(getattr(self.args, "stage2_ucl_lig_gap_sigma", 2.0))
            gap = gap - self._gaussian_lowpass_sigma(gap, sigma)
        if bool(getattr(self.args, "stage2_ucl_lig_gap_norm", False)):
            gap = gap / gap.abs().amax().clamp_min(1.0e-6)

        prop_src = (
            product_hr.detach()
            if bool(getattr(self.args, "stage2_ucl_lig_gap_detach_prop", False))
            else product_hr
        ).clamp_min(eps)
        prop = prop_src / self._to_pan_estimate(prop_src).clamp_min(eps)
        gain = self._render_ucl_lig_gap_gain(
            img_h, img_w, tile_bounds, render_scale_ratio
        )
        if gain is not None:
            prop = prop * gain.clamp_min(eps)
        weights = self._pan_weights().view(-1, 1, 1)
        prop = prop / (prop * weights).sum(dim=0, keepdim=True).clamp_min(eps)
        amp = float(getattr(self.args, "stage2_ucl_lig_gap_amp", 1.0))
        clamp_val = float(getattr(self.args, "stage2_ucl_lig_gap_clamp", 0.0))
        residual = amp * prop * gap
        if clamp_val > 0:
            residual = residual.clamp(-clamp_val, clamp_val)
        return product_hr + residual

    def _stage2_lig_gap_reg_loss(self):
        if self.stage2_lig_gap_coeff is None:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        coeff = self.stage2_lig_gap_coeff
        loss = coeff.new_tensor(0.0)
        l2_w = float(getattr(self.args, "stage2_ucl_lig_gap_l2_weight", 0.0))
        l1_w = float(getattr(self.args, "stage2_ucl_lig_gap_l1_weight", 0.0))
        tv_w = float(getattr(self.args, "stage2_ucl_lig_gap_tv_weight", 0.0))
        if l2_w > 0:
            loss = loss + l2_w * coeff.pow(2).mean()
        if l1_w > 0:
            loss = loss + l1_w * coeff.abs().mean()
        if tv_w > 0:
            pan_h, pan_w = self.ref_images.shape[-2:]
            tb = self._stage2_tile_bounds(pan_h, pan_w)
            gain = self._render_ucl_lig_gap_gain(pan_h, pan_w, tb, 1.0)
            loss = loss + tv_w * self._tv_loss(gain)
        return loss

    def _stage2_pan_detail_reg_loss(self):
        if self.stage2_pan_detail_coeff is None:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        coeff = self.stage2_pan_detail_coeff
        l2_w = float(getattr(self.args, "stage2_ucl_pan_detail_l2_weight", 0.0))
        l1_w = float(getattr(self.args, "stage2_ucl_pan_detail_l1_weight", 0.0))
        tv_w = float(getattr(self.args, "stage2_ucl_pan_detail_tv_weight", 0.0))
        loss = coeff.new_tensor(0.0)
        if l2_w > 0:
            loss = loss + l2_w * coeff.pow(2).mean()
        if l1_w > 0:
            loss = loss + l1_w * coeff.abs().mean()
        if tv_w > 0:
            pan_h, pan_w = self.ref_images.shape[-2:]
            tb = self._stage2_tile_bounds(pan_h, pan_w)
            gain = self._render_ucl_pan_detail_gain(pan_h, pan_w, tb, 1.0)
            loss = loss + tv_w * self._tv_loss(gain)
        return loss

    def _stage2_pan_detail_constraint_loss(self, ms_hr, product_pan):
        corr_w = float(getattr(self.args, "stage2_ucl_pan_detail_corr_weight", 0.0))
        l1_w = float(getattr(self.args, "stage2_ucl_pan_detail_l1_weight_loss", 0.0))
        if corr_w <= 0 and l1_w <= 0:
            return ms_hr.new_tensor(0.0)
        pred_hp = self._highpass(self._to_pan_estimate(ms_hr))
        pan_hp = self._pan_highpass_for_detail(product_pan)
        loss = ms_hr.new_tensor(0.0)
        if corr_w > 0:
            loss = loss + corr_w * self._corr_loss(pred_hp, pan_hp)
        if l1_w > 0:
            pred_norm = pred_hp / pred_hp.abs().mean().clamp_min(1.0e-6)
            pan_norm = pan_hp / pan_hp.abs().mean().clamp_min(1.0e-6)
            loss = loss + l1_w * F.l1_loss(pred_norm, pan_norm)
        return loss

    def _stage2_ucl_plain_tv_loss(self, ratio_hr):
        tv = (
            torch.pow(ratio_hr[:, :, :-1] - ratio_hr[:, :, 1:], 2).mean()
            + torch.pow(ratio_hr[:, :-1, :] - ratio_hr[:, 1:, :], 2).mean()
        )
        return tv

    def _stage2_ucl_pan_guided_tv_loss(self, ratio_hr, guide_pan):
        alpha = float(getattr(self.args, "stage2_ucl_pan_guided_tv_alpha", 10.0))
        eps = float(getattr(self.args, "stage2_ucl_tv_charbonnier_eps", 1.0e-3))
        guide = guide_pan.detach()
        gx = torch.abs(guide[:, :, 1:] - guide[:, :, :-1])
        gy = torch.abs(guide[:, 1:, :] - guide[:, :-1, :])
        gx = gx / gx.mean().clamp_min(1.0e-6)
        gy = gy / gy.mean().clamp_min(1.0e-6)
        wx = torch.exp(-alpha * gx)
        wy = torch.exp(-alpha * gy)

        dx = ratio_hr[:, :, 1:] - ratio_hr[:, :, :-1]
        dy = ratio_hr[:, 1:, :] - ratio_hr[:, :-1, :]
        tv_x = (wx * torch.sqrt(dx * dx + eps * eps)).mean()
        tv_y = (wy * torch.sqrt(dy * dy + eps * eps)).mean()
        return tv_x + tv_y

    def _stage2_ucl_tv_loss(self, ratio_hr, guide_pan=None, force_pan_guided=False):
        weight = float(getattr(self.args, "stage2_ucl_tv_weight", 0.001))
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        weight = weight * float(self._stage2_ucl_tv_schedule_scale())
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        use_pan_guided = force_pan_guided or bool(
            getattr(self.args, "stage2_ucl_pan_guided_tv", False)
        )
        if use_pan_guided:
            if guide_pan is None:
                raise ValueError("PAN-guided TV requires a guide PAN image")
            return weight * self._stage2_ucl_pan_guided_tv_loss(ratio_hr, guide_pan)
        return weight * self._stage2_ucl_plain_tv_loss(ratio_hr)

    def _stage2_inmask_ratio_tv_loss(self, ratio_hr):
        """Extra squared TV on ratio_hr restricted to PAN top-K% bright mask.

        Reduces in-mask variance of the rendered ratio. Default off
        (stage2_ucl_tv_inmask_extra_weight=0). Independent from base
        stage2_ucl_tv_weight, additive on top of it.
        """
        weight = float(getattr(
            self.args, "stage2_ucl_tv_inmask_extra_weight", 0.0
        ))
        if weight <= 0:
            return ratio_hr.new_tensor(0.0)
        topk = float(getattr(self.args, "stage2_ucl_tv_inmask_topk", 0.03))
        topk = min(max(topk, 1.0e-4), 1.0)
        cache = getattr(self, "_inmask_tv_cache", None)
        cache_topk = getattr(self, "_inmask_tv_cache_topk", -1.0)
        if cache is None or abs(cache_topk - topk) > 1.0e-9 \
                or cache.shape != ratio_hr.shape[-2:]:
            with torch.no_grad():
                pan_y = self.ref_images.detach()
                if pan_y.dim() == 3:
                    pan_y = pan_y.mean(0)
                thr = torch.quantile(pan_y.flatten(), 1.0 - topk)
                mask = (pan_y >= thr).to(ratio_hr.dtype)
                self._inmask_tv_cache = mask
                self._inmask_tv_cache_topk = topk
                n_in = int(mask.sum().item())
                self.worklog.info(
                    f">>> in-mask ratio TV active: topk={topk*100:.2f}% "
                    f"({n_in} px, {100*n_in/mask.numel():.2f}%), "
                    f"extra_weight={weight:.4g}"
                )
        mask = self._inmask_tv_cache  # [H, W]
        # Pair endpoint mask: count a pair if EITHER endpoint is in mask.
        mx = ((mask[:, 1:] + mask[:, :-1]) > 0).to(ratio_hr.dtype)  # [H, W-1]
        my = ((mask[1:, :] + mask[:-1, :]) > 0).to(ratio_hr.dtype)  # [H-1, W]
        dx2 = (ratio_hr[:, :, 1:] - ratio_hr[:, :, :-1]) ** 2
        dy2 = (ratio_hr[:, 1:, :] - ratio_hr[:, :-1, :]) ** 2
        cx = mx.sum().clamp_min(1.0) * ratio_hr.shape[0]
        cy = my.sum().clamp_min(1.0) * ratio_hr.shape[0]
        tv_x = (dx2 * mx.unsqueeze(0)).sum() / cx
        tv_y = (dy2 * my.unsqueeze(0)).sum() / cy
        return weight * (tv_x + tv_y)

    def _stage2_locality_xy(self):
        """Return the per-Gaussian xy used by the current ratio path.

        Picks self.xy_ratio when independent geo, else self.xy.
        """
        if bool(getattr(self.args, "stage2_ucl_independent_geo", True)) \
                and hasattr(self, "xy_ratio"):
            return self.xy_ratio
        return self.xy

    def _stage2_locality_build_knn(self):
        """Build kNN over Gaussian xy in pixel units. Called once on demand.

        Cache keyed by N so densification triggers a rebuild.
        """
        xy = self._stage2_locality_xy().detach()
        N = xy.shape[0]
        cache_N = getattr(self, "_locality_cache_N", -1)
        if cache_N == N and getattr(self, "_locality_idx", None) is not None:
            return
        K = int(getattr(self.args, "stage2_locality_k", 8))
        K = max(1, min(K, N - 1))
        max_r_frac = float(getattr(
            self.args, "stage2_locality_max_radius_frac", 0.05
        ))  # fraction of image extent; 0.05 ~ 5% of width
        H, W = self.ref_images.shape[-2:]
        xy_px = xy * xy.new_tensor([float(W), float(H)])
        # chunked pairwise to avoid N*N memory for large N
        chunk = 4096
        idxs = []
        dists = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            d = torch.cdist(xy_px[s:e], xy_px)
            # exclude self
            for i in range(s, e):
                d[i - s, i] = float("inf")
            kd, ki = d.topk(K, dim=1, largest=False)
            dists.append(kd)
            idxs.append(ki)
        knn_idx = torch.cat(idxs, dim=0)               # [N, K]
        knn_dist = torch.cat(dists, dim=0)             # [N, K], pixel units
        max_r_px = max_r_frac * max(H, W)
        valid = (knn_dist <= max_r_px).to(xy.dtype)
        self._locality_idx = knn_idx
        self._locality_dist = knn_dist
        self._locality_valid = valid
        self._locality_cache_N = N
        n_valid = int(valid.sum().item())
        self.worklog.info(
            f">>> locality kNN built: N={N}, K={K}, max_r={max_r_px:.1f}px, "
            f"valid_pairs={n_valid}/{N*K} ({100*n_valid/max(N*K,1):.1f}%)"
        )

    def _stage2_locality_loss(self, feat_ratio):
        """FewViewGS-style per-Gaussian locality regularizer.

        Pure geometric:
            L = mean_{ik} exp(-delta * d_ik) * ||f_i - f_k||_1
        Mixed (pan_alpha > 0) adds PAN-edge-aware weight:
            w_ik = exp(-delta * d_ik) * exp(-pan_alpha * |PAN_i - PAN_k|)

        - feat_ratio: [N, C] per-Gaussian ratio params (or [N, 1] in shared mode)
        - delta: spatial decay rate; FewViewGS used 2.0 in 3D unit cube.
                  Here d is in normalized image coords (0..1), so use larger.
        - pan_alpha: 0 disables PAN-aware weighting (= paper original).
        """
        weight = float(getattr(self.args, "stage2_locality_weight", 0.0))
        if weight <= 0:
            return feat_ratio.new_tensor(0.0)
        if feat_ratio is None or feat_ratio.shape[0] < 2:
            return feat_ratio.new_tensor(0.0)
        self._stage2_locality_build_knn()
        idx = self._locality_idx
        dist_px = self._locality_dist
        valid = self._locality_valid
        H, W = self.ref_images.shape[-2:]
        delta = float(getattr(self.args, "stage2_locality_delta", 2.0))
        # normalize dist to image extent so delta lives on a sane scale
        dist_norm = dist_px / float(max(H, W))
        w = torch.exp(-delta * dist_norm)              # [N, K]
        pan_alpha = float(getattr(self.args, "stage2_locality_pan_alpha", 0.0))
        if pan_alpha > 0:
            pan = self.ref_images.detach()
            if pan.dim() == 3:
                pan = pan.mean(0, keepdim=True)        # [1, H, W]
            xy = self._stage2_locality_xy().detach()
            grid = (xy * 2.0 - 1.0).reshape(1, 1, -1, 2)
            pan_at_g = F.grid_sample(
                pan.unsqueeze(0), grid, mode="bilinear",
                align_corners=False, padding_mode="border",
            ).reshape(-1)                              # [N]
            pan_self = pan_at_g.unsqueeze(1)           # [N, 1]
            pan_nb = pan_at_g[idx]                     # [N, K]
            w = w * torch.exp(-pan_alpha * (pan_self - pan_nb).abs())
        w = w * valid                                  # [N, K]
        # L1 over channel dim (vector L1 norm of diff), as in FewViewGS Eq. 13
        self_f = feat_ratio.unsqueeze(1)               # [N, 1, C]
        nb_f = feat_ratio[idx]                         # [N, K, C]
        diff = (self_f - nb_f).abs().sum(dim=-1)       # [N, K]
        denom = w.sum().clamp_min(1.0)
        loss = (w * diff).sum() / denom
        return weight * loss

    def _stage2_inmask_sam_loss(self, ms_hr):
        # F2: mask-internal spectral angle constraint.
        # Target = bicubic(MS_LR); SAM is scale-invariant in target magnitude,
        # so it only constrains the spectral direction at PAN-bright pixels.
        # Cannot bias the mean down (unlike TV/locality smoothing).
        weight = float(getattr(self.args, "stage2_inmask_sam_weight", 0.0))
        if weight <= 0:
            return ms_hr.new_tensor(0.0)
        topk = float(getattr(self.args, "stage2_inmask_sam_topk", 0.03))
        H, W = ms_hr.shape[-2:]
        cache_key = (round(topk, 5), H, W)
        if (not hasattr(self, "_inmask_sam_cache")
                or getattr(self, "_inmask_sam_cache_key", None) != cache_key):
            with torch.no_grad():
                pan_y = self.ref_images.detach()
                if pan_y.dim() == 3:
                    pan_y = pan_y.mean(0)
                thr = torch.quantile(pan_y.flatten(), 1.0 - topk)
                self._inmask_sam_cache = (pan_y >= thr)
                self._inmask_sam_cache_key = cache_key
        mask = self._inmask_sam_cache
        target = self._bicubic_hr_target().to(ms_hr.device).detach()
        if target.shape[-2:] != ms_hr.shape[-2:]:
            target = F.interpolate(
                target.unsqueeze(0), size=ms_hr.shape[-2:],
                mode="bicubic", align_corners=False,
            ).squeeze(0)
        pred_in = ms_hr[:, mask]                    # [C, N_in]
        target_in = target[:, mask]                 # [C, N_in]
        dot = (pred_in * target_in).sum(dim=0)
        pred_norm = torch.linalg.norm(pred_in, dim=0)
        target_norm = torch.linalg.norm(target_in, dim=0)
        cos = dot / (pred_norm * target_norm + 1.0e-8)
        sam = torch.acos(cos.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)).mean()
        return weight * sam

    def _stage2_ucl_tv_schedule_scale(self):
        enabled = bool(getattr(self.args, "stage2_ucl_tv_schedule_enabled", False))
        if not enabled:
            return 1.0
        start_frac = float(getattr(self.args, "stage2_ucl_tv_schedule_start_frac", 0.7))
        ramp_frac = float(getattr(self.args, "stage2_ucl_tv_schedule_ramp_frac", 0.0))
        step_idx = float(getattr(self, "_stage2_ucl_step_idx", 1))
        total_steps = max(1.0, float(getattr(self, "_stage2_ucl_total_steps", 1)))
        progress = step_idx / total_steps
        if progress <= start_frac:
            return 0.0
        if ramp_frac > 0:
            return max(0.0, min(1.0, (progress - start_frac) / ramp_frac))
        return 1.0

    def _channel_corr_matrix(self, image):
        flat = image.reshape(image.shape[0], -1)
        flat = flat - flat.mean(dim=1, keepdim=True)
        flat = flat / flat.std(dim=1, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        return torch.matmul(flat, flat.t()) / max(1, flat.shape[1])

    def _stage2_ucl_spectral_corr_loss(self, pred_image):
        weight = float(getattr(self.args, "stage2_ucl_spec_corr_weight", 0.0))
        if weight <= 0:
            return pred_image.new_tensor(0.0)
        pred_corr = self._channel_corr_matrix(pred_image)
        target_corr = self._channel_corr_matrix(self.gt_images.detach())
        return weight * F.mse_loss(pred_corr, target_corr)

    def _stage2_ucl_chroma_prior_loss(self, ms_hr, guide_pan):
        tv_w = float(getattr(self.args, "stage2_ucl_chroma_tv_weight", 0.0))
        hp_w = float(getattr(self.args, "stage2_ucl_chroma_hf_weight", 0.0))
        if tv_w <= 0 and hp_w <= 0:
            return ms_hr.new_tensor(0.0)
        eps = float(getattr(self.args, "stage2_ucl_chroma_eps", 1.0e-3))
        intensity = self._to_pan_estimate(ms_hr).clamp_min(eps)
        if bool(getattr(self.args, "stage2_ucl_chroma_detach_intensity", True)):
            intensity = intensity.detach()
        chroma = ms_hr / intensity
        loss = ms_hr.new_tensor(0.0)
        if tv_w > 0:
            if bool(getattr(self.args, "stage2_ucl_chroma_tv_plain", False)):
                # Uniform (non-PAN-guided) Charbonnier TV: chroma should be
                # smooth across PAN edges too — false colors live exactly at
                # edges, which the guided weighting exp(-a|gradPAN|) exempts.
                eps = float(getattr(self.args, "stage2_ucl_tv_charbonnier_eps", 1.0e-3))
                dx = chroma[:, :, 1:] - chroma[:, :, :-1]
                dy = chroma[:, 1:, :] - chroma[:, :-1, :]
                loss = loss + tv_w * (
                    torch.sqrt(dx * dx + eps * eps).mean()
                    + torch.sqrt(dy * dy + eps * eps).mean()
                )
            else:
                loss = loss + tv_w * self._stage2_ucl_pan_guided_tv_loss(chroma, guide_pan)
        if hp_w > 0:
            chroma_hp = self._highpass(chroma)
            edge = self._pan_grad_mag(guide_pan).detach()
            edge = edge / edge.mean().clamp_min(1.0e-6)
            alpha = float(getattr(self.args, "stage2_ucl_chroma_edge_alpha", 10.0))
            smooth_mask = torch.exp(-alpha * edge)
            loss = loss + hp_w * (smooth_mask * chroma_hp.abs()).mean()
        return loss

    def _stage2_bright_chroma_anchor_loss(self, ms_hr):
        """Anchor spectral ratios in PAN-bright pixels to LRMS chroma.

        PAN L1 only constrains the weighted band sum. Bright false-color
        spots can keep that sum plausible while individual bands oscillate,
        so this constrains chroma instead of intensity.
        """
        weight = float(getattr(self.args, "stage2_bright_chroma_anchor_weight", 0.0))
        hf_weight = float(getattr(
            self.args, "stage2_bright_chroma_anchor_hf_weight", 0.0
        ))
        if weight <= 0.0 and hf_weight <= 0.0:
            return ms_hr.new_tensor(0.0)

        eps = float(getattr(self.args, "stage2_bright_chroma_anchor_eps", 1.0e-3))
        target = self._bicubic_hr_target().to(ms_hr.device).detach()
        if target.shape[-2:] != ms_hr.shape[-2:]:
            target = F.interpolate(
                target.unsqueeze(0),
                size=ms_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)

        pred_intensity = self._to_pan_estimate(ms_hr).clamp_min(eps)
        target_intensity = self._to_pan_estimate(target).clamp_min(eps)
        if bool(getattr(self.args, "stage2_bright_chroma_anchor_detach_intensity", True)):
            pred_intensity = pred_intensity.detach()
            target_intensity = target_intensity.detach()
        pred_chroma = ms_hr / pred_intensity
        target_chroma = target / target_intensity

        ref_pan = self.ref_images.detach()
        if ref_pan.shape[-2:] != ms_hr.shape[-2:]:
            ref_pan = F.interpolate(
                ref_pan.unsqueeze(0),
                size=ms_hr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        topk = float(getattr(self.args, "stage2_bright_chroma_anchor_topk", 0.05))
        topk = min(max(topk, 1.0e-4), 1.0)
        thr = torch.quantile(ref_pan.flatten(), 1.0 - topk)
        mask = (ref_pan >= thr).to(ms_hr.dtype)
        mask = mask.expand_as(pred_chroma)
        denom = mask.sum().clamp_min(1.0)

        loss = ms_hr.new_tensor(0.0)
        if weight > 0.0:
            diff = (pred_chroma - target_chroma).abs() * mask
            loss = loss + weight * diff.sum() / denom
        if hf_weight > 0.0:
            pred_hp = self._highpass(pred_chroma)
            target_hp = self._highpass(target_chroma)
            diff_hp = (pred_hp - target_hp).abs() * mask
            loss = loss + hf_weight * diff_hp.sum() / denom
        return loss

    def _stage2_ratio_init_mode(self):
        mode = str(getattr(self.args, "stage2_ratio_init_mode", "zero")).lower()
        if mode != "zero":
            return mode
        legacy_mode = str(getattr(self.args, "stage2_ucl_ratio_init", "zero")).lower()
        return legacy_mode

    def _stage2_base_ratio_init_mode(self):
        """Stage2 base-ratio initialization mode.

        Shape contract: returns a string mode used to initialize feat_ratio
        rows [N,C]. The default is "zero", which preserves the baseline.
        """
        mode = str(getattr(self.args, "stage2_base_ratio_init_mode", "zero")).lower()
        if mode not in ("", "zero", "zeros", "none"):
            return mode
        return self._stage2_ratio_init_mode()

    def _stage2_pan_guided_lr_ratio_image(self, product_pan_ucl):
        """PAN-guided upsampling of the observable LR coefficient LRMS / D(PAN).

        This builds a coefficient prior before Gaussian optimization. It is
        intentionally deterministic and non-neural: each HR coefficient pixel is
        a joint-bilateral average of nearby LR coefficients, weighted by spatial
        distance and PAN radiance similarity.
        """
        with torch.no_grad():
            eps = float(getattr(self.args, "stage2_pan_guided_lr_ratio_eps", 1.0e-3))
            clamp_val = float(
                getattr(self.args, "stage2_pan_guided_lr_ratio_clamp", 5.0)
            )
            radius = int(getattr(self.args, "stage2_pan_guided_lr_ratio_radius", 2))
            radius = max(0, radius)
            sigma_s = float(
                getattr(self.args, "stage2_pan_guided_lr_ratio_sigma_s", 1.25)
            )
            sigma_r = float(
                getattr(self.args, "stage2_pan_guided_lr_ratio_sigma_r", 0.04)
            )
            sigma_s = max(sigma_s, 1.0e-6)
            sigma_r = max(sigma_r, 1.0e-6)

            pan_hr = product_pan_ucl.detach().to(dtype=self.dtype, device=self.device)
            ms_lr = self.gt_images.detach().to(dtype=self.dtype, device=self.device)
            pan_down = str(
                getattr(self.args, "stage2_pan_guided_lr_ratio_pan_down", "area")
            ).lower()
            if pan_down in ("bicubic", "bilinear"):
                pan_lr = F.interpolate(
                    pan_hr.unsqueeze(0),
                    size=ms_lr.shape[-2:],
                    mode=pan_down,
                    align_corners=False,
                ).squeeze(0)
            elif pan_down in ("area", "avg", "average"):
                pan_lr = F.interpolate(
                    pan_hr.unsqueeze(0), size=ms_lr.shape[-2:], mode="area"
                ).squeeze(0)
            else:
                raise ValueError(
                    f"Unsupported stage2_pan_guided_lr_ratio_pan_down: {pan_down}"
                )
            pan_lr = pan_lr.clamp_min(eps)
            ratio_lr = (ms_lr / pan_lr).clamp(0.0, clamp_val)
            c, h_lr, w_lr = ratio_lr.shape
            _, h_hr, w_hr = pan_hr.shape

            yy, xx = torch.meshgrid(
                torch.arange(h_hr, dtype=self.dtype, device=self.device),
                torch.arange(w_hr, dtype=self.dtype, device=self.device),
                indexing="ij",
            )
            # align_corners=False inverse coordinate mapping.
            y_lr = (yy + 0.5) * (float(h_lr) / float(h_hr)) - 0.5
            x_lr = (xx + 0.5) * (float(w_lr) / float(w_hr)) - 0.5
            y0 = torch.round(y_lr).long()
            x0 = torch.round(x_lr).long()

            accum = ratio_lr.new_zeros(c, h_hr, w_hr)
            wsum = ratio_lr.new_zeros(1, h_hr, w_hr)
            sat = (ms_lr / pan_lr > clamp_val).to(dtype=self.dtype).mean()
            for dy in range(-radius, radius + 1):
                yy_lr = (y0 + dy).clamp(0, h_lr - 1)
                dy_real = y_lr - yy_lr.to(dtype=self.dtype)
                for dx in range(-radius, radius + 1):
                    xx_lr = (x0 + dx).clamp(0, w_lr - 1)
                    dx_real = x_lr - xx_lr.to(dtype=self.dtype)
                    spatial = torch.exp(
                        -0.5 * (dx_real * dx_real + dy_real * dy_real) / (sigma_s * sigma_s)
                    )
                    pan_candidate = pan_lr[:, yy_lr, xx_lr]
                    range_w = torch.exp(
                        -0.5 * (pan_hr - pan_candidate).pow(2) / (sigma_r * sigma_r)
                    )
                    weight = spatial.unsqueeze(0) * range_w
                    accum = accum + ratio_lr[:, yy_lr, xx_lr] * weight
                    wsum = wsum + weight
            guided = (accum / wsum.clamp_min(1.0e-8)).clamp(0.0, clamp_val)

            blend = float(getattr(self.args, "stage2_pan_guided_lr_ratio_blend", 1.0))
            blend = max(0.0, min(1.0, blend))
            if blend < 1.0:
                bicubic = F.interpolate(
                    ratio_lr.unsqueeze(0),
                    size=(h_hr, w_hr),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).clamp(0.0, clamp_val)
                guided = blend * guided + (1.0 - blend) * bicubic

            self._stage2_last_ratio_init_stats = {
                "mode": "pan_guided_lr_ratio",
                "img_min": float(guided.min().item()),
                "img_max": float(guided.max().item()),
                "img_mean": float(guided.mean().item()),
                "lr_sat": float(sat.item()),
                "radius": int(radius),
                "sigma_s": float(sigma_s),
                "sigma_r": float(sigma_r),
                "blend": float(blend),
                "pan_down": pan_down,
            }
            return guided, self._stage2_last_ratio_init_stats

    def _stage2_init_ratio_features(self, pan_rendered_s1, product_pan_ucl, xy):
        mode = self._stage2_base_ratio_init_mode()
        self._stage2_last_ratio_init_stats = {}
        if mode in ("pan_bright_init", "pan_bright", "pan_bright_top"):
            topk = float(getattr(self.args, "stage2_pan_bright_init_topk", 0.25))
            topk = min(max(topk, 1.0e-4), 1.0)
            value = float(getattr(self.args, "stage2_pan_bright_init_value", 1.0))
            pan_ref = self.ref_images.detach().to(
                dtype=product_pan_ucl.dtype, device=product_pan_ucl.device
            )
            pan_y = pan_ref.mean(0) if pan_ref.dim() == 3 else pan_ref
            thresh = torch.quantile(pan_y.flatten(), 1.0 - topk)
            mask = (pan_y >= thresh).to(dtype=product_pan_ucl.dtype)
            C = sum(self.input_channels)
            ratio_init_img = (value * mask).unsqueeze(0).expand(C, -1, -1).contiguous()
            init_ratio = self._sample_features_from_image(ratio_init_img, xy)
            self._stage2_last_ratio_init_stats = {
                "mode": mode,
                "topk": topk,
                "value": value,
                "thresh": float(thresh.item()),
                "mask_active": float(mask.mean().item()),
                "feat_mean": float(init_ratio.mean().item()),
                "feat_max": float(init_ratio.max().item()),
            }
            return init_ratio, ratio_init_img, self._stage2_last_ratio_init_stats
        if mode in ("lr_ratio", "lowres_ratio", "lrms_ratio"):
            eps = float(getattr(self.args, "stage2_base_ratio_init_eps", 1.0e-6))
            clamp_val = float(getattr(self.args, "stage2_base_ratio_init_clamp", 5.0))
            ms_h, ms_w = self.gt_images.shape[-2], self.gt_images.shape[-1]
            pan_lr = F.interpolate(
                product_pan_ucl.detach().unsqueeze(0),
                size=(ms_h, ms_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp_min(eps)
            ratio_lr = self.gt_images.detach().to(
                dtype=product_pan_ucl.dtype, device=product_pan_ucl.device
            ) / pan_lr
            sat = (ratio_lr > clamp_val).to(dtype=product_pan_ucl.dtype).mean()
            ratio_lr = ratio_lr.clamp(0.0, clamp_val)
            ratio_init_img = F.interpolate(
                ratio_lr.unsqueeze(0),
                size=product_pan_ucl.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0.0, clamp_val)
            init_ratio = self._sample_features_from_image(ratio_init_img, xy)
            self._stage2_last_ratio_init_stats = {
                "mode": mode,
                "img_min": float(ratio_init_img.min().item()),
                "img_max": float(ratio_init_img.max().item()),
                "img_mean": float(ratio_init_img.mean().item()),
                "lr_sat": float(sat.item()),
            }
            return init_ratio, ratio_init_img, self._stage2_last_ratio_init_stats
        if mode in (
            "pan_guided_lr_ratio",
            "pan_guided_ratio",
            "jbu_lr_ratio",
            "joint_bilateral_lr_ratio",
            "graph_lr_ratio",
        ):
            ratio_init_img, stats = self._stage2_pan_guided_lr_ratio_image(
                product_pan_ucl
            )
            init_ratio = self._sample_features_from_image(ratio_init_img, xy)
            stats.update({
                "feat_min": float(init_ratio.min().item()),
                "feat_max": float(init_ratio.max().item()),
                "feat_mean": float(init_ratio.mean().item()),
            })
            self._stage2_last_ratio_init_stats = stats
            return init_ratio, ratio_init_img, stats
        if mode in ("bicubic_ratio", "bicubic", "lms_ratio", "sampled_ratio", "sample_ratio"):
            eps = float(getattr(self.args, "stage2_ratio_init_eps", 1.0e-6))
            clamp_val = float(getattr(self.args, "stage2_ratio_init_clamp", 5.0))
            lms_hr = self._bicubic_hr_target().to(
                dtype=product_pan_ucl.dtype, device=product_pan_ucl.device
            )
            ratio_init_img = lms_hr / product_pan_ucl.clamp_min(eps)
            ratio_init_img = ratio_init_img.clamp(0.0, clamp_val)
            init_ratio = self._sample_features_from_image(ratio_init_img, xy)
            stats = {
                "mode": mode,
                "img_min": float(ratio_init_img.min().item()),
                "img_max": float(ratio_init_img.max().item()),
                "img_mean": float(ratio_init_img.mean().item()),
            }
            self._stage2_last_ratio_init_stats = stats
            return init_ratio, ratio_init_img, stats
        if mode in ("mtf_glp_ratio", "mtf_glp_div_pan", "vivone_ratio", "vivone"):
            # Vivone 2018: use MTF-GLP HR target (optionally with full-scale
            # iterative g_k refinement via stage2_vivone_full_scale_iters)
            # divided by current PAN, as ratio init.
            eps = float(getattr(self.args, "stage2_ratio_init_eps", 1.0e-6))
            clamp_val = float(getattr(self.args, "stage2_ratio_init_clamp", 5.0))
            out_h, out_w = product_pan_ucl.shape[-2:]
            mtf_target = self._build_mtf_glp_target_at(out_h, out_w, log=False).to(
                dtype=product_pan_ucl.dtype, device=product_pan_ucl.device
            )
            ratio_init_img = (
                mtf_target / product_pan_ucl.clamp_min(eps)
            ).clamp(0.0, clamp_val)
            init_ratio = self._sample_features_from_image(ratio_init_img, xy)
            stats = {
                "mode": mode,
                "img_min": float(ratio_init_img.min().item()),
                "img_max": float(ratio_init_img.max().item()),
                "img_mean": float(ratio_init_img.mean().item()),
                "vivone_iters": int(getattr(
                    self.args, "stage2_vivone_full_scale_iters", 0
                )),
            }
            self._stage2_last_ratio_init_stats = stats
            return init_ratio, ratio_init_img, stats
        init_ratio = torch.zeros(
            xy.shape[0], sum(self.input_channels), dtype=self.dtype, device=self.device
        )
        self._stage2_last_ratio_init_stats = {"mode": mode}
        return init_ratio, None, {"mode": mode}

    def _stage2_product_pan_mode(self):
        return str(getattr(self.args, "stage2_product_pan_mode", "fixed_s1")).lower()

    def _stage2_residual_branch_enabled(self):
        return bool(getattr(self.args, "stage2_residual_branch", False))

    def _stage2_residual_init_mode(self):
        return str(getattr(self.args, "stage2_residual_init", "zero")).lower()

    def _stage2_residual_basis_rank(self):
        return int(getattr(self.args, "stage2_residual_basis_rank", 0))

    def _stage2_uses_residual_lowrank(self):
        return self._stage2_residual_branch_enabled() and self._stage2_residual_basis_rank() > 0

    def _make_stage2_residual_basis(self):
        if not self._stage2_uses_residual_lowrank():
            return None
        if self.stage2_residual_basis_B is None:
            R = self._stage2_residual_basis_rank()
            C = self.output_dim
            init = str(getattr(self.args, "stage2_residual_basis_init", "identity")).lower()
            mod = _Stage2SpectralBasisB(
                R=R, C=C, init="uniform" if init == "uniform" else "identity"
            ).to(self.device).to(dtype=self.dtype)
            if init in ("uniform", "avg", "average"):
                with torch.no_grad():
                    mod.B.fill_(1.0 / max(R, 1))
            elif init in ("identity", "eye", "diag", "truncated_identity"):
                with torch.no_grad():
                    mod.B.zero_()
                    for r in range(min(R, C)):
                        mod.B[r, r] = 1.0
                    if R > C:
                        mod.B[C:, :] = 1.0e-3 * torch.randn(R - C, C, dtype=self.dtype, device=self.device)
                    if R < C:
                        mod.B.add_(1.0e-3 * torch.randn(R, C, dtype=self.dtype, device=self.device))
            else:
                with torch.no_grad():
                    mod.B.copy_(0.05 * torch.randn(R, C, dtype=self.dtype, device=self.device))
            self.stage2_residual_basis_B = mod
        return self.stage2_residual_basis_B

    def _stage2_residual_gate_mode(self):
        return str(
            getattr(
                self.args,
                "stage2_residual_gate_mode",
                getattr(self.args, "stage2_residual_gate", "pan_hf"),
            )
        ).lower()

    def _stage2_pan_structure_gate(self, product_pan, mode=None, sigma=None, power=None):
        """Build a detached PAN structure gate [1,H,W] in [0,1].

        product_pan may be [1,H,W] or [C,H,W]. The output broadcasts over
        HRMS channels and intentionally does not backpropagate into PAN.
        """
        eps = 1.0e-8
        mode = self._stage2_residual_gate_mode() if mode is None else str(mode).lower()
        if product_pan.dim() == 2:
            pan = product_pan.unsqueeze(0)
        else:
            pan = product_pan
        pan = pan.detach().clamp(0.0, 1.0)
        if mode in ("none", "off", "false", ""):
            gate = pan.new_ones(1, pan.shape[-2], pan.shape[-1])
        elif mode in ("pan_hf", "pan_highfreq", "pan_highpass"):
            sigma = float(
                getattr(self.args, "stage2_residual_gate_sigma", 2.0)
                if sigma is None else sigma
            )
            low = self._gaussian_lowpass_sigma(pan, sigma)
            score = (pan - low).abs().mean(dim=0, keepdim=True)
            score = score / score.mean().clamp_min(eps)
            max_gate = float(getattr(self.args, "stage2_residual_gate_max", 8.0))
            gate = score.clamp(0.0, max_gate)
            gate = gate / gate.max().clamp_min(eps)
        elif mode in ("pan_grad", "grad", "gradient"):
            score = self._pan_grad_mag(pan).mean(dim=0, keepdim=True)
            score = score / score.mean().clamp_min(eps)
            max_gate = float(getattr(self.args, "stage2_residual_gate_max", 8.0))
            gate = score.clamp(0.0, max_gate)
            gate = gate / gate.max().clamp_min(eps)
        else:
            raise ValueError(
                "stage2_residual_gate_mode must be none, pan_hf, or pan_grad"
            )
        power = float(
            getattr(self.args, "stage2_residual_gate_power", 1.0)
            if power is None else power
        )
        gate = gate.clamp(0.0, 1.0).pow(power).detach()
        self._stage2_last_residual_gate_stats = {
            "mode": mode,
            "min": float(gate.min().item()),
            "max": float(gate.max().item()),
            "mean": float(gate.mean().item()),
            "active": float((gate > 0.5).to(dtype=gate.dtype).mean().item()),
        }
        return gate

    def _stage2_residual_gate(self, product_pan):
        return self._stage2_pan_structure_gate(product_pan)

    def _stage2_residual_carrier(self, product_pan):
        """Fixed carrier for the Stage2 residual branch.

        Default keeps the legacy residual branch as a free additive Gaussian
        residual. The pan_hf mode turns residual_raw into a learned per-band
        amplitude field multiplied by a fixed PAN high-pass carrier, so the
        residual cannot invent its own spatial high-frequency structure.
        """
        mode = str(getattr(self.args, "stage2_residual_carrier_mode", "none")).lower()
        if mode in ("none", "off", "false", "free", "additive", ""):
            return None
        if product_pan.dim() == 2:
            pan = product_pan.unsqueeze(0)
        else:
            pan = product_pan
        pan = pan.detach().clamp(0.0, 1.0)
        if pan.shape[0] > 1:
            pan = pan.mean(dim=0, keepdim=True)
        if mode in ("pan_hf", "panhf", "pan_highpass", "highpass"):
            sigma = float(getattr(self.args, "stage2_residual_carrier_sigma", 2.0))
            carrier = pan - self._gaussian_lowpass_sigma(pan, sigma)
        else:
            raise ValueError(
                "stage2_residual_carrier_mode must be none or pan_hf"
            )

        norm = str(
            getattr(self.args, "stage2_residual_carrier_norm", "mean_abs")
        ).lower()
        if norm in ("mean_abs", "l1", "absmean"):
            carrier = carrier / carrier.abs().mean().clamp_min(1.0e-6)
        elif norm in ("std", "zstd"):
            carrier = carrier / carrier.std(unbiased=False).clamp_min(1.0e-6)
        elif norm in ("none", "off", "raw"):
            pass
        else:
            raise ValueError(
                "stage2_residual_carrier_norm must be mean_abs, std, or none"
            )
        clamp_val = float(getattr(self.args, "stage2_residual_carrier_clamp", 6.0))
        if clamp_val > 0:
            carrier = carrier.clamp(-clamp_val, clamp_val)
        carrier = carrier.detach()
        self._stage2_last_residual_carrier_stats = {
            "mode": mode,
            "sigma": float(getattr(self.args, "stage2_residual_carrier_sigma", 2.0)),
            "norm": norm,
            "min": float(carrier.min().item()),
            "max": float(carrier.max().item()),
            "mean": float(carrier.mean().item()),
            "mean_abs": float(carrier.abs().mean().item()),
        }
        return carrier

    def _stage2_bright_mask_map(self, pan_image):
        """Build a bright-region mask [1,H,W] from PAN-like image.

        The mask is used only for Stage2 densification/initialization routing.
        Default behavior is disabled.
        """
        mode = str(getattr(self.args, "stage2_densify_bright_mask_mode", "off")).lower()
        if mode in ("off", "none", "false", "0", ""):
            return None
        pan = pan_image.detach()
        if pan.dim() == 2:
            pan = pan.unsqueeze(0)
        if pan.shape[0] > 1:
            pan = pan.mean(dim=0, keepdim=True)
        sigma = float(getattr(self.args, "stage2_densify_bright_blur_sigma", 3.0))
        if sigma > 0:
            pan = self._gaussian_lowpass_sigma(pan, sigma)
        score = pan.squeeze(0)
        if mode in ("otsu", "pan_otsu", "bright_otsu"):
            s_np = score.detach().cpu().numpy().astype(np.float32)
            hist, bin_edges = np.histogram(s_np.reshape(-1), bins=256)
            hist = hist.astype(np.float64)
            prob = hist / max(hist.sum(), 1.0)
            omega = np.cumsum(prob)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            mu = np.cumsum(prob * centers)
            mu_t = mu[-1]
            denom = omega * (1.0 - omega)
            sigma_b = np.zeros_like(mu)
            valid = denom > 1.0e-12
            sigma_b[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
            thr = float(centers[int(np.argmax(sigma_b))])
        else:
            top_percent = float(getattr(self.args, "stage2_densify_bright_top_percent", 10.0))
            thr = float(
                torch.quantile(
                    score.reshape(-1), 1.0 - top_percent / 100.0
                ).item()
            )
        mask = (score >= thr).to(dtype=pan.dtype).unsqueeze(0)
        self._stage2_last_bright_mask_stats = {
            "mode": mode,
            "threshold": thr,
            "area": float(mask.mean().item()),
        }
        return mask

    def _stage2_residual_regularizer_loss(
        self, residual_raw, gate=None, residual_contribution=None
    ):
        l1_w = float(getattr(self.args, "stage2_residual_l1_weight", 0.0))
        tv_w = float(getattr(self.args, "stage2_residual_tv_weight", 0.0))
        lr_zero_w = float(getattr(self.args, "stage2_residual_lr_zero_weight", 0.0))
        loss = residual_raw.new_tensor(0.0)
        if l1_w > 0:
            loss = loss + l1_w * residual_raw.abs().mean()
        if tv_w > 0:
            loss = loss + tv_w * self._tv_loss(residual_raw)
        if lr_zero_w > 0:
            weight = float(getattr(self.args, "stage2_residual_weight", 1.0))
            if residual_contribution is not None:
                residual_contrib = residual_contribution
            else:
                if gate is None:
                    gate = residual_raw.new_ones(1, residual_raw.shape[-2], residual_raw.shape[-1])
                residual_contrib = weight * gate * residual_raw
            residual_lr = self._ucl_degrade_once(residual_contrib)
            loss = loss + lr_zero_w * F.mse_loss(
                residual_lr, torch.zeros_like(residual_lr)
            )
        return loss

    def _stage2_residual_basis_regularizer_loss(self):
        if not self._stage2_uses_residual_lowrank() or self.stage2_residual_basis_B is None:
            return None
        weight = float(getattr(self.args, "stage2_residual_basis_ortho_weight", 0.0))
        if weight <= 0.0:
            return self.stage2_residual_basis_B().new_tensor(0.0)
        B = self.stage2_residual_basis_B()
        eye = torch.eye(B.shape[0], dtype=B.dtype, device=B.device)
        loss = F.mse_loss(B @ B.t(), eye)
        return weight * loss

    def _apply_stage2_residual_branch(
        self,
        base_hr,
        product_pan,
        img_h,
        img_w,
        tile_bounds,
        render_scale_ratio=1.0,
    ):
        self._stage2_last_residual_raw = None
        self._stage2_last_residual_gate = None
        self._stage2_last_residual_carrier = None
        self._stage2_last_residual_contribution = None
        if not self._stage2_residual_branch_enabled():
            return base_hr
        if self.feat_residual is None or int(self.feat_residual.shape[0]) == 0:
            zero = base_hr.new_zeros(base_hr.shape)
            self._stage2_last_residual_raw = zero
            self._stage2_last_residual_gate = product_pan.new_ones(
                1, product_pan.shape[-2], product_pan.shape[-1]
            )
            self._stage2_last_residual_carrier = None
            self._stage2_last_residual_contribution = zero
            return base_hr

        start = int(getattr(self, "_stage2_base_gaussian_count", self.num_gaussians))
        n_res = int(self.feat_residual.shape[0])
        end = min(int(self.xy.shape[0]), start + n_res)
        if end <= start:
            return base_hr

        train_geo = bool(getattr(self.args, "stage2_densify_train_new_geo", False))
        xy = self.xy[start:end] if train_geo else self.xy.detach()[start:end]
        sc = self._get_scale(render_scale_ratio)[start:end]
        sc = sc if train_geo else sc.detach()
        rot = self.rot[start:end] if train_geo else self.rot.detach()[start:end]
        feat = self.feat_residual[: end - start]
        residual_code, _ = self._rasterize_features(
            xy, sc, rot, feat, img_h, img_w, tile_bounds
        )
        if self._stage2_uses_residual_lowrank():
            B_mod = self._make_stage2_residual_basis()
            if B_mod is None:
                raise RuntimeError("stage2 residual basis is required but missing")
            B = B_mod()
            residual_raw = torch.einsum("rhw,rc->chw", residual_code, B)
        else:
            residual_raw = residual_code
        carrier = self._stage2_residual_carrier(product_pan)
        gate = self._stage2_residual_gate(product_pan)
        weight = float(getattr(self.args, "stage2_residual_weight", 1.0))
        self._stage2_last_residual_raw = residual_raw
        self._stage2_last_residual_gate = gate
        self._stage2_last_residual_carrier = carrier
        if carrier is not None:
            self._stage2_last_residual_contribution = weight * residual_raw * carrier
        else:
            self._stage2_last_residual_contribution = weight * gate * residual_raw
        return (base_hr + self._stage2_last_residual_contribution).clamp(0.0, 1.0)

    def _stage2_current_product_pan(
        self,
        pan_rendered_s1,
        img_h=None,
        img_w=None,
        tile_bounds=None,
        render_scale_ratio=1.0,
        fixed_product_pan=None,
    ):
        mode = self._stage2_product_pan_mode()
        if mode == "current_render":
            if img_h is None or img_w is None:
                img_h, img_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
            if tile_bounds is None:
                tile_bounds = self._stage2_tile_bounds(img_h, img_w)
            with torch.no_grad():
                product_pan, _ = self._rasterize_features(
                    self.xy.detach(),
                    self._get_scale(render_scale_ratio).detach(),
                    self.rot.detach(),
                    self.feat_pan.detach(),
                    img_h,
                    img_w,
                    tile_bounds,
                )
                product_pan = product_pan.clamp(0.0, 1.0)
            self._stage2_last_product_pan = product_pan.detach()
            return product_pan.detach()
        product_pan = fixed_product_pan if fixed_product_pan is not None else pan_rendered_s1
        self._stage2_last_product_pan = product_pan.detach()
        return product_pan.detach()

    def _stage2_log_ratio_init(self, ratio_init_img, init_ratio, product_pan_ucl):
        if ratio_init_img is not None:
            self.worklog.info(
                "Stage2 ratio init | "
                f"mode={self._stage2_base_ratio_init_mode()} "
                f"img_min={float(ratio_init_img.min().item()):.6f} "
                f"img_max={float(ratio_init_img.max().item()):.6f} "
                f"img_mean={float(ratio_init_img.mean().item()):.6f} "
                f"feat_min={float(init_ratio.min().item()):.6f} "
                f"feat_max={float(init_ratio.max().item()):.6f} "
                f"feat_mean={float(init_ratio.mean().item()):.6f}"
            )
        else:
            self.worklog.info(
                f"Stage2 ratio init | mode={self._stage2_base_ratio_init_mode()} "
                f"feat_min={float(init_ratio.min().item()):.6f} "
                f"feat_max={float(init_ratio.max().item()):.6f} "
                f"feat_mean={float(init_ratio.mean().item()):.6f}"
            )
        init_stats = getattr(self, "_stage2_last_ratio_init_stats", {})
        if "lr_sat" in init_stats:
            self.worklog.info(
                f"Stage2 ratio init lr_ratio | clamp_saturation={init_stats['lr_sat']:.6f}"
            )
        tb_hr = self._stage2_tile_bounds(
            self.ref_images.shape[-2], self.ref_images.shape[-1]
        )
        with torch.no_grad():
            ms_hr0, _, _, _ = self._render_ucl_current_output(
                self.ref_images.shape[-2],
                self.ref_images.shape[-1],
                tb_hr,
                product_pan_ucl,
                product_pan_ucl,
                render_scale_ratio=1.0,
            )
            ms_hr0_down = self._ucl_degrade_once(ms_hr0)
            mse0 = F.mse_loss(ms_hr0_down, self.gt_images).clamp_min(1.0e-12)
            self.worklog.info(
                f"Stage2 init LR_PSNR: {float((-10.0 * torch.log10(mse0)).item()):.4f}"
            )

    def _stage2_rr_constructed_loss(
        self,
        product_pan_ucl,
        pan_rendered_s1,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
    ):
        """Reduced-resolution constructed supervision.

        Renders the current Stage2 output directly at LRMS resolution [C,ms_h,ms_w]
        and compares it with observed LRMS. Weight defaults to zero, preserving
        baseline behavior.
        """
        weight = float(getattr(self.args, "stage2_rr_loss_weight", 0.0))
        if weight <= 0.0:
            zero = product_pan_ucl.new_tensor(0.0)
            self._stage2_last_rr_stats = {
                "loss": 0.0,
                "psnr": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
            }
            return zero, zero, zero, None
        tb_lr = self._stage2_tile_bounds(ms_h, ms_w)
        product_pan_lr = F.interpolate(
            product_pan_ucl.detach().unsqueeze(0),
            size=(ms_h, ms_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        render_scale_ratio = 0.5 * (float(ms_h) / float(pan_h) + float(ms_w) / float(pan_w))
        x_rr, ratio_rr, base_rr, _ = self._render_ucl_current_output(
            ms_h,
            ms_w,
            tb_lr,
            product_pan_lr,
            pan_rendered_s1,
            render_scale_ratio=render_scale_ratio,
        )
        loss_type = str(getattr(self.args, "stage2_rr_loss_type", "l1")).lower()
        if loss_type in ("mse", "l2"):
            raw_loss = F.mse_loss(x_rr, self.gt_images)
        else:
            raw_loss = F.l1_loss(x_rr, self.gt_images)
        mse = F.mse_loss(x_rr.detach(), self.gt_images.detach()).clamp_min(1.0e-12)
        rr_psnr = -10.0 * torch.log10(mse)
        self._stage2_last_rr_stats = {
            "loss": float(raw_loss.detach().item()),
            "psnr": float(rr_psnr.item()),
            "min": float(x_rr.detach().min().item()),
            "max": float(x_rr.detach().max().item()),
            "mean": float(x_rr.detach().mean().item()),
        }
        return weight * raw_loss, raw_loss, rr_psnr, x_rr

    def _render_ucl_current_output(
        self,
        img_h,
        img_w,
        tile_bounds,
        product_pan,
        pan_rendered_s1,
        render_scale_ratio=1.0,
    ):
        product_pan = self._stage2_current_product_pan(
            pan_rendered_s1,
            img_h=img_h,
            img_w=img_w,
            tile_bounds=tile_bounds,
            render_scale_ratio=render_scale_ratio,
            fixed_product_pan=product_pan,
        )
        ucl_local_ratio = self._stage2_uses_ucl_local_ratio()
        ucl_scaffold_mlp = self._stage2_uses_ucl_scaffold_mlp()
        ucl_independent_geo = bool(
            getattr(self.args, "stage2_ucl_independent_geo", True)
        )

        if self.stage2_ratio_inr is not None:
            ratio_hr = self._render_ucl_inr_ratio(
                img_h, img_w, tile_bounds, render_scale_ratio=render_scale_ratio
            )
            ratio_hr = self._apply_ucl_dense_residuals(ratio_hr, product_pan)
            product_hr = ratio_hr * product_pan
            product_hr = self._apply_ucl_pan_gap_residual(
                product_hr, ratio_hr, product_pan, img_h, img_w
            )
            product_hr = self._apply_ucl_lig_gap_residual(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            product_hr = self._apply_ucl_pan_detail(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            return product_hr.clamp(0.0, 1.0), ratio_hr, product_hr, None

        if ucl_local_ratio:
            ratio_hr, product_hr = self._render_ucl_local_ratio(
                img_h,
                img_w,
                tile_bounds,
                product_pan,
                render_scale_ratio=render_scale_ratio,
            )
            ratio_hr = self._apply_ucl_dense_residuals(ratio_hr, product_pan)
            product_hr = ratio_hr * product_pan
            product_hr = self._apply_ucl_pan_gap_residual(
                product_hr, ratio_hr, product_pan, img_h, img_w
            )
            product_hr = self._apply_ucl_lig_gap_residual(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            product_hr = self._apply_ucl_pan_detail(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            return product_hr.clamp(0.0, 1.0), ratio_hr, product_hr, None

        if ucl_scaffold_mlp:
            ratio_hr, product_hr, pan_new = self._render_ucl_scaffold_mlp(
                img_h,
                img_w,
                tile_bounds,
                product_pan=product_pan,
                render_scale_ratio=render_scale_ratio,
            )
            ratio_hr = self._apply_ucl_dense_residuals(ratio_hr, product_pan)
            product_hr = ratio_hr * product_pan
            product_hr = self._apply_ucl_pan_gap_residual(
                product_hr, ratio_hr, product_pan, img_h, img_w
            )
            product_hr = self._apply_ucl_lig_gap_residual(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            product_hr = self._apply_ucl_pan_detail(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            return product_hr.clamp(0.0, 1.0), ratio_hr, product_hr, pan_new

        if ucl_independent_geo:
            sc_ratio = (
                render_scale_ratio / self.scale_ratio.clamp_min(1.0e-8)
            )
            ratio_hr, product_hr = self._render_ucl_product(
                self.xy_ratio,
                sc_ratio,
                self.rot_ratio,
                self._stage2_effective_ratio_features(),
                img_h,
                img_w,
                tile_bounds,
                product_pan,
            )
            ratio_hr = self._apply_ucl_dense_residuals(ratio_hr, product_pan)
            product_hr = ratio_hr * product_pan
            product_hr = self._apply_ucl_pan_gap_residual(
                product_hr, ratio_hr, product_pan, img_h, img_w
            )
            product_hr = self._apply_ucl_lig_gap_residual(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            product_hr = self._apply_ucl_pan_detail(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            return product_hr.clamp(0.0, 1.0), ratio_hr, product_hr, None

        sc_ratio = self._get_scale(render_scale_ratio).detach()
        if bool(getattr(self.args, "stage2_ucl_joint_ratio_pan", False)):
            joint_feat = torch.cat([
                self._stage2_effective_ratio_features(),
                self.feat_pan.detach(),
            ], dim=1)
            joint_hr, _ = self._rasterize_features(
                self.xy.detach(),
                sc_ratio,
                self.rot.detach(),
                joint_feat,
                img_h,
                img_w,
                tile_bounds,
            )
            raw_dim = int(self.feat_ratio.shape[1])
            ratio_hr = self._expand_ucl_ratio_image(joint_hr[:raw_dim])
            product_pan = joint_hr[raw_dim:raw_dim + 1].clamp(0.0, 1.0)
            product_hr = ratio_hr * product_pan
            output_base = getattr(self, "stage2_ucl_output_ratio_base", None)
            if output_base is not None:
                ratio_hr = (ratio_hr + output_base.to(dtype=ratio_hr.dtype, device=ratio_hr.device)).clamp_min(0.0)
                product_hr = ratio_hr * product_pan
            ratio_hr = self._apply_ucl_dense_residuals(ratio_hr, product_pan)
            product_hr = ratio_hr * product_pan
            product_hr = self._apply_ucl_pan_gap_residual(
                product_hr, ratio_hr, product_pan, img_h, img_w
            )
            product_hr = self._apply_ucl_lig_gap_residual(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            product_hr = self._apply_ucl_pan_detail(
                product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
            )
            base_hr = self._render_ucl_additive_base(sc_ratio, img_h, img_w, tile_bounds)
            if base_hr is not None:
                ms_hr = (base_hr + product_hr).clamp(0.0, 1.0)
            else:
                ms_hr = product_hr.clamp(0.0, 1.0)
            return ms_hr, ratio_hr, product_hr, None

        ratio_hr, product_hr = self._render_ucl_product(
            self.xy.detach(),
            sc_ratio,
            self.rot.detach(),
            self._stage2_effective_ratio_features(),
            img_h,
            img_w,
            tile_bounds,
            product_pan,
            pan_feat=self.feat_pan.detach(),
        )
        if self._stage2_ratio_split_enabled():
            ratio_hr, product_hr = self._render_stage2_ratio_split(
                sc_ratio,
                img_h,
                img_w,
                tile_bounds,
                product_pan,
            )
        output_base = getattr(self, "stage2_ucl_output_ratio_base", None)
        if output_base is not None:
            ratio_hr = (ratio_hr + output_base.to(dtype=ratio_hr.dtype, device=ratio_hr.device)).clamp_min(0.0)
            product_hr = ratio_hr * product_pan
        ratio_hr = self._apply_ucl_dense_residuals(ratio_hr, product_pan)
        if self._stage2_ratio_split_enabled() and output_base is None:
            product_hr = self._stage2_ratio_split_product(ratio_hr, product_pan)
        else:
            product_hr = ratio_hr * product_pan
        product_hr = self._apply_ucl_pan_gap_residual(
            product_hr, ratio_hr, product_pan, img_h, img_w
        )
        product_hr = self._apply_ucl_lig_gap_residual(
            product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
        )
        product_hr = self._apply_ucl_pan_detail(
            product_hr, product_pan, img_h, img_w, tile_bounds, render_scale_ratio
        )
        base_hr = self._render_ucl_additive_base(sc_ratio, img_h, img_w, tile_bounds)
        if base_hr is not None:
            ms_hr = (base_hr + product_hr).clamp(0.0, 1.0)
        else:
            ms_hr = product_hr.clamp(0.0, 1.0)
        ms_hr = self._apply_stage2_residual_branch(
            ms_hr,
            product_pan,
            img_h,
            img_w,
            tile_bounds,
            render_scale_ratio=render_scale_ratio,
        )
        return ms_hr, ratio_hr, product_hr, None

    def _render_ucl_multiscale_output(
        self, img_h, img_w, full_h, full_w, pan_rendered_s1
    ):
        tb = self._stage2_tile_bounds(img_h, img_w)
        render_scale_ratio = 0.5 * (
            float(img_h) / float(full_h) + float(img_w) / float(full_w)
        )
        pan_source = getattr(
            self.args,
            "stage2_ucl_multilevel_product_pan",
            getattr(self.args, "stage2_ucl_product_pan", "rendered"),
        )
        product_pan = self._select_ucl_pan_image_at(
            pan_source, pan_rendered_s1, img_h, img_w, full_h, full_w
        )
        return self._render_ucl_current_output(
            img_h,
            img_w,
            tb,
            product_pan,
            pan_rendered_s1,
            render_scale_ratio=render_scale_ratio,
        )

    def _stage2_ucl_multilevel_loss(
        self, ms_hr, pan_rendered_s1, pan_h, pan_w, ms_h, ms_w
    ):
        reduced_w = float(getattr(self.args, "stage2_ucl_reduced_weight", 0.0))
        lower_w = float(getattr(self.args, "stage2_ucl_lower_weight", 0.0))
        cross_w = float(getattr(self.args, "stage2_ucl_cross_scale_weight", 0.0))
        loss_reduced = ms_hr.new_tensor(0.0)
        loss_lower = ms_hr.new_tensor(0.0)
        loss_cross = ms_hr.new_tensor(0.0)

        if reduced_w <= 0 and lower_w <= 0 and cross_w <= 0:
            return loss_reduced, loss_lower, loss_cross

        ms_direct, _, _, _ = self._render_ucl_multiscale_output(
            ms_h, ms_w, pan_h, pan_w, pan_rendered_s1
        )
        if reduced_w > 0:
            loss_reduced = reduced_w * F.mse_loss(ms_direct, self.gt_images)

        if cross_w > 0:
            loss_cross = loss_cross + cross_w * F.mse_loss(
                self._ucl_degrade_once(ms_hr), ms_direct
            )

        if lower_w > 0 or cross_w > 0:
            ds = max(1, int(round(self.image_ratio)))
            low_h = max(1, int(round(float(ms_h) / float(ds))))
            low_wid = max(1, int(round(float(ms_w) / float(ds))))
            ms_low, _, _, _ = self._render_ucl_multiscale_output(
                low_h, low_wid, pan_h, pan_w, pan_rendered_s1
            )
            gt_low = self._ucl_degrade_once(self.gt_images)
            if gt_low.shape[-2:] != (low_h, low_wid):
                gt_low = F.interpolate(
                    gt_low.unsqueeze(0), size=(low_h, low_wid), mode="area"
                ).squeeze(0)
            if lower_w > 0:
                loss_lower = lower_w * F.mse_loss(ms_low, gt_low)
            if cross_w > 0:
                loss_cross = loss_cross + cross_w * F.mse_loss(
                    self._ucl_degrade_once(ms_direct), ms_low
                )

        return loss_reduced, loss_lower, loss_cross

    def _stage2_uses_detail_branch(self):
        mode = str(getattr(self.args, "stage2_g_detail_mode", "none")).lower()
        return (
            mode in ("highpass", "output_highpass", "detail_highpass")
            or self._stage2_uses_pan_residual_gain()
            or self._stage2_uses_kernel_affine_pan()
        )

    def _phat_from_pan_image(self):
        """Band-wise histogram-matched input PAN, matching PSDIP's P_hat."""
        pan = self.ref_images
        pan_mu = pan.mean()
        pan_std = pan.std().clamp(min=1e-6)
        ms_mu = self.gt_images.mean(dim=(-2, -1), keepdim=True)
        ms_std = self.gt_images.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        phat = (pan.repeat(self.output_dim, 1, 1) - pan_mu) * (ms_std / pan_std)
        phat = phat + ms_mu
        floor = float(getattr(self.args, "stage2_phat_floor", 1.0e-2))
        return phat + floor

    def _phat_from_pan_features(self, pan_feat):
        """Band-wise histogram-matched PAN features."""
        pan_mu = self.ref_images.mean()
        pan_std = self.ref_images.std().clamp(min=1e-6)
        ms_mu = self.gt_images.mean(dim=(-2, -1))
        ms_std = self.gt_images.std(dim=(-2, -1)).clamp(min=1e-6)
        if pan_feat.dim() != 2 or pan_feat.shape[1] != 1:
            raise ValueError("pan_feat must have shape [N, 1]")
        phat = (pan_feat - pan_mu) * (ms_std / pan_std).view(1, -1)
        phat = phat + ms_mu.view(1, -1)
        floor = float(getattr(self.args, "stage2_phat_floor", 1.0e-2))
        return phat + floor

    def _stage2_base_features(self):
        mode = self._stage2_fuse_mode()
        if (
            mode in ("base_only", "x_base", "none")
            or self._stage2_uses_pan_residual_gain()
            or self._stage2_uses_kernel_affine_pan()
        ):
            return torch.ones(
                self.num_gaussians, self.output_dim,
                dtype=self.dtype, device=self.device,
            )
        if mode in (
            "g_times_input_phat", "input_phat", "psdip_phat",
            "render_g_times_input_phat", "dense_g_times_input_phat",
            "render_g_times_phat_image", "dense_phat",
        ):
            return self._sample_features_from_image(
                self._phat_from_pan_image(), self.xy.detach()
            )
        if mode in ("g_times_phat", "g_times_p_hat", "extended_pan", "phat"):
            return self._phat_from_pan_features(self.feat_pan.detach())
        return self.feat_pan.detach()

    def _stage2_base_features_at(self, positions):
        mode = self._stage2_fuse_mode()
        if (
            mode in ("base_only", "x_base", "none")
            or self._stage2_uses_pan_residual_gain()
            or self._stage2_uses_kernel_affine_pan()
        ):
            return torch.ones(
                positions.shape[0], self.output_dim,
                dtype=self.dtype, device=self.device,
            )
        if mode in (
            "g_times_input_phat", "input_phat", "psdip_phat",
            "render_g_times_input_phat", "dense_g_times_input_phat",
            "render_g_times_phat_image", "dense_phat",
        ):
            return self._sample_features_from_image(
                self._phat_from_pan_image(), positions.detach()
            )
        pan_feat = self._sample_features_from_image(self.ref_images, positions.detach())
        if mode in ("g_times_phat", "g_times_p_hat", "extended_pan", "phat"):
            return self._phat_from_pan_features(pan_feat)
        return pan_feat

    def _stage2_base_features_safe(self):
        base = self._stage2_base_features()
        return self._safe_base_features(base)

    def _fused_feat_ms(self):
        """Gaussian-space coefficient fusion: G*P or PSDIP-style G*P_hat."""
        return self._stage2_coeff_features() * self._stage2_base_features()

    def _stage2_coeff_features(self):
        if self.feat_ms is None:
            raise RuntimeError("feat_ms is disabled for this UCL ratio configuration")
        return self.feat_ms[:, : self.output_dim]

    def _rasterize_features(
        self, xy, scale, rot, feat, img_h, img_w, tile_bounds, benchmark=False,
        topk_norm_override=None,
    ):
        begin = perf_counter()
        xy_proj, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            xy, scale, rot, img_h, img_w, tile_bounds
        )
        if not self.disable_tiles:
            topk_norm = (
                not self.disable_topk_norm
                if topk_norm_override is None
                else bool(topk_norm_override)
            )
            out_image = rasterize_gaussians_sum(
                xy_proj, radii, conics, num_tiles_hit, feat,
                img_h, img_w, self.block_h, self.block_w,
                topk_norm,
            )
        else:
            out_image = rasterize_gaussians_no_tiles(
                xy_proj, conics, feat, img_h, img_w
            )
        render_time = perf_counter() - begin
        if benchmark:
            return render_time
        feat_dim = feat.shape[1]
        out_image = (
            out_image.view(-1, img_h, img_w, feat_dim)
            .permute(0, 3, 1, 2).contiguous()
        )
        return out_image.squeeze(dim=0), render_time

    # -- Optimization setup ----------------------------------------------------

    def _init_optimization(self, args):
        self.disable_tiles = args.disable_tiles
        self.max_steps = args.max_steps
        self.pos_lr = args.pos_lr
        self.scale_lr = args.scale_lr
        self.rot_lr = args.rot_lr
        self.feat_lr = args.feat_lr
        params = [
            {"params": self.xy, "lr": self.pos_lr},
            {"params": self.scale, "lr": self.scale_lr},
            {"params": self.rot, "lr": self.rot_lr},
            {"params": self.feat_pan, "lr": self.feat_lr},
        ]
        if self.feat_ms is not None:
            params.append({"params": self.feat_ms, "lr": self.feat_lr})
        if self.feat_x is not None:
            params.append({"params": self.feat_x, "lr": self.feat_lr})
        if self.stage2_decoder is not None:
            params.append({"params": self.stage2_decoder.parameters(), "lr": self.feat_lr})
        self.optimizer = torch.optim.Adam(params)

    def _init_pos_scale_feat(self, args):
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio
        self.pixel_xy = (
            get_grid(h=self.img_h, w=self.img_w)
            .to(dtype=self.dtype, device=self.device)
            .reshape(-1, 2)
        )
        self.worklog.info(
            f"Init Gaussians | mode={self.init_mode} | initial={self.num_gaussians:d} "
            f"| target={self.total_num_gaussians:d}"
        )
        if self.skip_unused_ms_x:
            self.worklog.info("Init Gaussians | UCL ratio path: feat_ms/feat_x disabled")
        with torch.no_grad():
            scaffold_parent_init = str(
                getattr(self.args, "stage1_scaffold_parent_init", "default")
            ).lower()
            if (
                bool(getattr(self.args, "stage1_scaffold_expand", False))
                and scaffold_parent_init in ("pan_gradient", "pan", "pan_hf")
            ):
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                gy, gx = compute_image_gradients(
                    np.power(self.ref_images.detach().cpu().numpy(), 1.0 / self.gamma)
                )
                g_norm = np.hypot(gy, gx).astype(np.float32)
                max_val = g_norm.max()
                if max_val > 1e-12:
                    g_norm /= max_val
                score, prob = self._coverage_aware_prob(
                    score_map=np.power(g_norm.reshape(-1), 2.0),
                    pan_h=pan_h,
                    pan_w=pan_w,
                    xy=None,
                    coverage_alpha=float(
                        getattr(self.args, "stage1_init_coverage_alpha", 0.0)
                    ),
                    coverage_sigma_px=float(
                        getattr(self.args, "stage1_init_coverage_sigma_px", 4.0)
                    ),
                    density_gamma=float(
                        getattr(self.args, "stage1_init_density_gamma", 1.0)
                    ),
                )
                selected = self._sample_spatial_indices(
                    score_map=score.reshape(pan_h, pan_w).detach().cpu().numpy(),
                    pixel_prob=prob.detach().cpu().numpy(),
                    h=pan_h,
                    w=pan_w,
                    n=int(self.num_gaussians),
                    uniform_ratio=float(
                        getattr(self.args, "stage1_init_uniform_ratio", self.init_random_ratio)
                    ),
                    block=int(getattr(self.args, "stage1_init_block", 0)),
                    min_per_block=int(getattr(self.args, "stage1_init_min_per_block", 0)),
                    quota_power=float(getattr(self.args, "stage1_init_quota_power", 1.0)),
                    uniform_within=bool(
                        getattr(self.args, "stage1_init_uniform_within", False)
                    ),
                    available_mask=None,
                    max_per_block=int(getattr(self.args, "stage1_init_max_per_block", 0)),
                )
                self.xy.copy_(pan_grid[selected])
            elif self.init_mode == "gradient":
                self._compute_gmap()
                self.xy.copy_(self._sample_pos(prob=self.image_gradients))
            elif self.init_mode in ("pan_gradient", "pan_highpass"):
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                pan_np = self.ref_images.detach().cpu().numpy()
                gy, gx = compute_image_gradients(np.power(pan_np, 1.0 / self.gamma))
                g_norm = np.hypot(gy, gx).astype(np.float32)
                sigma = float(getattr(self.args, "init_pan_hf_sigma", 0.0))
                if sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    g_lp = gaussian_filter(g_norm, sigma=sigma)
                    g_norm = np.clip(g_norm - g_lp, 0.0, None)
                g_flat = g_norm.reshape(-1)
                prob = np.power(g_flat, 2.0)
                s = float(prob.sum())
                if s < 1e-12:
                    prob = np.ones_like(g_flat) / float(g_flat.size)
                else:
                    prob = prob / s
                n_init = int(self.num_gaussians)
                alloc_block = int(getattr(self.args, "init_alloc_block", 0))
                if alloc_block > 0:
                    selected = self._sample_block_alloc_indices(
                        score_map=g_norm,
                        pixel_prob=prob,
                        h=pan_h,
                        w=pan_w,
                        n=n_init,
                        block=alloc_block,
                        random_ratio=float(getattr(self.args, "init_random_ratio", 0.3)),
                        min_per_block=int(getattr(self.args, "init_alloc_min_per_block", 1)),
                        quota_power=float(getattr(self.args, "init_alloc_quota_power", 1.0)),
                        uniform_within=bool(
                            getattr(self.args, "init_alloc_uniform_within", False)
                        ),
                        max_per_block=int(getattr(self.args, "init_alloc_max_per_block", 0)),
                    )
                else:
                    num_random = round(self.init_random_ratio * n_init)
                    num_prob = n_init - num_random
                    selected = []
                    if num_random > 0:
                        selected.append(np.random.choice(pan_h * pan_w, num_random, replace=False))
                    if num_prob > 0:
                        selected.append(np.random.choice(pan_h * pan_w, num_prob, replace=False, p=prob))
                    selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
                self.xy.copy_(pan_grid[selected])
            elif self.init_mode in (
                "pan_saliency",
                "saliency",
                "pan_sg_alloc",
                "sg_alloc",
                "pan_entropy_alloc",
                "entropy_alloc",
            ):
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                pan_np = np.power(self.ref_images.detach().cpu().numpy(), 1.0 / self.gamma)
                gy, gx = compute_image_gradients(pan_np)
                grad = np.hypot(gy, gx).astype(np.float32)
                if grad.max() > 1.0e-12:
                    grad = grad / grad.max()
                if self.init_mode in ("pan_saliency", "saliency"):
                    sal_target = str(
                        getattr(self.args, "init_saliency_target", "pan")
                    ).lower()
                    if sal_target in ("lms", "bicubic", "lrms"):
                        target_img = self._to_pan_estimate(self._bicubic_hr_target())
                    elif sal_target in ("mtf_glp", "trad", "traditional"):
                        target_img = self._to_pan_estimate(
                            self._build_stage2_target_at(pan_h, pan_w, log=False)
                        )
                    else:
                        target_img = self.ref_images
                    score_map = self._spectral_residual_saliency_map(
                        target_img.detach().cpu().numpy()
                    )
                    selected = self._sample_saliency_indices(
                        saliency=score_map,
                        h=pan_h,
                        w=pan_w,
                        n=int(self.num_gaussians),
                        saliency_ratio=float(
                            getattr(self.args, "init_saliency_ratio", 0.7)
                        ),
                        saliency_power=float(
                            getattr(self.args, "init_saliency_power", 1.0)
                        ),
                    )
                    self.xy.copy_(pan_grid[selected])
                else:
                    if self.init_mode in ("pan_entropy_alloc", "entropy_alloc"):
                        score_map = self._block_entropy_score_map(pan_np)
                    else:
                        score_map = grad
                    selected = self._sample_block_alloc_indices(
                        score_map=score_map,
                        pixel_prob=np.power(grad.reshape(-1), 2.0),
                        h=pan_h,
                        w=pan_w,
                        n=int(self.num_gaussians),
                        block=int(getattr(self.args, "init_alloc_block", 16)),
                        random_ratio=float(getattr(self.args, "init_random_ratio", 0.3)),
                        min_per_block=int(getattr(self.args, "init_alloc_min_per_block", 1)),
                        quota_power=float(getattr(self.args, "init_alloc_quota_power", 1.0)),
                        uniform_within=bool(
                            getattr(
                                self.args,
                                "init_alloc_uniform_within",
                                self.init_mode in ("pan_entropy_alloc", "entropy_alloc"),
                            )
                        ),
                    )
                    self.xy.copy_(pan_grid[selected])
            elif self.init_mode in ("hrms_trad_gradient", "trad_hr_gradient", "mtf_glp_gradient"):
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                trad_hr = self._build_stage2_target_at(pan_h, pan_w, log=False)
                trad_pan = self._to_pan_estimate(trad_hr)
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                gy, gx = compute_image_gradients(
                    np.power(trad_pan.detach().cpu().numpy(), 1.0 / self.gamma)
                )
                g_norm = np.hypot(gy, gx).astype(np.float32)
                max_val = g_norm.max()
                if max_val > 1e-12:
                    g_norm /= max_val
                prob = np.power(g_norm.reshape(-1), 2.0)
                s = float(prob.sum())
                if s < 1e-12:
                    prob = np.ones_like(prob) / float(prob.size)
                else:
                    prob = prob / s
                n_init = int(self.num_gaussians)
                num_random = round(self.init_random_ratio * n_init)
                num_prob = n_init - num_random
                selected = []
                if num_random > 0:
                    selected.append(np.random.choice(pan_h * pan_w, num_random, replace=False))
                if num_prob > 0:
                    selected.append(np.random.choice(pan_h * pan_w, num_prob, replace=False, p=prob))
                selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
                self.xy.copy_(pan_grid[selected])
            elif self.init_mode == "pan_highpass_stratified":
                # Hybrid A: uniform PAN-HR coverage + PAN-HF bias remainder.
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                pan_np = self.ref_images.detach().cpu().numpy()
                gy, gx = compute_image_gradients(np.power(pan_np, 1.0 / self.gamma))
                g_norm = np.hypot(gy, gx).astype(np.float32)
                sigma = float(getattr(self.args, "init_pan_hf_sigma", 0.0))
                if sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    g_lp = gaussian_filter(g_norm, sigma=sigma)
                    g_norm = np.clip(g_norm - g_lp, 0.0, None)
                g_flat = g_norm.reshape(-1)
                prob = np.power(g_flat, 2.0)
                s = float(prob.sum())
                if s < 1e-12:
                    prob = np.ones_like(g_flat) / float(g_flat.size)
                else:
                    prob = prob / s
                n_init = int(self.num_gaussians)
                # Reuse init_random_ratio as "uniform PAN-HR coverage" share.
                num_uniform = round(self.init_random_ratio * n_init)
                num_prob = n_init - num_uniform
                selected = []
                if num_uniform > 0:
                    selected.append(np.random.choice(pan_h * pan_w, num_uniform, replace=False))
                if num_prob > 0:
                    selected.append(np.random.choice(pan_h * pan_w, num_prob, replace=False, p=prob))
                selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
                self.xy.copy_(pan_grid[selected])
            elif self.init_mode == "ms_lr_plus_pan_hf":
                # Hybrid B: copy baseline MS_LR full-grid coverage, then add
                # remaining Gaussians on PAN-HR grid biased by |∇PAN|².
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                ms_pixels = int(self.num_pixels)
                n_init = int(self.num_gaussians)
                ms_take = min(ms_pixels, n_init)
                ms_xy = self.pixel_xy.detach().clone()
                # Cover every MS_LR pixel deterministically (matches baseline).
                ms_pos = ms_xy[:ms_take] if ms_take == ms_pixels else ms_xy[
                    np.random.choice(ms_pixels, ms_take, replace=False)
                ]
                if n_init > ms_take:
                    pan_np = self.ref_images.detach().cpu().numpy()
                    gy, gx = compute_image_gradients(np.power(pan_np, 1.0 / self.gamma))
                    g_norm = np.hypot(gy, gx).astype(np.float32)
                    sigma = float(getattr(self.args, "init_pan_hf_sigma", 0.0))
                    if sigma > 0:
                        from scipy.ndimage import gaussian_filter
                        g_lp = gaussian_filter(g_norm, sigma=sigma)
                        g_norm = np.clip(g_norm - g_lp, 0.0, None)
                    g_flat = g_norm.reshape(-1)
                    prob = np.power(g_flat, 2.0)
                    s = float(prob.sum())
                    if s < 1e-12:
                        prob = np.ones_like(g_flat) / float(g_flat.size)
                    else:
                        prob = prob / s
                    extra = n_init - ms_take
                    sel_extra = np.random.choice(pan_h * pan_w, extra, replace=False, p=prob)
                    pan_pos = pan_grid[sel_extra]
                    self.xy.copy_(torch.cat([ms_pos, pan_pos], dim=0))
                else:
                    self.xy.copy_(ms_pos)
            elif self.init_mode == "pan_uniform":
                # Diagnostic: uniform random sampling on PAN-HR grid (no HF bias).
                # Tests whether topk-neighbor uniformity is what makes baseline good.
                pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
                pan_grid = (
                    get_grid(h=pan_h, w=pan_w)
                    .to(dtype=self.dtype, device=self.device)
                    .reshape(-1, 2)
                )
                selected = np.random.choice(
                    pan_h * pan_w, int(self.num_gaussians), replace=False
                )
                self.xy.copy_(pan_grid[selected])
            elif self.init_mode == "random":
                selected = np.random.choice(
                    self.num_pixels, self.num_gaussians, replace=False
                )
                self.xy.copy_(self.pixel_xy.detach().clone()[selected])
            else:
                raise ValueError(
                    "init_mode must be 'gradient', 'pan_gradient', 'pan_highpass', "
                    "'pan_highpass_stratified', 'ms_lr_plus_pan_hf', "
                    "'hrms_trad_gradient', 'pan_uniform', or 'random'"
                )

            self.scale.fill_(
                self.init_scale if self.disable_inverse_scale else 1.0 / self.init_scale
            )
            self.feat_pan.copy_(
                self._get_target_features(positions=self.xy, level="pan")
            )
            if not self.disable_color_init and self.feat_ms is not None and self.feat_x is not None:
                ms_init = self._get_target_features(positions=self.xy, level="ms")
                self._copy_stage2_coeff_features(ms_init / self._stage2_base_features_safe())
                self.feat_x.copy_(ms_init)

    def _expand_pan_scaffold_gaussians(self):
        target_n = int(self.total_num_gaussians)
        add_num = target_n - int(self.num_gaussians)
        if add_num <= 0:
            self.worklog.info(
                f"PAN scaffold skipped: current={self.num_gaussians:d}, target={target_n:d}."
            )
            return

        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        jitter_scale = float(getattr(self.args, "stage1_scaffold_jitter_scale", 0.75))
        child_scale_mult = float(
            getattr(self.args, "stage1_scaffold_child_scale_mult", 0.65)
        )
        rot_jitter = float(getattr(self.args, "stage1_scaffold_rot_jitter", 0.10))
        score_power = float(getattr(self.args, "stage1_scaffold_score_power", 1.5))

        with torch.no_grad():
            pan_hp = torch.abs(self._highpass(self.ref_images))
            parent_score = self._sample_features_from_image(
                pan_hp, self.xy.detach()
            ).mean(dim=1).clamp_min(1.0e-8)
            probs = torch.pow(parent_score, score_power)
            probs = probs / probs.sum().clamp_min(1.0e-8)
            parent_idx = torch.multinomial(probs, add_num, replacement=True)

            parent_xy = self.xy.detach()[parent_idx]
            parent_scale_raw = self.scale.detach()[parent_idx]
            parent_scale_actual = self._get_scale(1.0).detach()[parent_idx]
            parent_rot = self.rot.detach()[parent_idx]

            std_x = jitter_scale * parent_scale_actual[:, 0] / float(pan_w)
            std_y = jitter_scale * parent_scale_actual[:, 1] / float(pan_h)
            noise = torch.randn(add_num, 2, dtype=self.dtype, device=self.device)
            noise[:, 0] = noise[:, 0] * std_x
            noise[:, 1] = noise[:, 1] * std_y

            theta = parent_rot[:, 0]
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            rot_noise = torch.empty_like(noise)
            rot_noise[:, 0] = cos_t * noise[:, 0] - sin_t * noise[:, 1]
            rot_noise[:, 1] = sin_t * noise[:, 0] + cos_t * noise[:, 1]
            new_xy = (parent_xy + rot_noise).clamp(0.0, 1.0)

            if self.disable_inverse_scale:
                new_s = parent_scale_raw * child_scale_mult
            else:
                new_s = parent_scale_raw / max(child_scale_mult, 1.0e-6)
            new_r = parent_rot + rot_jitter * torch.randn(
                add_num, 1, dtype=self.dtype, device=self.device
            )
            new_fp = self._get_target_features(new_xy, level="pan")
            new_ms = self._get_target_features(new_xy, level="ms")
            new_base = self._safe_base_features(self._stage2_base_features_at(new_xy))
            new_fm = self._pack_stage2_coeff_features(new_ms / new_base)
            new_fx = new_ms
            new_fm_detail = torch.zeros(
                add_num, self.output_dim, dtype=self.dtype, device=self.device
            )

            self.xy = nn.Parameter(torch.cat([self.xy.detach(), new_xy], 0))
            self.scale = nn.Parameter(torch.cat([self.scale.detach(), new_s], 0))
            self.rot = nn.Parameter(torch.cat([self.rot.detach(), new_r], 0))
            self.feat_pan = nn.Parameter(torch.cat([self.feat_pan.detach(), new_fp], 0))
            self.feat_ms = nn.Parameter(torch.cat([self.feat_ms.detach(), new_fm], 0))
            self.feat_x = nn.Parameter(torch.cat([self.feat_x.detach(), new_fx], 0))
            self.feat_ms_detail = nn.Parameter(
                torch.cat([self.feat_ms_detail.detach(), new_fm_detail], 0)
            )
            self.num_gaussians = target_n

        self.worklog.info(
            "PAN scaffold expansion: "
            f"parents={target_n - add_num:d}, children={add_num:d}, "
            f"jitter={jitter_scale:.2f}, child_scale={child_scale_mult:.2f}."
        )

    def _refine_pan_scaffold_gaussians(self, steps, pan_h, pan_w, tb_pan):
        steps = int(steps)
        if steps <= 0:
            return
        self._set_trainable_modules(stage=1)
        self.worklog.info(f"PAN scaffold refine: {steps:d} steps after expansion.")
        for step in range(1, steps + 1):
            self.optimizer.zero_grad()
            pan_pred, _ = self._rasterize_features(
                self.xy, self._get_scale(1.0), self.rot, self.feat_pan,
                pan_h, pan_w, tb_pan,
            )
            loss = self.l1_loss_ratio * F.l1_loss(pan_pred, self.ref_images)
            if self.pan_grad_weight > 0:
                loss = loss + self.pan_grad_weight * F.l1_loss(
                    self._pan_grad_mag(pan_pred), self._pan_grad_mag(self.ref_images)
                )
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.xy.clamp_(0.0, 1.0)
            if step == steps or step % max(1, steps // 2) == 0:
                self.worklog.info(
                    f"PAN scaffold refine {step:d}/{steps:d} | Loss: {loss.item():.5f}"
                )

    def _init_aux_pos_scale_feat_(self, target_image, residual_hint=None):
        if residual_hint is not None:
            h, w = residual_hint.shape[-2], residual_hint.shape[-1]
        else:
            h, w = target_image.shape[-2], target_image.shape[-1]

        self.pixel_xy_aux = (
            get_grid(h=h, w=w)
            .to(dtype=self.dtype, device=self.device).reshape(-1, 2)
        )
        num_pixels_aux = h * w

        with torch.no_grad():
            prob = None
            if residual_hint is not None:
                error_map = torch.abs(residual_hint).mean(dim=0).squeeze()
                prob_map = torch.pow(error_map, 2.0)
                prob_sum = prob_map.sum()
                if prob_sum > 1e-7:
                    prob = (prob_map / prob_sum).cpu().numpy().flatten()
            elif self.init_mode == "gradient":
                gy, gx = compute_image_gradients(
                    np.power(target_image.detach().cpu().numpy(), 1.0 / self.gamma)
                )
                g_norm = np.hypot(gy, gx).astype(np.float32)
                max_val = g_norm.max()
                if max_val > 1e-12:
                    g_norm /= max_val
                g_norm = np.power(g_norm.reshape(-1), 2.0)
                prob = g_norm / (g_norm.sum() + 1e-12)

            num_random = round(self.init_random_ratio * self.num_gaussians_aux)
            num_prob = self.num_gaussians_aux - num_random

            selected = []
            if num_random > 0:
                selected.append(np.random.choice(num_pixels_aux, num_random, replace=False))
            if num_prob > 0:
                selected.append(np.random.choice(num_pixels_aux, num_prob, replace=False, p=prob))
            selected_indices = (
                np.concatenate(selected) if selected else np.array([], dtype=np.int64)
            )
            if selected_indices.size == 0:
                selected_indices = np.random.choice(
                    num_pixels_aux, self.num_gaussians_aux, replace=False
                )
            self.xy_aux.copy_(self.pixel_xy_aux.detach().clone()[selected_indices])

            init_sv = self.init_scale * 0.5 if residual_hint is not None else self.init_scale
            self.scale_aux.fill_(
                init_sv if self.disable_inverse_scale else 1.0 / init_sv
            )

            if not self.disable_color_init:
                if residual_hint is not None:
                    self.feat_aux.zero_()
                else:
                    tf = F.grid_sample(
                        target_image.unsqueeze(0),
                        self.xy_aux.detach().clone()[None, None, ...] * 2.0 - 1.0,
                        align_corners=False,
                    )
                    sampled = tf[0, :, 0, :].permute(1, 0)  # [N, C_target]
                    feat_ch = self.feat_aux.shape[1]
                    if sampled.shape[1] > feat_ch:
                        # Average channels to match feat_aux dim (e.g. 4-ch → 1-ch)
                        sampled = sampled.mean(dim=1, keepdim=True)
                    self.feat_aux.copy_(sampled[:, :feat_ch])

    # -- Trainable module switching --------------------------------------------

    def _set_trainable_modules(self, stage):
        if stage == 1:
            self.xy.requires_grad = True
            self.scale.requires_grad = True
            self.rot.requires_grad = True
            self.feat_pan.requires_grad = True
            if self.feat_ms is not None:
                self.feat_ms.requires_grad = False
            if self.feat_x is not None:
                self.feat_x.requires_grad = False
            if self.feat_ms_detail is not None:
                self.feat_ms_detail.requires_grad = False
            if self.stage2_decoder is not None:
                self.stage2_decoder.requires_grad_(False)
            params = [
                {"params": self.xy, "lr": self.pos_lr},
                {"params": self.scale, "lr": self.scale_lr},
                {"params": self.rot, "lr": self.rot_lr},
                {"params": self.feat_pan, "lr": self.feat_lr},
            ]
            self.optimizer = torch.optim.Adam(params) if params else None

        elif stage == 2:
            s2_finetune_geo = bool(
                getattr(self.args, "stage2_finetune_geometry", False)
            )
            train_xy = bool(getattr(self.args, "stage2_finetune_xy", s2_finetune_geo))
            train_scale = bool(
                getattr(self.args, "stage2_finetune_scale", s2_finetune_geo)
            )
            train_rot = bool(getattr(self.args, "stage2_finetune_rot", False))
            self.xy.requires_grad = train_xy
            self.scale.requires_grad = train_scale
            self.rot.requires_grad = train_rot
            self.feat_pan.requires_grad = False
            if self.feat_ms is not None:
                self.feat_ms.requires_grad = True
            if self.feat_x is not None:
                self.feat_x.requires_grad = bool(getattr(self.args, "stage2_use_x_field", False))
            if self.feat_ms_detail is not None:
                self.feat_ms_detail.requires_grad = self._stage2_uses_detail_branch()
            if self.stage2_decoder is not None:
                self.stage2_decoder.requires_grad_(True)
            params = []
            if self.feat_ms is not None:
                params.append({"params": self.feat_ms, "lr": self.feat_lr})
            if bool(getattr(self.args, "stage2_use_x_field", False)) and self.feat_x is not None:
                params.append(
                    {
                        "params": self.feat_x,
                        "lr": self.feat_lr * float(
                            getattr(self.args, "stage2_x_lr_mult", 1.0)
                        ),
                    }
                )
            if self._stage2_uses_detail_branch() and self.feat_ms_detail is not None:
                params.append(
                    {
                        "params": self.feat_ms_detail,
                        "lr": self.feat_lr * float(
                            getattr(self.args, "stage2_detail_lr_mult", 0.2)
                        ),
                    }
                )
            if self.stage2_decoder is not None:
                params.append(
                    {
                        "params": self.stage2_decoder.parameters(),
                        "lr": self.feat_lr * float(
                            getattr(self.args, "stage2_decoder_lr_mult", 1.0)
                        ),
                    }
                )
            if train_xy or train_scale or train_rot:
                geo_lr_mult = float(
                    getattr(self.args, "stage2_geo_lr_mult", 0.1)
                )
                if train_xy:
                    params.append(
                        {"params": self.xy, "lr": self.pos_lr * geo_lr_mult}
                    )
                if train_scale:
                    params.append(
                        {"params": self.scale, "lr": self.scale_lr * geo_lr_mult}
                    )
                if train_rot:
                    params.append(
                        {"params": self.rot, "lr": self.rot_lr * geo_lr_mult}
                    )
            self.optimizer = torch.optim.Adam(params) if params else None

        elif stage == 3:
            for p in [
                self.xy, self.scale, self.rot, self.feat_pan,
                self.feat_ms, self.feat_x, self.feat_ms_detail,
            ]:
                if p is not None:
                    p.requires_grad = False
            if self.stage2_decoder is not None:
                self.stage2_decoder.requires_grad_(False)
            for p in [self.xy_aux, self.scale_aux, self.rot_aux, self.feat_aux]:
                p.requires_grad = True
            self.band_coeff.requires_grad = True
            self.feat_aux_res.requires_grad = True
            res_lr = (
                float(getattr(self.args, "stage3_aux_res_lr_mult", 0.1)) * self.feat_lr
            )
            params = [
                {"params": self.xy_aux, "lr": self.pos_lr},
                {"params": self.scale_aux, "lr": self.scale_lr},
                {"params": self.rot_aux, "lr": self.rot_lr},
                {"params": self.feat_aux, "lr": self.feat_lr},
                {"params": self.band_coeff, "lr": self.feat_lr},
                {"params": self.feat_aux_res, "lr": res_lr},
            ]
            self.optimizer = torch.optim.Adam(params)

    # -- Utility methods -------------------------------------------------------

    def _spectral_residual_saliency_map(self, image):
        """Classical spectral-residual saliency map for image-guided sampling."""
        if image.ndim == 3:
            gray = image.mean(axis=0)
        else:
            gray = image
        gray = np.asarray(gray, dtype=np.float32)
        gray = gray - float(gray.min())
        mx = float(gray.max())
        if mx > 1.0e-12:
            gray = gray / mx

        fft = np.fft.fft2(gray)
        log_amp = np.log(np.abs(fft) + 1.0e-8)
        phase = np.angle(fft)
        from scipy.ndimage import uniform_filter, gaussian_filter

        avg_log_amp = uniform_filter(log_amp, size=3, mode="nearest")
        residual = log_amp - avg_log_amp
        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        sigma = float(getattr(self.args, "init_saliency_blur_sigma", 2.5))
        if sigma > 0:
            sal = gaussian_filter(sal, sigma=sigma)
        sal = sal.astype(np.float32)
        sal = sal - float(sal.min())
        mx = float(sal.max())
        if mx > 1.0e-12:
            sal = sal / mx
        return sal

    def _sample_saliency_indices(self, saliency, h, w, n, saliency_ratio, saliency_power):
        n = int(min(n, h * w))
        sal = np.asarray(saliency, dtype=np.float64).reshape(-1)
        sal = np.power(np.clip(sal, 0.0, None) + 1.0e-8, saliency_power)
        if float(sal.sum()) <= 1.0e-12:
            sal = np.ones_like(sal) / float(sal.size)
        else:
            sal = sal / sal.sum()

        selected = np.zeros(h * w, dtype=bool)
        selected_ids = []
        n_sal = int(round(max(0.0, min(1.0, saliency_ratio)) * n))
        if n_sal > 0:
            chosen = np.random.choice(h * w, n_sal, replace=False, p=sal)
            selected[chosen] = True
            selected_ids.extend(chosen.tolist())
        n_uniform = n - len(selected_ids)
        if n_uniform > 0:
            pool = np.flatnonzero(~selected)
            chosen = np.random.choice(pool, min(n_uniform, pool.size), replace=False)
            selected_ids.extend(chosen.tolist())
        return np.asarray(selected_ids[:n], dtype=np.int64)

    def _block_entropy_score_map(self, image):
        """Local-contrast proxy for entropy-based Gaussian allocation."""
        from scipy.ndimage import uniform_filter

        if image.ndim == 3:
            gray = image.mean(axis=0)
        else:
            gray = image
        block = int(getattr(self.args, "init_alloc_block", 16))
        block = max(3, block)
        mean = uniform_filter(gray, size=block, mode="nearest")
        mean2 = uniform_filter(gray * gray, size=block, mode="nearest")
        std = np.sqrt(np.clip(mean2 - mean * mean, 0.0, None)).astype(np.float32)
        mx = float(std.max())
        if mx > 1.0e-12:
            std = std / mx
        return std

    def _sample_block_alloc_indices(
        self,
        score_map,
        pixel_prob,
        h,
        w,
        n,
        block,
        random_ratio,
        min_per_block,
        quota_power,
        uniform_within,
        available_mask=None,
        max_per_block=0,
    ):
        """Sample PAN-HR positions with per-block capacity quotas."""
        available = (
            np.ones(h * w, dtype=bool)
            if available_mask is None
            else np.asarray(available_mask, dtype=bool).reshape(-1)
        )
        n = int(min(n, int(np.count_nonzero(available))))
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        block = max(1, int(block))
        nby = (h + block - 1) // block
        nbx = (w + block - 1) // block
        n_blocks = nby * nbx
        max_per_block = int(max_per_block)

        selected = np.zeros(h * w, dtype=bool)
        selected_ids = []

        n_random = int(round(max(0.0, min(1.0, random_ratio)) * n))
        if n_random > 0:
            pool = np.flatnonzero(available)
            ids = np.random.choice(pool, min(n_random, pool.size), replace=False)
            selected[ids] = True
            selected_ids.extend(ids.tolist())

        remaining = n - len(selected_ids)
        if remaining <= 0:
            return np.asarray(selected_ids[:n], dtype=np.int64)

        score_map = np.asarray(score_map, dtype=np.float64)
        if score_map.ndim == 3:
            score_map = score_map.mean(axis=0)
        pixel_prob = np.asarray(pixel_prob, dtype=np.float64).reshape(-1)
        if pixel_prob.size != h * w or float(pixel_prob.sum()) <= 1.0e-12:
            pixel_prob = np.ones(h * w, dtype=np.float64)

        block_scores = []
        block_indices = []
        def _block_room(ids):
            ids_avail = ids[available[ids] & ~selected[ids]]
            if ids_avail.size == 0:
                return 0
            if max_per_block > 0:
                already = int(np.count_nonzero(selected[ids]))
                cap_left = max(0, max_per_block - already)
                return min(int(ids_avail.size), cap_left)
            return int(ids_avail.size)
        for by in range(nby):
            y0, y1 = by * block, min(h, (by + 1) * block)
            for bx in range(nbx):
                x0, x1 = bx * block, min(w, (bx + 1) * block)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                ids = (yy.reshape(-1) * w + xx.reshape(-1)).astype(np.int64)
                block_indices.append(ids)
                vals = score_map[y0:y1, x0:x1].reshape(-1)
                vals = vals[available[ids]]
                if vals.size == 0:
                    block_scores.append(0.0)
                    continue
                mean_score = float(np.mean(vals))
                var_score = float(np.var(vals))
                block_scores.append(mean_score + var_score)

        block_scores = np.asarray(block_scores, dtype=np.float64)
        room = np.asarray([_block_room(ids) for ids in block_indices], dtype=np.int64)
        block_scores = np.power(np.clip(block_scores, 0.0, None) + 1.0e-8, quota_power)
        block_scores = block_scores * (room > 0).astype(np.float64)
        if float(block_scores.sum()) <= 1.0e-12:
            block_scores = (room > 0).astype(np.float64)
        block_probs = block_scores / block_scores.sum()

        quota = np.zeros(n_blocks, dtype=np.int64)
        if min_per_block > 0:
            usable = room.copy()
            active = np.flatnonzero(usable > 0)
            base = min(int(min_per_block), max(0, remaining // max(1, len(active))))
            if base > 0:
                quota[active] = np.minimum(base, usable[active])

        left = remaining - int(quota.sum())
        if left > 0:
            raw = block_probs * left
            add = np.floor(raw).astype(np.int64)
            add = np.minimum(add, np.maximum(room - quota, 0))
            quota += add
            left = remaining - int(quota.sum())
            if left > 0:
                frac = raw - add
                order = np.argsort(-frac)
                for idx in order.tolist():
                    if left <= 0:
                        break
                    if quota[idx] < room[idx]:
                        quota[idx] += 1
                        left -= 1

        def _take_from_block(block_id, count):
            if count <= 0:
                return []
            ids = block_indices[block_id]
            ids = ids[available[ids] & ~selected[ids]]
            if ids.size == 0:
                return []
            if max_per_block > 0:
                already = int(np.count_nonzero(selected[block_indices[block_id]]))
                count = min(int(count), max(0, max_per_block - already))
            count = min(int(count), int(ids.size))
            if count <= 0:
                return []
            if uniform_within:
                chosen = np.random.choice(ids, count, replace=False)
            else:
                p = pixel_prob[ids].astype(np.float64)
                if float(p.sum()) <= 1.0e-12:
                    p = np.ones_like(p) / float(p.size)
                else:
                    p = p / p.sum()
                chosen = np.random.choice(ids, count, replace=False, p=p)
            selected[chosen] = True
            return chosen.tolist()

        for block_id, count in enumerate(quota.tolist()):
            selected_ids.extend(_take_from_block(block_id, count))

        while len(selected_ids) < n:
            if max_per_block > 0:
                elig = []
                for ids_blk in block_indices:
                    if _block_room(ids_blk) <= 0:
                        continue
                    ids_open = ids_blk[available[ids_blk] & ~selected[ids_blk]]
                    if ids_open.size > 0:
                        elig.append(ids_open)
                ids = np.concatenate(elig) if elig else np.empty((0,), dtype=np.int64)
            else:
                ids = np.flatnonzero(available & ~selected)
            if ids.size == 0:
                break
            need = min(n - len(selected_ids), ids.size)
            p = pixel_prob[ids].astype(np.float64)
            if float(p.sum()) <= 1.0e-12:
                p = None
            else:
                p = p / p.sum()
            chosen = np.random.choice(ids, need, replace=False, p=p)
            selected[chosen] = True
            selected_ids.extend(chosen.tolist())

        return np.asarray(selected_ids[:n], dtype=np.int64)

    def _sample_mixed_indices(self, pixel_prob, h, w, n, uniform_ratio, available_mask=None):
        available = (
            np.ones(h * w, dtype=bool)
            if available_mask is None
            else np.asarray(available_mask, dtype=bool).reshape(-1)
        )
        n = int(min(n, int(np.count_nonzero(available))))
        if n <= 0:
            return np.empty((0,), dtype=np.int64)

        prob = np.asarray(pixel_prob, dtype=np.float64).reshape(-1)
        if prob.size != h * w:
            prob = np.ones(h * w, dtype=np.float64)
        prob = np.where(available, np.clip(prob, 0.0, None), 0.0)
        if float(prob.sum()) <= 1.0e-12:
            prob = available.astype(np.float64)
        prob = prob / prob.sum()

        uniform_ratio = max(0.0, min(1.0, float(uniform_ratio)))
        uniform_num = int(round(n * uniform_ratio))
        guided_num = max(0, n - uniform_num)

        selected = np.zeros(h * w, dtype=bool)
        selected_ids = []
        if guided_num > 0:
            guided = np.random.choice(h * w, guided_num, replace=False, p=prob)
            selected[guided] = True
            selected_ids.extend(guided.tolist())
        if uniform_num > 0:
            pool = np.flatnonzero(available & ~selected)
            if pool.size > 0:
                chosen = np.random.choice(pool, min(uniform_num, pool.size), replace=False)
                selected[chosen] = True
                selected_ids.extend(chosen.tolist())
        if len(selected_ids) < n:
            pool = np.flatnonzero(available & ~selected)
            if pool.size > 0:
                p = prob[pool]
                if float(p.sum()) <= 1.0e-12:
                    p = None
                else:
                    p = p / p.sum()
                chosen = np.random.choice(
                    pool,
                    min(n - len(selected_ids), pool.size),
                    replace=False,
                    p=p,
                )
                selected_ids.extend(chosen.tolist())
        return np.asarray(selected_ids[:n], dtype=np.int64)

    def _sample_spatial_indices(
        self,
        score_map,
        pixel_prob,
        h,
        w,
        n,
        uniform_ratio,
        block,
        min_per_block,
        quota_power,
        uniform_within,
        available_mask=None,
        max_per_block=0,
    ):
        if int(block) > 0:
            return self._sample_block_alloc_indices(
                score_map=score_map,
                pixel_prob=pixel_prob,
                h=h,
                w=w,
                n=n,
                block=block,
                random_ratio=uniform_ratio,
                min_per_block=min_per_block,
                quota_power=quota_power,
                uniform_within=uniform_within,
                available_mask=available_mask,
                max_per_block=max_per_block,
            )
        return self._sample_mixed_indices(
            pixel_prob=pixel_prob,
            h=h,
            w=w,
            n=n,
            uniform_ratio=uniform_ratio,
            available_mask=available_mask,
        )

    def _coverage_aware_prob(
        self,
        score_map,
        pan_h,
        pan_w,
        xy=None,
        coverage_alpha=0.0,
        coverage_sigma_px=4.0,
        density_gamma=1.0,
    ):
        score = torch.as_tensor(
            score_map, dtype=self.dtype, device=self.device
        ).reshape(-1)
        if score.numel() != int(pan_h * pan_w):
            raise ValueError("score_map size does not match pan_h * pan_w")
        score = score.clamp_min(0.0)
        if (
            float(coverage_alpha) > 0.0
            and xy is not None
            and int(xy.shape[0]) > 0
        ):
            density = self._local_density_map(
                xy.detach(), pan_h, pan_w, float(coverage_sigma_px)
            ).reshape(-1)
            density_n = density / density.mean().clamp_min(1.0e-6)
            density_n = density_n.clamp_min(1.0e-3).pow(float(density_gamma))
            score = score / density_n.pow(float(coverage_alpha))
        total = float(score.sum().item())
        if total <= 1.0e-12:
            prob = torch.full_like(score, 1.0 / float(score.numel()))
        else:
            prob = score / (total + 1.0e-12)
        return score, prob

    def _sample_pos(self, prob):
        num_random = round(self.init_random_ratio * self.num_gaussians)
        sel_r = np.random.choice(self.num_pixels, num_random, replace=False)
        sel_p = np.random.choice(
            self.num_pixels, self.num_gaussians - num_random, replace=False, p=prob
        )
        return torch.cat(
            [self.pixel_xy.detach().clone()[sel_r],
             self.pixel_xy.detach().clone()[sel_p]], dim=0,
        )

    def _compute_gmap(self):
        gy, gx = compute_image_gradients(
            np.power(self.gt_images.detach().cpu().numpy(), 1.0 / self.gamma)
        )
        g_norm = np.hypot(gy, gx).astype(np.float32)
        mx = g_norm.max()
        if mx > 1e-12:
            g_norm /= mx
        g_norm = np.power(g_norm.reshape(-1), 2.0)
        self.image_gradients = g_norm / (g_norm.sum() + 1e-12)

    def _get_target_features(self, positions, level="ms"):
        with torch.no_grad():
            src = self.gt_images if level in ("ms", "rgb") else self.ref_images
            features = F.grid_sample(
                src.unsqueeze(0),
                positions[None, None, ...] * 2.0 - 1.0,
                align_corners=False,
            )
            return features[0, :, 0, :].permute(1, 0)

    def _sample_features_from_image(self, image, positions):
        with torch.no_grad():
            features = F.grid_sample(
                image.unsqueeze(0),
                positions[None, None, ...] * 2.0 - 1.0,
                align_corners=False, mode="bilinear",
            )
            return features[0, :, 0, :].permute(1, 0)

    def _pack_stage2_coeff_features(self, coeff):
        if self.stage2_feat_dim == self.output_dim:
            return coeff
        packed = torch.zeros(
            coeff.shape[0], self.stage2_feat_dim,
            dtype=coeff.dtype, device=coeff.device,
        )
        packed[:, : self.output_dim] = coeff
        init_std = float(getattr(self.args, "stage2_latent_init_std", 1.0e-3))
        if init_std > 0 and self.stage2_feat_dim > self.output_dim:
            packed[:, self.output_dim:] = init_std * torch.randn(
                coeff.shape[0], self.stage2_feat_dim - self.output_dim,
                dtype=coeff.dtype, device=coeff.device,
            )
        return packed

    def _copy_stage2_coeff_features(self, coeff):
        if self.feat_ms is None:
            return
        with torch.no_grad():
            self.feat_ms.copy_(self._pack_stage2_coeff_features(coeff))
            if (
                self.feat_ms_detail is not None
                and bool(getattr(self.args, "stage2_detail_reset_on_reinit", True))
            ):
                self.feat_ms_detail.zero_()

    def _copy_stage2_x_features(self, image):
        if self.feat_x is None:
            return
        with torch.no_grad():
            sampled = self._sample_features_from_image(image, self.xy.detach())
            self.feat_x.copy_(sampled)

    def _safe_base_features(self, base_features):
        eps = float(getattr(self.args, "stage2_base_eps", 1.0e-2))
        sign = torch.where(
            base_features >= 0,
            torch.ones_like(base_features),
            -torch.ones_like(base_features),
        )
        return sign * torch.clamp(base_features.abs(), min=eps)

    def _stage2_base_image(self):
        mode = self._stage2_fuse_mode()
        if mode in ("base_only", "x_base", "none") or self._stage2_uses_pan_residual_gain():
            return torch.ones(
                self.output_dim,
                self.ref_images.shape[-2],
                self.ref_images.shape[-1],
                dtype=self.dtype,
                device=self.device,
            )
        if mode in (
            "g_times_input_phat", "input_phat", "psdip_phat",
            "render_g_times_input_phat", "dense_g_times_input_phat",
            "render_g_times_phat_image", "dense_phat",
            "g_times_phat", "g_times_p_hat", "extended_pan", "phat",
        ):
            return self._phat_from_pan_image()
        return self.ref_images.repeat(self.output_dim, 1, 1)

    def _stage2_mtf_gnyq(self, channels):
        sensor = str(getattr(self.args, "sensor", "")).upper()
        if sensor == "QB":
            vals = [0.34, 0.32, 0.30, 0.22]
        elif sensor in ("WV2", "WV02"):
            vals = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.27]
        elif sensor in ("WV3", "WV03"):
            vals = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
        else:
            vals = [0.30] * channels
        if len(vals) < channels:
            vals = vals + [vals[-1]] * (channels - len(vals))
        return vals[:channels]

    def _stage2_mtf_blur(self, image):
        ratio = max(1.0, float(self.ref_images.shape[-1]) / float(self.gt_images.shape[-1]))
        ksize = int(getattr(self.args, "stage2_psdip_mtf_kernel_size", 41))
        if ksize % 2 == 0:
            ksize += 1
        coords = (
            torch.arange(ksize, dtype=image.dtype, device=image.device)
            - (ksize - 1) / 2.0
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        win_1d = torch.kaiser_window(
            ksize, beta=0.5, periodic=False, dtype=image.dtype, device=image.device
        )
        window = win_1d[:, None] * win_1d[None, :]
        fcut = 1.0 / ratio
        kernels = []
        for gnyq in self._stage2_mtf_gnyq(image.shape[0]):
            alpha = math.sqrt(
                ((ksize - 1) * (fcut / 2.0)) ** 2
                / max(1.0e-12, -2.0 * math.log(float(gnyq)))
            )
            kernel = torch.exp(-0.5 * ((xx / alpha) ** 2 + (yy / alpha) ** 2))
            kernel = kernel * window
            kernel = kernel / kernel.sum().clamp(min=1.0e-12)
            kernels.append(kernel)
        kernel = torch.stack(kernels, dim=0).unsqueeze(1)
        pad = ksize // 2
        return F.conv2d(
            F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="replicate"),
            kernel,
            groups=image.shape[0],
        ).squeeze(0)

    def _stage2_lr_phat_image(self, lr_h, lr_w):
        phat_hr = self._phat_from_pan_image()
        if bool(getattr(self.args, "stage2_psdip_blur_base", True)):
            phat_hr = self._stage2_mtf_blur(phat_hr)
        return F.interpolate(
            phat_hr.unsqueeze(0), size=(lr_h, lr_w), mode="area"
        ).squeeze(0)

    def _bicubic_hr_target(self):
        return F.interpolate(
            self.gt_images.unsqueeze(0),
            size=self.ref_images.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)

    def _reinit_feat_ms_from_target(self, new_hr_target, base_image=None):
        """Re-initialize feat_ms by sampling from a new HR target.
        G*P mode: feat_ms = sampled_ms / feat_pan."""
        with torch.no_grad():
            new_ms = self._sample_features_from_image(new_hr_target, self.xy.detach())
            if self._stage2_uses_hypergs_decoder():
                self._copy_stage2_coeff_features(new_ms)
                if bool(getattr(self.args, "stage2_use_x_field", False)):
                    self._copy_stage2_x_features(new_hr_target)
                return
            if base_image is None:
                base = self._stage2_base_features_safe()
            else:
                base = self._safe_base_features(
                    self._sample_features_from_image(base_image, self.xy.detach())
                )
            self._copy_stage2_coeff_features(new_ms / base)
            if bool(getattr(self.args, "stage2_use_x_field", False)):
                self._copy_stage2_x_features(new_hr_target)

    def _initialize_stage2_feat_ms(self, hr_target, use_bicubic=None):
        mode = str(getattr(self.args, "stage2_g_init_mode", "target_div_base")).lower()
        if mode in ("target_div_base", "target", "default"):
            self._reinit_feat_ms_from_target(hr_target)
            return

        if mode not in ("psdip_ratio", "psdip_prefit", "psdip", "psdip_cnn"):
            raise ValueError(f"Unknown stage2_g_init_mode: {mode}")

        target_mode = str(getattr(self.args, "stage2_g_init_target", "bicubic")).lower()
        if use_bicubic is None:
            use_bicubic = target_mode == "bicubic"
        init_target = self._bicubic_hr_target() if use_bicubic else hr_target
        target_name = "bicubic" if use_bicubic else "target"
        base_image = self._stage2_base_image()
        if bool(getattr(self.args, "stage2_psdip_blur_base", True)):
            base_image = self._stage2_mtf_blur(base_image)

        smooth_ksize = int(getattr(self.args, "stage2_g_init_smooth_ksize", 0))
        if smooth_ksize > 1:
            if self._stage2_uses_hypergs_decoder():
                coeff_image = init_target
            else:
                safe_base = self._safe_base_features(base_image)
                coeff_image = init_target / safe_base
            coeff_image = self._blur_image(coeff_image, smooth_ksize)
            with torch.no_grad():
                self._copy_stage2_coeff_features(
                    self._sample_features_from_image(coeff_image, self.xy.detach())
                )
        else:
            self._reinit_feat_ms_from_target(init_target, base_image=base_image)

        steps = int(getattr(self.args, "stage2_g_init_steps", 0))
        if not use_bicubic and not bool(
            getattr(self.args, "stage2_g_init_prefit_each_round", False)
        ):
            steps = 0
        if mode == "psdip_cnn" and steps > 0:
            gnet = _PSDipFusionNet(self.output_dim).to(self.device)
            optimizer = torch.optim.Adam(
                gnet.parameters(),
                lr=float(getattr(self.args, "stage2_g_init_lr", 1.0e-3)),
            )
            log_every = max(1, int(getattr(self.args, "stage2_g_init_log_every", 200)))
            target_b = init_target.unsqueeze(0)
            pan_b = self.ref_images.unsqueeze(0)
            base_b = self._safe_base_features(base_image).unsqueeze(0)
            for step in range(1, steps + 1):
                optimizer.zero_grad()
                coeff = gnet(target_b, pan_b)
                loss = torch.norm(target_b - coeff * base_b)
                loss.backward()
                optimizer.step()
                if step % log_every == 0 or step == steps:
                    self.worklog.info(
                        f"Stage2 PSDIP-CNN G init {step:d}/{steps:d} | "
                        f"Loss: {loss.item():.5f}"
                    )
            with torch.no_grad():
                coeff_image = gnet(target_b, pan_b).squeeze(0)
                self._copy_stage2_coeff_features(
                    self._sample_features_from_image(coeff_image, self.xy.detach())
                )
            return

        if mode == "psdip_ratio" or steps <= 0:
            self.worklog.info(
                f"Stage2 G init: PSDIP ratio target={target_name}, "
                f"blur_base={bool(getattr(self.args, 'stage2_psdip_blur_base', True))}, "
                f"smooth={smooth_ksize:d}."
            )
            return

        self.feat_ms.requires_grad_(True)
        if self.stage2_decoder is not None:
            self.stage2_decoder.requires_grad_(True)
        lr = float(getattr(self.args, "stage2_g_init_lr", self.feat_lr))
        params = [{"params": self.feat_ms, "lr": lr}]
        if self.stage2_decoder is not None:
            params.append(
                {
                    "params": self.stage2_decoder.parameters(),
                    "lr": lr * float(getattr(self.args, "stage2_decoder_lr_mult", 1.0)),
                }
            )
        optimizer = torch.optim.Adam(params)
        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        ms_h, ms_w = self.gt_images.shape[-2], self.gt_images.shape[-1]
        tb_pan = (
            (pan_w + self.block_w - 1) // self.block_w,
            (pan_h + self.block_h - 1) // self.block_h,
            1,
        )
        tb_ms = (
            (ms_w + self.block_w - 1) // self.block_w,
            (ms_h + self.block_h - 1) // self.block_h,
            1,
        )
        base_feat = self._safe_base_features(
            self._sample_features_from_image(base_image, self.xy.detach())
        )
        log_every = max(1, int(getattr(self.args, "stage2_g_init_log_every", 200)))
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            if self._stage2_uses_post_decoder():
                _, pred = self._render_stage2_pair(
                    pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    train_geom=False,
                )
            else:
                pred, _ = self._rasterize_features(
                    self.xy.detach(), self._get_scale(1.0).detach(), self.rot.detach(),
                    self._stage2_coeff_features() * base_feat, pan_h, pan_w, tb_pan,
                )
            loss = torch.norm(pred - init_target)
            loss.backward()
            optimizer.step()
            if step % log_every == 0 or step == steps:
                self.worklog.info(
                    f"Stage2 G init prefit {step:d}/{steps:d} | Loss: {loss.item():.5f}"
                )

    def _current_stage2_mtf_gain_scale(self):
        if getattr(self, "_hard_patch_active", False):
            return float(getattr(self.args, "hard_patch_mtf_gain_scale", 1.5))
        return float(getattr(self.args, "stage2_mtf_gain_scale", 1.0))

    def _current_stage3_output_mix(self):
        if getattr(self, "_hard_patch_active", False):
            return float(getattr(self.args, "hard_patch_stage3_output_mix", 0.0))
        return float(getattr(self.args, "stage3_output_mix", 1.0))

    def _get_scale(self, upsample_ratio=None):
        scale = self.scale
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return scale

    def _get_scale_(self, upsample_ratio=None):
        scale = self.scale_aux
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return scale

    def _save_pansharpened(self, image, stage_tag, h, w):
        save_path = (
            f"{self.log_dir}/pansharp_{stage_tag}_res-{h:d}x{w:d}"
            f".{self.save_image_format}"
        )
        save_image(image, save_path, gamma=self.gamma)

    # -- Detail injection (Stage 3) --------------------------------------------

    def _aux_to_ms(self, aux_I, aux_res):
        return aux_I * self.band_coeff.view(-1, 1, 1) + aux_res

    def _pan_weights(self):
        """Normalized PAN channel weights."""
        wt = self.pan_channel_weight[: self.output_dim].clamp_min(0.0)
        if wt.sum().item() <= 0:
            wt = torch.ones_like(wt)
        return wt / wt.sum()

    def _inject_stage3_detail(self, base_ms_hr, I_base, aux_ms):
        """Dispatch to gain-masked or LIG-style additive injection."""
        mode = str(getattr(self.args, "stage3_injection_mode", "gain")).lower()
        if mode in ("lig", "additive"):
            return self._inject_stage3_detail_lig(base_ms_hr, I_base, aux_ms)
        return self._inject_stage3_detail_gain(base_ms_hr, I_base, aux_ms)

    def _inject_stage3_detail_gain(self, base_ms_hr, I_base, aux_ms):
        """Original gain-masked detail injection with dark-region adaptive mask."""
        neg_clip = float(getattr(self.args, "stage3_neg_gain_clip", 0.05))
        pos_clip = float(getattr(self.args, "stage3_pos_gain_clip", 0.15))
        dark_th = float(getattr(self.args, "stage3_dark_th", 0.12))
        dark_gamma = float(getattr(self.args, "stage3_dark_gamma", 0.5))

        eps = 1e-6
        base_ms_safe = torch.clamp(base_ms_hr, 0.0, 1.0)
        I_base_safe = torch.clamp(I_base, min=eps)
        mask = (I_base_safe / dark_th).clamp(0.0, 1.0) ** dark_gamma

        weights = self._pan_weights()
        detail_ms = aux_ms * mask
        base_clamped = torch.clamp(base_ms_safe, min=eps)
        gain = (detail_ms / base_clamped).clamp(-neg_clip, pos_clip)
        final_ms_hr = torch.clamp(base_ms_safe * (1.0 + gain), 0.0, 1.0)
        final_I = (final_ms_hr * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        return final_ms_hr, final_I

    def _inject_stage3_detail_lig(self, base_ms_hr, I_base, aux_ms):
        """LIG-style pure additive (residual) injection."""
        clamp_val = float(getattr(self.args, "stage3_additive_clamp", 0.10))
        dark_th = float(getattr(self.args, "stage3_dark_th", 0.12))
        dark_gamma = float(getattr(self.args, "stage3_dark_gamma", 0.5))

        if dark_th > 0:
            eps = 1e-6
            mask = (I_base.clamp(min=eps) / dark_th).clamp(0.0, 1.0) ** dark_gamma
            detail = aux_ms * mask
        else:
            detail = aux_ms

        detail_clamped = detail.clamp(-clamp_val, clamp_val)
        final_ms_hr = (base_ms_hr + detail_clamped).clamp(0.0, 1.0)
        weights = self._pan_weights()
        final_I = (final_ms_hr * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        return final_ms_hr, final_I

    # -- LIG residual normalization (for aux Gaussian target fitting) -----------

    def _build_residual_target(self, base_ms_hr, hr_target):
        """Compute normalized residual for aux Gaussians to fit.
        LIG-style: residual → +0.5 → min-max normalize to [0,1]."""
        residual = hr_target - base_ms_hr            # [C, H, W]
        shifted = residual + 0.5
        res_min = shifted.amin(dim=(-2, -1), keepdim=True)
        res_max = shifted.amax(dim=(-2, -1), keepdim=True)
        norm = (shifted - res_min) / (res_max - res_min + 1e-8)
        return norm, res_min, res_max

    def _denorm_residual(self, aux_render, res_min, res_max):
        """Reverse: aux render [0,1] → actual residual."""
        return aux_render * (res_max - res_min) + res_min - 0.5

    def _mix_stage3_output(self, base_ms_hr, I_base, final_ms_hr, final_I):
        mix = self._current_stage3_output_mix()
        return self._mix_stage3_output_with_ratio(
            base_ms_hr, I_base, final_ms_hr, final_I, mix
        )

    def _mix_stage3_output_with_ratio(self, base_ms_hr, I_base, final_ms_hr, final_I, mix):
        if mix >= 1.0:
            return final_ms_hr, final_I
        if mix <= 0.0:
            return base_ms_hr, I_base
        return (
            torch.clamp(base_ms_hr + mix * (final_ms_hr - base_ms_hr), 0.0, 1.0),
            torch.clamp(I_base + mix * (final_I - I_base), 0.0, 1.0),
        )

    def _build_stage3_pan_targets(self, I_base):
        pan_hp = self._highpass(self.ref_images)
        I_base_hp = self._highpass(I_base)
        residual_hint = (pan_hp - I_base_hp).detach()
        return pan_hp, I_base_hp, residual_hint, pan_hp

    def _blur_image(self, image, blur_ksize):
        if blur_ksize <= 1:
            return image
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        return gaussian_blur(image, [blur_ksize, blur_ksize])

    def _gaussian_lowpass_sigma(self, image, sigma):
        sigma = float(sigma)
        if sigma <= 1.0e-6:
            return image
        squeeze = image.dim() == 3
        image_b = image.unsqueeze(0) if squeeze else image
        _, channels, _, _ = image_b.shape
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        coords = torch.arange(
            ksize, dtype=image_b.dtype, device=image_b.device
        ) - ksize // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-12)
        kernel = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
        kernel = kernel.expand(channels, 1, ksize, ksize)
        pad = ksize // 2
        out = F.conv2d(
            F.pad(image_b, (pad, pad, pad, pad), mode="reflect"),
            kernel,
            groups=channels,
        )
        return out.squeeze(0) if squeeze else out

    def _smooth_stage3_aux(self, aux_I):
        bk = int(getattr(self.args, "stage3_aux_blur_ksize", 0))
        return self._blur_image(aux_I, bk)

    def _spectral_angle_loss(self, pred, target):
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        dot = (pred_flat * target_flat).sum(dim=0)
        pred_norm = torch.linalg.norm(pred_flat, dim=0)
        target_norm = torch.linalg.norm(target_flat, dim=0)
        cos = dot / (pred_norm * target_norm + 1e-8)
        return torch.acos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).mean()

    def _charbonnier_loss(self, pred, target, eps=1.0e-3):
        diff = pred - target
        return torch.sqrt(diff * diff + eps * eps).mean()

    def _spectral_cosine_loss(self, pred, target):
        squeeze = pred.dim() == 3
        pred_b = pred.unsqueeze(0) if squeeze else pred
        target_b = target.unsqueeze(0) if target.dim() == 3 else target
        pred_flat = pred_b.flatten(2)
        target_flat = target_b.flatten(2)
        pred_flat = F.normalize(pred_flat, dim=1, eps=1.0e-8)
        target_flat = F.normalize(target_flat, dim=1, eps=1.0e-8)
        return (1.0 - (pred_flat * target_flat).sum(dim=1).clamp(-1.0, 1.0)).mean()

    def _stage2_hr_sam_prior_loss(self, ms_hr):
        weight = float(getattr(self.args, "stage2_hr_sam_prior_weight", 0.0))
        if weight <= 0:
            return ms_hr.new_tensor(0.0)
        target = self._bicubic_hr_target().to(dtype=ms_hr.dtype, device=ms_hr.device)
        if target.shape[-2:] != ms_hr.shape[-2:]:
            target = F.interpolate(
                target.unsqueeze(0),
                size=ms_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        return weight * self._spectral_cosine_loss(ms_hr, target.detach())

    def _stage2_ratio_direction_reference(self, ratio_hr, product_pan):
        eps = float(getattr(self.args, "stage2_ratio_split_ref_eps", 1.0e-6))
        lms_hr = self._bicubic_hr_target().to(
            dtype=ratio_hr.dtype, device=ratio_hr.device
        )
        if lms_hr.shape[-2:] != ratio_hr.shape[-2:]:
            lms_hr = F.interpolate(
                lms_hr.unsqueeze(0),
                size=ratio_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        pan = product_pan.to(dtype=ratio_hr.dtype, device=ratio_hr.device)
        if pan.shape[-2:] != ratio_hr.shape[-2:]:
            pan = F.interpolate(
                pan.unsqueeze(0),
                size=ratio_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        sigma = float(getattr(self.args, "stage2_ratio_split_ref_pan_sigma", 2.0))
        pan_low = self._gaussian_lowpass_sigma(pan, sigma).clamp_min(eps)
        return (lms_hr / pan_low).detach()

    def _stage2_ratio_split_regularizer_loss(self, ratio_hr, product_pan):
        if not self._stage2_ratio_split_enabled():
            return ratio_hr.new_tensor(0.0), ratio_hr.new_tensor(0.0)

        loss_dir = ratio_hr.new_tensor(0.0)
        dir_w = float(getattr(self.args, "stage2_ratio_split_dir_weight", 0.0))
        if dir_w > 0:
            sigma = float(getattr(self.args, "stage2_ratio_split_dir_sigma", 2.0))
            ratio_low = self._gaussian_lowpass_sigma(ratio_hr, sigma)
            ref = self._stage2_ratio_direction_reference(ratio_hr, product_pan)
            loss_dir = dir_w * self._spectral_cosine_loss(ratio_low, ref)

        loss_hf_zero = ratio_hr.new_tensor(0.0)
        hf_w = float(getattr(self.args, "stage2_ratio_hf_low_zero_weight", 0.0))
        ratio_hf = getattr(self, "_stage2_last_ratio_hf", None)
        if hf_w > 0 and ratio_hf is not None:
            sigma = float(getattr(self.args, "stage2_ratio_hf_low_zero_sigma", 4.0))
            hf_low = self._gaussian_lowpass_sigma(ratio_hf, sigma)
            loss_hf_zero = hf_w * (hf_low * hf_low).mean()

        return loss_dir, loss_hf_zero

    def _tv_loss(self, image):
        dh = torch.abs(image[:, 1:, :] - image[:, :-1, :]).mean()
        dw = torch.abs(image[:, :, 1:] - image[:, :, :-1]).mean()
        return dh + dw

    def _edge_aware_tv_loss(self, image, edge):
        edge = edge.detach()
        if edge.shape[-2:] != image.shape[-2:]:
            edge = F.interpolate(
                edge.unsqueeze(0), size=image.shape[-2:], mode="area"
            ).squeeze(0)
        alpha = float(getattr(self.args, "stage2_g_edge_alpha", 10.0))
        grad_y = torch.abs(edge[:, 1:, :] - edge[:, :-1, :]).mean(dim=0, keepdim=True)
        grad_x = torch.abs(edge[:, :, 1:] - edge[:, :, :-1]).mean(dim=0, keepdim=True)
        grad_y = grad_y / grad_y.mean().clamp_min(1.0e-6)
        grad_x = grad_x / grad_x.mean().clamp_min(1.0e-6)
        wy = torch.exp(-alpha * grad_y)
        wx = torch.exp(-alpha * grad_x)
        tv_y = (wy * torch.abs(image[:, 1:, :] - image[:, :-1, :])).mean()
        tv_x = (wx * torch.abs(image[:, :, 1:] - image[:, :, :-1])).mean()
        return tv_y + tv_x

    def _stage2_g_ratio_prior_loss(self, g_lr, g_hr, lr_h, lr_w):
        ratio_w = float(getattr(self.args, "stage2_g_lr_ratio_weight", 0.0))
        tv_w = float(getattr(self.args, "stage2_g_tv_weight", 0.0))
        if ratio_w <= 0 and tv_w <= 0:
            return g_hr.new_tensor(0.0)

        loss = g_hr.new_tensor(0.0)
        if ratio_w > 0:
            eps = float(getattr(self.args, "stage2_g_ratio_eps", 1.0e-3))
            clip = float(getattr(self.args, "stage2_g_ratio_clip", 4.0))
            phat_lr = self._stage2_lr_phat_image(lr_h, lr_w).clamp_min(eps)
            target = self.gt_images / phat_lr
            if clip > 0:
                target = target.clamp(0.0, clip)
                g_cmp = g_lr.clamp(0.0, clip)
            else:
                g_cmp = g_lr
            loss = loss + ratio_w * self.l1_loss_ratio * F.l1_loss(g_cmp, target.detach())

        if tv_w > 0:
            if bool(getattr(self.args, "stage2_g_edge_aware", True)):
                loss = loss + tv_w * self._edge_aware_tv_loss(g_hr, self.ref_images)
            else:
                loss = loss + tv_w * self._tv_loss(g_hr)
        return loss

    def _corr_loss(self, a, b, eps=1.0e-8):
        a = a - a.mean()
        b = b - b.mean()
        return 1.0 - (a * b).mean() / (
            a.std(unbiased=False) * b.std(unbiased=False) + eps
        )

    def _stage2_kernel_affine_pan_features(self):
        gain_scale = float(getattr(self.args, "stage2_kernel_affine_gain_scale", 1.0))
        gain_clip = float(getattr(self.args, "stage2_kernel_affine_gain_clip", 0.0))
        pan_feat = self.feat_pan.detach()
        if bool(getattr(self.args, "stage2_kernel_affine_center_pan", False)):
            pan_feat = pan_feat - pan_feat.mean()
        gain_feat = self.feat_ms_detail
        if gain_clip > 0:
            gain_feat = gain_feat.clamp(-gain_clip, gain_clip)
        base_feat = self._stage2_coeff_features()
        detail_feat = gain_scale * gain_feat * pan_feat
        return base_feat, detail_feat

    def _render_stage2_kernel_affine_parts_hr(self, pan_h, pan_w, tb_pan):
        base_feat, detail_feat = self._stage2_kernel_affine_pan_features()
        base_hr, _ = self._rasterize_features(
            self.xy.detach(),
            self._get_scale(1.0).detach(),
            self.rot.detach(),
            base_feat,
            pan_h,
            pan_w,
            tb_pan,
        )
        detail_hr, _ = self._rasterize_features(
            self.xy.detach(),
            self._get_scale(1.0).detach(),
            self.rot.detach(),
            detail_feat,
            pan_h,
            pan_w,
            tb_pan,
        )
        return base_hr, detail_hr

    def _kernel_affine_disentangle_loss(self, gt_hr, lms_hr, pan_h, pan_w, tb_pan):
        if not self._stage2_uses_kernel_affine_pan():
            return gt_hr.new_tensor(0.0)
        base_low_w = float(getattr(self.args, "stage2_kernel_affine_base_low_weight", 0.0))
        detail_low_w = float(getattr(self.args, "stage2_kernel_affine_detail_low_weight", 0.0))
        detail_l1_w = float(getattr(self.args, "stage2_kernel_affine_detail_l1_weight", 0.0))
        detail_tv_w = float(getattr(self.args, "stage2_kernel_affine_detail_tv_weight", 0.0))
        detail_pan_corr_w = float(
            getattr(self.args, "stage2_kernel_affine_detail_pan_corr_weight", 0.0)
        )
        if (
            base_low_w <= 0
            and detail_low_w <= 0
            and detail_l1_w <= 0
            and detail_tv_w <= 0
            and detail_pan_corr_w <= 0
        ):
            return gt_hr.new_tensor(0.0)

        base_hr, detail_hr = self._render_stage2_kernel_affine_parts_hr(
            pan_h, pan_w, tb_pan
        )
        sigma = float(getattr(self.args, "stage2_kernel_affine_low_sigma", 4.0))
        loss = gt_hr.new_tensor(0.0)

        if base_low_w > 0:
            target_mode = str(
                getattr(self.args, "stage2_kernel_affine_base_low_target", "lms")
            ).lower()
            base_target = gt_hr if target_mode in ("target", "teacher", "hr") else lms_hr
            loss = loss + base_low_w * F.l1_loss(
                self._gaussian_lowpass_sigma(base_hr, sigma),
                self._gaussian_lowpass_sigma(base_target, sigma).detach(),
            )

        if detail_low_w > 0:
            loss = loss + detail_low_w * self._gaussian_lowpass_sigma(
                detail_hr, sigma
            ).abs().mean()

        if detail_l1_w > 0:
            loss = loss + detail_l1_w * self.feat_ms_detail.abs().mean()

        if detail_tv_w > 0:
            loss = loss + detail_tv_w * self._edge_aware_tv_loss(
                detail_hr, self.ref_images
            )

        if detail_pan_corr_w > 0:
            detail_i_hp = self._highpass(self._to_pan_estimate(detail_hr))
            pan_hp = self._highpass(self.ref_images)
            loss = loss + detail_pan_corr_w * self._corr_loss(detail_i_hp, pan_hp)

        return loss

    def _stage2_regularizer_loss(self, ms_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan):
        extra = ms_hr.new_tensor(0.0)

        sam_w = float(getattr(self.args, "stage2_sam_weight", 0.0))
        if sam_w > 0:
            extra = extra + sam_w * self._spectral_angle_loss(
                ms_hr.clamp(0.0, 1.0), gt_hr.clamp(0.0, 1.0)
            )

        charbonnier_w = float(getattr(self.args, "stage2_charbonnier_weight", 0.0))
        if charbonnier_w > 0:
            extra = extra + charbonnier_w * self._charbonnier_loss(
                ms_hr.clamp(0.0, 1.0), gt_hr.clamp(0.0, 1.0)
            )

        cosine_w = float(getattr(self.args, "stage2_cosine_weight", 0.0))
        if cosine_w > 0:
            extra = extra + cosine_w * self._spectral_cosine_loss(
                ms_hr.clamp(0.0, 1.0), gt_hr.clamp(0.0, 1.0)
            )

        lms_w = float(getattr(self.args, "stage2_lms_weight", 0.0))
        if lms_w > 0:
            ksize = int(getattr(self.args, "stage2_lms_blur_ksize", 11))
            ms_lp = self._blur_image(ms_hr, ksize)
            lms_lp = self._blur_image(lms_hr, ksize)
            extra = extra + lms_w * F.l1_loss(ms_lp, lms_lp)

        coeff_tv_w = float(getattr(self.args, "stage2_coeff_tv_weight", 0.0))
        if coeff_tv_w > 0:
            coeff_hr, _ = self._rasterize_features(
                self.xy.detach(), self._get_scale(1.0).detach(), self.rot.detach(),
                self.feat_ms, pan_h, pan_w, tb_pan,
            )
            extra = extra + coeff_tv_w * self._tv_loss(coeff_hr)

        detail_l1_w = float(getattr(self.args, "stage2_detail_l1_weight", 0.0))
        if detail_l1_w > 0 and self._stage2_uses_detail_branch():
            extra = extra + detail_l1_w * self.feat_ms_detail.abs().mean()

        extra = extra + self._kernel_affine_disentangle_loss(
            gt_hr, lms_hr, pan_h, pan_w, tb_pan
        )

        return extra

    def _gain_consistency_loss(self, base_ms_hr, final_ms_hr):
        rel_gain = (final_ms_hr - base_ms_hr) / base_ms_hr.clamp(min=1e-4)
        shared_gain = rel_gain.mean(dim=0, keepdim=True)
        return torch.mean(torch.abs(rel_gain - shared_gain))

    def _degrade_to_lr(self, image, lr_h, lr_w):
        mode = str(getattr(self.args, "stage3_lr_degrade_mode", "area")).lower()
        if mode != "mtf":
            return F.interpolate(
                image.unsqueeze(0), size=(lr_h, lr_w), mode="area"
            ).squeeze(0)

        ratio_h = image.shape[-2] / float(lr_h)
        ratio_w = image.shape[-1] / float(lr_w)
        ratio = max(1.0, 0.5 * (ratio_h + ratio_w))
        sigma = max(1e-6, float(getattr(self.args, "stage3_mtf_sigma_scale", 0.39)) * ratio)
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1

        x = torch.arange(ksize, dtype=image.dtype, device=image.device) - ksize // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d.view(1, 1, -1, 1) * gauss_1d.view(1, 1, 1, -1)
        kernel = kernel.expand(image.shape[0], 1, -1, -1)
        pad = ksize // 2
        image_lp = F.conv2d(
            F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"),
            kernel,
            groups=image.shape[0],
        )
        return F.interpolate(image_lp, size=(lr_h, lr_w), mode="area").squeeze(0)

    def _stage2_lr_consistency_image(self, ms_lr, ms_hr, lr_h, lr_w):
        mode = str(getattr(self.args, "stage2_lr_consistency_mode", "direct_lr")).lower()
        if mode in ("degrade_hr", "mtf_hr", "psdip_like"):
            return self._degrade_to_lr(ms_hr, lr_h, lr_w)
        return ms_lr

    def _stage2_loss_curriculum_scales(self, step):
        if not bool(getattr(self.args, "stage2_loss_curriculum", False)):
            return 1.0, 1.0, False

        switch_step = int(getattr(self.args, "stage2_loss_switch_step", 1000))
        switch_step = max(1, switch_step)
        after_direct = float(
            getattr(self.args, "stage2_loss_direct_scale_after", 0.1)
        )
        after_direct = max(0.0, after_direct)
        ramp_steps = int(getattr(self.args, "stage2_loss_switch_ramp_steps", 0))
        ramp_steps = max(0, ramp_steps)
        if int(step) < switch_step:
            return 1.0, 0.0, False
        if ramp_steps <= 0:
            return after_direct, 1.0, True
        progress = max(0.0, min(1.0, float(int(step) - switch_step + 1) / ramp_steps))
        direct_scale = 1.0 + (after_direct - 1.0) * progress
        return direct_scale, progress, True

    def _stage2_curriculum_lr_loss(self, ms_lr, target_lr, step, base_weight):
        direct_scale, aux_scale, active = self._stage2_loss_curriculum_scales(step)
        raw_mse = F.mse_loss(ms_lr, target_lr)
        loss_lr = float(base_weight) * direct_scale * raw_mse

        low_loss = ms_lr.new_tensor(0.0)
        low_weight = float(getattr(self.args, "stage2_loss_lr_low_weight", 0.0))
        if aux_scale > 0.0 and low_weight > 0.0:
            sigma = float(getattr(self.args, "stage2_loss_lr_low_sigma", 1.0))
            pred_low = self._gaussian_lowpass_sigma(ms_lr, sigma)
            target_low = self._gaussian_lowpass_sigma(target_lr.detach(), sigma)
            loss_type = str(
                getattr(self.args, "stage2_loss_lr_low_type", "mse")
            ).lower()
            if loss_type in ("l1", "mae"):
                low_raw = F.l1_loss(pred_low, target_low)
            else:
                low_raw = F.mse_loss(pred_low, target_low)
            low_loss = low_weight * aux_scale * low_raw

        self._stage2_last_loss_curriculum = {
            "active": bool(active),
            "direct_scale": float(direct_scale),
            "aux_scale": float(aux_scale),
        }
        return loss_lr, low_loss, raw_mse

    def _stage2_curriculum_progress(self, step, total_steps):
        if bool(getattr(self.args, "stage2_freq_global", False)):
            step = int(step) + max(0, int(getattr(self, "_stage2_freq_step_offset", 0)))
            total_steps = int(getattr(self, "_stage2_freq_total_steps", total_steps))
        total_steps = max(1, int(total_steps))
        if total_steps <= 1:
            return 1.0
        return max(0.0, min(1.0, float(step - 1) / float(total_steps - 1)))

    def _stage2_curriculum_sigma(self, step, total_steps):
        if not bool(getattr(self.args, "stage2_freq_guided", False)):
            return 0.0
        progress = self._stage2_curriculum_progress(step, total_steps)
        until = float(getattr(self.args, "stage2_freq_until", 1.0))
        until = max(1.0e-6, min(1.0, until))
        if progress >= until:
            return 0.0
        local = progress / until
        start_sigma = float(getattr(self.args, "stage2_freq_start_sigma", 4.0))
        end_sigma = float(getattr(self.args, "stage2_freq_end_sigma", 0.0))
        power = max(1.0e-6, float(getattr(self.args, "stage2_freq_decay_power", 1.0)))
        return end_sigma + (start_sigma - end_sigma) * ((1.0 - local) ** power)

    def _stage2_frequency_pair(self, pred, target, step, total_steps):
        sigma = self._stage2_curriculum_sigma(step, total_steps)
        if sigma <= 1.0e-6:
            return pred, target, sigma
        return (
            self._gaussian_lowpass_sigma(pred, sigma),
            self._gaussian_lowpass_sigma(target, sigma),
            sigma,
        )

    def _stage2_pan_constraint_loss(self, ms_hr, step, total_steps):
        pan_pred = self._to_pan_estimate(ms_hr)
        if not bool(getattr(self.args, "stage2_freq_guided", False)):
            return self._compute_pan_loss(pan_pred, self.ref_images)

        progress = self._stage2_curriculum_progress(step, total_steps)
        ramp_start = float(getattr(self.args, "stage2_pan_hf_start_frac", 0.25))
        ramp_start = max(0.0, min(0.99, ramp_start))
        ramp_power = max(1.0e-6, float(getattr(self.args, "stage2_pan_hf_ramp_power", 1.0)))
        if progress <= ramp_start:
            hf_ramp = 0.0
        else:
            hf_ramp = ((progress - ramp_start) / (1.0 - ramp_start)) ** ramp_power
        loss = hf_ramp * self._compute_pan_loss(pan_pred, self.ref_images)

        pan_l1_w = float(getattr(self.args, "stage2_pan_l1_weight", 0.0))
        if pan_l1_w > 0:
            pan_cmp, ref_cmp, _ = self._stage2_frequency_pair(
                pan_pred, self.ref_images, step, total_steps
            )
            loss = loss + pan_l1_w * F.l1_loss(pan_cmp, ref_cmp)
        return loss

    def _stage2_pan_corr_constraint_loss(self, ms_hr):
        weight = float(getattr(self.args, "stage2_pan_corr_weight", 0.0))
        if weight <= 0.0:
            return ms_hr.new_tensor(0.0)
        sigma = float(getattr(self.args, "stage2_pan_corr_sigma", 2.0))
        pan_pred = self._to_pan_estimate(ms_hr)
        pred_hp = pan_pred - self._gaussian_lowpass_sigma(pan_pred, sigma)
        ref_hp = self.ref_images - self._gaussian_lowpass_sigma(self.ref_images, sigma)
        return weight * self._corr_loss(pred_hp, ref_hp.detach())

    def _stage2_pan_l1_constraint_losses(self, ms_hr):
        full_w = float(getattr(self.args, "stage2_pan_full_l1_weight", 0.0))
        hf_w = float(getattr(self.args, "stage2_pan_hf_l1_weight", 0.0))
        if full_w <= 0.0 and hf_w <= 0.0:
            zero = ms_hr.new_tensor(0.0)
            return zero, zero
        pan_pred = self._to_pan_estimate(ms_hr)
        loss_full = ms_hr.new_tensor(0.0)
        if full_w > 0.0:
            loss_full = full_w * F.l1_loss(pan_pred, self.ref_images.detach())
        loss_hf = ms_hr.new_tensor(0.0)
        if hf_w > 0.0:
            sigma = float(getattr(self.args, "stage2_pan_hf_l1_sigma", 2.0))
            pred_hp = pan_pred - self._gaussian_lowpass_sigma(pan_pred, sigma)
            ref_hp = self.ref_images - self._gaussian_lowpass_sigma(self.ref_images, sigma)
            loss_hf = hf_w * F.l1_loss(pred_hp, ref_hp.detach())
        return loss_full, loss_hf

    def _stage2_bright_intensity_upper_loss(self, ms_hr):
        weight = float(getattr(self.args, "stage2_bright_intensity_upper_weight", 0.0))
        if weight <= 0.0:
            return ms_hr.new_tensor(0.0)
        pred_pan = self._to_pan_estimate(ms_hr)
        ref_pan = self.ref_images.detach()
        if ref_pan.shape[-2:] != pred_pan.shape[-2:]:
            ref_pan = F.interpolate(
                ref_pan.unsqueeze(0),
                size=pred_pan.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        topk = float(getattr(self.args, "stage2_bright_intensity_upper_topk", 0.05))
        topk = min(max(topk, 1.0e-4), 1.0)
        margin = float(getattr(self.args, "stage2_bright_intensity_upper_margin", 0.02))
        mult = float(getattr(self.args, "stage2_bright_intensity_upper_mult", 1.05))
        thr = torch.quantile(ref_pan.flatten(), 1.0 - topk)
        mask = (ref_pan >= thr).to(pred_pan.dtype)
        upper = mult * ref_pan + margin
        excess = F.relu(pred_pan - upper)
        denom = mask.sum().clamp_min(1.0)
        return weight * (excess.pow(2) * mask).sum() / denom

    def _parse_stage2_float_list(self, attr_name, default):
        raw = getattr(self.args, attr_name, default)
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            return [float(p) for p in parts]
        if isinstance(raw, (list, tuple)):
            return [float(p) for p in raw]
        return [float(raw)]

    def _stage2_tile_bounds(self, img_h, img_w):
        return (
            (img_w + self.block_w - 1) // self.block_w,
            (img_h + self.block_h - 1) // self.block_h,
            1,
        )

    def _stage2_resized_pan(self, img_h, img_w):
        if self.ref_images.shape[-2:] == (img_h, img_w):
            return self.ref_images
        return F.interpolate(
            self.ref_images.unsqueeze(0), size=(img_h, img_w), mode="area"
        ).squeeze(0)

    def _stage2_mtf_target_cached(self, img_h, img_w):
        cache = getattr(self, "_stage2_mtf_target_cache", None)
        if cache is None:
            cache = {}
            self._stage2_mtf_target_cache = cache
        key = (
            int(img_h),
            int(img_w),
            str(getattr(self.args, "stage2_hr_target_mode", "mtf_glp")).lower(),
            round(float(self._current_stage2_mtf_gain_scale()), 6),
            round(float(getattr(self.args, "stage2_sfim_sigma", 3.0)), 6),
            round(float(getattr(self.args, "stage2_target_blend_alpha", 0.5)), 6),
        )
        if key not in cache:
            cache[key] = self._build_stage2_target_at(img_h, img_w, log=True).detach()
        return cache[key]

    def _stage2_resolution_level(self, step, total_steps, pan_h, pan_w, ms_h, ms_w):
        scales = self._parse_stage2_float_list("stage2_resolution_scales", "4,2,1")
        if not scales:
            scales = [1.0]
        fracs = self._parse_stage2_float_list(
            "stage2_resolution_step_fracs", ",".join(["1"] * len(scales))
        )
        if len(fracs) != len(scales):
            fracs = [1.0] * len(scales)
        fracs = [max(0.0, f) for f in fracs]
        frac_sum = sum(fracs)
        if frac_sum <= 1.0e-12:
            fracs = [1.0] * len(scales)
            frac_sum = float(len(scales))

        progress = self._stage2_curriculum_progress(step, total_steps)
        acc = 0.0
        idx = len(scales) - 1
        for i, frac in enumerate(fracs):
            acc += frac / frac_sum
            if progress <= acc or i == len(fracs) - 1:
                idx = i
                break

        scale = max(1.0, float(scales[idx]))
        native_ratio = 0.5 * (float(pan_h) / float(ms_h) + float(pan_w) / float(ms_w))
        if scale >= native_ratio - 1.0e-6:
            return "lrms", ms_h, ms_w, scale, idx
        img_h = max(1, int(round(float(pan_h) / scale)))
        img_w = max(1, int(round(float(pan_w) / scale)))
        if scale <= 1.0 + 1.0e-6:
            return "pan", pan_h, pan_w, scale, idx
        return "mtf_glp", img_h, img_w, scale, idx

    def _render_stage2_single_resolution(
        self,
        img_h,
        img_w,
        tile_bounds,
        full_h,
        full_w,
        train_geom=False,
        train_rot=False,
        train_scale=None,
        use_x_field=False,
    ):
        if train_scale is None:
            train_scale = train_geom
        xy_s2 = self.xy if train_geom else self.xy.detach()
        rot_s2 = self.rot if train_rot else self.rot.detach()
        sc = self._get_scale(float(img_h) / float(full_h))
        if not train_scale:
            sc = sc.detach()

        if use_x_field:
            image, _ = self._rasterize_features(
                xy_s2, sc, rot_s2, self.feat_x, img_h, img_w, tile_bounds
            )
            return image.clamp(0.0, 1.0)

        if (
            self._stage2_uses_post_decoder()
            or self._stage2_uses_dense_phat_multiply()
            or self._stage2_uses_glp_adaptive_gain()
            or self._stage2_uses_kernel_affine_pan()
            or self._stage2_uses_pan_residual_gain()
        ):
            tb_full = self._stage2_tile_bounds(full_h, full_w)
            image, _ = self._render_stage2_pair(
                full_h, full_w, img_h, img_w, tb_full, tile_bounds,
                train_geom=train_geom,
                train_rot=train_rot,
                train_scale=train_scale,
            )
            return image.clamp(0.0, 1.0)

        feat = self._fused_feat_ms()
        image, _ = self._rasterize_features(
            xy_s2, sc, rot_s2, feat, img_h, img_w, tile_bounds
        )
        if self._stage2_uses_detail_branch():
            detail_feat = self.feat_ms_detail * self._stage2_base_features()
            detail, _ = self._rasterize_features(
                xy_s2, sc, rot_s2, detail_feat, img_h, img_w, tile_bounds
            )
            image = image + self._stage2_detail_highpass(detail)
        return image.clamp(0.0, 1.0)

    def _stage2_resolution_curriculum_loss(
        self,
        step,
        total_steps,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
        train_geom,
        train_rot,
        train_scale,
        s2_hr_w,
        s2_pan_w,
        x_prior_w,
    ):
        mode, img_h, img_w, scale, level_idx = self._stage2_resolution_level(
            step, total_steps, pan_h, pan_w, ms_h, ms_w
        )
        tb_cur = self._stage2_tile_bounds(img_h, img_w)
        use_x_field = bool(getattr(self.args, "stage2_use_x_field", False))

        g_img = None
        if use_x_field and x_prior_w > 0:
            g_img = self._render_stage2_single_resolution(
                img_h, img_w, tb_cur, pan_h, pan_w,
                train_geom=False,
                train_rot=False,
                train_scale=False,
                use_x_field=False,
            )

        ms_img = self._render_stage2_single_resolution(
            img_h, img_w, tb_cur, pan_h, pan_w,
            train_geom=train_geom,
            train_rot=train_rot,
            train_scale=train_scale,
            use_x_field=use_x_field,
        )

        target = None
        target_weight = 0.0
        if mode == "lrms":
            target = self.gt_images
            target_weight = float(
                getattr(self.args, "stage2_resolution_lr_weight", s2_hr_w)
            )
        elif mode == "mtf_glp":
            target = self._stage2_mtf_target_cached(img_h, img_w)
            target_weight = float(
                getattr(self.args, "stage2_resolution_mtf_weight", s2_hr_w)
            )
        else:
            full_target_weight = float(
                getattr(self.args, "stage2_resolution_full_target_weight", 0.0)
            )
            if full_target_weight > 0:
                target = self._stage2_mtf_target_cached(img_h, img_w)
                target_weight = full_target_weight

        loss = ms_img.new_tensor(0.0)
        if target is not None and target_weight > 0:
            band_mean = target.mean(dim=(-2, -1), keepdim=True).clamp(min=1.0e-6)
            band_w_cur = (1.0 / band_mean) / (1.0 / band_mean).mean()
            loss = loss + target_weight * self.l1_loss_ratio * (
                band_w_cur * torch.abs(ms_img - target)
            ).mean()

        pan_weight = float(getattr(self.args, "stage2_resolution_pan_weight", s2_pan_w))
        if mode == "pan" and pan_weight > 0:
            pan_ref = self._stage2_resized_pan(img_h, img_w)
            loss = loss + pan_weight * self._compute_pan_loss(
                self._to_pan_estimate(ms_img), pan_ref
            )

        if use_x_field and x_prior_w > 0 and g_img is not None:
            loss = loss + x_prior_w * self.l1_loss_ratio * F.l1_loss(
                ms_img, g_img.detach()
            )

        lms_cur = F.interpolate(
            self.gt_images.unsqueeze(0),
            size=(img_h, img_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        reg_target = target if target is not None else lms_cur
        loss = loss + self._stage2_regularizer_loss(
            ms_img, reg_target, lms_cur, img_h, img_w, tb_cur
        )

        tag = f"{level_idx + 1}:{mode}@1/{scale:g}"
        return loss, tag

    def _project_output_to_lrms(self, image, lr_h, lr_w):
        mode = str(getattr(self.args, "output_lr_projection_mode", "none")).lower()
        if mode in ("", "none", "off", "false"):
            return image

        projected = image
        strength = float(getattr(self.args, "output_lr_projection_strength", 1.0))

        if mode in ("affine", "affine_residual", "affine_backproject"):
            lr_pred = self._degrade_to_lr(projected, lr_h, lr_w)
            pred = lr_pred.flatten(1)
            target = self.gt_images.flatten(1)
            pred_mean = pred.mean(dim=1, keepdim=True)
            target_mean = target.mean(dim=1, keepdim=True)
            pred_centered = pred - pred_mean
            target_centered = target - target_mean
            gain = (
                (pred_centered * target_centered).mean(dim=1)
                / pred_centered.var(dim=1, unbiased=False).clamp_min(1.0e-8)
            )
            gain_clip = float(getattr(self.args, "output_lr_projection_gain_clip", 1.25))
            if gain_clip > 0:
                gain = gain.clamp(1.0 / gain_clip, gain_clip)
            bias = target_mean.squeeze(1) - gain * pred_mean.squeeze(1)
            affine = gain.view(-1, 1, 1) * projected + bias.view(-1, 1, 1)
            projected = projected + strength * (affine - projected)

        if mode in ("residual", "backproject", "backprojection", "affine_residual", "affine_backproject"):
            lr_pred = self._degrade_to_lr(projected, lr_h, lr_w)
            residual_lr = self.gt_images - lr_pred
            residual_hr = F.interpolate(
                residual_lr.unsqueeze(0),
                size=projected.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            residual_clip = float(getattr(self.args, "output_lr_projection_residual_clip", 0.0))
            if residual_clip > 0:
                residual_hr = residual_hr.clamp(-residual_clip, residual_clip)
            projected = projected + strength * residual_hr

        return projected.clamp(0.0, 1.0)

    def _make_stage2_alternating_optimizers(self, train_xy, train_scale, train_rot):
        x_lr_mult = float(getattr(self.args, "stage2_x_lr_mult", 1.0))
        geo_lr_mult = float(getattr(self.args, "stage2_geo_lr_mult", 0.1))

        x_params = [{"params": self.feat_x, "lr": self.feat_lr * x_lr_mult}]
        if train_xy:
            x_params.append({"params": self.xy, "lr": self.pos_lr * geo_lr_mult})
        if train_scale:
            x_params.append({"params": self.scale, "lr": self.scale_lr * geo_lr_mult})
        if train_rot:
            x_params.append({"params": self.rot, "lr": self.rot_lr * geo_lr_mult})

        g_params = [{"params": self.feat_ms, "lr": self.feat_lr}]
        if self._stage2_uses_detail_branch():
            g_params.append(
                {
                    "params": self.feat_ms_detail,
                    "lr": self.feat_lr * float(
                        getattr(self.args, "stage2_detail_lr_mult", 0.2)
                    ),
                }
            )
        if self.stage2_decoder is not None:
            g_params.append(
                {
                    "params": self.stage2_decoder.parameters(),
                    "lr": self.feat_lr * float(
                        getattr(self.args, "stage2_decoder_lr_mult", 1.0)
                    ),
                }
            )

        return torch.optim.Adam(x_params), torch.optim.Adam(g_params)

    def _stage3_proxy_score(self, final_hr, gt_hr):
        """Proxy score for adaptive mix selection — use HR directly."""
        mse = F.mse_loss(final_hr, gt_hr).clamp(min=1e-12)
        psnr = -10.0 * torch.log10(mse)
        sam_deg = self._spectral_angle_loss(final_hr, gt_hr) * (180.0 / math.pi)
        sam_weight = float(getattr(self.args, "stage3_proxy_sam_weight", 0.0))
        return psnr - sam_weight * sam_deg

    def _stage3_mix_candidates(self):
        spec = str(getattr(self.args, "stage3_mix_candidates", "0.0,0.25,0.5,0.75,1.0"))
        candidates = []
        for item in spec.split(","):
            item = item.strip()
            if item:
                candidates.append(float(item))
        if not candidates:
            candidates = [self._current_stage3_output_mix()]
        return sorted(set(max(0.0, min(1.0, value)) for value in candidates))

    def _select_stage3_output(
        self, base_ms_hr, I_base, raw_final_ms_hr, raw_final_I, lr_h, lr_w,
        gt_hr=None,
    ):
        if not bool(getattr(self.args, "stage3_adaptive_output_mix", False)):
            final_ms_hr, final_I = self._mix_stage3_output(
                base_ms_hr, I_base, raw_final_ms_hr, raw_final_I
            )
            final_ms_hr = self._project_output_to_lrms(final_ms_hr, lr_h, lr_w)
            final_I = (
                final_ms_hr * self._pan_weights().view(-1, 1, 1)
            ).sum(dim=0, keepdim=True)
            final_lr = self._degrade_to_lr(final_ms_hr, lr_h, lr_w)
            return final_ms_hr, final_I, final_lr, self._current_stage3_output_mix()

        best = None
        for mix in self._stage3_mix_candidates():
            cand_ms_hr, cand_I = self._mix_stage3_output_with_ratio(
                base_ms_hr, I_base, raw_final_ms_hr, raw_final_I, mix
            )
            cand_ms_hr = self._project_output_to_lrms(cand_ms_hr, lr_h, lr_w)
            cand_I = (
                cand_ms_hr * self._pan_weights().view(-1, 1, 1)
            ).sum(dim=0, keepdim=True)
            cand_lr = self._degrade_to_lr(cand_ms_hr, lr_h, lr_w)
            if gt_hr is not None:
                score = self._stage3_proxy_score(cand_ms_hr, gt_hr)
            else:
                score = self._stage3_proxy_score(cand_ms_hr, cand_ms_hr)
            if best is None or score.item() > best[0]:
                best = (score.item(), mix, cand_ms_hr, cand_I, cand_lr)

        _, mix, final_ms_hr, final_I, final_lr = best
        return final_ms_hr, final_I, final_lr, mix

    # -- PSDIP-style Stage 2 variational ablation ------------------------------

    def _render_stage2_pair(
        self,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
        tb_pan,
        tb_ms,
        train_geom=False,
        train_rot=False,
        train_scale=None,
    ):
        if train_scale is None:
            train_scale = train_geom
        xy_s2 = self.xy if train_geom else self.xy.detach()
        rot_s2 = self.rot if train_rot else self.rot.detach()
        sc_ms = self._get_scale(float(ms_h) / float(pan_h))
        sc_hr = self._get_scale(1.0)
        if not train_scale:
            sc_ms = sc_ms.detach()
            sc_hr = sc_hr.detach()

        if self._stage2_uses_post_decoder():
            latent_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, self.feat_ms, ms_h, ms_w, tb_ms
            )
            latent_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, self.feat_ms, pan_h, pan_w, tb_pan
            )
            return (
                self._decode_stage2_latent_image(latent_lr, ms_h, ms_w),
                self._decode_stage2_latent_image(latent_hr, pan_h, pan_w),
            )

        if self._stage2_uses_dense_phat_multiply():
            coeff = self._stage2_coeff_features()
            g_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, coeff, ms_h, ms_w, tb_ms
            )
            g_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, coeff, pan_h, pan_w, tb_pan
            )
            ms_lr = (g_lr * self._stage2_lr_phat_image(ms_h, ms_w)).clamp(0.0, 1.0)
            ms_hr = (g_hr * self._phat_from_pan_image()).clamp(0.0, 1.0)
            if self._stage2_uses_detail_branch():
                detail_feat = self.feat_ms_detail * self._stage2_base_features()
                detail_lr, _ = self._rasterize_features(
                    xy_s2, sc_ms, rot_s2, detail_feat, ms_h, ms_w, tb_ms
                )
                detail_hr, _ = self._rasterize_features(
                    xy_s2, sc_hr, rot_s2, detail_feat, pan_h, pan_w, tb_pan
                )
                ms_lr = (ms_lr + self._stage2_detail_highpass(detail_lr)).clamp(0.0, 1.0)
                ms_hr = (ms_hr + self._stage2_detail_highpass(detail_hr)).clamp(0.0, 1.0)
            return ms_lr, ms_hr

        if self._stage2_uses_glp_adaptive_gain():
            gain_delta_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, self._stage2_coeff_features(), ms_h, ms_w, tb_ms
            )
            gain_delta_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, self._stage2_coeff_features(), pan_h, pan_w, tb_pan
            )
            base_gain, pan_hp_hr, pan_lr_up = self._mtf_glp_gain_image()
            base_hr = self._bicubic_hr_target()
            pan_hp_lr = F.interpolate(
                pan_hp_hr.unsqueeze(0), size=(ms_h, ms_w), mode="area"
            ).squeeze(0)
            base_lr = F.interpolate(
                base_hr.unsqueeze(0), size=(ms_h, ms_w), mode="area"
            ).squeeze(0)
            delta_scale = float(getattr(self.args, "stage2_gain_delta_scale", 0.1))
            delta_clip = float(getattr(self.args, "stage2_gain_delta_clip", 0.1))
            gain_hr = (base_gain + delta_scale * gain_delta_hr).clamp(-delta_clip, delta_clip)
            gain_lr = (base_gain + delta_scale * gain_delta_lr).clamp(-delta_clip, delta_clip)
            ms_hr = (base_hr + gain_hr * pan_hp_hr).clamp(0.0, 1.0)
            ms_lr = (base_lr + gain_lr * pan_hp_lr).clamp(0.0, 1.0)
            return ms_lr, ms_hr

        if self._stage2_uses_kernel_affine_pan():
            base_feat, detail_feat = self._stage2_kernel_affine_pan_features()
            feat = base_feat + detail_feat
            ms_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, feat, ms_h, ms_w, tb_ms
            )
            ms_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, feat, pan_h, pan_w, tb_pan
            )
            return ms_lr.clamp(0.0, 1.0), ms_hr.clamp(0.0, 1.0)

        if self._stage2_uses_pan_residual_gain():
            base_feat = self._stage2_coeff_features()
            base_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, base_feat, ms_h, ms_w, tb_ms
            )
            base_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, base_feat, pan_h, pan_w, tb_pan
            )
            gain_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, self.feat_ms_detail, ms_h, ms_w, tb_ms
            )
            gain_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, self.feat_ms_detail, pan_h, pan_w, tb_pan
            )
            gain_scale = float(getattr(self.args, "stage2_pan_res_gain_scale", 1.0))
            gain_clip = float(getattr(self.args, "stage2_pan_res_gain_clip", 0.5))
            if gain_clip > 0:
                gain_lr = gain_lr.clamp(-gain_clip, gain_clip)
                gain_hr = gain_hr.clamp(-gain_clip, gain_clip)

            pan_hp_hr = self._highpass(self.ref_images)
            pan_hp_lr = F.interpolate(
                pan_hp_hr.unsqueeze(0), size=(ms_h, ms_w), mode="area"
            ).squeeze(0)
            res_lr = gain_scale * gain_lr * pan_hp_lr
            res_hr = gain_scale * gain_hr * pan_hp_hr

            res_clip = float(getattr(self.args, "stage2_pan_residual_clip", 0.08))
            if res_clip > 0:
                res_lr = res_lr.clamp(-res_clip, res_clip)
                res_hr = res_hr.clamp(-res_clip, res_clip)
            return (base_lr + res_lr).clamp(0.0, 1.0), (base_hr + res_hr).clamp(0.0, 1.0)

        # UCL-style: render ratio field, then multiply PAN in pixel domain
        use_ucl_ratio = bool(getattr(self.args, "stage2_ucl_ratio", False))
        if use_ucl_ratio:
            ratio_feat = self._stage2_coeff_features()  # feat_ms[:, :C], no feat_pan multiply
            ratio_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, ratio_feat, ms_h, ms_w, tb_ms
            )
            ratio_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, ratio_feat, pan_h, pan_w, tb_pan
            )
            pan_lr = F.interpolate(
                self.ref_images.unsqueeze(0), size=(ms_h, ms_w), mode="area"
            ).squeeze(0)
            ms_lr = (ratio_lr * pan_lr).clamp(0.0, 1.0)
            ms_hr = (ratio_hr * self.ref_images).clamp(0.0, 1.0)
            return ms_lr, ms_hr

        feat = self._fused_feat_ms()
        ms_lr, _ = self._rasterize_features(
            xy_s2, sc_ms, rot_s2, feat, ms_h, ms_w, tb_ms
        )
        ms_hr, _ = self._rasterize_features(
            xy_s2, sc_hr, rot_s2, feat, pan_h, pan_w, tb_pan
        )
        if self._stage2_uses_detail_branch():
            detail_feat = self.feat_ms_detail * self._stage2_base_features()
            detail_lr, _ = self._rasterize_features(
                xy_s2, sc_ms, rot_s2, detail_feat, ms_h, ms_w, tb_ms
            )
            detail_hr, _ = self._rasterize_features(
                xy_s2, sc_hr, rot_s2, detail_feat, pan_h, pan_w, tb_pan
            )
            ms_lr = (ms_lr + self._stage2_detail_highpass(detail_lr)).clamp(0.0, 1.0)
            ms_hr = (ms_hr + self._stage2_detail_highpass(detail_hr)).clamp(0.0, 1.0)
        return ms_lr, ms_hr

    def _stage2_detail_highpass(self, detail_image):
        ksize = int(getattr(self.args, "stage2_detail_highpass_ksize", 7))
        if ksize <= 1:
            detail = detail_image
        else:
            detail = detail_image - self._blur_image(detail_image, ksize)
        scale = float(getattr(self.args, "stage2_detail_scale", 1.0))
        clamp_val = float(getattr(self.args, "stage2_detail_clamp", 0.08))
        detail = scale * detail
        if clamp_val > 0:
            detail = detail.clamp(-clamp_val, clamp_val)
        return detail

    def _decode_stage2_latent_image(self, latent_image, img_h, img_w):
        if self._stage2_uses_hypergs_decoder():
            return self.stage2_decoder(latent_image).clamp(0.0, 1.0)
        base_image = self._stage2_base_image()
        pan_image = self.ref_images
        if base_image.shape[-2:] != (img_h, img_w):
            base_image = F.interpolate(
                base_image.unsqueeze(0), size=(img_h, img_w), mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        if pan_image.shape[-2:] != (img_h, img_w):
            pan_image = F.interpolate(
                pan_image.unsqueeze(0), size=(img_h, img_w), mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        coeff = self.stage2_decoder(latent_image, base_image, pan_image)
        return coeff * base_image

    def _render_stage2_x_pair(self, pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False, train_rot=False, train_scale=None):
        if train_scale is None:
            train_scale = train_geom
        xy_s2 = self.xy if train_geom else self.xy.detach()
        rot_s2 = self.rot if train_rot else self.rot.detach()
        sc_ms = self._get_scale(float(ms_h) / float(pan_h))
        sc_hr = self._get_scale(1.0)
        if not train_scale:
            sc_ms = sc_ms.detach()
            sc_hr = sc_hr.detach()
        x_lr, _ = self._rasterize_features(
            xy_s2, sc_ms, rot_s2, self.feat_x, ms_h, ms_w, tb_ms
        )
        x_hr, _ = self._rasterize_features(
            xy_s2, sc_hr, rot_s2, self.feat_x, pan_h, pan_w, tb_pan
        )
        return x_lr.clamp(0.0, 1.0), x_hr.clamp(0.0, 1.0)

    def _render_stage2_coeff_pair(
        self,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
        tb_pan,
        tb_ms,
        train_geom=False,
        train_rot=False,
        train_scale=None,
    ):
        if train_scale is None:
            train_scale = train_geom
        xy_s2 = self.xy if train_geom else self.xy.detach()
        rot_s2 = self.rot if train_rot else self.rot.detach()
        sc_ms = self._get_scale(float(ms_h) / float(pan_h))
        sc_hr = self._get_scale(1.0)
        if not train_scale:
            sc_ms = sc_ms.detach()
            sc_hr = sc_hr.detach()
        coeff = self._stage2_coeff_features()
        g_lr, _ = self._rasterize_features(
            xy_s2, sc_ms, rot_s2, coeff, ms_h, ms_w, tb_ms
        )
        g_hr, _ = self._rasterize_features(
            xy_s2, sc_hr, rot_s2, coeff, pan_h, pan_w, tb_pan
        )
        return g_lr, g_hr

    def _stage2_self_optimizer(self):
        train_xy = bool(getattr(self.args, "stage2_self_train_xy", True))
        train_rot = bool(getattr(self.args, "stage2_self_train_rot", True))
        train_scale = bool(getattr(self.args, "stage2_self_train_scale", False))
        geo_lr_mult = float(getattr(self.args, "stage2_self_geo_lr_mult", 0.05))
        feat_lr_mult = float(getattr(self.args, "stage2_self_feat_lr_mult", 0.5))

        self.xy.requires_grad_(train_xy)
        self.scale.requires_grad_(train_scale)
        self.rot.requires_grad_(train_rot)
        self.feat_pan.requires_grad_(False)
        self.feat_ms.requires_grad_(True)
        if self.stage2_decoder is not None:
            self.stage2_decoder.requires_grad_(True)

        params = [{"params": self.feat_ms, "lr": self.feat_lr * feat_lr_mult}]
        if self.stage2_decoder is not None:
            params.append(
                {
                    "params": self.stage2_decoder.parameters(),
                    "lr": self.feat_lr * feat_lr_mult * float(
                        getattr(self.args, "stage2_decoder_lr_mult", 1.0)
                    ),
                }
            )
        if train_xy:
            params.append({"params": self.xy, "lr": self.pos_lr * geo_lr_mult})
        if train_rot:
            params.append({"params": self.rot, "lr": self.rot_lr * geo_lr_mult})
        if train_scale:
            params.append({"params": self.scale, "lr": self.scale_lr * geo_lr_mult})
        self.optimizer = torch.optim.Adam(params)
        return train_xy, train_rot, train_scale

    def _optimize_stage2_self_supervised_only(
        self,
        target_hr,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
        tb_pan,
        tb_ms,
        s1_done_steps,
        s2_steps,
        s2_hr_w,
        s2_pan_w,
        s2_pan_every,
    ):
        """Directly optimize Gaussian HR output X=render(G*P) with LRMS supervision."""
        warmup_steps = max(0, int(getattr(self.args, "stage2_self_warmup_steps", s2_steps)))
        refine_steps = max(0, int(getattr(self.args, "stage2_self_steps", 800)))
        lr_w = float(getattr(self.args, "stage2_self_lr_weight", 1.0))
        target_w = float(getattr(self.args, "stage2_self_target_weight", 0.0))
        pan_loss_w = float(getattr(self.args, "stage2_self_pan_weight", s2_pan_w))
        geo_reg_w = float(getattr(self.args, "stage2_self_geo_reg_weight", 0.0))
        coeff_reg_w = float(getattr(self.args, "stage2_self_coeff_reg_weight", 0.0))
        sam_w = float(getattr(self.args, "stage2_self_sam_weight", 0.0))
        eval_every = max(1, int(getattr(self.args, "stage2_self_eval_every", 50)))
        select_lr_floor = float(getattr(self.args, "stage2_self_select_lr_floor", 30.0))
        select_target_w = float(getattr(self.args, "stage2_self_select_target_weight", 50.0))
        select_pan_w = float(getattr(self.args, "stage2_self_select_pan_weight", 1.0))
        select_sam_w = float(getattr(self.args, "stage2_self_select_sam_weight", 0.02))
        want_train_xy = bool(getattr(self.args, "stage2_self_train_xy", True))
        want_train_rot = bool(getattr(self.args, "stage2_self_train_rot", True))
        want_train_scale = bool(getattr(self.args, "stage2_self_train_scale", False))

        self._initialize_stage2_feat_ms(target_hr)
        warmup_train_geom = bool(getattr(self.args, "stage2_finetune_geometry", False))
        self._set_trainable_modules(stage=2)
        train_xy = warmup_train_geom
        train_rot = False
        train_scale = warmup_train_geom
        xy_anchor = self.xy.detach().clone()
        rot_anchor = self.rot.detach().clone()
        scale_anchor = self.scale.detach().clone()
        feat_anchor = self.feat_ms.detach().clone()
        self_phase_started = False

        self.worklog.info(
            ">>> Stage2 self-supervised Gaussian X: "
            f"warmup={warmup_steps}, refine={refine_steps}, lr_w={lr_w:.3f}, "
            f"target_w={target_w:.3f}, pan_w={pan_loss_w:.3f}, "
            f"xy={want_train_xy}, rot={want_train_rot}, scale={want_train_scale}"
        )

        def _render_train_pair():
            return self._render_stage2_pair(
                pan_h,
                pan_w,
                ms_h,
                ms_w,
                tb_pan,
                tb_ms,
                train_geom=train_xy,
                train_rot=train_rot,
                train_scale=train_scale,
            )

        total_steps = warmup_steps + refine_steps
        best_score = -float("inf")
        best_hr = None

        for step in range(1, total_steps + 1):
            self.step = s1_done_steps + step
            if step > warmup_steps and not self_phase_started:
                train_xy, train_rot, train_scale = self._stage2_self_optimizer()
                xy_anchor = self.xy.detach().clone()
                rot_anchor = self.rot.detach().clone()
                scale_anchor = self.scale.detach().clone()
                feat_anchor = self.feat_ms.detach().clone()
                self_phase_started = True
            self.optimizer.zero_grad()
            ms_lr, ms_hr = _render_train_pair()

            if step <= warmup_steps:
                total_loss = s2_hr_w * self.l1_loss_ratio * F.l1_loss(ms_hr, target_hr)
                if s2_pan_w > 0 and step % s2_pan_every == 0:
                    total_loss = total_loss + s2_pan_w * self._compute_pan_loss(
                        self._to_pan_estimate(ms_hr), self.ref_images
                    )
            else:
                lr_pred = self._stage2_lr_consistency_image(ms_lr, ms_hr, ms_h, ms_w)
                total_loss = lr_w * F.l1_loss(lr_pred, self.gt_images)
                if target_w > 0:
                    total_loss = total_loss + target_w * F.l1_loss(ms_hr, target_hr)
                if pan_loss_w > 0 and step % s2_pan_every == 0:
                    total_loss = total_loss + pan_loss_w * self._compute_pan_loss(
                        self._to_pan_estimate(ms_hr), self.ref_images
                    )
                if sam_w > 0:
                    total_loss = total_loss + sam_w * self._spectral_angle_loss(
                        ms_lr, self.gt_images
                    )
                if coeff_reg_w > 0:
                    total_loss = total_loss + coeff_reg_w * F.l1_loss(
                        self.feat_ms, feat_anchor
                    )
                if geo_reg_w > 0:
                    if train_xy:
                        total_loss = total_loss + geo_reg_w * F.mse_loss(self.xy, xy_anchor)
                    if train_rot:
                        total_loss = total_loss + geo_reg_w * F.mse_loss(self.rot, rot_anchor)
                    if train_scale:
                        total_loss = total_loss + geo_reg_w * F.mse_loss(
                            self.scale, scale_anchor
                        )

            total_loss.backward()
            self.optimizer.step()
            if train_xy:
                with torch.no_grad():
                    self.xy.clamp_(0.0, 1.0)

            should_eval = (
                step == warmup_steps
                or step == total_steps
                or self.step % self.eval_steps == 0
                or (step > warmup_steps and (step - warmup_steps) % eval_every == 0)
            )
            if should_eval:
                with torch.no_grad():
                    ms_lr_eval, ms_hr_eval = self._render_stage2_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms
                    )
                    mse = F.mse_loss(ms_lr_eval, self.gt_images).clamp(min=1e-12)
                    psnr = -10.0 * torch.log10(mse)
                    sam_deg = self._spectral_angle_loss(ms_lr_eval, self.gt_images) * (
                        180.0 / math.pi
                    )
                    target_l1 = F.l1_loss(ms_hr_eval, target_hr)
                    pan_proxy = self._compute_pan_loss(
                        self._to_pan_estimate(ms_hr_eval), self.ref_images
                    )
                    lr_penalty = torch.relu(
                        torch.tensor(select_lr_floor, dtype=psnr.dtype, device=psnr.device) - psnr
                    )
                    score = (
                        -lr_penalty
                        - select_target_w * target_l1
                        - select_pan_w * pan_proxy
                        - select_sam_w * sam_deg
                    ).item()
                    phase = "warmup" if step <= warmup_steps else "self"
                    self.worklog.info(
                        f"Stage2-X {phase} step {step}/{total_steps} | "
                        f"LR PSNR: {psnr.item():.4f} | drift={target_l1.item():.5f} | "
                        f"pan={pan_proxy.item():.5f} | score={score:.4f} | "
                        f"loss={total_loss.item():.5f}"
                    )
                    if score > best_score:
                        best_score = score
                        best_hr = ms_hr_eval.detach().clone()

        if best_hr is None:
            with torch.no_grad():
                _, best_hr = self._render_stage2_pair(
                    pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms
                )
                best_hr = best_hr.detach().clone()

        best_hr = self._project_output_to_lrms(best_hr, ms_h, ms_w)
        self._save_pansharpened(best_hr, "stage2_hr", pan_h, pan_w)
        self._save_pansharpened(best_hr, "stage3", pan_h, pan_w)
        final_lr = self._degrade_to_lr(best_hr, ms_h, ms_w)
        mse = F.mse_loss(final_lr, self.gt_images).clamp(min=1e-12)
        return (-10.0 * torch.log10(mse)).item(), 0.0

    def _optimize_stage2_variational_only(
        self,
        target_hr,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
        tb_pan,
        tb_ms,
        s1_done_steps,
        s2_steps,
        s2_lr_w,
        s2_hr_w,
        s2_pan_w,
        s2_pan_every,
    ):
        """PSDIP-style ablation: optimize explicit HRMS X and Stage2 G only.

        X can be updated with just degrade(X) ~= LRMS, leaving high-frequency
        content mostly inherited from its initialization. G is then fit to X.
        """
        rounds = max(1, int(getattr(self.args, "stage2_variational_rounds", 5)))
        x_steps = max(0, int(getattr(self.args, "stage2_variational_x_steps", 100)))
        x_lr = float(getattr(self.args, "stage2_variational_x_lr", 5e-2))
        x_lr_w = float(getattr(self.args, "stage2_variational_x_lr_weight", 1.0))
        x_prior_w = float(getattr(self.args, "stage2_variational_x_prior_weight", 0.0))
        x_pan_w = float(getattr(self.args, "stage2_variational_x_pan_weight", 0.0))
        output_mode = str(getattr(self.args, "stage2_variational_output", "x")).lower()
        s2_finetune_geo = bool(getattr(self.args, "stage2_finetune_geometry", False))
        s2_train_xy = bool(getattr(self.args, "stage2_finetune_xy", s2_finetune_geo))
        s2_train_scale = bool(
            getattr(self.args, "stage2_finetune_scale", s2_finetune_geo)
        )
        s2_train_rot = bool(getattr(self.args, "stage2_finetune_rot", False))

        self.worklog.info(
            ">>> Stage2 variational-only ablation: "
            f"rounds={rounds}, x_steps={x_steps}, x_prior={x_prior_w:.3f}, "
            f"x_pan={x_pan_w:.3f}, output={output_mode}"
        )

        x_target = target_hr.detach().clone().clamp(0.0, 1.0)
        best_score = -float("inf")
        best_out = None

        for rnd in range(rounds):
            self._initialize_stage2_feat_ms(x_target, use_bicubic=False)
            self._set_trainable_modules(stage=2)
            self.worklog.info(f">>> Var round {rnd+1}/{rounds}: fit G to X")

            for step in range(1, s2_steps + 1):
                self.step = s1_done_steps + step
                self.optimizer.zero_grad()
                ms_lr, ms_hr = self._render_stage2_pair(
                    pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    train_geom=s2_train_xy,
                    train_rot=s2_train_rot,
                    train_scale=s2_train_scale,
                )
                total_loss = s2_hr_w * self.l1_loss_ratio * F.l1_loss(ms_hr, x_target)
                if s2_lr_w > 0:
                    total_loss = total_loss + s2_lr_w * self.l1_loss_ratio * F.l1_loss(
                        ms_lr, self.gt_images
                    )
                if s2_pan_w > 0 and step % s2_pan_every == 0:
                    total_loss = total_loss + s2_pan_w * self._compute_pan_loss(
                        self._to_pan_estimate(ms_hr), self.ref_images
                    )
                total_loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                ms_lr, ms_hr = self._render_stage2_pair(
                    pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                )
                fr = full_ref_metrics(ms_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                self.worklog.info(
                    f"Var round {rnd+1} G Eval | LR PSNR: {fr['PSNR'].item():.4f} | "
                    f"SSIM: {fr['SSIM'].item():.4f}"
                )

            if x_steps > 0:
                x_param = nn.Parameter(x_target.detach().clone())
                opt_x = torch.optim.Adam([x_param], lr=x_lr)
                ms_hr_det = ms_hr.detach()
                for _ in range(x_steps):
                    opt_x.zero_grad()
                    x_clamped = x_param.clamp(0.0, 1.0)
                    x_lr_pred = self._degrade_to_lr(x_clamped, ms_h, ms_w)
                    loss_x = x_lr_w * F.l1_loss(x_lr_pred, self.gt_images)
                    if x_prior_w > 0:
                        loss_x = loss_x + x_prior_w * F.l1_loss(x_clamped, ms_hr_det)
                    if x_pan_w > 0:
                        loss_x = loss_x + x_pan_w * self._compute_pan_loss(
                            self._to_pan_estimate(x_clamped), self.ref_images
                        )
                    loss_x.backward()
                    opt_x.step()
                    with torch.no_grad():
                        x_param.clamp_(0.0, 1.0)
                x_target = x_param.detach().clone()
            else:
                x_target = ms_hr.detach().clone().clamp(0.0, 1.0)

            with torch.no_grad():
                out_hr = x_target if output_mode == "x" else ms_hr.detach()
                out_lr = self._degrade_to_lr(out_hr, ms_h, ms_w)
                mse = F.mse_loss(out_lr, self.gt_images).clamp(min=1e-12)
                psnr = -10.0 * torch.log10(mse)
                sam_deg = self._spectral_angle_loss(out_lr, self.gt_images) * (
                    180.0 / math.pi
                )
                sam_weight = float(getattr(self.args, "stage3_proxy_sam_weight", 0.0))
                score = (psnr - sam_weight * sam_deg).item()
                fr = full_ref_metrics(out_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                self.worklog.info(
                    f"Var round {rnd+1} X Eval | LR PSNR: {fr['PSNR'].item():.4f} | "
                    f"score={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_out = out_hr.detach().clone()

        if best_out is None:
            best_out = x_target.detach().clone()
        best_out = self._project_output_to_lrms(best_out, ms_h, ms_w)
        self._save_pansharpened(best_out, "stage2_hr", pan_h, pan_w)
        self._save_pansharpened(best_out, "stage3", pan_h, pan_w)
        final_lr = self._degrade_to_lr(best_out, ms_h, ms_w)
        mse = F.mse_loss(final_lr, self.gt_images).clamp(min=1e-12)
        return (-10.0 * torch.log10(mse)).item(), 0.0

    # -- Progressive densification ---------------------------------------------

    def _add_gaussians_pan_guided(self, add_num):
        s1_target = int(getattr(self.args, "stage1_target_gaussians", 0))
        cap = self.total_num_gaussians if s1_target <= 0 else min(s1_target, self.total_num_gaussians)
        add_num = min(add_num, self.max_add_num, cap - self.num_gaussians)
        if add_num <= 0:
            return

        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        tile_bounds = (
            (pan_w + self.block_w - 1) // self.block_w,
            (pan_h + self.block_h - 1) // self.block_h, 1,
        )

        with torch.no_grad():
            render_pan, _ = self._rasterize_features(
                self.xy, self._get_scale(upsample_ratio=1.0), self.rot, self.feat_pan,
                pan_h, pan_w, tile_bounds,
            )
            est = self._to_pan_estimate(render_pan)
            imgs = torch.pow(torch.clamp(est, 0.0, 1.0), 1.0 / self.gamma)
            ref = torch.pow(self.ref_images, 1.0 / self.gamma)
            diff = (self._highpass(ref) - self._highpass(imgs)).detach()
            err = torch.pow(torch.abs(diff).mean(dim=0).reshape(-1), 2.0)
            err_prob = (err / (err.sum() + 1e-7)).cpu().numpy().astype(np.float64)

            sal_w = float(getattr(self.args, "densify_saliency_weight", 0.0))
            uni_w = float(getattr(self.args, "densify_uniform_weight", 0.0))
            res_w = float(getattr(self.args, "densify_residual_weight", 1.0))
            prob = max(res_w, 0.0) * err_prob
            if sal_w > 0.0:
                sal_target = str(getattr(self.args, "densify_saliency_target", "pan")).lower()
                if sal_target in ("residual", "error", "pan_residual"):
                    sal_img = torch.abs(diff).mean(dim=0).detach().cpu().numpy()
                elif sal_target in ("render", "rendered", "pan_render"):
                    sal_img = imgs.detach().cpu().numpy()
                else:
                    sal_img = ref.detach().cpu().numpy()
                sal = self._spectral_residual_saliency_map(sal_img).reshape(-1).astype(np.float64)
                sal_power = float(getattr(self.args, "densify_saliency_power", 1.0))
                sal = np.power(np.clip(sal, 0.0, None) + 1.0e-8, sal_power)
                sal = sal / (float(sal.sum()) + 1.0e-12)
                prob = prob + sal_w * sal
            if uni_w > 0.0:
                prob = prob + uni_w / float(pan_h * pan_w)
            prob_sum = float(prob.sum())
            if prob_sum <= 1.0e-12:
                prob = np.ones(pan_h * pan_w, dtype=np.float64) / float(pan_h * pan_w)
            else:
                prob = prob / prob_sum
            cov_alpha = float(getattr(self.args, "densify_coverage_alpha", 0.0))
            if cov_alpha > 0.0 and self.xy is not None and self.xy.shape[0] > 0:
                cov_sigma = float(getattr(self.args, "densify_coverage_sigma_px", 4.0))
                density = self._local_density_map(
                    self.xy.detach(), pan_h, pan_w, cov_sigma
                ).reshape(-1).detach().cpu().numpy()
                density = density / max(float(density.mean()), 1.0e-6)
                prob = prob / np.power(np.clip(density, 1.0e-3, None), cov_alpha)
                prob = prob / max(float(prob.sum()), 1.0e-12)

        available_mask = self._available_sampling_mask(
            pan_h,
            pan_w,
            xy=self.xy.detach(),
            radius_px=int(getattr(self.args, "densify_exclude_existing_radius_px", -1)),
        )
        sel = self._sample_spatial_indices(
            score_map=prob.reshape(pan_h, pan_w),
            pixel_prob=prob,
            h=pan_h,
            w=pan_w,
            n=add_num,
            uniform_ratio=float(getattr(self.args, "densify_uniform_ratio", 0.0)),
            block=int(getattr(self.args, "densify_block", 0)),
            min_per_block=int(getattr(self.args, "densify_min_per_block", 0)),
            quota_power=float(getattr(self.args, "densify_quota_power", 1.0)),
            uniform_within=bool(getattr(self.args, "densify_uniform_within", False)),
            available_mask=available_mask,
            max_per_block=int(getattr(self.args, "densify_max_per_block", 0)),
        )
        add_num = int(sel.shape[0])
        if add_num <= 0:
            return
        new_xy = torch.zeros(add_num, 2, dtype=self.dtype, device=self.device)
        new_xy[:, 0] = (torch.tensor(sel % pan_w, device=self.device) + 0.5) / pan_w
        new_xy[:, 1] = (torch.tensor(sel // pan_w, device=self.device) + 0.5) / pan_h

        new_scale_px = getattr(self.args, "densify_new_scale_pixels", None)
        if new_scale_px is not None and float(new_scale_px) > 0.0:
            anneal = str(
                getattr(self.args, "densify_new_scale_anneal", "constant")
            ).lower()
            if anneal == "narrow_each_round":
                round_idx = int(self._densify_round_count)
                decay = float(getattr(self.args, "densify_new_scale_anneal_factor", 0.7))
                eff_px = float(new_scale_px) * (decay ** round_idx)
            else:
                eff_px = float(new_scale_px)
            pan_w_local = float(self.ref_images.shape[-1])
            new_actual = max(eff_px, 1.0e-6) / pan_w_local
            isv = (1.0 / new_actual) if not self.disable_inverse_scale else new_actual
            new_s = torch.full((add_num, 2), isv, dtype=self.dtype, device=self.device)
            self._densify_round_count += 1
        else:
            cm = self._get_scale().mean().detach()
            isv = cm if not self.disable_inverse_scale else 1.0 / cm
            new_s = torch.full((add_num, 2), isv, dtype=self.dtype, device=self.device)
        new_r = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_fp = self._get_target_features(new_xy, level="pan")
        if self.feat_ms is not None or self.feat_x is not None:
            new_ms = self._get_target_features(new_xy, level="ms")
        else:
            new_ms = None
        if self.feat_ms is not None:
            new_base = self._safe_base_features(self._stage2_base_features_at(new_xy))
            new_fm = self._pack_stage2_coeff_features(new_ms / new_base)
        if self.feat_x is not None:
            new_fx = new_ms
        if self.feat_ms_detail is not None:
            new_fm_detail = torch.zeros(
                add_num, self.output_dim, dtype=self.dtype, device=self.device
            )

        self.num_gaussians += add_num
        self.xy = nn.Parameter(torch.cat([self.xy.detach(), new_xy], 0))
        self.scale = nn.Parameter(torch.cat([self.scale.detach(), new_s], 0))
        self.rot = nn.Parameter(torch.cat([self.rot.detach(), new_r], 0))
        self.feat_pan = nn.Parameter(torch.cat([self.feat_pan.detach(), new_fp], 0))
        if self.feat_ms is not None:
            self.feat_ms = nn.Parameter(torch.cat([self.feat_ms.detach(), new_fm], 0))
        if self.feat_x is not None:
            self.feat_x = nn.Parameter(torch.cat([self.feat_x.detach(), new_fx], 0))
        if self.feat_ms_detail is not None:
            self.feat_ms_detail = nn.Parameter(
                torch.cat([self.feat_ms_detail.detach(), new_fm_detail], 0)
            )

        params = [
            {"params": self.xy, "lr": self.pos_lr},
            {"params": self.scale, "lr": self.scale_lr},
            {"params": self.rot, "lr": self.rot_lr},
            {"params": self.feat_pan, "lr": self.feat_lr},
        ]
        if self.feat_ms is not None:
            params.append({"params": self.feat_ms, "lr": self.feat_lr})
        if self.feat_x is not None:
            params.append({"params": self.feat_x, "lr": self.feat_lr})
        self.optimizer = torch.optim.Adam(params)
        self.worklog.info(f"Densification: Added {add_num} Gaussians.")

    def _add_gaussians_ms_guided(
        self, add_num, ms_hr_rendered, s2_ratio_optimizer, ratio_hr=None, gt_hr=None
    ):
        """Stage 2 densification: sample new Gaussians ∝ |∇MS_HR|² (mean over bands).

        Targets MS-spectral-variation regions that PAN gradient may miss. New
        rows in feat_ratio are initialised to the current rendered ratio at
        each new Gaussian's pixel so that insertion does not perturb the
        existing topk-normalised output. Returns a fresh optimizer over the
        resized feat_ratio.
        """
        add_num = min(add_num, self.total_num_gaussians - self.num_gaussians)
        if add_num <= 0:
            return s2_ratio_optimizer
        train_new_geo = bool(getattr(self.args, "stage2_densify_train_new_geo", False))
        frozen_geo_n = int(
            getattr(self, "_stage2_base_gaussian_count", int(self.num_gaussians))
        )

        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        with torch.no_grad():
            mode = str(getattr(self.args, "stage2_densify_mode", "ms_rendered")).lower()
            if mode in (
                "lr_residual_pan_hf",
                "lr_residual_panhf",
                "ucl_residual_pan_hf",
                "obs_residual_pan_hf",
            ):
                x = ms_hr_rendered.detach()
                if x.dim() == 4:
                    x = x.squeeze(0)
                lr_pred = self._ucl_degrade_once(x).detach()
                lr_err = (lr_pred - self.gt_images.detach()).abs().mean(
                    dim=0, keepdim=True
                )
                err_up = F.interpolate(
                    lr_err.unsqueeze(0),
                    size=(pan_h, pan_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).clamp_min(0.0)
                pan = torch.pow(self.ref_images.detach().clamp(0.0, 1.0), 1.0 / self.gamma)
                sigma = float(getattr(self.args, "stage2_densify_pan_hf_sigma", 2.0))
                pan_hf = (pan - self._gaussian_lowpass_sigma(pan, sigma)).abs().mean(
                    dim=0, keepdim=True
                )
                pan_hf = pan_hf / pan_hf.mean().clamp_min(1.0e-6)
                err_power = float(getattr(self.args, "stage2_densify_lr_err_power", 1.0))
                pan_power = float(getattr(self.args, "stage2_densify_pan_hf_power", 1.0))
                g2 = (
                    err_up.clamp_min(0.0).pow(err_power)
                    * pan_hf.clamp_min(0.0).pow(pan_power)
                ).reshape(-1)
                score = g2.detach()
                s = float(g2.sum().item())
                if s <= 1e-12:
                    return s2_ratio_optimizer
                prob = (g2 / (s + 1e-12)).cpu().numpy()
            elif mode in ("pan_highfreq", "pan_highpass", "pan_hf", "pan_gradient", "pan"):
                pan = torch.pow(self.ref_images.detach().clamp(0.0, 1.0), 1.0 / self.gamma)
                hf_power = float(getattr(self.args, "stage2_densify_pan_hf_power", 2.0))
                score_stats = {"score_mode": mode}
                if mode in ("pan_highfreq", "pan_highpass", "pan_hf"):
                    score_mode = str(
                        getattr(self.args, "stage2_densify_score_mode", "pan_hf")
                    ).lower()
                    sigma = float(getattr(self.args, "stage2_densify_pan_hf_sigma", 2.0))
                    pan_hf = (pan - self._gaussian_lowpass_sigma(pan, sigma)).abs().mean(
                        dim=0, keepdim=True
                    )
                    pan_hf = pan_hf / pan_hf.mean().clamp_min(1.0e-6)
                    ms = ms_hr_rendered.detach()
                    if ms.dim() == 4:
                        ms = ms.squeeze(0)
                    intensity = ms.mean(dim=0, keepdim=True)
                    intensity_hf = (intensity - self._gaussian_lowpass_sigma(intensity, sigma)).abs()
                    hf_error = (intensity_hf - pan_hf).abs()
                    lr_err_hr = None
                    if score_mode in ("pan_hf_lr_error", "mixed", "mixed_observable_error"):
                        lr_pred = self._ucl_degrade_once(ms).detach()
                        lr_err = (lr_pred - self.gt_images.detach()).abs().mean(
                            dim=0, keepdim=True
                        )
                        lr_err_hr = F.interpolate(
                            lr_err.unsqueeze(0),
                            size=(pan_h, pan_w),
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze(0).clamp_min(0.0)
                    gt_hf = None
                    gt_hf_missing = None
                    if score_mode in (
                        "hr_gt_hf_missing",
                        "gt_hf_missing",
                        "hr_gt_hf_error",
                        "gt_hf_error",
                    ):
                        if gt_hr is None:
                            raise RuntimeError(
                                "stage2_densify_score_mode=hr_gt_hf_missing requires gt_hr"
                            )
                        gt = gt_hr.detach()
                        if gt.dim() == 4:
                            gt = gt.squeeze(0)
                        gt_intensity = gt.mean(dim=0, keepdim=True)
                        gt_hf = (
                            gt_intensity
                            - self._gaussian_lowpass_sigma(gt_intensity, sigma)
                        ).abs()
                        gt_hf_missing = (gt_hf - intensity_hf).clamp_min(0.0)
                    pan_power = float(getattr(self.args, "stage2_densify_pan_power", 1.0))
                    err_power = float(getattr(self.args, "stage2_densify_error_power", 1.0))
                    score_eps = float(getattr(self.args, "stage2_densify_score_eps", 1.0e-8))
                    lr_error_weight = float(getattr(self.args, "stage2_densify_lr_error_weight", 1.0))
                    pan_error_weight = float(
                        getattr(
                            self.args,
                            "stage2_densify_pan_error_weight",
                            getattr(self.args, "stage2_densify_fusion_error_weight", 1.0),
                        )
                    )
                    score_stats = {
                        "score_mode": score_mode,
                        "pan_hf_mean": float(pan_hf.mean().item()),
                        "pan_hf_max": float(pan_hf.max().item()),
                        "hf_error_mean": float(hf_error.mean().item()),
                        "hf_error_max": float(hf_error.max().item()),
                    }
                    if score_mode == "pan_hf":
                        g2 = torch.pow(pan_hf.reshape(-1).clamp_min(0.0), hf_power)
                    elif score_mode == "pan_hf_fusion_error":
                        g2 = torch.pow(pan_hf.reshape(-1).clamp_min(score_eps), pan_power) * torch.pow(
                            hf_error.reshape(-1).clamp_min(score_eps), err_power
                        )
                    elif score_mode == "pan_hf_lr_error":
                        if lr_err_hr is None:
                            raise RuntimeError("lr_err_hr was not computed")
                        g2 = torch.pow(pan_hf.reshape(-1).clamp_min(score_eps), pan_power) * torch.pow(
                            lr_err_hr.mean(dim=0).reshape(-1).clamp_min(score_eps), err_power
                        )
                        score_stats["lr_err_mean"] = float(lr_err_hr.mean().item())
                        score_stats["lr_err_max"] = float(lr_err_hr.max().item())
                    elif score_mode in ("mixed", "mixed_observable_error"):
                        if lr_err_hr is None:
                            raise RuntimeError("lr_err_hr was not computed")
                        g2 = (
                            pan_error_weight
                            * torch.pow(pan_hf.reshape(-1).clamp_min(score_eps), pan_power)
                            * torch.pow(hf_error.reshape(-1).clamp_min(score_eps), err_power)
                            + lr_error_weight
                            * torch.pow(pan_hf.reshape(-1).clamp_min(score_eps), pan_power)
                            * torch.pow(lr_err_hr.mean(dim=0).reshape(-1).clamp_min(score_eps), err_power)
                        )
                        score_stats["lr_err_mean"] = float(lr_err_hr.mean().item())
                        score_stats["lr_err_max"] = float(lr_err_hr.max().item())
                        score_stats["lr_error_weight"] = float(lr_error_weight)
                        score_stats["pan_error_weight"] = float(pan_error_weight)
                    elif score_mode in ("hr_gt_hf_missing", "gt_hf_missing"):
                        if gt_hf_missing is None:
                            raise RuntimeError("gt_hf_missing was not computed")
                        g2 = gt_hf_missing.reshape(-1).clamp_min(score_eps)
                        score_stats["gt_hf_mean"] = float(gt_hf.mean().item())
                        score_stats["gt_hf_max"] = float(gt_hf.max().item())
                        score_stats["ms_hf_mean"] = float(intensity_hf.mean().item())
                        score_stats["ms_hf_max"] = float(intensity_hf.max().item())
                        score_stats["missing_mean"] = float(gt_hf_missing.mean().item())
                        score_stats["missing_max"] = float(gt_hf_missing.max().item())
                    elif score_mode in ("hr_gt_hf_error", "gt_hf_error"):
                        if gt_hf is None:
                            raise RuntimeError("gt_hf was not computed")
                        gt_hf_error = (gt_hf - intensity_hf).abs()
                        g2 = gt_hf_error.reshape(-1).clamp_min(score_eps)
                        score_stats["gt_hf_mean"] = float(gt_hf.mean().item())
                        score_stats["gt_hf_max"] = float(gt_hf.max().item())
                        score_stats["ms_hf_mean"] = float(intensity_hf.mean().item())
                        score_stats["ms_hf_max"] = float(intensity_hf.max().item())
                        score_stats["gt_hf_error_mean"] = float(gt_hf_error.mean().item())
                        score_stats["gt_hf_error_max"] = float(gt_hf_error.max().item())
                    else:
                        raise ValueError(
                            "stage2_densify_score_mode must be pan_hf, "
                            "pan_hf_fusion_error, pan_hf_lr_error, mixed, "
                            "mixed_observable_error, hr_gt_hf_missing, or hr_gt_hf_error"
                        )
                else:
                    grad = self._pan_grad_mag(pan)
                    g2 = torch.pow(grad.mean(dim=0).reshape(-1).clamp_min(0.0), hf_power)
                score, prob_t = self._coverage_aware_prob(
                    score_map=g2,
                    pan_h=pan_h,
                    pan_w=pan_w,
                    xy=self.xy,
                    coverage_alpha=float(
                        getattr(self.args, "stage2_densify_coverage_alpha", 0.0)
                    ),
                    coverage_sigma_px=float(
                        getattr(self.args, "stage2_densify_coverage_sigma_px", 4.0)
                    ),
                    density_gamma=float(
                        getattr(self.args, "stage2_densify_density_gamma", 1.0)
                    ),
                )
                if float(score.sum().item()) <= 1.0e-12:
                    return s2_ratio_optimizer
                prob = prob_t.detach().cpu().numpy()
                score_stats["score_mean"] = float(score.mean().item())
                score_stats["score_max"] = float(score.max().item())
                sample_stats = {}
            elif mode in (
                "pan_render_residual",
                "pan_rendered_residual",
                "render_residual",
                "pan_residual",
                "pan_gap",
            ):
                tb_hr = self._stage2_tile_bounds(pan_h, pan_w)
                rendered_pan, _ = self._rasterize_features(
                    self.xy.detach(),
                    self._get_scale(1.0).detach(),
                    self.rot.detach(),
                    self.feat_pan.detach(),
                    pan_h,
                    pan_w,
                    tb_hr,
                )
                raw_pan = torch.pow(
                    self.ref_images.detach().clamp(0.0, 1.0), 1.0 / self.gamma
                )
                rendered_pan = torch.pow(
                    rendered_pan.detach().clamp(0.0, 1.0), 1.0 / self.gamma
                )
                if bool(getattr(self.args, "stage2_densify_residual_highpass", True)):
                    residual = self._highpass(raw_pan) - self._highpass(rendered_pan)
                else:
                    residual = raw_pan - rendered_pan
                res_power = float(
                    getattr(self.args, "stage2_densify_pan_residual_power", 2.0)
                )
                g2 = torch.pow(
                    residual.abs().mean(dim=0).reshape(-1).clamp_min(0.0),
                    res_power,
                )
                hf_weight = float(
                    getattr(self.args, "stage2_densify_residual_pan_hf_weight", 0.0)
                )
                if hf_weight > 0.0:
                    sigma = float(getattr(self.args, "stage2_densify_pan_hf_sigma", 2.0))
                    low = self._gaussian_lowpass_sigma(raw_pan, sigma)
                    pan_hf = (raw_pan - low).abs().mean(dim=0).reshape(-1)
                    pan_hf = pan_hf / pan_hf.mean().clamp_min(1.0e-6)
                    g2 = g2 * pan_hf.clamp_min(0.0).pow(hf_weight)
                score, prob_t = self._coverage_aware_prob(
                    score_map=g2,
                    pan_h=pan_h,
                    pan_w=pan_w,
                    xy=self.xy,
                    coverage_alpha=float(
                        getattr(self.args, "stage2_densify_coverage_alpha", 0.0)
                    ),
                    coverage_sigma_px=float(
                        getattr(self.args, "stage2_densify_coverage_sigma_px", 4.0)
                    ),
                    density_gamma=float(
                        getattr(self.args, "stage2_densify_density_gamma", 1.0)
                    ),
                )
                if float(score.sum().item()) <= 1.0e-12:
                    return s2_ratio_optimizer
                prob = prob_t.detach().cpu().numpy()
            elif mode in ("lrms_gradient", "lms_gradient", "bicubic_lrms", "lrms"):
                x = self._bicubic_hr_target().detach()
                if x.dim() == 4:
                    x = x.squeeze(0)
                gx = x[:, :, 1:] - x[:, :, :-1]
                gy = x[:, 1:, :] - x[:, :-1, :]
                gx = F.pad(gx, (0, 1, 0, 0))
                gy = F.pad(gy, (0, 0, 0, 1))
                g2 = (gx ** 2 + gy ** 2).mean(dim=0).reshape(-1)
                score = g2.detach()
                s = float(g2.sum().item())
                if s <= 1e-12:
                    return s2_ratio_optimizer
                prob = (g2 / (s + 1e-12)).cpu().numpy()
            else:
                x = ms_hr_rendered.detach()
                if x.dim() == 4:
                    x = x.squeeze(0)
                gx = x[:, :, 1:] - x[:, :, :-1]
                gy = x[:, 1:, :] - x[:, :-1, :]
                gx = F.pad(gx, (0, 1, 0, 0))
                gy = F.pad(gy, (0, 0, 0, 1))
                g2 = (gx ** 2 + gy ** 2).mean(dim=0).reshape(-1)
                score = g2.detach()
                s = float(g2.sum().item())
                if s <= 1e-12:
                    return s2_ratio_optimizer
                prob = (g2 / (s + 1e-12)).cpu().numpy()

        with torch.no_grad():
            bright_mask = self._stage2_bright_mask_map(
                torch.pow(self.ref_images.detach().clamp(0.0, 1.0), 1.0 / self.gamma)
            )
            bright_flat = None
            if bright_mask is not None:
                bright_weight = float(getattr(self.args, "stage2_densify_bright_score_weight", 1.0))
                if bright_weight > 1.0:
                    bright_flat = bright_mask.reshape(-1).to(device=self.device, dtype=score.dtype)
                    score = score * (1.0 + (bright_weight - 1.0) * bright_flat)
                    s = float(score.sum().item())
                    if s > 1.0e-12:
                        prob = (score / (s + 1.0e-12)).detach().cpu().numpy()
                elif bright_weight <= 0.0:
                    bright_flat = bright_mask.reshape(-1).to(device=self.device, dtype=score.dtype)
                if "score_stats" in locals():
                    stats = getattr(self, "_stage2_last_bright_mask_stats", {})
                    score_stats["bright_area"] = float(stats.get("area", 0.0))
                    score_stats["bright_weight"] = float(bright_weight)
        sample_mode = str(
            getattr(self.args, "stage2_densify_sample_mode", "prob")
        ).lower()
        default_exclude = 0 if sample_mode in ("topk", "top_k", "d3", "distortion_topk") else -1
        exclude_radius = int(
            getattr(self.args, "stage2_densify_exclude_existing_radius_px", default_exclude)
        )
        available_mask = self._available_sampling_mask(
            pan_h,
            pan_w,
            xy=self.xy.detach(),
            radius_px=exclude_radius,
        )
        uniform_ratio = float(getattr(self.args, "stage2_densify_uniform_ratio", 0.0))
        uniform_ratio = max(0.0, min(1.0, uniform_ratio))

        def _sample_stage2_part(n_select, part_available_mask):
            n_select = int(n_select)
            if n_select <= 0:
                return np.empty((0,), dtype=np.int64)
            uniform_num = int(round(n_select * uniform_ratio))
            uniform_num = max(0, min(n_select, uniform_num))
            guided_num = max(0, n_select - uniform_num)
            if sample_mode in ("topk", "top_k", "d3", "distortion_topk"):
                score_sel = score.clone()
                if exclude_radius >= 0:
                    avail_t = torch.from_numpy(part_available_mask).to(self.device)
                    score_sel[~avail_t] = -1.0
                guided_num = min(guided_num, int((score_sel >= 0).sum().item()))
                if guided_num > 0:
                    guided = torch.topk(score_sel, k=guided_num, largest=True).indices.cpu().numpy()
                else:
                    guided = np.empty((0,), dtype=np.int64)
                if uniform_num > 0:
                    mask = part_available_mask.copy()
                    mask[guided] = False
                    candidates = np.flatnonzero(mask)
                    if candidates.size > 0:
                        take = min(uniform_num, candidates.size)
                        uniform = np.random.choice(candidates, take, replace=False)
                        sel_part = np.concatenate([guided, uniform])
                    else:
                        sel_part = guided
                else:
                    sel_part = guided
                if sel_part.shape[0] < n_select:
                    mask = part_available_mask.copy()
                    mask[sel_part] = False
                    candidates = np.flatnonzero(mask)
                    fill = min(n_select - sel_part.shape[0], candidates.size)
                    if fill > 0:
                        extra = np.random.choice(candidates, fill, replace=False, p=None)
                        sel_part = np.concatenate([sel_part, extra])
                return sel_part
            return self._sample_spatial_indices(
                score_map=score.reshape(pan_h, pan_w).detach().cpu().numpy(),
                pixel_prob=prob,
                h=pan_h,
                w=pan_w,
                n=n_select,
                uniform_ratio=uniform_ratio,
                block=int(getattr(self.args, "stage2_densify_block", 0)),
                min_per_block=int(getattr(self.args, "stage2_densify_min_per_block", 0)),
                quota_power=float(getattr(self.args, "stage2_densify_quota_power", 1.0)),
                uniform_within=bool(
                    getattr(self.args, "stage2_densify_uniform_within", False)
                ),
                available_mask=part_available_mask,
                max_per_block=int(getattr(self.args, "stage2_densify_max_per_block", 0)),
            )

        bright_quota_ratio = float(
            getattr(self.args, "stage2_densify_bright_add_ratio", 0.0)
        )
        bright_quota_ratio = max(0.0, min(1.0, bright_quota_ratio))
        bright_mask_np = None
        if "bright_mask" in locals() and bright_mask is not None:
            bright_mask_np = bright_mask.reshape(-1).detach().cpu().numpy().astype(bool)

        if bright_mask_np is not None and bright_quota_ratio > 0.0:
            bright_num = int(round(add_num * bright_quota_ratio))
            bright_num = max(0, min(add_num, bright_num))
            bright_avail = available_mask & bright_mask_np
            hf_avail = available_mask & (~bright_mask_np)
            bright_sel = _sample_stage2_part(bright_num, bright_avail)
            remaining_mask = hf_avail.copy()
            if bright_sel.size > 0:
                remaining_mask[bright_sel] = False
            hf_num = add_num - bright_num
            hf_sel = _sample_stage2_part(hf_num, remaining_mask)
            sel = np.concatenate([hf_sel, bright_sel]) if bright_sel.size > 0 else hf_sel
            if sel.shape[0] < add_num:
                fill_mask = available_mask.copy()
                fill_mask[sel] = False
                fill = add_num - sel.shape[0]
                if fill > 0:
                    extra = _sample_stage2_part(fill, fill_mask)
                    if extra.size > 0:
                        sel = np.concatenate([sel, extra])
            if "score_stats" in locals():
                score_stats["bright_quota_ratio"] = float(bright_quota_ratio)
                score_stats["bright_quota_num"] = float(bright_num)
                score_stats["hf_quota_num"] = float(hf_num)
        else:
            sel = _sample_stage2_part(add_num, available_mask)
        add_num = int(sel.shape[0])
        if add_num <= 0:
            return s2_ratio_optimizer
        if "score_stats" in locals():
            sel_t = torch.as_tensor(sel, device=self.device, dtype=torch.long)
            flat_score = score.reshape(-1)
            score_stats["sample_score_mean"] = float(flat_score[sel_t].mean().item())
            score_stats["sample_score_max"] = float(flat_score[sel_t].max().item())
            if "pan_hf" in locals():
                flat_pan_hf = pan_hf.reshape(-1)
                score_stats["sample_pan_hf_mean"] = float(flat_pan_hf[sel_t].mean().item())
            if "hf_error" in locals():
                flat_hf_error = hf_error.reshape(-1)
                score_stats["sample_hf_error_mean"] = float(flat_hf_error[sel_t].mean().item())
            if "lr_err_hr" in locals() and lr_err_hr is not None:
                flat_lr_err = lr_err_hr.mean(dim=0).reshape(-1)
                score_stats["sample_lr_err_mean"] = float(flat_lr_err[sel_t].mean().item())
            if "gt_hf" in locals() and gt_hf is not None:
                flat_gt_hf = gt_hf.reshape(-1)
                score_stats["sample_gt_hf_mean"] = float(flat_gt_hf[sel_t].mean().item())
            if "gt_hf_missing" in locals() and gt_hf_missing is not None:
                flat_missing = gt_hf_missing.reshape(-1)
                score_stats["sample_missing_mean"] = float(flat_missing[sel_t].mean().item())
            log_parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in score_stats.items()]
            self.worklog.info("Stage2 densify score | " + " ".join(log_parts))
        new_xy = torch.zeros(add_num, 2, dtype=self.dtype, device=self.device)
        new_xy[:, 0] = (torch.tensor(sel % pan_w, device=self.device) + 0.5) / pan_w
        new_xy[:, 1] = (torch.tensor(sel // pan_w, device=self.device) + 0.5) / pan_h
        bright_sel = None
        if "bright_mask" in locals() and bright_mask is not None:
            bright_sel = bright_mask.reshape(-1)[torch.as_tensor(sel, device=self.device, dtype=torch.long)] > 0.5

        new_scale_px = float(getattr(self.args, "stage2_densify_new_scale_pixels", 0.0))
        if new_scale_px > 0.0:
            new_actual = max(new_scale_px, 1.0e-6) / float(pan_w)
            isv = new_actual if self.disable_inverse_scale else 1.0 / new_actual
        else:
            cm = self._get_scale().mean().detach()
            isv = cm if not self.disable_inverse_scale else 1.0 / cm
            s_mult = float(getattr(self.args, "stage2_densify_new_scale_mult", 1.0))
            if s_mult > 0.0 and abs(s_mult - 1.0) > 1.0e-6:
                if self.disable_inverse_scale:
                    isv = isv * s_mult
                else:
                    isv = isv / max(s_mult, 1.0e-6)
        new_s = torch.full((add_num, 2), isv, dtype=self.dtype, device=self.device)
        if bright_sel is not None and bool(bright_sel.any().item()):
            bright_scale_mult = float(getattr(self.args, "stage2_densify_bright_scale_mult", 1.0))
            if bright_scale_mult > 0.0 and abs(bright_scale_mult - 1.0) > 1.0e-6:
                if self.disable_inverse_scale:
                    new_s[bright_sel] = new_s[bright_sel] * bright_scale_mult
                else:
                    new_s[bright_sel] = new_s[bright_sel] / max(bright_scale_mult, 1.0e-6)
        new_r = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        shape_mode = str(
            getattr(self.args, "stage2_densify_shape_mode", "default")
        ).lower()
        if shape_mode in ("edge_aligned", "pan_edge", "edge", "anisotropic_edge"):
            along_px = max(
                1.0e-3,
                float(getattr(self.args, "stage2_densify_edge_along_pixels", 2.0)),
            )
            across_px = max(
                1.0e-3,
                float(getattr(self.args, "stage2_densify_edge_across_pixels", 0.6)),
            )
            actual = torch.empty(add_num, 2, dtype=self.dtype, device=self.device)
            actual[:, 0] = along_px / float(pan_w)
            actual[:, 1] = across_px / float(pan_w)
            new_s = actual if self.disable_inverse_scale else 1.0 / actual

            grad_x, grad_y = self._pan_grad_xy(self.ref_images.detach())
            gx = grad_x.mean(dim=0).reshape(-1)[
                torch.as_tensor(sel, device=self.device, dtype=torch.long)
            ]
            gy = grad_y.mean(dim=0).reshape(-1)[
                torch.as_tensor(sel, device=self.device, dtype=torch.long)
            ]
            axis = str(
                getattr(self.args, "stage2_densify_edge_axis", "tangent")
            ).lower()
            angle = torch.atan2(gy, gx)
            if axis in ("tangent", "edge", "along"):
                angle = angle + 0.5 * math.pi
            elif axis not in ("normal", "gradient", "grad", "across"):
                raise ValueError(
                    "stage2_densify_edge_axis must be tangent or normal"
                )
            new_r = angle.unsqueeze(1).to(dtype=self.dtype)
            self.worklog.info(
                "Stage2 densify shape | "
                f"mode={shape_mode} axis={axis} "
                f"along_px={along_px:.3f} across_px={across_px:.3f} "
                f"rot_mean={float(new_r.mean().item()):.4f} "
                f"rot_std={float(new_r.std(unbiased=False).item()):.4f}"
            )
        new_fp = self._get_target_features(new_xy, level="pan")
        new_fm = self._get_target_features(new_xy, level="ms")
        new_fx = self._get_target_features(new_xy, level="ms")
        new_base = self._safe_base_features(self._stage2_base_features_at(new_xy))
        new_fm = self._pack_stage2_coeff_features(new_fm / new_base)
        new_fm_detail = torch.zeros(
            add_num, self.output_dim, dtype=self.dtype, device=self.device
        )
        C = self.feat_ratio.shape[1]
        ratio_init_mode = str(
            getattr(self.args, "stage2_densify_ratio_init", "current")
        ).lower()
        if ratio_init_mode in ("zero", "zeros", "none"):
            new_fr = torch.zeros(add_num, C, dtype=self.dtype, device=self.device)
        elif ratio_hr is not None:
            with torch.no_grad():
                rh = ratio_hr.detach()
                if rh.dim() == 4:
                    rh = rh.squeeze(0)
                # Sample current ratio_hr at the new Gaussian pixel positions
                cols = torch.tensor(sel % pan_w, device=self.device, dtype=torch.long)
                rows = torch.tensor(sel // pan_w, device=self.device, dtype=torch.long)
                new_fr = rh[:, rows, cols].t().contiguous().to(self.dtype)
            if new_fr.shape[1] != C:
                new_fr = torch.zeros(add_num, C, dtype=self.dtype, device=self.device)
        else:
            new_fr = torch.zeros(add_num, C, dtype=self.dtype, device=self.device)
        if bright_sel is not None and bool(bright_sel.any().item()):
            bright_init_mode = str(
                getattr(self.args, "stage2_densify_bright_ratio_init", "off")
            ).lower()
            if bright_init_mode not in ("off", "none", "false", "0", ""):
                lr_h, lr_w = self.gt_images.shape[-2], self.gt_images.shape[-1]
                pan_lr = F.interpolate(
                    self.ref_images.detach().unsqueeze(0),
                    size=(lr_h, lr_w),
                    mode="area",
                ).squeeze(0)
                ratio_lr = self.gt_images.detach() / pan_lr.clamp_min(
                    float(getattr(self.args, "stage2_densify_bright_ratio_eps", 1.0e-6))
                )
                ratio_lr = ratio_lr.clamp(
                    0.0,
                    float(getattr(self.args, "stage2_densify_bright_ratio_clamp", 5.0)),
                )
                ratio_hr = F.interpolate(
                    ratio_lr.unsqueeze(0),
                    size=(pan_h, pan_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
                sampled_ratio = self._sample_features_from_image(ratio_hr, new_xy)
                if bright_init_mode in ("ms_pan_lr_ratio", "lr_ratio", "pan_ms_ratio", "replace"):
                    new_fr[bright_sel] = sampled_ratio[bright_sel].to(self.dtype)
                elif bright_init_mode in ("blend", "current_blend"):
                    alpha = float(getattr(self.args, "stage2_densify_bright_ratio_blend_alpha", 0.5))
                    alpha = max(0.0, min(1.0, alpha))
                    new_fr[bright_sel] = (
                        (1.0 - alpha) * new_fr[bright_sel] + alpha * sampled_ratio[bright_sel]
                    ).to(self.dtype)

        self.num_gaussians += add_num
        self.xy = nn.Parameter(torch.cat([self.xy.detach(), new_xy], 0))
        self.scale = nn.Parameter(torch.cat([self.scale.detach(), new_s], 0))
        self.rot = nn.Parameter(torch.cat([self.rot.detach(), new_r], 0))
        self.feat_pan = nn.Parameter(torch.cat([self.feat_pan.detach(), new_fp], 0))
        self.feat_ms = nn.Parameter(torch.cat([self.feat_ms.detach(), new_fm], 0))
        self.feat_x = nn.Parameter(torch.cat([self.feat_x.detach(), new_fx], 0))
        self.feat_ms_detail = nn.Parameter(
            torch.cat([self.feat_ms_detail.detach(), new_fm_detail], 0)
        )
        old_band_mask = (
            self.feat_ratio_band_mask.detach()
            if self.feat_ratio_band_mask is not None
            else None
        )
        self.feat_ratio = nn.Parameter(torch.cat([self.feat_ratio.detach(), new_fr], 0))
        if self._stage2_ratio_band_mask_enabled():
            if old_band_mask is None:
                old_band_mask = torch.zeros(
                    self.feat_ratio.shape[0] - add_num,
                    C,
                    dtype=self.dtype,
                    device=self.device,
                )
            new_band_mask = torch.zeros(add_num, C, dtype=self.dtype, device=self.device)
            self.feat_ratio_band_mask = nn.Parameter(
                torch.cat([old_band_mask, new_band_mask], dim=0)
            )
        if self._stage2_ratio_poly_enabled():
            old_dx = (
                self.feat_ratio_dx.detach()
                if self.feat_ratio_dx is not None
                else torch.zeros(
                    self.num_gaussians,
                    C,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            old_dy = (
                self.feat_ratio_dy.detach()
                if self.feat_ratio_dy is not None
                else torch.zeros(
                    self.num_gaussians,
                    C,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            new_dx = torch.zeros(add_num, C, dtype=self.dtype, device=self.device)
            new_dy = torch.zeros(add_num, C, dtype=self.dtype, device=self.device)
            self.feat_ratio_dx = nn.Parameter(torch.cat([old_dx, new_dx], dim=0))
            self.feat_ratio_dy = nn.Parameter(torch.cat([old_dy, new_dy], dim=0))
        self._append_stage2_ratio_clusters(
            new_xy,
            new_fr,
            self.ref_images.detach(),
            pan_h,
            pan_w,
        )
        self._build_stage2_feat_ratio_graph()
        if self._stage2_residual_branch_enabled():
            residual_init_mode = self._stage2_residual_init_mode()
            residual_dim = (
                self._stage2_residual_basis_rank()
                if self._stage2_uses_residual_lowrank()
                else self.output_dim
            )
            basis = self._make_stage2_residual_basis() if self._stage2_uses_residual_lowrank() else None
            if residual_init_mode in ("current_error", "error", "bicubic_error"):
                err_img = self._bicubic_hr_target() - ms_hr_rendered.detach().clamp(0.0, 1.0)
                if err_img.dim() == 4:
                    err_img = err_img.squeeze(0)
                sampled_err = self._sample_features_from_image(err_img, new_xy)
                if basis is not None:
                    B = basis().detach()
                    B_pinv = torch.linalg.pinv(B)
                    new_res = sampled_err @ B_pinv
                else:
                    new_res = sampled_err
            else:
                new_res = torch.zeros(
                    add_num, residual_dim, dtype=self.dtype, device=self.device
                )
            old_res = (
                self.feat_residual.detach()
                if self.feat_residual is not None
                else torch.empty(0, residual_dim, dtype=self.dtype, device=self.device)
            )
            self.feat_residual = nn.Parameter(torch.cat([old_res, new_res], 0))
            self.worklog.info(
                f"Stage2 residual branch append: init={residual_init_mode}, "
                f"new_absmean={float(new_res.abs().mean().item()):.6f}, "
                f"dim={residual_dim}, "
                f"total_residual_gaussians={int(self.feat_residual.shape[0])}"
            )
        if self.stage2_pan_detail_coeff is not None:
            new_detail = torch.zeros(
                add_num,
                self.output_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.stage2_pan_detail_coeff = nn.Parameter(
                torch.cat([self.stage2_pan_detail_coeff.detach(), new_detail], 0)
            )
        if self.stage2_pan_gap_gain_coeff is not None:
            new_gap_gain = torch.zeros(
                add_num,
                self.output_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.stage2_pan_gap_gain_coeff = nn.Parameter(
                torch.cat([self.stage2_pan_gap_gain_coeff.detach(), new_gap_gain], 0)
            )
        if self.stage2_lig_gap_coeff is not None:
            new_lig_gap = torch.zeros(
                add_num,
                self.output_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.stage2_lig_gap_coeff = nn.Parameter(
                torch.cat([self.stage2_lig_gap_coeff.detach(), new_lig_gap], 0)
            )

        feat_lr = float(getattr(self.args, "feat_lr", 5e-3))
        params = [{"params": self.feat_ratio, "lr": feat_lr}]
        if self._stage2_ratio_band_mask_enabled() and self.feat_ratio_band_mask is not None:
            params.append({
                "params": self.feat_ratio_band_mask,
                "lr": feat_lr * float(getattr(self.args, "stage2_ratio_band_mask_lr_mult", 1.0)),
            })
        if (
            self._stage2_ratio_poly_enabled()
            and self.feat_ratio_dx is not None
            and self.feat_ratio_dy is not None
        ):
            poly_mult = float(getattr(self.args, "stage2_ratio_poly_lr_mult", 0.2))
            params.append({"params": self.feat_ratio_dx, "lr": feat_lr * poly_mult})
            params.append({"params": self.feat_ratio_dy, "lr": feat_lr * poly_mult})
        if (
            self._stage2_ratio_cluster_enabled()
            and self.stage2_ratio_cluster_scalar is not None
            and self.stage2_ratio_cluster_gain is not None
        ):
            mult = float(getattr(self.args, "stage2_ratio_cluster_lr_mult", 1.0))
            params.append({
                "params": self.stage2_ratio_cluster_scalar,
                "lr": feat_lr * mult,
            })
            params.append({
                "params": self.stage2_ratio_cluster_gain,
                "lr": feat_lr * mult,
            })
        if self._stage2_residual_branch_enabled() and self.feat_residual is not None:
            params.append({
                "params": self.feat_residual,
                "lr": feat_lr * float(getattr(self.args, "stage2_residual_lr_mult", 1.0)),
            })
            if self._stage2_uses_residual_lowrank() and self.stage2_residual_basis_B is not None:
                params.append({
                    "params": list(self.stage2_residual_basis_B.parameters()),
                    "lr": feat_lr * float(getattr(self.args, "stage2_residual_basis_lr_mult", 0.5)),
                })
        if self.stage2_pixel_residual is not None:
            params.append({
                "params": self.stage2_pixel_residual,
                "lr": feat_lr
                * float(getattr(self.args, "stage2_ucl_pixel_residual_lr_mult", 1.0)),
            })
        if self.stage2_grid_residual is not None:
            params.append({
                "params": self.stage2_grid_residual,
                "lr": feat_lr
                * float(getattr(self.args, "stage2_ucl_grid_lr_mult", 1.0)),
            })
        if self.stage2_pan_detail_coeff is not None:
            params.append({
                "params": self.stage2_pan_detail_coeff,
                "lr": feat_lr
                * float(getattr(self.args, "stage2_ucl_pan_detail_lr_mult", 1.0)),
            })
        if self.stage2_pan_gap_gain_coeff is not None:
            params.append({
                "params": self.stage2_pan_gap_gain_coeff,
                "lr": feat_lr
                * float(getattr(self.args, "stage2_ucl_pan_gap_gain_lr_mult", 1.0)),
            })
        if self.stage2_lig_gap_coeff is not None:
            params.append({
                "params": self.stage2_lig_gap_coeff,
                "lr": feat_lr
                * float(getattr(self.args, "stage2_ucl_lig_gap_lr_mult", 1.0)),
            })
        if train_new_geo:
            if not bool(getattr(self.args, "stage2_unfreeze_base_geo", False)):
                self._register_stage2_new_geo_hooks(frozen_geo_n)
            params.extend([
                {
                    "params": self.xy,
                    "lr": self.pos_lr
                    * float(getattr(self.args, "stage2_new_geo_pos_lr_mult", 1.0)),
                },
                {
                    "params": self.scale,
                    "lr": self.scale_lr
                    * float(getattr(self.args, "stage2_new_geo_scale_lr_mult", 1.0)),
                },
                {
                    "params": self.rot,
                    "lr": self.rot_lr
                    * float(getattr(self.args, "stage2_new_geo_rot_lr_mult", 1.0)),
                },
            ])
        s2_ratio_optimizer = torch.optim.Adam(params)
        bright_count = (
            int(bright_sel.to(dtype=torch.int32).sum().item())
            if bright_sel is not None else 0
        )
        self.worklog.info(
            f"Stage2 densification: Added {add_num} {mode}-guided Gaussians "
            f"(total {self.num_gaussians}, train_new_geo={train_new_geo}, "
            f"ratio_init={ratio_init_mode}, bright={bright_count})."
        )
        return s2_ratio_optimizer

    def _register_stage2_new_geo_hooks(self, frozen_count):
        """Freeze Stage1 geometry rows while Stage2 optimizes appended rows."""
        for handle in getattr(self, "_stage2_new_geo_hook_handles", []):
            handle.remove()
        self._stage2_new_geo_hook_handles = []
        frozen_count = max(0, int(frozen_count))

        def _mask_grad(grad):
            if grad is None or frozen_count <= 0:
                return grad
            out = grad.clone()
            out[: min(frozen_count, out.shape[0])] = 0
            return out

        for param in (self.xy, self.scale, self.rot):
            self._stage2_new_geo_hook_handles.append(param.register_hook(_mask_grad))

    def _add_aux_gaussians(self, add_num, final_pan_estim, ref_pan):
        add_num = min(add_num, self.max_add_num, self.total_num_gaussians - self.aux_num_gaussians)
        if add_num <= 0:
            return

        with torch.no_grad():
            diff = self._highpass(ref_pan) - self._highpass(final_pan_estim)
            err = torch.pow(torch.abs(diff).squeeze(), 2.0)
            if err.sum() < 1e-7:
                return
            prob = (err / err.sum()).cpu().numpy().flatten()
            ph, pw = ref_pan.shape[-2], ref_pan.shape[-1]

        sel = np.random.choice(ph * pw, add_num, replace=False, p=prob)
        new_xy = torch.zeros(add_num, 2, dtype=self.dtype, device=self.device)
        new_xy[:, 0] = (torch.tensor(sel % pw, device=self.device) + 0.5) / pw
        new_xy[:, 1] = (torch.tensor(sel // pw, device=self.device) + 0.5) / ph

        csm = self.scale_aux.mean().detach()
        new_s = torch.full((add_num, 2), csm * 0.8, dtype=self.dtype, device=self.device)
        new_r = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_f = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_fr = torch.zeros(add_num, self.output_dim, dtype=self.dtype, device=self.device)

        self.xy_aux = nn.Parameter(torch.cat([self.xy_aux.detach(), new_xy], 0))
        self.scale_aux = nn.Parameter(torch.cat([self.scale_aux.detach(), new_s], 0))
        self.rot_aux = nn.Parameter(torch.cat([self.rot_aux.detach(), new_r], 0))
        self.feat_aux = nn.Parameter(torch.cat([self.feat_aux.detach(), new_f], 0))
        self.feat_aux_res = nn.Parameter(torch.cat([self.feat_aux_res.detach(), new_fr], 0))
        self.aux_num_gaussians += add_num

        res_lr = float(getattr(self.args, "stage3_aux_res_lr_mult", 0.1)) * self.feat_lr
        self.optimizer = torch.optim.Adam([
            {"params": self.xy_aux, "lr": self.pos_lr},
            {"params": self.scale_aux, "lr": self.scale_lr},
            {"params": self.rot_aux, "lr": self.rot_lr},
            {"params": self.feat_aux, "lr": self.feat_lr},
            {"params": self.band_coeff, "lr": self.feat_lr},
            {"params": self.feat_aux_res, "lr": res_lr},
        ])

    def _run_stage3_detail_once(
        self,
        base_ms_hr,
        gt_hr,
        pan_h,
        pan_w,
        ms_h,
        ms_w,
        tb_pan,
        s1_done_steps,
        s2_done_steps,
        s3_steps,
        lambda_ms,
    ):
        s3_inject_mode = str(getattr(self.args, "stage3_injection_mode", "gain")).lower()
        s3_use_lig = s3_inject_mode in ("lig", "additive")
        s3_lig_normalize = bool(getattr(self.args, "stage3_lig_normalize", False))
        s3_aux_highpass = bool(getattr(self.args, "stage3_aux_highpass", True))
        self.worklog.info(
            f">>> Stage 3: Separate detail injection "
            f"(mode={s3_inject_mode}, highpass={s3_aux_highpass}, "
            f"lig_norm={s3_lig_normalize and s3_use_lig})."
        )
        aux_res_clamp = float(getattr(self.args, "stage3_aux_res_clamp", 0.02))
        res_smooth_w = float(getattr(self.args, "stage3_aux_res_smooth_weight", 0.3))
        sam_loss_w = float(getattr(self.args, "stage3_sam_loss_weight", 0.0))
        gain_consistency_w = float(
            getattr(self.args, "stage3_gain_consistency_weight", 0.0)
        )
        lambda_aux = 0.01

        base_ms_hr = base_ms_hr.detach()
        gt_hr = gt_hr.detach()
        self._init_aux_gaussians(self.args)
        wt = self._pan_weights()
        I_base = (base_ms_hr * wt.view(-1, 1, 1)).sum(dim=0, keepdim=True)

        _, _, res_hint, pan_target = self._build_stage3_pan_targets(I_base)

        lig_res_target = lig_res_min = lig_res_max = None
        if s3_use_lig and s3_lig_normalize:
            with torch.no_grad():
                lig_res_target, lig_res_min, lig_res_max = (
                    self._build_residual_target(base_ms_hr, gt_hr)
                )
            self.worklog.info(
                f"  LIG residual target: "
                f"min=[{lig_res_min.squeeze().tolist()}], "
                f"max=[{lig_res_max.squeeze().tolist()}]"
            )
            self._init_aux_pos_scale_feat_(lig_res_target, residual_hint=None)
        else:
            self._init_aux_pos_scale_feat_(self.ref_images, residual_hint=res_hint)

        with torch.no_grad():
            edge = self._pan_grad_mag(self.ref_images)
            edge = (edge / edge.max().clamp_min(1e-6)).clamp(0, 1)
            smooth_mask = 1.0 - edge

        self._set_trainable_modules(stage=3)
        t0 = perf_counter()

        for step in range(1, s3_steps + 1):
            self.step = s1_done_steps + s2_done_steps + step
            self.optimizer.zero_grad()

            aux_I, _ = self._rasterize_features(
                self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux,
                pan_h, pan_w, tb_pan,
            )
            if s3_aux_highpass:
                aux_I = self._smooth_stage3_aux(self._highpass(aux_I))
            else:
                aux_I = self._smooth_stage3_aux(aux_I)

            aux_res, _ = self._rasterize_features(
                self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux_res,
                pan_h, pan_w, tb_pan,
            )
            if s3_aux_highpass:
                aux_res = self._smooth_stage3_aux(self._highpass(aux_res))
            else:
                aux_res = self._smooth_stage3_aux(aux_res)
            if aux_res_clamp > 0:
                aux_res = aux_res.clamp(-aux_res_clamp, aux_res_clamp)

            if s3_use_lig and s3_lig_normalize and lig_res_target is not None:
                aux_combined = self._aux_to_ms(aux_I, aux_res)
                aux_residual = self._denorm_residual(
                    aux_combined, lig_res_min, lig_res_max
                )
                loss_res_fit = self.l1_loss_ratio * F.l1_loss(
                    aux_combined, lig_res_target
                )
                raw_final_ms_hr = (base_ms_hr + aux_residual).clamp(0.0, 1.0)
                raw_final_I = (
                    raw_final_ms_hr * wt.view(-1, 1, 1)
                ).sum(dim=0, keepdim=True)
            else:
                loss_res_fit = None
                aux_ms = self._aux_to_ms(aux_I, aux_res)
                raw_final_ms_hr, raw_final_I = self._inject_stage3_detail(
                    base_ms_hr, I_base, aux_ms
                )

            train_mix = (
                1.0
                if bool(getattr(self.args, "stage3_adaptive_output_mix", False))
                else self._current_stage3_output_mix()
            )
            final_ms_hr, final_I = self._mix_stage3_output_with_ratio(
                base_ms_hr, I_base, raw_final_ms_hr, raw_final_I, train_mix
            )

            loss_pan = self.l1_loss_ratio * F.l1_loss(self._highpass(final_I), pan_target)
            loss_ms = self.l1_loss_ratio * F.l1_loss(final_ms_hr, gt_hr)
            loss_sam = self._spectral_angle_loss(final_ms_hr, gt_hr)
            loss_gain = self._gain_consistency_loss(base_ms_hr, final_ms_hr)
            loss_aux = torch.mean(torch.abs(aux_I)) + torch.mean(torch.abs(aux_res))

            loss_tv = aux_I.new_tensor(0.0)
            if res_smooth_w > 0:
                if aux_res.dim() == 4:
                    dh = torch.abs(aux_res[:, :, 1:, :] - aux_res[:, :, :-1, :])
                    dw = torch.abs(aux_res[:, :, :, 1:] - aux_res[:, :, :, :-1])
                else:
                    dh = torch.abs(aux_res[:, 1:, :] - aux_res[:, :-1, :])
                    dw = torch.abs(aux_res[:, :, 1:] - aux_res[:, :, :-1])
                loss_tv = (
                    torch.mean(dh * smooth_mask[..., :-1, :])
                    + torch.mean(dw * smooth_mask[..., :, :-1])
                )

            total_loss = (
                loss_pan
                + lambda_ms * loss_ms
                + sam_loss_w * loss_sam
                + gain_consistency_w * loss_gain
                + lambda_aux * loss_aux
                + res_smooth_w * loss_tv
            )
            if loss_res_fit is not None:
                total_loss = total_loss + loss_res_fit
            total_loss.backward()
            self.optimizer.step()

            if (not self.disable_prog_optim
                and step % self.add_steps == 0
                and self.aux_num_gaussians < self.total_num_gaussians):
                self._add_aux_gaussians(self.max_add_num, final_I, self.ref_images)

            if self.step % self.eval_steps == 0:
                with torch.no_grad():
                    t1 = perf_counter()
                    _eval_lr = self._degrade_to_lr(final_ms_hr, ms_h, ms_w)
                    fr = full_ref_metrics(_eval_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                    self.psnr_curr = fr["PSNR"].item()
                    self.worklog.info(
                        f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                        f"Loss: {total_loss.item():.5f} | LR_PSNR: {self.psnr_curr:.4f}"
                    )
                    t0 = t1

        with torch.no_grad():
            aux_I, _ = self._rasterize_features(
                self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux,
                pan_h, pan_w, tb_pan,
            )
            aux_I = self._smooth_stage3_aux(
                self._highpass(aux_I) if s3_aux_highpass else aux_I
            )
            aux_res, _ = self._rasterize_features(
                self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux_res,
                pan_h, pan_w, tb_pan,
            )
            aux_res = self._smooth_stage3_aux(
                self._highpass(aux_res) if s3_aux_highpass else aux_res
            )
            if aux_res_clamp > 0:
                aux_res = aux_res.clamp(-aux_res_clamp, aux_res_clamp)

            if s3_use_lig and s3_lig_normalize and lig_res_target is not None:
                aux_combined = self._aux_to_ms(aux_I, aux_res)
                aux_residual = self._denorm_residual(
                    aux_combined, lig_res_min, lig_res_max
                )
                raw_final_ms_hr = (base_ms_hr + aux_residual).clamp(0.0, 1.0)
                raw_final_I = (
                    raw_final_ms_hr * wt.view(-1, 1, 1)
                ).sum(dim=0, keepdim=True)
            else:
                aux_ms = self._aux_to_ms(aux_I, aux_res)
                raw_final_ms_hr, raw_final_I = self._inject_stage3_detail(
                    base_ms_hr, I_base, aux_ms
                )
            final_ms_hr, final_I, final_lr, selected_mix = self._select_stage3_output(
                base_ms_hr, I_base, raw_final_ms_hr, raw_final_I, ms_h, ms_w,
                gt_hr=gt_hr,
            )
            fr = full_ref_metrics(final_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
            nr = no_ref_metrics(
                self.gt_images.unsqueeze(0), self.ref_images.unsqueeze(0),
                final_ms_hr.unsqueeze(0), fused_lr_override=final_lr.unsqueeze(0),
            )
            self.psnr_curr = fr["PSNR"].item()
            self._stage3_round_score = self._stage3_proxy_score(final_ms_hr, gt_hr).item()
            self.worklog.info(
                f"Stage3 Eval | mix={selected_mix:.2f} | LR_PSNR: {self.psnr_curr:.4f} | "
                f"LR_SSIM: {fr['SSIM'].item():.4f} | LR_SAM: {fr['SAM'].item():.4f} | "
                f"ERGAS: {fr['ERGAS'].item():.4f} | QNR: {nr['QNR'].item():.4f} | "
                f"SCC: {nr['SCC'].item():.4f}"
            )
            self._save_pansharpened(final_ms_hr, "stage3", pan_h, pan_w)

        return final_ms_hr

    # ==========================================================================
    # Main optimization: Stage 2 loop plus optional separate Stage 3
    # ==========================================================================
    # DashGaussian-style frequency-guided resolution schedule
    # ==========================================================================

    def _compute_freq_schedule(self, image, n_levels=4, max_scale=4):
        """Compute a coarse-to-fine resolution schedule from FFT spectral energy.

        DashGaussian insight: images with less low-frequency content can start
        at higher resolution. We compute the fraction of spectral energy within
        increasingly larger centered windows in the frequency domain.

        Returns list of integer downscale factors, e.g. [4, 2, 1, 1].
        """
        with torch.no_grad():
            fft = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))
            mag = fft.abs()
            h, w = mag.shape[-2], mag.shape[-1]
            E_total = mag.sum().item()
            if E_total < 1e-12:
                return [1] * n_levels

            scales = []
            for i in range(n_levels):
                frac = (i + 1) / n_levels
                wh = max(1, int(h * frac))
                ww = max(1, int(w * frac))
                ch, cw = h // 2, w // 2
                r0, r1 = max(0, ch - wh // 2), min(h, ch + wh // 2)
                c0, c1 = max(0, cw - ww // 2), min(w, cw + ww // 2)
                E_win = mag[..., r0:r1, c0:c1].sum().item()
                scale = max(1, int(round(max_scale * (1.0 - E_win / E_total))))
                scales.append(scale)

        return scales

    # ==========================================================================

    def optimize_three_stage_complete(self):
        pan_h, pan_w = self.ref_images.shape[-2], self.ref_images.shape[-1]
        ms_h, ms_w = self.gt_images.shape[-2], self.gt_images.shape[-1]
        tb_pan = (
            (pan_w + self.block_w - 1) // self.block_w,
            (pan_h + self.block_h - 1) // self.block_h, 1,
        )
        tb_ms = (
            (ms_w + self.block_w - 1) // self.block_w,
            (ms_h + self.block_h - 1) // self.block_h, 1,
        )

        s1_steps = int(getattr(self.args, "stage1_steps", 10000))
        s1_min_steps = min(s1_steps, max(1, int(getattr(self.args, "stage1_min_steps", s1_steps))))
        s1_patience = max(0, int(getattr(self.args, "stage1_early_stop_patience", 0)))
        s1_min_delta = float(getattr(self.args, "stage1_early_stop_min_delta", 0.0))
        s1_min_delta_rel = float(getattr(self.args, "stage1_early_stop_min_delta_rel", 0.0))
        s2_steps = int(getattr(self.args, "stage2_steps", 3000))
        s3_steps = int(getattr(self.args, "stage3_steps", 5000))
        s2_lr_w = float(getattr(self.args, "stage2_lr_weight", 0.0))
        s2_hr_w = float(getattr(self.args, "stage2_hr_weight", 0.7))
        s2_pan_w = float(getattr(self.args, "stage2_pan_weight", 0.06))
        s2_pan_every = max(1, int(getattr(self.args, "stage2_pan_every", 1)))
        s2_finetune_geo = bool(getattr(self.args, "stage2_finetune_geometry", False))
        s2_train_xy = bool(getattr(self.args, "stage2_finetune_xy", s2_finetune_geo))
        s2_train_scale = bool(
            getattr(self.args, "stage2_finetune_scale", s2_finetune_geo)
        )
        s2_train_rot = bool(getattr(self.args, "stage2_finetune_rot", False))
        lambda_ms = float(getattr(self.args, "lambda_ms", 1.0))
        hard_patch_threshold = float(
            getattr(self.args, "hard_patch_stage2_psnr_threshold", 0.0)
        )
        hard_patch_max_rounds = max(0, int(getattr(self.args, "hard_patch_max_rounds", 0)))
        self._hard_patch_active = False
        self._hard_patch_start_round = None

        # == Stage 1: Geometry from PAN ========================================

        s1_freq_guided = bool(getattr(self.args, "stage1_freq_guided", False))
        s1_freq_levels = max(1, int(getattr(self.args, "stage1_freq_levels", 4)))
        s1_freq_max_scale = max(1, int(getattr(self.args, "stage1_freq_max_scale", 4)))

        self.worklog.info(">>> Stage 1: Geometry from PAN only.")
        self.psnr_curr, self.ssim_curr = 0.0, 0.0
        self._set_trainable_modules(stage=1)
        t0 = perf_counter()
        s1_done_steps = s1_steps
        best_s1_loss = float("inf")
        stale_s1_checks = 0

        if s1_freq_guided:
            freq_scales = self._compute_freq_schedule(
                self.ref_images, n_levels=s1_freq_levels, max_scale=s1_freq_max_scale
            )
            self.worklog.info(
                f"  DashGaussian freq schedule: {freq_scales} "
                f"(levels={s1_freq_levels}, max_scale={s1_freq_max_scale})"
            )
        else:
            freq_scales = [1]  # single level = original behaviour

        # Distribute steps across frequency levels
        total_freq_steps = s1_steps
        steps_per_level = [total_freq_steps // len(freq_scales)] * len(freq_scales)
        steps_per_level[-1] += total_freq_steps - sum(steps_per_level)

        global_step = 0
        for level_idx, cur_scale in enumerate(freq_scales):
            cur_h = max(1, pan_h // cur_scale)
            cur_w = max(1, pan_w // cur_scale)

            if cur_scale > 1:
                # Downsampled PAN target for this level
                pan_target_ds = F.interpolate(
                    self.ref_images.unsqueeze(0), size=(cur_h, cur_w), mode="area"
                ).squeeze(0)
                tb_cur = (
                    (cur_w + self.block_w - 1) // self.block_w,
                    (cur_h + self.block_h - 1) // self.block_h, 1,
                )
                scale_ratio = float(cur_h) / float(pan_h)
            else:
                pan_target_ds = self.ref_images
                tb_cur = tb_pan
                scale_ratio = 1.0

            if s1_freq_guided and len(freq_scales) > 1:
                self.worklog.info(
                    f"  Stage1 level {level_idx+1}/{len(freq_scales)}: "
                    f"scale=1/{cur_scale} ({cur_h}×{cur_w}), "
                    f"steps={steps_per_level[level_idx]}"
                )

            # Precompute gradient reference for this resolution
            with torch.no_grad():
                ref_grad = self._pan_grad_mag(pan_target_ds) if self.pan_grad_weight > 0 else None

            for step_in_level in range(1, steps_per_level[level_idx] + 1):
                global_step += 1
                self.step = global_step
                self.optimizer.zero_grad()

                pan_pred, _ = self._rasterize_features(
                    self.xy, self._get_scale(scale_ratio), self.rot, self.feat_pan,
                    cur_h, cur_w, tb_cur,
                )
                loss = self.l1_loss_ratio * F.l1_loss(pan_pred, pan_target_ds)
                if self.pan_grad_weight > 0 and ref_grad is not None:
                    loss += self.pan_grad_weight * F.l1_loss(
                        self._pan_grad_mag(pan_pred), ref_grad,
                    )
                loss.backward()
                self.optimizer.step()

                _s1_target = int(getattr(self.args, "stage1_target_gaussians", 0))
                _s1_cap = self.total_num_gaussians if _s1_target <= 0 else min(_s1_target, self.total_num_gaussians)
                if (not self.disable_prog_optim
                    and not bool(getattr(self.args, "stage1_scaffold_expand", False))
                    and self.num_gaussians < _s1_cap
                    and global_step % self.add_steps == 0):
                    self._add_gaussians_pan_guided(self.max_add_num)

                if global_step % self.eval_steps == 0:
                    t1 = perf_counter()
                    self.worklog.info(
                        f"Step: {global_step:d} | {t1 - t0:.2f}s | "
                        f"PAN Loss: {loss.item():.5f} (scale=1/{cur_scale})"
                    )
                    t0 = t1
                    if s1_patience > 0:
                        loss_val = float(loss.detach().item())
                        min_delta = max(s1_min_delta, abs(best_s1_loss) * s1_min_delta_rel)
                        if math.isinf(best_s1_loss) or loss_val < best_s1_loss - min_delta:
                            best_s1_loss = loss_val
                            stale_s1_checks = 0
                        elif global_step >= s1_min_steps:
                            stale_s1_checks += 1
                            if stale_s1_checks >= s1_patience:
                                s1_done_steps = global_step
                                self.worklog.info(
                                    f"Stage1 early stop at step {global_step:d}/{s1_steps:d} "
                                    f"(best PAN loss={best_s1_loss:.5f})."
                                )
                                break
            else:
                # Inner loop completed without break → continue outer loop
                continue
            # Inner loop hit break (early stop) → break outer loop too
            break

        s1_done_steps = global_step

        with torch.no_grad():
            pan_pred, _ = self._rasterize_features(
                self.xy, self._get_scale(1.0), self.rot, self.feat_pan,
                pan_h, pan_w, tb_pan,
            )
            self.worklog.info(
                f"Stage1 Eval | PAN Loss: {F.l1_loss(pan_pred, self.ref_images).item():.5f}"
            )
            self._snapshot_stage1_pan_state()
            self._save_pansharpened(pan_pred, "stage1", pan_h, pan_w)

        if bool(getattr(self.args, "stage1_scaffold_expand", False)):
            self._expand_pan_scaffold_gaussians()
            self._refine_pan_scaffold_gaussians(
                int(getattr(self.args, "stage1_scaffold_refine_steps", 0)),
                pan_h,
                pan_w,
                tb_pan,
            )

        # == Stage 2 setup =====================================================

        self.worklog.info(">>> Stage 2: Spectral coefficients (G*P mode, MTF-GLP target).")
        self._set_trainable_modules(stage=2)
        self._stage2_densify_count = 0
        self._stage2_base_gaussian_count = int(self.num_gaussians)
        self.feat_residual = None
        self.stage2_ratio_cluster_scalar = None
        self.stage2_ratio_cluster_gain = None
        self.stage2_ratio_cluster_assign = None
        self.feat_ratio_dx = None
        self.feat_ratio_dy = None
        t0 = perf_counter()
        stage2_resolution_curriculum = bool(
            getattr(self.args, "stage2_resolution_curriculum", False)
        )
        self._stage2_mtf_target_cache = {}

        gt_hr = self._build_stage2_target()
        lms_hr = self._bicubic_hr_target()
        _bmu = gt_hr.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        band_w = (1.0 / _bmu) / (1.0 / _bmu).mean()
        densify_gt_hr = self._load_stage2_densify_gt_hr(gt_hr)
        stage2_debug_hr_trace_ref = None
        stage2_debug_hr_trace_every = 0
        if bool(getattr(self.args, "stage2_debug_hr_trace", False)):
            if self._stage2_debug_hr_trace_source_is_real_gt():
                self._stage2_debug_hr_trace = []
                stage2_debug_hr_trace_ref = densify_gt_hr.detach()
                stage2_debug_hr_trace_every = int(
                    getattr(self.args, "stage2_debug_hr_trace_every", 0)
                )
                if stage2_debug_hr_trace_every <= 0:
                    stage2_debug_hr_trace_every = int(self.eval_steps)
                stage2_debug_hr_trace_every = max(1, stage2_debug_hr_trace_every)
                self.worklog.info(
                    "Stage2 HR_TRACE enabled: "
                    f"every={stage2_debug_hr_trace_every:d}, "
                    f"path={self._stage2_debug_hr_trace_path()}. "
                    "Diagnostic only; HR_GT is not used in loss."
                )
            else:
                self.worklog.info(
                    "Stage2 HR_TRACE requested but disabled because "
                    "stage2_densify_gt_hr_source is not rr_gt/external."
                )

        # UCL-style Stage2: independent ratio Gaussian branch
        ucl_ratio = bool(getattr(self.args, "stage2_ucl_ratio", False))
        ucl_lam   = float(getattr(self.args, "stage2_ucl_lam", 0.0))
        ucl_independent_geo = bool(getattr(self.args, "stage2_ucl_independent_geo", True))
        ucl_scaffold_mlp = self._stage2_uses_ucl_scaffold_mlp()
        ucl_local_ratio = self._stage2_uses_ucl_local_ratio()
        ucl_inr = self._stage2_uses_ucl_inr()
        ucl_pixel_residual = self._stage2_uses_ucl_pixel_residual()
        ucl_grid_residual = self._stage2_uses_ucl_grid_residual()
        ucl_pan_detail = self._stage2_uses_ucl_pan_detail()
        ucl_pan_gap = self._stage2_uses_ucl_pan_gap_residual()
        ucl_lig_gap = self._stage2_uses_ucl_lig_gap_residual()
        stage2_admm_split = bool(getattr(self.args, "stage2_admm_split", False))
        stage2_ucl_spatial_every = max(
            1, int(getattr(self.args, "stage2_ucl_spatial_every", 1))
        )

        if ucl_ratio:
            self.worklog.info(
                ">>> Stage2 UCL flags: "
                f"local={ucl_local_ratio}, inr={ucl_inr}, "
                f"pixel_res={ucl_pixel_residual}, grid_res={ucl_grid_residual}, "
                f"pan_detail={ucl_pan_detail}, pan_gap={ucl_pan_gap}, "
                f"lig_gap={ucl_lig_gap}, "
                f"split={stage2_admm_split}"
            )
            if self._stage2_residual_branch_enabled():
                self.worklog.info(
                    ">>> Stage2 residual branch enabled: "
                    f"weight={float(getattr(self.args, 'stage2_residual_weight', 1.0)):.3g}, "
                    f"init={self._stage2_residual_init_mode()}, "
                    f"gate={self._stage2_residual_gate_mode()}, "
                    f"carrier={str(getattr(self.args, 'stage2_residual_carrier_mode', 'none')).lower()}, "
                    f"lowrank={self._stage2_uses_residual_lowrank()}, "
                    f"rank={self._stage2_residual_basis_rank()}, "
                    f"l1={float(getattr(self.args, 'stage2_residual_l1_weight', 0.0)):.3g}, "
                    f"tv={float(getattr(self.args, 'stage2_residual_tv_weight', 0.0)):.3g}, "
                    f"basis_ortho={float(getattr(self.args, 'stage2_residual_basis_ortho_weight', 0.0)):.3g}"
                )
            if self._stage2_ratio_split_enabled():
                self.worklog.info(
                    ">>> Stage2 ratio split product: "
                    f"mode={str(getattr(self.args, 'stage2_ratio_split_pan_mode', 'full')).lower()}, "
                    f"sigma={float(getattr(self.args, 'stage2_ratio_split_pan_sigma', 2.0)):.3g}, "
                    f"hf_alpha={float(getattr(self.args, 'stage2_ratio_hf_alpha', 0.1)):.3g}, "
                    f"hf_gain={float(getattr(self.args, 'stage2_ratio_split_pan_hf_gain', 1.0)):.3g}"
                )
            # Pre-render PAN from Stage1 (frozen), used as multiplier
            with torch.no_grad():
                sc_hr_pan = self._get_scale(1.0)
                tb_hr = (
                    (pan_w + self.block_w - 1) // self.block_w,
                    (pan_h + self.block_h - 1) // self.block_h, 1,
                )
                pan_rendered_s1, _ = self._rasterize_features(
                    self.xy.detach(), sc_hr_pan, self.rot.detach(),
                    self.feat_pan.detach(), pan_h, pan_w, tb_hr,
                )
                pan_rendered_s1 = pan_rendered_s1.clamp(0.0, 1.0)  # [1, H, W]
            self._stage2_ucl_pan_render_cache = {
                (int(pan_h), int(pan_w), int(pan_h), int(pan_w)):
                    pan_rendered_s1.detach()
            }

            C = sum(self.input_channels)
            feat_lr = float(getattr(self.args, "feat_lr", 5e-3))
            self.worklog.info(
                ">>> Stage2 UCL ratio render: "
                f"{getattr(self.args, 'stage2_ucl_render_norm_mode', 'topk_norm')}"
            )
            product_pan_ucl = self._stage2_ucl_product_pan(pan_rendered_s1)
            gamma_input_pan_ucl = self._stage2_ucl_gamma_input_pan(pan_rendered_s1)
            gamma_denom_pan_ucl = self._stage2_ucl_gamma_denom_pan(pan_rendered_s1)
            self.worklog.info(
                ">>> Stage2 UCL PAN sources: "
                f"product={getattr(self.args, 'stage2_ucl_product_pan', 'rendered')}, "
                f"gamma_in={getattr(self.args, 'stage2_ucl_gamma_input_pan', 'rendered')}, "
                f"gamma_den={getattr(self.args, 'stage2_ucl_gamma_denom_pan', 'rendered')}"
            )
            with torch.no_grad():
                current_pan_probe, _ = self._rasterize_features(
                    self.xy.detach(),
                    self._get_scale(1.0).detach(),
                    self.rot.detach(),
                    self.feat_pan.detach(),
                    pan_h,
                    pan_w,
                    tb_hr,
                )
                current_pan_probe = current_pan_probe.clamp(0.0, 1.0)
                self.worklog.info(
                    "Stage2 product PAN | "
                    f"mode={self._stage2_product_pan_mode()} "
                    f"fixed_s1_l1={float(F.l1_loss(pan_rendered_s1, self.ref_images).item()):.6f} "
                    f"current_l1={float(F.l1_loss(current_pan_probe, self.ref_images).item()):.6f} "
                    f"product_hf_energy={float(self._highpass(product_pan_ucl).abs().mean().item()):.6f}"
                )
            # Compute global ratio_base from LRMS/PAN statistics.
            ms_mean = self.gt_images.mean(dim=(-2, -1))  # [C]
            pan_mean = self.ref_images.mean()
            ratio_base_init = (
                ms_mean / pan_mean.clamp(min=1e-6)
            ).clamp(min=0.1, max=10.0)
            if ucl_inr:
                inr = self._make_stage2_ratio_inr(ratio_base_init)
                latent_lr_mult = float(
                    getattr(self.args, "stage2_ucl_inr_latent_lr_mult", 1.0)
                )
                net_lr_mult = float(
                    getattr(self.args, "stage2_ucl_inr_lr_mult", 1.0)
                )
                s2_ratio_optimizer = torch.optim.Adam([
                    {
                        "params": [self.stage2_ratio_latent],
                        "lr": feat_lr * latent_lr_mult,
                    },
                    {"params": inr.parameters(), "lr": feat_lr * net_lr_mult},
                ])
                self._ucl_ratio_mode = "full"
                self.worklog.info(
                    f">>> Stage2 UCL: implicit field + Gaussian anchors "
                    f"({self.xy.shape[0]} anchors, D="
                    f"{int(getattr(self.args, 'stage2_ucl_inr_latent_dim', 8))}, "
                    f"hidden={int(getattr(self.args, 'stage2_ucl_inr_hidden', 64))})"
                )
            elif ucl_local_ratio:
                coeff = self._make_stage2_local_ratio(ratio_base_init)
                # Optional pan_bright_init override: for anchors whose xy lands in
                # the top-K% PAN-bright pixels, replace the const-slot per-band
                # offset with `value` so feat_ratio starts at 1.0 there. Lets
                # Gaussians optimize downward from above in bright regions where
                # X under-predicts (top-25% PAN-bright = ~60% of MSE on patch12).
                pb_mode = str(
                    getattr(self.args, "stage2_base_ratio_init_mode", "zero")
                ).lower()
                if pb_mode in ("pan_bright_init", "pan_bright", "pan_bright_top"):
                    topk = float(
                        getattr(self.args, "stage2_pan_bright_init_topk", 0.25)
                    )
                    topk = min(max(topk, 1.0e-4), 1.0)
                    value = float(
                        getattr(self.args, "stage2_pan_bright_init_value", 1.0)
                    )
                    pan_ref = self.ref_images.detach()
                    pan_y = pan_ref.mean(0) if pan_ref.dim() == 3 else pan_ref
                    thresh = torch.quantile(pan_y.flatten(), 1.0 - topk)
                    pan_y_at_anchor = self._sample_features_from_image(
                        pan_y.unsqueeze(0), self.xy.detach()
                    ).squeeze(-1)  # [N_anchor]
                    bright_mask = (pan_y_at_anchor >= thresh).to(
                        dtype=coeff.dtype, device=coeff.device
                    )
                    basis_names = self.stage2_local_ratio_basis or \
                        self._parse_stage2_local_ratio_basis()
                    const_idx = basis_names.index("const")
                    with torch.no_grad():
                        if coeff.dim() == 3 and coeff.shape[1] == C:
                            override = bright_mask.view(-1, 1) * value
                            base = coeff[:, :, const_idx]
                            coeff[:, :, const_idx] = (
                                base * (1.0 - bright_mask.view(-1, 1))
                                + override
                            )
                        else:
                            override = bright_mask * value
                            base = coeff[:, 0, const_idx]
                            coeff[:, 0, const_idx] = (
                                base * (1.0 - bright_mask) + override
                            )
                    self.worklog.info(
                        f">>> Stage2 pan_bright_init: topk={topk:.3f} "
                        f"value={value:.3f} thresh={float(thresh.item()):.4f} "
                        f"active_anchors={float(bright_mask.mean().item()):.4f}"
                    )
                coeff_lr_mult = float(
                    getattr(self.args, "stage2_ucl_local_ratio_lr_mult", 1.0)
                )
                s2_ratio_optimizer = torch.optim.Adam([
                    {"params": [coeff], "lr": feat_lr * coeff_lr_mult},
                ])
                self._ucl_ratio_mode = "full"
                self.worklog.info(
                    f">>> Stage2 UCL: local ratio primitive "
                    f"({self.xy.shape[0]} anchors, C={C}, "
                    f"basis={','.join(self.stage2_local_ratio_basis)})"
                )
            elif ucl_scaffold_mlp:
                self.worklog.info(
                    f">>> Stage2 scaffold ratio_base init: {ratio_base_init.tolist()}"
                )
                decoder = self._make_stage2_scaffold_decoder(
                    ratio_base=ratio_base_init
                )
                decoder_lr_mult = float(
                    getattr(self.args, "stage2_scaffold_decoder_lr_mult", 1.0)
                )
                latent_lr_mult = float(
                    getattr(self.args, "stage2_scaffold_latent_lr_mult", 1.0)
                )
                bias_lr_mult = float(
                    getattr(self.args, "stage2_scaffold_bias_lr_mult", 1.0)
                )
                base_lr_mult = float(
                    getattr(self.args, "stage2_scaffold_base_lr_mult", 1.0)
                )
                opt_groups = [
                    {"params": decoder.parameters(), "lr": feat_lr * decoder_lr_mult},
                    {
                        "params": [self.stage2_scaffold_latent],
                        "lr": feat_lr * latent_lr_mult,
                    },
                    {
                        "params": [self.stage2_scaffold_ratio_bias],
                        "lr": feat_lr * bias_lr_mult,
                    },
                ]
                if self.stage2_scaffold_base_ratio is not None:
                    train_base_ratio = bool(
                        getattr(self.args, "stage2_scaffold_train_base_ratio", True)
                    )
                    if train_base_ratio:
                        opt_groups.append({
                            "params": [self.stage2_scaffold_base_ratio],
                            "lr": feat_lr * base_lr_mult,
                        })
                    else:
                        self.stage2_scaffold_base_ratio.requires_grad_(False)
                s2_ratio_optimizer = torch.optim.Adam([
                    *opt_groups
                ])
                self._ucl_ratio_mode = "full"
                children = int(getattr(self.args, "stage2_scaffold_children", 4))
                latent_dim = int(getattr(self.args, "stage2_scaffold_latent_dim", 8))
                residual_scale = float(
                    getattr(self.args, "stage2_scaffold_residual_scale", 0.1)
                )
                use_base_ratio = self.stage2_scaffold_base_ratio is not None
                train_base_ratio = (
                    self.stage2_scaffold_base_ratio is not None
                    and self.stage2_scaffold_base_ratio.requires_grad
                )
                self.worklog.info(
                    f">>> Stage2 UCL: scaffold MLP (fixed geo) "
                    f"({self.xy.shape[0]} anchors x {children} children, "
                    f"latent_dim={latent_dim}, ratio_bias=True, "
                    f"base_ratio={use_base_ratio}, train_base={train_base_ratio}, "
                    f"residual_scale={residual_scale:.3f})"
                )
            elif ucl_independent_geo:
                # 方案A：独立几何，在 HR 空间重新初始化（30000 个 Gaussian）
                from utils.image_utils import get_grid, compute_image_gradients
                n_ratio = int(getattr(self.args, "stage2_ucl_num_gaussians",
                                      int(getattr(self.args, "num_gaussians", 30000))))
                geo_init_mode = str(
                    getattr(self.args, "stage2_ucl_geo_init_mode", "pan_gradient")
                ).lower()
                if geo_init_mode in (
                    "ms_gradient",
                    "lrms_gradient",
                    "lms_gradient",
                    "bicubic_ms",
                    "bicubic_lrms",
                ):
                    init_img = self._to_pan_estimate(self._bicubic_hr_target())
                    init_np = init_img.detach().cpu().numpy()
                else:
                    init_np = self.ref_images.detach().cpu().numpy()
                gy, gx = compute_image_gradients(init_np)
                grad = np.hypot(gy, gx).astype(np.float64)
                if grad.max() > 1e-12:
                    grad /= grad.max()
                prob = np.power(grad.reshape(-1), 2.0)
                prob /= prob.sum() + 1e-12
                n_rand = int(round(float(getattr(self.args, "init_random_ratio", 0.3)) * n_ratio))
                n_prob = n_ratio - n_rand
                grid = get_grid(h=pan_h, w=pan_w).to(
                    device=self.device, dtype=self.dtype).reshape(-1, 2)
                idx = np.concatenate([
                    np.random.choice(pan_h * pan_w, n_rand, replace=False),
                    np.random.choice(pan_h * pan_w, n_prob, replace=False, p=prob),
                ])
                self.xy_ratio    = nn.Parameter(grid[idx].detach().clone())
                sv = float(getattr(self.args, "init_scale", 5.0))
                sv = sv if bool(getattr(self.args, "disable_inverse_scale", False)) else 1.0 / sv
                self.rot_ratio   = nn.Parameter(torch.zeros(n_ratio, 1, dtype=self.dtype, device=self.device))
                self.scale_ratio = nn.Parameter(torch.full((n_ratio, 2), sv, dtype=self.dtype, device=self.device))
                self.feat_ratio  = nn.Parameter(torch.zeros(n_ratio, C, dtype=self.dtype, device=self.device))
                self._stage2_init_ratio_band_mask(n_ratio, C)
                opt_groups = [
                    {"params": self.feat_ratio,  "lr": float(getattr(self.args, "feat_lr", 5e-3))},
                    {"params": self.xy_ratio,    "lr": float(getattr(self.args, "pos_lr",  5e-4))},
                    {"params": self.rot_ratio,   "lr": float(getattr(self.args, "rot_lr",  5e-4))},
                    {"params": self.scale_ratio, "lr": float(getattr(self.args, "scale_lr",5e-4))},
                ]
                mask_group = self._stage2_ratio_band_mask_param_group(
                    float(getattr(self.args, "feat_lr", 5e-3))
                )
                if mask_group is not None:
                    opt_groups.append(mask_group)
                s2_ratio_optimizer = torch.optim.Adam(opt_groups)
                self.worklog.info(
                    f">>> Stage2 UCL: independent geo ({n_ratio} Gaussians, "
                    f"init={geo_init_mode}, HR {pan_h}x{pan_w})"
                )
            else:
                # 方案B：复用 Stage1 几何（冻结），只学 feat_ratio
                # ucl_ratio_mode: "full"(Nx4), "shared"(Nx1+4gain), "lowrank"(NxK + Kx4)
                ucl_ratio_mode = str(getattr(self.args, "stage2_ucl_ratio_mode", "full")).lower()
                ucl_ratio_rank = int(getattr(self.args, "stage2_ucl_ratio_rank", 2))
                N = self.xy.shape[0]
                feat_lr = float(getattr(self.args, "feat_lr", 5e-3))
                if ucl_ratio_mode == "shared":
                    # 方案1: 1维空间ratio + per-band gain scalar
                    self.feat_ratio = nn.Parameter(
                        torch.zeros(N, 1, dtype=self.dtype, device=self.device)
                    )
                    self.ratio_band_gain = nn.Parameter(
                        torch.ones(C, dtype=self.dtype, device=self.device)
                    )
                    s2_ratio_optimizer = torch.optim.Adam([
                        {"params": self.feat_ratio,     "lr": feat_lr},
                        {"params": self.ratio_band_gain,"lr": feat_lr},
                    ])
                    self.worklog.info(
                        f">>> Stage2 UCL: reuse S1 geo ({N}G, shared ratio+{C}gain)"
                    )
                elif ucl_ratio_mode == "lowrank":
                    # 方案2: NxK basis + Kx4 band_mix
                    K = ucl_ratio_rank
                    self.feat_ratio = nn.Parameter(
                        torch.zeros(N, K, dtype=self.dtype, device=self.device)
                    )
                    self.ratio_band_mix = nn.Parameter(
                        torch.eye(K, C, dtype=self.dtype, device=self.device)[:K]
                        if K <= C else torch.randn(K, C, dtype=self.dtype, device=self.device) * 0.01
                    )
                    s2_ratio_optimizer = torch.optim.Adam([
                        {"params": self.feat_ratio,    "lr": feat_lr},
                        {"params": self.ratio_band_mix,"lr": feat_lr},
                    ])
                    self.worklog.info(
                        f">>> Stage2 UCL: reuse S1 geo ({N}G, lowrank K={K})"
                    )
                else:
                    # 方案3 / full: NxC，每个高斯独立学4维ratio
                    ucl_ratio_mode = "full"
                    explicit_ratio_init_mode = str(
                        getattr(self.args, "stage2_ratio_init_mode", "zero")
                    ).lower()
                    base_ratio_init_mode = str(
                        getattr(self.args, "stage2_base_ratio_init_mode", "zero")
                    ).lower()
                    ratio_init_mode = self._stage2_base_ratio_init_mode()
                    ratio_init_img_for_log = None
                    if (
                        base_ratio_init_mode in (
                            "lr_ratio",
                            "lowres_ratio",
                            "lrms_ratio",
                            "pan_bright_init",
                            "pan_bright",
                            "pan_bright_top",
                            "mtf_glp_ratio",
                            "mtf_glp_div_pan",
                            "vivone_ratio",
                            "vivone",
                            "pan_guided_lr_ratio",
                            "pan_guided_ratio",
                            "jbu_lr_ratio",
                            "joint_bilateral_lr_ratio",
                            "graph_lr_ratio",
                        )
                        or explicit_ratio_init_mode in (
                        "bicubic_ratio",
                        "bicubic",
                        "lms_ratio",
                        "sampled_ratio",
                        "sample_ratio",
                        "mtf_glp_ratio",
                        "mtf_glp_div_pan",
                        "vivone_ratio",
                        "vivone",
                        "pan_guided_lr_ratio",
                        "pan_guided_ratio",
                        "jbu_lr_ratio",
                        "joint_bilateral_lr_ratio",
                        "graph_lr_ratio",
                        )
                    ):
                        ratio_init, ratio_init_img, _ = self._stage2_init_ratio_features(
                            pan_rendered_s1,
                            product_pan_ucl,
                            self.xy.detach(),
                        )
                        ratio_init_img_for_log = ratio_init_img
                    elif ratio_init_mode in ("ratio_base", "base", "mean"):
                        ratio_init = ratio_base_init.to(
                            dtype=self.dtype, device=self.device
                        ).view(1, C).repeat(N, 1)
                    elif ratio_init_mode in ("delta_base", "base_delta", "output_base"):
                        ratio_init = torch.zeros(N, C, dtype=self.dtype, device=self.device)
                        self.stage2_ucl_output_ratio_base = ratio_base_init.to(
                            dtype=self.dtype, device=self.device
                        ).view(C, 1, 1)
                    elif ratio_init_mode in (
                        "sampled_ratio",
                        "sample_ratio",
                        "bicubic_ratio",
                        "lms_ratio",
                        "ratio_base_easy_zero",
                        "base_easy_zero",
                        "sampled_easy_zero",
                        "easy_zero",
                        "local_easy_zero",
                    ):
                        if ratio_init_mode in ("ratio_base_easy_zero", "base_easy_zero"):
                            ratio_init = ratio_base_init.to(
                                dtype=self.dtype, device=self.device
                            ).view(1, C).repeat(N, 1)
                        else:
                            ratio_eps = float(
                                getattr(self.args, "stage2_ucl_ratio_init_eps", 1.0e-3)
                            )
                            ratio_clip = float(
                                getattr(self.args, "stage2_ucl_ratio_init_clip", 4.0)
                            )
                            ratio_img = lms_hr / pan_rendered_s1.clamp_min(ratio_eps)
                            ratio_img = ratio_img.clamp(0.0, ratio_clip)
                            ratio_init = self._sample_features_from_image(
                                ratio_img, self.xy.detach()
                            )
                        if ratio_init_mode in (
                            "ratio_base_easy_zero",
                            "base_easy_zero",
                            "sampled_easy_zero",
                            "easy_zero",
                            "local_easy_zero",
                        ):
                            pan_grad = self._pan_grad_mag(self.ref_images).detach()
                            pan_res = (pan_rendered_s1 - self.ref_images).abs().detach()
                            grad_s = self._sample_features_from_image(
                                pan_grad / pan_grad.mean().clamp_min(1.0e-6),
                                self.xy.detach(),
                            ).mean(dim=1, keepdim=True)
                            res_s = self._sample_features_from_image(
                                pan_res / pan_res.mean().clamp_min(1.0e-6),
                                self.xy.detach(),
                            ).mean(dim=1, keepdim=True)
                            grad_alpha = float(
                                getattr(self.args, "stage2_ucl_ratio_easy_grad_alpha", 1.0)
                            )
                            res_alpha = float(
                                getattr(self.args, "stage2_ucl_ratio_easy_res_alpha", 1.0)
                            )
                            hard_score = grad_alpha * grad_s + res_alpha * res_s
                            easy_mode = str(
                                getattr(self.args, "stage2_ucl_ratio_easy_mask_mode", "soft")
                            ).lower()
                            if easy_mode in ("hard", "binary", "quantile"):
                                keep = float(
                                    getattr(self.args, "stage2_ucl_ratio_easy_keep", 0.5)
                                )
                                keep = min(1.0, max(0.0, keep))
                                thresh = torch.quantile(hard_score.detach().flatten(), keep)
                                easy = (hard_score <= thresh).to(dtype=self.dtype)
                            else:
                                easy = torch.exp(-hard_score)
                                floor = float(
                                    getattr(self.args, "stage2_ucl_ratio_easy_floor", 0.0)
                                )
                                if floor > 0:
                                    easy = floor + (1.0 - floor) * easy
                            ratio_init = ratio_init * easy
                            self.worklog.info(
                                f">>> Stage2 UCL ratio easy init: "
                                f"easy_mean={easy.mean().item():.4f}, "
                                f"easy_min={easy.min().item():.4f}, "
                                f"easy_max={easy.max().item():.4f}"
                            )
                    elif ratio_init_mode in (
                        "ucl_teacher",
                        "teacher",
                        "teacher_ratio",
                        "distill_init",
                        "ucl_distill",
                    ):
                        teacher = self._load_stage2_ucl_teacher((pan_h, pan_w))
                        if teacher is None:
                            raise ValueError(
                                "stage2_ucl_ratio_init=ucl_teacher requires "
                                "stage2_ucl_teacher_path"
                            )
                        denom_source = str(
                            getattr(
                                self.args,
                                "stage2_ucl_ratio_init_teacher_denom_pan",
                                "product",
                            )
                        ).lower()
                        if denom_source in ("product", "product_pan"):
                            denom_pan = self._stage2_ucl_product_pan(pan_rendered_s1)
                        else:
                            denom_pan = self._select_ucl_pan_image(
                                denom_source, pan_rendered_s1
                            )
                        ratio_eps = float(
                            getattr(self.args, "stage2_ucl_ratio_init_eps", 1.0e-3)
                        )
                        ratio_clip = float(
                            getattr(self.args, "stage2_ucl_ratio_init_clip", 4.0)
                        )
                        ratio_img = teacher / denom_pan.clamp_min(ratio_eps)
                        if ratio_clip > 0:
                            ratio_img = ratio_img.clamp(0.0, ratio_clip)
                        ratio_init = self._sample_features_from_image(
                            ratio_img, self.xy.detach()
                        )
                        self.worklog.info(
                            f">>> Stage2 UCL teacher ratio init: "
                            f"denom={denom_source}, "
                            f"mean={ratio_init.mean().item():.4f}, "
                            f"std={ratio_init.std().item():.4f}"
                        )
                    elif ratio_init_mode in ("random", "rand", "normal", "randn"):
                        random_std = float(
                            getattr(self.args, "stage2_ucl_ratio_init_random_std", 0.05)
                        )
                        random_center = str(
                            getattr(self.args, "stage2_ucl_ratio_init_random_center", "ratio_base")
                        ).lower()
                        if random_center in ("ratio_base", "base", "mean"):
                            center = ratio_base_init.to(
                                dtype=self.dtype, device=self.device
                            ).view(1, C).repeat(N, 1)
                        else:
                            center = torch.zeros(N, C, dtype=self.dtype, device=self.device)
                        ratio_init = center + random_std * torch.randn(
                            N, C, dtype=self.dtype, device=self.device
                        )
                        ratio_init = ratio_init.clamp_min(0.0)
                    else:
                        ratio_init = torch.zeros(N, C, dtype=self.dtype, device=self.device)
                    self.feat_ratio = nn.Parameter(
                        ratio_init
                    )
                    self._stage2_init_ratio_band_mask(N, C)
                    self._build_stage2_feat_ratio_graph()
                    self._stage2_log_ratio_init(
                        ratio_init_img_for_log, ratio_init, product_pan_ucl
                    )
                    free_lr_mult = 1.0
                    if self._stage2_pan_descriptor_binding_enabled():
                        free_lr_mult = float(
                            getattr(
                                self.args,
                                "stage2_pan_descriptor_free_ratio_lr_mult",
                                1.0,
                            )
                        )
                    params = [{"params": self.feat_ratio, "lr": feat_lr * free_lr_mult}]
                    mask_group = self._stage2_ratio_band_mask_param_group(feat_lr)
                    if mask_group is not None:
                        params.append(mask_group)
                    if self._stage2_ratio_poly_enabled():
                        poly_init = str(
                            getattr(self.args, "stage2_ratio_poly_init", "zero")
                        ).lower()
                        if poly_init in ("ratio_ref_sobel", "sobel", "lr_ratio_sobel"):
                            ref_img = ratio_init_img_for_log
                            if ref_img is None:
                                ref_img = self._stage2_lr_ratio_image(
                                    product_pan_ucl, product_pan_ucl.shape[-2:]
                                )
                            init_dx, init_dy = self._stage2_ratio_poly_sobel_init(
                                ref_img.to(dtype=self.dtype, device=self.device),
                                self.xy.detach(),
                            )
                        else:
                            init_dx = torch.zeros_like(ratio_init)
                            init_dy = torch.zeros_like(ratio_init)
                        self.feat_ratio_dx = nn.Parameter(init_dx)
                        self.feat_ratio_dy = nn.Parameter(init_dy)
                        poly_mult = float(
                            getattr(self.args, "stage2_ratio_poly_lr_mult", 0.2)
                        )
                        params.extend([
                            {"params": self.feat_ratio_dx, "lr": feat_lr * poly_mult},
                            {"params": self.feat_ratio_dy, "lr": feat_lr * poly_mult},
                        ])
                        self.worklog.info(
                            f">>> Stage2 ratio poly order=1 active "
                            f"(init={poly_init}, lr_mult={poly_mult:.3g}, scale="
                            f"{float(getattr(self.args, 'stage2_ratio_poly_scale', 1.0)):.3g}, "
                            f"dx_abs={float(init_dx.abs().mean().item()):.6f}, "
                            f"dy_abs={float(init_dy.abs().mean().item()):.6f})"
                        )
                    s2_ratio_optimizer = torch.optim.Adam(params)
                    self.worklog.info(
                        f">>> Stage2 UCL: reuse S1 geo ({N} Gaussians, frozen, "
                        f"ratio_init={ratio_init_mode})"
                    )
                # additive mode: HRMS = render(feat_ms_lr) + pan * render(feat_ratio)
                # feat_ms_lr is fixed from bicubic LRMS; feat_ratio is trained from zero.
                ucl_additive = bool(getattr(self.args, "stage2_ucl_additive", False))
                if ucl_additive:
                    lms_hr_init = self._bicubic_hr_target()  # [C, H, W]
                    feat_ms_lr_init = self._sample_features_from_image(
                        lms_hr_init, self.xy.detach()
                    )  # [N, C]
                    self.register_buffer(
                        "feat_ms_lr",
                        feat_ms_lr_init.detach().clone(),
                    )
                    # re-init feat_ratio to zero (only feat_ratio is trained)
                    self.feat_ratio = nn.Parameter(
                        torch.zeros(N, C, dtype=self.dtype, device=self.device)
                    )
                    self._stage2_init_ratio_band_mask(N, C)
                    params = [{"params": self.feat_ratio, "lr": feat_lr}]
                    mask_group = self._stage2_ratio_band_mask_param_group(feat_lr)
                    if mask_group is not None:
                        params.append(mask_group)
                    s2_ratio_optimizer = torch.optim.Adam(params)
                    self.worklog.info(
                        f">>> Stage2 UCL: reuse S1 geo ({N}G, additive: bicubic_lms+pan*ratio)"
                    )
                self._ucl_ratio_mode = ucl_ratio_mode
                self._ucl_ratio_rank = ucl_ratio_rank
                self._build_stage2_feat_ratio_graph()
                if self._stage2_pan_descriptor_binding_enabled() and self.feat_ratio is not None:
                    self._stage2_init_pan_descriptor_binding()
                    pan_desc_group = self._stage2_pan_descriptor_param_group(feat_lr)
                    if pan_desc_group is not None:
                        s2_ratio_optimizer.add_param_group(pan_desc_group)

                if self._stage2_residual_branch_enabled() and self.feat_residual is not None:
                    feat_residual_lr_mult = float(getattr(self.args, "stage2_residual_lr_mult", 1.0))
                    s2_ratio_optimizer.add_param_group({
                        "params": [self.feat_residual],
                        "lr": feat_lr * feat_residual_lr_mult,
                        "name": "stage2_residual",
                    })
                    if self._stage2_uses_residual_lowrank() and self.stage2_residual_basis_B is not None:
                        basis_lr_mult = float(getattr(self.args, "stage2_residual_basis_lr_mult", 0.5))
                        s2_ratio_optimizer.add_param_group({
                            "params": list(self.stage2_residual_basis_B.parameters()),
                            "lr": feat_lr * basis_lr_mult,
                            "name": "stage2_residual_basis_B",
                        })
                    self.worklog.info(
                        f">>> Stage2 residual branch params: "
                        f"dim={int(self.feat_residual.shape[1])}, "
                        f"lr_mult={feat_residual_lr_mult:.3g}, "
                        f"basis_lr_mult={float(getattr(self.args, 'stage2_residual_basis_lr_mult', 0.5)):.3g}"
                    )

            if ucl_pixel_residual:
                pix = self._make_stage2_pixel_residual(pan_h, pan_w)
                pix_lr_mult = float(
                    getattr(self.args, "stage2_ucl_pixel_residual_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": [pix],
                    "lr": feat_lr * pix_lr_mult,
                })
                self.worklog.info(
                    f">>> Stage2 UCL: PAN-guided dense highpass residual "
                    f"({self.output_dim}x{pan_h}x{pan_w}, "
                    f"mask={getattr(self.args, 'stage2_ucl_pixel_residual_mask', 'grad')})"
                )

            if ucl_grid_residual:
                grid = self._make_stage2_grid_residual(pan_h, pan_w)
                grid_lr_mult = float(
                    getattr(self.args, "stage2_ucl_grid_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": [grid],
                    "lr": feat_lr * grid_lr_mult,
                })
                self.worklog.info(
                    f">>> Stage2 UCL: low-res grid residual "
                    f"({grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]})"
                )

            if ucl_pan_detail:
                pan_detail = self._make_stage2_pan_detail(ratio_base_init)
                pan_detail_lr_mult = float(
                    getattr(self.args, "stage2_ucl_pan_detail_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": [pan_detail],
                    "lr": feat_lr * pan_detail_lr_mult,
                })
                self.worklog.info(
                    f">>> Stage2 UCL: explicit PAN high-frequency detail "
                    f"(amp={float(getattr(self.args, 'stage2_ucl_pan_detail_amp', 1.0)):.3f}, "
                    f"sigma={float(getattr(self.args, 'stage2_ucl_pan_detail_sigma', 2.0)):.2f}, "
                    f"corr_w={float(getattr(self.args, 'stage2_ucl_pan_detail_corr_weight', 0.0)):.6g})"
                )

            if ucl_pan_gap and bool(getattr(self.args, "stage2_ucl_pan_gap_train_gain", False)):
                pan_gap_gain = self._make_stage2_pan_gap_gain_coeff()
                pan_gap_gain_lr_mult = float(
                    getattr(self.args, "stage2_ucl_pan_gap_gain_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": [pan_gap_gain],
                    "lr": feat_lr * pan_gap_gain_lr_mult,
                })
                self.worklog.info(
                    ">>> Stage2 UCL: trainable PAN-gap channel gain "
                    f"(gain_amp={float(getattr(self.args, 'stage2_ucl_pan_gap_gain_amp', 0.1)):.3g}, "
                    f"l2={float(getattr(self.args, 'stage2_ucl_pan_gap_gain_l2_weight', 0.0)):.3g})"
                )

            if self._stage2_uses_inr_residual():
                _aux_tokens = self._stage2_inr_aux_feature_tokens()
                inr_mod = self._make_stage2_ratio_context_inr(n_aux=len(_aux_tokens))
                inr_lr_mult = float(
                    getattr(self.args, "stage2_ucl_inr_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": list(inr_mod.parameters()),
                    "lr": feat_lr * inr_lr_mult,
                })
                self.worklog.info(
                    ">>> Stage2 UCL: coord-INR HF ratio residual "
                    f"(aux={_aux_tokens}, "
                    f"n_freq={int(getattr(self.args, 'stage2_ucl_inr_n_freq', 6))}, "
                    f"hidden={int(getattr(self.args, 'stage2_ucl_inr_hidden', 64))}, "
                    f"layers={int(getattr(self.args, 'stage2_ucl_inr_layers', 3))}, "
                    f"amp={float(getattr(self.args, 'stage2_ucl_inr_amp', 0.05)):.3g}, "
                    f"lr_mult={inr_lr_mult:.3g})"
                )

            if self._stage2_uses_pan_stat_residual():
                _tokens = self._stage2_pan_stat_feature_tokens()
                if len(_tokens) > 0:
                    pan_stat_mlp = self._make_stage2_pan_stat_mlp(n_feat=len(_tokens))
                    pan_stat_lr_mult = float(
                        getattr(self.args, "stage2_ucl_pan_stat_lr_mult", 1.0)
                    )
                    s2_ratio_optimizer.add_param_group({
                        "params": list(pan_stat_mlp.parameters()),
                        "lr": feat_lr * pan_stat_lr_mult,
                    })
                    self.worklog.info(
                        ">>> Stage2 UCL: per-Gaussian PAN-stat MLP residual "
                        f"(features={_tokens}, "
                        f"hidden={int(getattr(self.args, 'stage2_ucl_pan_stat_hidden', 32))}, "
                        f"layers={int(getattr(self.args, 'stage2_ucl_pan_stat_layers', 2))}, "
                        f"amp={float(getattr(self.args, 'stage2_ucl_pan_stat_amp', 0.05)):.3g}, "
                        f"lr_mult={pan_stat_lr_mult:.3g})"
                    )

            if self._stage2_uses_m2p():
                m2p_mod = self._make_stage2_m2p_weights()
                m2p_lr_mult = float(
                    getattr(self.args, "stage2_m2p_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": list(m2p_mod.parameters()),
                    "lr": feat_lr * m2p_lr_mult,
                    "name": "m2p",
                })
                _hf_w = float(getattr(self.args, "stage2_m2p_hf_weight", 0.0))
                _master_w = float(getattr(self.args, "stage2_m2p_weight", 0.0))
                if _master_w > 0.0 and _hf_w == 0.0:
                    _hf_w = _master_w
                self.worklog.info(
                    ">>> Stage2 UCL: per-channel PAN HF match head "
                    "(L = sum_c ||HP(R_c*PAN) - alpha_c*HP(PAN)||^2) "
                    f"(hf_w={_hf_w:.3g}, "
                    f"hf_sigma={float(getattr(self.args, 'stage2_m2p_hf_sigma', 1.5)):.2f}, "
                    f"alpha_init={float(getattr(self.args, 'stage2_m2p_alpha_init', 1.0)):.2f}, "
                    f"lr_mult={m2p_lr_mult:.3g})"
                )

            if self._stage2_uses_p2m():
                p2m_mod = self._make_stage2_p2m_colors()
                p2m_lr_mult = float(getattr(self.args, "stage2_p2m_lr_mult", 1.0))
                s2_ratio_optimizer.add_param_group({
                    "params": list(p2m_mod.parameters()),
                    "lr": feat_lr * p2m_lr_mult,
                    "name": "p2m",
                })
                _obs_w = float(getattr(self.args, "stage2_p2m_obs_weight", 0.0))
                _cross_w = float(getattr(self.args, "stage2_p2m_weight", 0.0))
                self.worklog.info(
                    ">>> Stage2 UCL: P2M head (per-Gaussian free MS colors via PAN geometry) "
                    f"(obs_w={_obs_w:.3g}, cross_w={_cross_w:.3g}, "
                    f"init_std={float(getattr(self.args, 'stage2_p2m_init_std', 1.0e-3)):.1e}, "
                    f"lr_mult={p2m_lr_mult:.3g})"
                )

            if self._stage2_uses_spectral_basis():
                basis_mod = self._make_stage2_spectral_basis()
                basis_lr_mult = float(getattr(self.args, "stage2_basis_lr_mult", 0.5))
                s2_ratio_optimizer.add_param_group({
                    "params": list(basis_mod.parameters()),
                    "lr": feat_lr * basis_lr_mult,
                    "name": "basis_B",
                })
                self.worklog.info(
                    ">>> Stage2 UCL: low-rank spectral basis B "
                    f"(R={int(getattr(self.args, 'stage2_basis_rank', 0))}, "
                    f"C={self.output_dim}, "
                    f"init={str(getattr(self.args, 'stage2_basis_init', 'uniform'))}, "
                    f"lr_mult={basis_lr_mult:.3g})"
                )

            if ucl_lig_gap and bool(getattr(self.args, "stage2_ucl_lig_gap_train_gain", False)):
                lig_gap_coeff = self._make_stage2_lig_gap_coeff()
                lig_gap_lr_mult = float(
                    getattr(self.args, "stage2_ucl_lig_gap_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": [lig_gap_coeff],
                    "lr": feat_lr * lig_gap_lr_mult,
                })
                self.worklog.info(
                    f">>> Stage2 UCL: trainable LIG-gap channel gain "
                    f"(gain_amp={float(getattr(self.args, 'stage2_ucl_lig_gap_gain_amp', 0.25)):.3g})"
                )

            gamma_prior_w = float(
                getattr(self.args, "stage2_ucl_gamma_prior_weight", 0.0)
            )
            if gamma_prior_w > 0:
                gamma_net = self._make_stage2_gamma_net(C)
                gamma_lr_mult = float(
                    getattr(self.args, "stage2_ucl_gamma_lr_mult", 1.0)
                )
                s2_ratio_optimizer.add_param_group({
                    "params": gamma_net.parameters(),
                    "lr": feat_lr * gamma_lr_mult,
                })
                self.worklog.info(
                    f">>> Stage2 UCL: Gamma(PAN) ratio prior "
                    f"(weight={gamma_prior_w:.6g}, hidden="
                    f"{int(getattr(self.args, 'stage2_ucl_gamma_hidden', 32))}, "
                    f"mode={getattr(self.args, 'stage2_ucl_gamma_prior_mode', 'ratio')})"
                )
            if stage2_admm_split:
                self.worklog.info(
                    ">>> Stage2 UCL: ADMM-style spectral/spatial split "
                    f"(R_spec={float(getattr(self.args, 'stage2_ucl_spec_corr_weight', 0.0)):.6g}, "
                    f"pan_guided_tv={float(getattr(self.args, 'stage2_ucl_tv_weight', 0.001)):.6g}, "
                    f"tv_alpha={float(getattr(self.args, 'stage2_ucl_pan_guided_tv_alpha', 10.0)):.3g}, "
                    f"spatial_every={stage2_ucl_spatial_every})"
                )
            if ucl_pan_gap:
                _sigmas_cfg = getattr(self.args, "stage2_ucl_pan_gap_sigmas", None)
                if _sigmas_cfg is not None and len(list(_sigmas_cfg)) > 0:
                    _amps_cfg = getattr(self.args, "stage2_ucl_pan_gap_amps", None)
                    _band_desc = (
                        f"sigmas={list(_sigmas_cfg)}, "
                        f"amps={list(_amps_cfg) if _amps_cfg is not None else 'uniform'}"
                    )
                else:
                    _band_desc = (
                        f"highpass={bool(getattr(self.args, 'stage2_ucl_pan_gap_highpass', True))}, "
                        f"sigma={float(getattr(self.args, 'stage2_ucl_pan_gap_sigma', 2.0)):.3g}"
                    )
                self.worklog.info(
                    ">>> Stage2 UCL: raw-rendered PAN gap residual "
                    f"(amp={float(getattr(self.args, 'stage2_ucl_pan_gap_amp', 1.0)):.3g}, "
                    f"{_band_desc}, "
                    f"start={float(getattr(self.args, 'stage2_ucl_pan_gap_start_frac', 0.0)):.2g}, "
                    f"ramp={float(getattr(self.args, 'stage2_ucl_pan_gap_ramp_frac', 0.0)):.2g}, "
                    f"train_gain={bool(getattr(self.args, 'stage2_ucl_pan_gap_train_gain', False))})"
                )
            if ucl_lig_gap:
                self.worklog.info(
                    ">>> Stage2 UCL: LIG-style PAN residual injection "
                    f"(amp={float(getattr(self.args, 'stage2_ucl_lig_gap_amp', 1.0)):.3g}, "
                    f"highpass={bool(getattr(self.args, 'stage2_ucl_lig_gap_highpass', True))}, "
                    f"sigma={float(getattr(self.args, 'stage2_ucl_lig_gap_sigma', 2.0)):.3g})"
                )

        if bool(getattr(self.args, "stage2_reinit_on_start", True)):
            if ucl_ratio:
                pass  # feat_ratio already zero-initialized above
            else:
                init_target = gt_hr
                if stage2_resolution_curriculum:
                    init_mode = str(
                        getattr(self.args, "stage2_resolution_init_target", "mtf_glp")
                    ).lower()
                    if init_mode in ("lrms", "lms", "bicubic", "upsampled_lrms"):
                        init_target = lms_hr
                self._initialize_stage2_feat_ms(init_target)

        if bool(getattr(self.args, "stage2_self_supervised_only", False)):
            return self._optimize_stage2_self_supervised_only(
                gt_hr,
                pan_h,
                pan_w,
                ms_h,
                ms_w,
                tb_pan,
                tb_ms,
                s1_done_steps,
                s2_steps,
                s2_hr_w,
                s2_pan_w,
                s2_pan_every,
            )

        if bool(getattr(self.args, "stage2_variational_only", False)):
            return self._optimize_stage2_variational_only(
                gt_hr,
                pan_h,
                pan_w,
                ms_h,
                ms_w,
                tb_pan,
                tb_ms,
                s1_done_steps,
                s2_steps,
                s2_lr_w,
                s2_hr_w,
                s2_pan_w,
                s2_pan_every,
            )

        # == Iterative Stage2 ==================================================

        n_rounds = int(getattr(self.args, "iterative_rounds", 1))
        s2_decay = float(getattr(self.args, "iterative_s2_decay", 1.0))
        s3_decay = float(getattr(self.args, "iterative_s3_decay", 1.0))
        target_mix = float(getattr(self.args, "iterative_target_mix", 1.0))
        min_rounds = min(n_rounds, max(1, int(getattr(self.args, "iterative_min_rounds", n_rounds))))
        round_patience = max(0, int(getattr(self.args, "iterative_early_stop_patience", 0)))
        round_min_delta = float(getattr(self.args, "iterative_early_stop_min_delta", 0.0))
        ucl_early_stop_lr_loss = float(
            getattr(self.args, "stage2_ucl_early_stop_lr_loss", 0.0)
        )
        ucl_early_stop_lr_psnr = float(
            getattr(self.args, "stage2_ucl_early_stop_lr_psnr", 0.0)
        )
        ucl_early_stop_min_steps = max(
            0, int(getattr(self.args, "stage2_ucl_early_stop_min_steps", 0))
        )
        ucl_early_stop_patience = max(
            1, int(getattr(self.args, "stage2_ucl_early_stop_patience", 1))
        )
        stage3_disabled = bool(getattr(self.args, "stage3_skip", False)) or s3_steps <= 0
        stage3_loop_mode = str(getattr(self.args, "stage3_loop_mode", "after")).lower()
        stage3_after_loop = (
            not stage3_disabled
            and stage3_loop_mode in ("after", "post", "single", "separate", "outside")
        )
        skip_stage3 = stage3_disabled or stage3_after_loop
        resolution_curriculum = stage2_resolution_curriculum
        use_x_field = bool(getattr(self.args, "stage2_use_x_field", False))
        x_prior_w = float(getattr(self.args, "stage2_x_prior_weight", 0.1))
        alternate_g_lr_w = float(getattr(self.args, "stage2_alternate_g_lr_weight", 0.0))
        alternate_xg = bool(getattr(self.args, "stage2_alternate_xg", False)) and use_x_field
        if resolution_curriculum and alternate_xg:
            self.worklog.info(
                "Stage2 resolution curriculum disables alternate X/G updates."
            )
            alternate_xg = False
        mtf_glp_target = gt_hr.detach().clone()
        prev_final = None
        best_round = 0
        best_round_metric = -float("inf")
        best_round_final = None
        stale_rounds = 0
        freq_global = bool(getattr(self.args, "stage2_freq_global", False))
        freq_total_rounds = int(getattr(self.args, "stage2_freq_total_rounds", 0))
        if freq_global:
            if freq_total_rounds <= 0:
                freq_total_rounds = n_rounds
            self._stage2_freq_step_offset = 0
            self._stage2_freq_total_steps = max(1, s2_steps * max(1, freq_total_rounds))
        else:
            self._stage2_freq_step_offset = 0
            self._stage2_freq_total_steps = 0
        freq_seen_steps = 0

        if n_rounds > 1:
            self.worklog.info(
                f">>> Iterative Stage2: {n_rounds} rounds "
                f"(s2_decay={s2_decay}, mix={target_mix}, "
                f"stage3_mode={stage3_loop_mode})"
            )
        if bool(getattr(self.args, "stage2_freq_guided", False)):
            self.worklog.info(
                "Stage2 frequency curriculum: "
                f"sigma {float(getattr(self.args, 'stage2_freq_start_sigma', 4.0)):.2f}"
                f" -> {float(getattr(self.args, 'stage2_freq_end_sigma', 0.0)):.2f}, "
                f"until={float(getattr(self.args, 'stage2_freq_until', 1.0)):.2f}, "
                f"pan_l1={float(getattr(self.args, 'stage2_pan_l1_weight', 0.0)):.3f}, "
                f"global={freq_global}, total_rounds={freq_total_rounds if freq_global else 1}."
            )
        if ucl_ratio and (ucl_early_stop_lr_loss > 0 or ucl_early_stop_lr_psnr > 0):
            stop_parts = []
            if ucl_early_stop_lr_loss > 0:
                stop_parts.append(f"lr_loss <= {ucl_early_stop_lr_loss:.6g}")
            if ucl_early_stop_lr_psnr > 0:
                stop_parts.append(f"LR_PSNR >= {ucl_early_stop_lr_psnr:.4f}")
            self.worklog.info(
                "Stage2 UCL early stop: "
                f"{' or '.join(stop_parts)}, "
                f"min_steps={ucl_early_stop_min_steps}, "
                f"patience={ucl_early_stop_patience}"
            )
        if resolution_curriculum:
            self.worklog.info(
                "Stage2 resolution curriculum: "
                f"scales={getattr(self.args, 'stage2_resolution_scales', '4,2,1')}, "
                f"fracs={getattr(self.args, 'stage2_resolution_step_fracs', '1,1,1')}, "
                f"init={getattr(self.args, 'stage2_resolution_init_target', 'mtf_glp')}, "
                f"pan_w={float(getattr(self.args, 'stage2_resolution_pan_weight', s2_pan_w)):.3f}."
            )

        for rnd in range(n_rounds):
            cs2 = max(200, int(s2_steps * (s2_decay ** rnd)))
            cs3 = 0 if skip_stage3 else max(200, int(s3_steps * (s3_decay ** rnd)))

            if rnd > 0:
                gt_hr = (target_mix * prev_final + (1 - target_mix) * mtf_glp_target).clamp(0, 1)
                _bmu = gt_hr.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                band_w = (1.0 / _bmu) / (1.0 / _bmu).mean()
                self._initialize_stage2_feat_ms(gt_hr, use_bicubic=False)
                self._set_trainable_modules(stage=2)
                self.worklog.info(
                    f">>> Round {rnd+1}/{n_rounds}: S2={cs2}, "
                    f"S3={'post' if stage3_after_loop else cs3} "
                    f"(mix={target_mix:.2f})"
                )
            else:
                cs2 = s2_steps
                cs3 = 0 if skip_stage3 else s3_steps
                if n_rounds > 1:
                    self.worklog.info(
                        f">>> Round 1/{n_rounds}: S2={cs2}, "
                        f"S3={'post' if stage3_after_loop else cs3}"
                    )

            # -- Stage 2 training loop -----------------------------------------

            if freq_global:
                self._stage2_freq_step_offset = freq_seen_steps

            if alternate_xg:
                x_optimizer, g_optimizer = self._make_stage2_alternating_optimizers(
                    s2_train_xy, s2_train_scale, s2_train_rot
                )

            cross_warmup_frac = float(getattr(self.args, "stage2_cross_warmup_frac", 0.0))
            cross_warmup_steps = int(round(cs2 * cross_warmup_frac))
            cross_warmup_done = False
            ucl_early_stop_hits = 0
            if ucl_ratio and stage2_debug_hr_trace_ref is not None:
                with torch.no_grad():
                    tb_trace = (
                        (pan_w + self.block_w - 1) // self.block_w,
                        (pan_h + self.block_h - 1) // self.block_h, 1,
                    )
                    trace_ms_hr, _, _, _ = self._render_ucl_current_output(
                        pan_h,
                        pan_w,
                        tb_trace,
                        product_pan_ucl,
                        pan_rendered_s1,
                        render_scale_ratio=1.0,
                    )
                    self._record_stage2_debug_hr_trace(
                        "round_start",
                        rnd + 1,
                        0,
                        trace_ms_hr,
                        stage2_debug_hr_trace_ref,
                        ms_lr=self._ucl_degrade_once(trace_ms_hr),
                    )
            for step in range(1, cs2 + 1):
                self.step = s1_done_steps + step
                is_cross_warmup = (cross_warmup_steps > 0) and (step <= cross_warmup_steps)
                if (cross_warmup_steps > 0) and (step == cross_warmup_steps + 1) and not cross_warmup_done:
                    post_mult = float(getattr(self.args, "stage2_cross_post_warmup_lr_mult", 0.3))
                    for _g in s2_ratio_optimizer.param_groups:
                        if _g.get("name") in ("p2m", "m2p"):
                            _g["lr"] = _g["lr"] * post_mult
                    cross_warmup_done = True
                    self.worklog.info(
                        f">>> Stage2 cross-warmup ended at step {step}; "
                        f"P2M/M2P lr *= {post_mult:.3g}"
                    )

                if alternate_xg:
                    with torch.no_grad():
                        _, g_hr_det = self._render_stage2_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                        )
                    x_optimizer.zero_grad()
                    ms_lr, ms_hr = self._render_stage2_x_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        train_geom=s2_train_xy,
                        train_rot=s2_train_rot,
                        train_scale=s2_train_scale,
                    )
                    lr_pred = self._stage2_lr_consistency_image(ms_lr, ms_hr, ms_h, ms_w)
                    total_loss = s2_lr_w * self.l1_loss_ratio * F.l1_loss(
                        lr_pred, self.gt_images
                    )
                    if s2_hr_w > 0:
                        ms_hr_loss, gt_hr_loss, _ = self._stage2_frequency_pair(
                            ms_hr, gt_hr, step, cs2
                        )
                        total_loss += s2_hr_w * self.l1_loss_ratio * (
                            band_w * torch.abs(ms_hr_loss - gt_hr_loss)
                        ).mean()
                    if s2_pan_w > 0 and step % s2_pan_every == 0:
                        total_loss += s2_pan_w * self._stage2_pan_constraint_loss(
                            ms_hr, step, cs2
                        )
                    if x_prior_w > 0:
                        total_loss += x_prior_w * self.l1_loss_ratio * F.l1_loss(
                            ms_hr, g_hr_det
                        )
                    total_loss += self._stage2_regularizer_loss(
                        ms_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan
                    )
                    total_loss.backward()
                    x_optimizer.step()
                    if s2_train_xy:
                        with torch.no_grad():
                            self.xy.clamp_(0.0, 1.0)

                    with torch.no_grad():
                        _, x_hr_det = self._render_stage2_x_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                        )
                    g_optimizer.zero_grad()
                    g_lr, g_hr = self._render_stage2_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                    )
                    g_loss = x_prior_w * self.l1_loss_ratio * F.l1_loss(g_hr, x_hr_det)
                    if alternate_g_lr_w > 0:
                        g_loss += alternate_g_lr_w * self.l1_loss_ratio * F.l1_loss(
                            self._stage2_lr_consistency_image(g_lr, g_hr, ms_h, ms_w),
                            self.gt_images,
                        )
                    if s2_hr_w > 0:
                        g_hr_loss, gt_hr_loss, _ = self._stage2_frequency_pair(
                            g_hr, gt_hr, step, cs2
                        )
                        g_loss += s2_hr_w * self.l1_loss_ratio * (
                            band_w * torch.abs(g_hr_loss - gt_hr_loss)
                        ).mean()
                    if s2_pan_w > 0 and step % s2_pan_every == 0:
                        g_loss += s2_pan_w * self._stage2_pan_constraint_loss(
                            g_hr, step, cs2
                        )
                    g_loss += self._stage2_regularizer_loss(
                        g_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan
                    )
                    g_loss.backward()
                    g_optimizer.step()

                    if self.step % self.eval_steps == 0:
                        t1 = perf_counter()
                        self.worklog.info(
                            f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                            f"XLoss: {total_loss.item():.5f} | GLoss: {g_loss.item():.5f}"
                        )
                        t0 = t1
                    continue

                # UCL ratio branch: independent or reuse-geo, both use pan_rendered_s1
                if ucl_ratio:
                    self._stage2_ucl_step_idx = step
                    self._stage2_ucl_total_steps = cs2
                    if stage2_admm_split and str(
                        getattr(self.args, "stage2_admm_split", False)
                    ).strip().lower() in ("true", "1", "yes", "y"):
                        tb_hr = (
                            (pan_w + self.block_w - 1) // self.block_w,
                            (pan_h + self.block_h - 1) // self.block_h, 1,
                        )

                        # Spectral substep: LR observation consistency and
                        # spectral priors only. Spatial/PAN regularizers are
                        # intentionally evaluated after this optimizer step.
                        s2_ratio_optimizer.zero_grad()
                        ms_hr, ratio_hr, product_hr, _ = self._render_ucl_current_output(
                            pan_h,
                            pan_w,
                            tb_hr,
                            product_pan_ucl,
                            pan_rendered_s1,
                            render_scale_ratio=1.0,
                        )
                        if is_cross_warmup:
                            loss_lr = ms_hr.new_tensor(0.0)
                            loss_spec_corr = ms_hr.new_tensor(0.0)
                            loss_teacher = ms_hr.new_tensor(0.0)
                            loss_obs_ratio = ms_hr.new_tensor(0.0)
                            loss_reduced = ms_hr.new_tensor(0.0)
                            loss_lower = ms_hr.new_tensor(0.0)
                            loss_cross = ms_hr.new_tensor(0.0)
                        else:
                            ms_hr_down = self._ucl_degrade_once(ms_hr)
                            loss_lr = ucl_lam * F.mse_loss(ms_hr_down, self.gt_images)
                            loss_spec_corr = self._stage2_ucl_spectral_corr_loss(ms_hr)
                            loss_teacher = self._stage2_ucl_teacher_loss(
                                ratio_hr, product_hr, pan_rendered_s1
                            )
                            loss_obs_ratio = self._stage2_ucl_observed_ratio_loss(
                                ratio_hr, product_pan_ucl
                            )
                            loss_reduced, loss_lower, loss_cross = (
                                self._stage2_ucl_multilevel_loss(
                                    ms_hr, pan_rendered_s1, pan_h, pan_w, ms_h, ms_w
                                )
                            )
                        loss_m2p_spec = self._stage2_m2p_loss(ms_hr, product_pan_ucl)
                        if self._stage2_uses_p2m():
                            p2m_out = self._render_stage2_p2m(pan_h, pan_w, tb_hr)
                            loss_p2m = self._stage2_p2m_loss(p2m_out, ms_hr)
                        else:
                            loss_p2m = ms_hr.new_tensor(0.0)
                        loss_ratio_band_mask_l2 = self._stage2_ratio_band_mask_regularizer_loss()
                        loss_pan_desc = self._stage2_pan_descriptor_regularizer_loss()
                        spectral_loss = (
                            loss_lr
                            + loss_spec_corr
                            + loss_teacher
                            + loss_obs_ratio
                            + loss_reduced
                            + loss_lower
                            + loss_cross
                            + loss_m2p_spec
                            + loss_p2m
                            + loss_ratio_band_mask_l2
                            + loss_pan_desc
                        )
                        spectral_loss.backward()
                        s2_ratio_optimizer.step()
                        if ucl_independent_geo and not ucl_scaffold_mlp and not ucl_local_ratio:
                            with torch.no_grad():
                                self.xy_ratio.clamp_(0.0, 1.0)
                                self.scale_ratio.clamp_(1e-3, 10.0)

                        loss_pan_scaffold = ms_hr.new_tensor(0.0)
                        loss_tv = ms_hr.new_tensor(0.0)
                        loss_gamma_prior = ms_hr.new_tensor(0.0)
                        loss_fixed_gain_tv = ms_hr.new_tensor(0.0)
                        loss_graph = ms_hr.new_tensor(0.0)
                        loss_feat_graph = ms_hr.new_tensor(0.0)
                        loss_local_reg = ms_hr.new_tensor(0.0)
                        loss_dense_reg = ms_hr.new_tensor(0.0)
                        loss_pan_detail = ms_hr.new_tensor(0.0)
                        loss_chroma = ms_hr.new_tensor(0.0)
                        loss_bright_chroma = ms_hr.new_tensor(0.0)
                        loss_pan_constraint = ms_hr.new_tensor(0.0)
                        spatial_loss = ms_hr.new_tensor(0.0)

                        do_spatial_step = (
                            (step % stage2_ucl_spatial_every == 0 or step == cs2)
                            and not is_cross_warmup
                        )
                        if do_spatial_step:
                            # Spatial substep: edge-preserving PAN-guided ratio
                            # smoothing and PAN/Gaussian-space priors.
                            s2_ratio_optimizer.zero_grad()
                            ms_hr_sp, ratio_hr_sp, _, pan_new_sp = self._render_ucl_current_output(
                                pan_h,
                                pan_w,
                                tb_hr,
                                product_pan_ucl,
                                pan_rendered_s1,
                                render_scale_ratio=1.0,
                            )
                            if ucl_scaffold_mlp and pan_new_sp is not None:
                                loss_pan_scaffold = self._stage2_scaffold_pan_loss(pan_new_sp)
                            loss_tv = self._stage2_ucl_tv_loss(
                                ratio_hr_sp,
                                self.ref_images,
                                force_pan_guided=True,
                            )
                            loss_gamma_prior = self._stage2_ucl_gamma_prior_loss(
                                ratio_hr_sp, gamma_input_pan_ucl, gamma_denom_pan_ucl
                            )
                            loss_fixed_gain_tv = self._stage2_ucl_fixed_pan_gain_tv_loss(
                                ratio_hr_sp, product_pan_ucl
                            )
                            loss_graph = self._stage2_local_ratio_graph_loss()
                            loss_feat_graph = self._stage2_feat_ratio_graph_loss()
                            loss_local_reg = self._stage2_local_ratio_basis_reg_loss()
                            loss_dense_reg = (
                                self._stage2_dense_residual_reg_loss()
                                + self._stage2_pan_gap_gain_reg_loss()
                                + self._stage2_lig_gap_reg_loss()
                            )
                            loss_pan_detail = (
                                self._stage2_pan_detail_reg_loss()
                                + self._stage2_pan_detail_constraint_loss(ms_hr_sp, product_pan_ucl)
                            )
                            loss_chroma = self._stage2_ucl_chroma_prior_loss(
                                ms_hr_sp, self.ref_images
                            )
                            _residual_raw = getattr(self, "_stage2_last_residual_raw", None)
                            _residual_gate = getattr(self, "_stage2_last_residual_gate", None)
                            _residual_contrib = getattr(
                                self, "_stage2_last_residual_contribution", None
                            )
                            if _residual_raw is not None:
                                loss_residual_reg = self._stage2_residual_regularizer_loss(
                                    _residual_raw,
                                    _residual_gate,
                                    residual_contribution=_residual_contrib,
                                )
                            else:
                                loss_residual_reg = ms_hr.new_tensor(0.0)
                            loss_residual_basis_reg = self._stage2_residual_basis_regularizer_loss()
                            if loss_residual_basis_reg is None:
                                loss_residual_basis_reg = ms_hr.new_tensor(0.0)
                            if s2_pan_w > 0 and step % s2_pan_every == 0:
                                loss_pan_constraint = s2_pan_w * self._stage2_pan_constraint_loss(
                                    ms_hr_sp, step, cs2
                                )
                            loss_inmask_tv = self._stage2_inmask_ratio_tv_loss(ratio_hr_sp)
                            loss_locality = self._stage2_locality_loss(self.feat_ratio)
                            loss_inmask_sam = self._stage2_inmask_sam_loss(ms_hr_sp)
                            loss_bright_chroma = self._stage2_bright_chroma_anchor_loss(ms_hr_sp)
                            spatial_loss = (
                                loss_pan_scaffold
                                + loss_tv
                                + loss_gamma_prior
                                + loss_fixed_gain_tv
                                + loss_graph
                                + loss_feat_graph
                                + loss_local_reg
                                + loss_dense_reg
                                + loss_pan_detail
                                + loss_chroma
                                + loss_residual_reg
                                + loss_residual_basis_reg
                                + loss_pan_constraint
                                + loss_inmask_tv
                                + loss_locality
                                + loss_inmask_sam
                                + loss_bright_chroma
                            )
                            if spatial_loss.requires_grad:
                                spatial_loss.backward()
                                s2_ratio_optimizer.step()
                                if ucl_independent_geo and not ucl_scaffold_mlp and not ucl_local_ratio:
                                    with torch.no_grad():
                                        self.xy_ratio.clamp_(0.0, 1.0)
                                        self.scale_ratio.clamp_(1e-3, 10.0)
                        total_loss = spectral_loss.detach() + spatial_loss.detach()
                        if (
                            stage2_debug_hr_trace_ref is not None
                            and stage2_debug_hr_trace_every > 0
                            and (
                                step % stage2_debug_hr_trace_every == 0
                                or step == cs2
                            )
                        ):
                            with torch.no_grad():
                                trace_ms_hr, _, _, _ = self._render_ucl_current_output(
                                    pan_h,
                                    pan_w,
                                    tb_hr,
                                    product_pan_ucl,
                                    pan_rendered_s1,
                                    render_scale_ratio=1.0,
                                )
                                self._record_stage2_debug_hr_trace(
                                    "train_update",
                                    rnd + 1,
                                    step,
                                    trace_ms_hr,
                                    stage2_debug_hr_trace_ref,
                                    ms_lr=self._ucl_degrade_once(trace_ms_hr),
                                )
                        if self.step % self.eval_steps == 0:
                            lr_mse_for_stop = (
                                loss_lr.detach()
                                / max(float(getattr(self.args, "stage2_ucl_lam", 1.0)), 1.0e-12)
                            ).clamp_min(1.0e-12)
                            lr_psnr_for_stop = -10.0 * torch.log10(lr_mse_for_stop)
                            t1 = perf_counter()
                            self.worklog.info(
                                f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                                f"UCL_split: {total_loss.item():.6f} | "
                                f"spectral: {spectral_loss.item():.6f} | "
                                f"spatial: {spatial_loss.item():.6f} | "
                                f"lr: {loss_lr.item():.6f} | "
                                f"lr_psnr: {float(lr_psnr_for_stop.item()):.4f} | "
                                f"R_spec: {loss_spec_corr.item():.9f} | "
                                f"pan_scf: {loss_pan_scaffold.item():.6f} | "
                                f"pan_hf: {loss_pan_constraint.item():.6f} | "
                                f"tv_pan: {loss_tv.item():.6f} | "
                                f"gamma_tv: {loss_gamma_prior.item():.6f} | "
                                f"fixed_gain_tv: {loss_fixed_gain_tv.item():.6f} | "
                                f"ratio_band_mask_l2: {loss_ratio_band_mask_l2.item():.6f} | "
                                f"pan_desc: {loss_pan_desc.item():.6f} | "
                                f"teacher: {loss_teacher.item():.6f} | "
                                f"obs_ratio: {loss_obs_ratio.item():.6f} | "
                                f"graph: {loss_graph.item():.6f} | "
                                f"feat_graph: {loss_feat_graph.item():.6f} | "
                                f"local_reg: {loss_local_reg.item():.6f} | "
                                f"dense_reg: {loss_dense_reg.item():.6f} | "
                                f"pan_detail: {loss_pan_detail.item():.6f} | "
                                f"residual_reg: {loss_residual_reg.item():.6f} | "
                                f"residual_basis: {loss_residual_basis_reg.item():.6f} | "
                                f"chroma: {loss_chroma.item():.6f} | "
                                f"bright_chroma: {loss_bright_chroma.item():.6f} | "
                                f"hf_match: {loss_m2p_spec.item():.6f} | "
                                f"p2m: {loss_p2m.item():.6f} | "
                                f"red: {loss_reduced.item():.6f} | "
                                f"low: {loss_lower.item():.6f} | "
                                f"xscale: {loss_cross.item():.6f}"
                            )
                            t0 = t1
                            if (
                                (ucl_early_stop_lr_loss > 0 or ucl_early_stop_lr_psnr > 0)
                                and not is_cross_warmup
                                and step >= ucl_early_stop_min_steps
                            ):
                                lr_loss_ok = (
                                    ucl_early_stop_lr_loss > 0
                                    and float(loss_lr.detach().item()) <= ucl_early_stop_lr_loss
                                )
                                lr_psnr_ok = (
                                    ucl_early_stop_lr_psnr > 0
                                    and float(lr_psnr_for_stop.item()) >= ucl_early_stop_lr_psnr
                                )
                                if lr_loss_ok or lr_psnr_ok:
                                    ucl_early_stop_hits += 1
                                else:
                                    ucl_early_stop_hits = 0
                                if ucl_early_stop_hits >= ucl_early_stop_patience:
                                    self.worklog.info(
                                        f"Stage2 UCL early stop at step {step:d}/{cs2:d} "
                                        f"(lr={float(loss_lr.detach().item()):.6g}, "
                                        f"lr_psnr={float(lr_psnr_for_stop.item()):.4f}, "
                                        f"loss_threshold={ucl_early_stop_lr_loss:.6g}, "
                                        f"psnr_threshold={ucl_early_stop_lr_psnr:.4f})."
                                    )
                                    break
                        continue

                    tb_hr = (
                        (pan_w + self.block_w - 1) // self.block_w,
                        (pan_h + self.block_h - 1) // self.block_h, 1,
                    )

                    # MS-error-guided densification (Phase B)
                    s2_add_times = int(getattr(self.args, "stage2_add_times", 0))
                    s2_add_steps = int(getattr(self.args, "stage2_add_steps", 0))
                    if (
                        s2_add_times > 0
                        and s2_add_steps > 0
                        and step > 0
                        and step % s2_add_steps == 0
                        and self.num_gaussians < self.total_num_gaussians
                        and getattr(self, "_stage2_densify_count", 0) < s2_add_times
                    ):
                        with torch.no_grad():
                            ms_hr_for_prob, ratio_hr_for_init, _, _ = self._render_ucl_current_output(
                                pan_h,
                                pan_w,
                                tb_hr,
                                product_pan_ucl,
                                pan_rendered_s1,
                                render_scale_ratio=1.0,
                            )
                        s2_remaining_rounds = s2_add_times - getattr(
                            self, "_stage2_densify_count", 0
                        )
                        s2_budget = self.total_num_gaussians - self.num_gaussians
                        s2_add_num = max(1, s2_budget // max(s2_remaining_rounds, 1))
                        s2_ratio_optimizer = self._add_gaussians_ms_guided(
                            s2_add_num,
                            ms_hr_for_prob,
                            s2_ratio_optimizer,
                            ratio_hr=ratio_hr_for_init,
                            gt_hr=densify_gt_hr,
                        )
                        self._stage2_densify_count = (
                            getattr(self, "_stage2_densify_count", 0) + 1
                        )
                        if self._stage2_ucl_uses_current_product_pan():
                            product_pan_ucl = self._stage2_ucl_product_pan(
                                pan_rendered_s1
                            )

                    s2_ratio_optimizer.zero_grad()
                    ms_hr, ratio_hr, product_hr, pan_new = self._render_ucl_current_output(
                        pan_h,
                        pan_w,
                        tb_hr,
                        product_pan_ucl,
                        pan_rendered_s1,
                        render_scale_ratio=1.0,
                    )
                    product_pan_for_loss = getattr(
                        self, "_stage2_last_product_pan", product_pan_ucl
                    )
                    ms_hr_down = self._ucl_degrade_once(ms_hr)
                    loss_lr, loss_lr_low, loss_lr_raw_mse = (
                        self._stage2_curriculum_lr_loss(
                            ms_hr_down, self.gt_images, step, ucl_lam
                        )
                    )
                    loss_hr_sam_prior = self._stage2_hr_sam_prior_loss(ms_hr)
                    loss_lr_hp = ms_hr.new_tensor(0.0)
                    lr_hp_w = float(getattr(self.args, "stage2_lr_hp_weight", 0.0))
                    if lr_hp_w > 0:
                        lr_hp_sigma = float(
                            getattr(self.args, "stage2_lr_hp_sigma", 1.0)
                        )
                        lr_hp_norm = str(
                            getattr(self.args, "stage2_lr_hp_norm", "l1")
                        ).lower()
                        pred_lp = self._gaussian_lowpass_sigma(
                            ms_hr_down, lr_hp_sigma
                        )
                        gt_lp = self._gaussian_lowpass_sigma(
                            self.gt_images.detach(), lr_hp_sigma
                        )
                        pred_hp = ms_hr_down - pred_lp
                        gt_hp = self.gt_images.detach() - gt_lp
                        if lr_hp_norm == "l2":
                            loss_lr_hp = lr_hp_w * F.mse_loss(pred_hp, gt_hp)
                        else:
                            loss_lr_hp = lr_hp_w * F.l1_loss(pred_hp, gt_hp)
                    loss_pan_scaffold = ms_hr.new_tensor(0.0)
                    if ucl_scaffold_mlp and pan_new is not None:
                        loss_pan_scaffold = self._stage2_scaffold_pan_loss(pan_new)
                    loss_tv = ms_hr.new_tensor(0.0)
                    # TV regularization on ratio map (same as UCL: mu=0.001)
                    ucl_tv_weight = float(getattr(self.args, "stage2_ucl_tv_weight", 0.001))
                    if ucl_tv_weight > 0:
                        r = ratio_hr  # [C, H, W]
                        tv = (torch.pow(r[:, :, :-1] - r[:, :, 1:], 2).mean() +
                              torch.pow(r[:, :-1, :] - r[:, 1:, :], 2).mean())
                        loss_tv = ucl_tv_weight * tv
                    loss_gamma_prior = self._stage2_ucl_gamma_prior_loss(
                        ratio_hr, gamma_input_pan_ucl, gamma_denom_pan_ucl
                    )
                    loss_fixed_gain_tv = self._stage2_ucl_fixed_pan_gain_tv_loss(
                        ratio_hr, product_pan_for_loss
                    )
                    loss_ratio_split_dir, loss_ratio_hf_zero = (
                        self._stage2_ratio_split_regularizer_loss(
                            ratio_hr, product_pan_for_loss
                        )
                    )
                    loss_ratio_poly_l2 = self._stage2_ratio_poly_regularizer_loss()
                    loss_ratio_band_mask_l2 = self._stage2_ratio_band_mask_regularizer_loss()
                    loss_pan_desc = self._stage2_pan_descriptor_regularizer_loss()
                    loss_teacher = self._stage2_ucl_teacher_loss(
                        ratio_hr, product_hr, pan_rendered_s1
                    )
                    loss_teacher_output = self._stage2_ucl_teacher_output_loss(ms_hr)
                    loss_obs_ratio = self._stage2_ucl_observed_ratio_loss(
                        ratio_hr, product_pan_for_loss
                    )
                    loss_graph = self._stage2_local_ratio_graph_loss()
                    loss_feat_graph = self._stage2_feat_ratio_graph_loss()
                    loss_local_reg = self._stage2_local_ratio_basis_reg_loss()
                    loss_dense_reg = (
                        self._stage2_dense_residual_reg_loss()
                        + self._stage2_pan_gap_gain_reg_loss()
                        + self._stage2_lig_gap_reg_loss()
                    )
                    loss_pan_detail = (
                        self._stage2_pan_detail_reg_loss()
                        + self._stage2_pan_detail_constraint_loss(ms_hr, product_pan_for_loss)
                    )
                    residual_raw = getattr(self, "_stage2_last_residual_raw", None)
                    residual_gate = getattr(self, "_stage2_last_residual_gate", None)
                    residual_contrib = getattr(
                        self, "_stage2_last_residual_contribution", None
                    )
                    loss_residual_reg = (
                        self._stage2_residual_regularizer_loss(
                            residual_raw,
                            residual_gate,
                            residual_contribution=residual_contrib,
                        )
                        if residual_raw is not None
                        else ms_hr.new_tensor(0.0)
                    )
                    loss_residual_basis_reg = self._stage2_residual_basis_regularizer_loss()
                    if loss_residual_basis_reg is None:
                        loss_residual_basis_reg = ms_hr.new_tensor(0.0)
                    loss_rr, loss_rr_raw, rr_psnr, x_rr = self._stage2_rr_constructed_loss(
                        product_pan_for_loss,
                        pan_rendered_s1,
                        pan_h,
                        pan_w,
                        ms_h,
                        ms_w,
                    )
                    self._stage2_last_residual_raw = residual_raw
                    self._stage2_last_residual_gate = residual_gate
                    self._stage2_last_residual_contribution = residual_contrib
                    loss_reduced, loss_lower, loss_cross = (
                        self._stage2_ucl_multilevel_loss(
                            ms_hr, pan_rendered_s1, pan_h, pan_w, ms_h, ms_w
                        )
                    )
                    loss_pan_constraint = ms_hr.new_tensor(0.0)
                    # PAN high-frequency constraint (gradient + edge matching)
                    if s2_pan_w > 0 and step % s2_pan_every == 0:
                        loss_pan_constraint = s2_pan_w * self._stage2_pan_constraint_loss(
                            ms_hr, step, cs2
                        )
                    loss_pan_corr = self._stage2_pan_corr_constraint_loss(ms_hr)
                    loss_pan_full_l1, loss_pan_hf_l1 = (
                        self._stage2_pan_l1_constraint_losses(ms_hr)
                    )

                    # ORACLE_INVALID: in-mask GT supervision diagnostic.
                    # NOT a publishable result — uses HR_GT as training signal
                    # in top-K%% PAN-bright pixels to measure ceiling of
                    # in-mask repair. Gated by stage2_oracle_inmask_gt_weight=0.
                    loss_inmask_tv = self._stage2_inmask_ratio_tv_loss(ratio_hr)
                    loss_locality = self._stage2_locality_loss(self.feat_ratio)
                    loss_inmask_sam = self._stage2_inmask_sam_loss(ms_hr)
                    loss_chroma = self._stage2_ucl_chroma_prior_loss(
                        ms_hr, product_pan_for_loss
                    )
                    loss_bright_upper = self._stage2_bright_intensity_upper_loss(ms_hr)
                    loss_bright_chroma = self._stage2_bright_chroma_anchor_loss(ms_hr)
                    if bool(getattr(self.args, "stage2_loss_gate_aux", False)):
                        _aux_scale = float(
                            getattr(
                                self,
                                "_stage2_last_loss_curriculum",
                                {"aux_scale": 1.0},
                            ).get("aux_scale", 1.0)
                        )
                        loss_hr_sam_prior = loss_hr_sam_prior * _aux_scale
                        loss_pan_corr = loss_pan_corr * _aux_scale
                        loss_pan_full_l1 = loss_pan_full_l1 * _aux_scale
                        loss_pan_hf_l1 = loss_pan_hf_l1 * _aux_scale
                        loss_inmask_sam = loss_inmask_sam * _aux_scale
                        loss_chroma = loss_chroma * _aux_scale
                        loss_bright_upper = loss_bright_upper * _aux_scale
                        loss_bright_chroma = loss_bright_chroma * _aux_scale
                        loss_teacher_output = loss_teacher_output * _aux_scale

                    loss_oracle_inmask = ms_hr.new_tensor(0.0)
                    _oracle_w = float(getattr(
                        self.args, "stage2_oracle_inmask_gt_weight", 0.0
                    ))
                    # ORACLE_INVALID: use densify_gt_hr (real HR_GT only when cfg
                    # sets stage2_densify_gt_hr_source=rr_gt and provides a path,
                    # which eval_benchmark.py wires to the patch's HR_GT). NOT
                    # gt_hr — that local was reassigned to _build_stage2_target()
                    # at line 7796 and is a synthetic MTF-GLP target, not real GT.
                    if _oracle_w > 0 and densify_gt_hr is not None:
                        _oracle_topk = float(getattr(
                            self.args, "stage2_oracle_inmask_gt_topk", 0.03
                        ))
                        _oracle_topk = min(max(_oracle_topk, 1.0e-4), 1.0)
                        if not hasattr(self, "_oracle_inmask_cache"):
                            with torch.no_grad():
                                _pan_y = self.ref_images.detach()
                                if _pan_y.dim() == 3:
                                    _pan_y = _pan_y.mean(0)
                                _thr = torch.quantile(
                                    _pan_y.flatten(), 1.0 - _oracle_topk
                                )
                                self._oracle_inmask_cache = (
                                    _pan_y >= _thr
                                ).to(ms_hr.dtype)
                                _n_in = int(self._oracle_inmask_cache.sum().item())
                                self.worklog.info(
                                    f">>> ORACLE_INVALID: in-mask GT supervision "
                                    f"active. topk={_oracle_topk*100:.2f}% "
                                    f"({_n_in} px), weight={_oracle_w:.3g}. "
                                    f"This run is for diagnostic only; HR_GT in loss."
                                )
                                _src = str(getattr(
                                    self.args, "stage2_densify_gt_hr_source",
                                    "stage2_target",
                                )).lower()
                                if _src not in ("rr_gt", "external", "gt", "hr_gt"):
                                    self.worklog.info(
                                        f">>> ORACLE WARNING: "
                                        f"stage2_densify_gt_hr_source={_src}, "
                                        f"so oracle is supervising against the "
                                        f"synthetic MTF-GLP target, NOT real "
                                        f"HR_GT. Set source=rr_gt for a real "
                                        f"oracle test."
                                    )
                        _mask = self._oracle_inmask_cache  # [H, W]
                        _gt_det = densify_gt_hr.detach()
                        _diff = (ms_hr - _gt_det).abs() * _mask.unsqueeze(0)
                        _denom = (_mask.sum() * ms_hr.shape[0]).clamp_min(1.0)
                        loss_oracle_inmask = _oracle_w * (_diff.sum() / _denom)

                    total_loss = (
                        loss_lr
                        + loss_lr_low
                        + loss_hr_sam_prior
                        + loss_lr_hp
                        + loss_pan_scaffold
                        + loss_tv
                        + loss_gamma_prior
                        + loss_fixed_gain_tv
                        + loss_ratio_split_dir
                        + loss_ratio_hf_zero
                        + loss_ratio_poly_l2
                        + loss_ratio_band_mask_l2
                        + loss_pan_desc
                        + loss_teacher
                        + loss_teacher_output
                        + loss_obs_ratio
                        + loss_graph
                        + loss_feat_graph
                        + loss_local_reg
                        + loss_dense_reg
                        + loss_pan_detail
                        + loss_residual_reg
                        + loss_residual_basis_reg
                        + loss_rr
                        + loss_reduced
                        + loss_lower
                        + loss_cross
                        + loss_pan_constraint
                        + loss_pan_corr
                        + loss_pan_full_l1
                        + loss_pan_hf_l1
                        + loss_inmask_tv
                        + loss_locality
                        + loss_inmask_sam
                        + loss_chroma
                        + loss_bright_upper
                        + loss_bright_chroma
                        + loss_oracle_inmask
                    )
                    total_loss.backward()
                    s2_ratio_optimizer.step()
                    if bool(getattr(self.args, "stage2_densify_train_new_geo", False)):
                        with torch.no_grad():
                            self.xy.clamp_(0.0, 1.0)
                            self.scale.clamp_(1.0e-3, 10.0)
                    if ucl_independent_geo and not ucl_scaffold_mlp and not ucl_local_ratio:
                        with torch.no_grad():
                            self.xy_ratio.clamp_(0.0, 1.0)
                            self.scale_ratio.clamp_(1e-3, 10.0)
                    if (
                        stage2_debug_hr_trace_ref is not None
                        and stage2_debug_hr_trace_every > 0
                        and (
                            step % stage2_debug_hr_trace_every == 0
                            or step == cs2
                        )
                    ):
                        with torch.no_grad():
                            trace_ms_hr, _, _, _ = self._render_ucl_current_output(
                                pan_h,
                                pan_w,
                                tb_hr,
                                product_pan_ucl,
                                pan_rendered_s1,
                                render_scale_ratio=1.0,
                            )
                            self._record_stage2_debug_hr_trace(
                                "train_update",
                                rnd + 1,
                                step,
                                trace_ms_hr,
                                stage2_debug_hr_trace_ref,
                                ms_lr=self._ucl_degrade_once(trace_ms_hr),
                            )
                    if self.step % self.eval_steps == 0:
                        lr_mse_for_stop = loss_lr_raw_mse.detach().clamp_min(1.0e-12)
                        lr_psnr_for_stop = -10.0 * torch.log10(lr_mse_for_stop)
                        _cur_stats = getattr(
                            self,
                            "_stage2_last_loss_curriculum",
                            {
                                "active": False,
                                "direct_scale": 1.0,
                                "aux_scale": 1.0,
                            },
                        )
                        t1 = perf_counter()
                        self.worklog.info(
                            f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                            f"UCL_loss: {total_loss.item():.6f} | "
                            f"lr: {loss_lr.item():.6f} | "
                            f"lr_low: {loss_lr_low.item():.6f} | "
                            f"cur_direct: {float(_cur_stats.get('direct_scale', 1.0)):.3f} | "
                            f"cur_aux: {float(_cur_stats.get('aux_scale', 1.0)):.3f} | "
                            f"hr_sam: {loss_hr_sam_prior.item():.6f} | "
                            f"lr_hp: {loss_lr_hp.item():.6f} | "
                            f"lr_psnr: {float(lr_psnr_for_stop.item()):.4f} | "
                            f"pan_scf: {loss_pan_scaffold.item():.6f} | "
                            f"pan_hf: {loss_pan_constraint.item():.6f} | "
                            f"pan_corr: {loss_pan_corr.item():.6f} | "
                            f"pan_full_l1: {loss_pan_full_l1.item():.6f} | "
                            f"pan_hf_l1: {loss_pan_hf_l1.item():.6f} | "
                            f"tv: {loss_tv.item():.6f} | "
                            f"inmask_tv: {loss_inmask_tv.item():.6f} | "
                            f"locality: {loss_locality.item():.6f} | "
                            f"inmask_sam: {loss_inmask_sam.item():.6f} | "
                            f"chroma: {loss_chroma.item():.6f} | "
                            f"bright_upper: {loss_bright_upper.item():.6f} | "
                            f"bright_chroma: {loss_bright_chroma.item():.6f} | "
                            f"gamma_tv: {loss_gamma_prior.item():.6f} | "
                            f"fixed_gain_tv: {loss_fixed_gain_tv.item():.6f} | "
                            f"ratio_split_dir: {loss_ratio_split_dir.item():.6f} | "
                            f"ratio_hf_zero: {loss_ratio_hf_zero.item():.6f} | "
                            f"ratio_poly_l2: {loss_ratio_poly_l2.item():.6f} | "
                            f"ratio_band_mask_l2: {loss_ratio_band_mask_l2.item():.6f} | "
                            f"pan_desc: {loss_pan_desc.item():.6f} | "
                            f"teacher: {loss_teacher.item():.6f} | "
                            f"teacher_out: {loss_teacher_output.item():.6f} | "
                            f"obs_ratio: {loss_obs_ratio.item():.6f} | "
                            f"graph: {loss_graph.item():.6f} | "
                            f"local_reg: {loss_local_reg.item():.6f} | "
                            f"dense_reg: {loss_dense_reg.item():.6f} | "
                                f"pan_detail: {loss_pan_detail.item():.6f} | "
                                f"residual_reg: {loss_residual_reg.item():.6f} | "
                                f"residual_basis: {loss_residual_basis_reg.item():.6f} | "
                                f"rr: {loss_rr.item():.6f} | "
                                f"rr_raw: {loss_rr_raw.item():.6f} | "
                                f"rr_psnr: {float(rr_psnr.item()):.4f} | "
                                f"red: {loss_reduced.item():.6f} | "
                            f"low: {loss_lower.item():.6f} | "
                            f"xscale: {loss_cross.item():.6f}"
                        )
                        mask_stats = self._stage2_ratio_band_mask_stats()
                        if mask_stats is not None:
                            self.worklog.info(
                                "Stage2 ratio band mask | "
                                f"scale_min={mask_stats[0]:.6f} "
                                f"scale_max={mask_stats[1]:.6f} "
                                f"scale_mean={mask_stats[2]:.6f}"
                            )
                        pan_desc_stats = self._stage2_pan_descriptor_stats()
                        if pan_desc_stats is not None:
                            self.worklog.info(
                                "Stage2 PAN descriptor | "
                                f"delta_abs_mean={pan_desc_stats['delta_abs_mean']:.6f} "
                                f"delta_min={pan_desc_stats['delta_min']:.6f} "
                                f"delta_max={pan_desc_stats['delta_max']:.6f} "
                                f"weight_abs_mean={pan_desc_stats['weight_abs_mean']:.6f} "
                                f"weight_max_abs={pan_desc_stats['weight_max_abs']:.6f}"
                            )
                        if self._stage2_residual_branch_enabled() and residual_raw is not None:
                            gate = getattr(self, "_stage2_last_residual_gate", None)
                            carrier = getattr(self, "_stage2_last_residual_carrier", None)
                            contrib = getattr(self, "_stage2_last_residual_contribution", None)
                            contrib_ratio = (
                                contrib.abs().mean() / ms_hr.abs().mean().clamp_min(1.0e-8)
                                if contrib is not None else ms_hr.new_tensor(0.0)
                            )
                            self.worklog.info(
                                "Stage2 residual branch | "
                                f"raw_mean={float(residual_raw.mean().item()):.6f} "
                                f"raw_absmean={float(residual_raw.abs().mean().item()):.6f} "
                                f"raw_max={float(residual_raw.max().item()):.6f} "
                                f"gate_mean={float(gate.mean().item()) if gate is not None else 0.0:.6f} "
                                f"gate_max={float(gate.max().item()) if gate is not None else 0.0:.6f} "
                                f"gate_active={float((gate > 0.5).to(dtype=gate.dtype).mean().item()) if gate is not None else 0.0:.6f} "
                                f"carrier_absmean={float(carrier.abs().mean().item()) if carrier is not None else 0.0:.6f} "
                                f"contrib_ratio={float(contrib_ratio.item()):.6f}"
                            )
                        t0 = t1
                        if (
                            (ucl_early_stop_lr_loss > 0 or ucl_early_stop_lr_psnr > 0)
                            and step >= ucl_early_stop_min_steps
                        ):
                            lr_loss_ok = (
                                ucl_early_stop_lr_loss > 0
                                and float(loss_lr.detach().item()) <= ucl_early_stop_lr_loss
                            )
                            lr_psnr_ok = (
                                ucl_early_stop_lr_psnr > 0
                                and float(lr_psnr_for_stop.item()) >= ucl_early_stop_lr_psnr
                            )
                            if lr_loss_ok or lr_psnr_ok:
                                ucl_early_stop_hits += 1
                            else:
                                ucl_early_stop_hits = 0
                            if ucl_early_stop_hits >= ucl_early_stop_patience:
                                self.worklog.info(
                                    f"Stage2 UCL early stop at step {step:d}/{cs2:d} "
                                    f"(lr={float(loss_lr.detach().item()):.6g}, "
                                    f"lr_psnr={float(lr_psnr_for_stop.item()):.4f}, "
                                    f"loss_threshold={ucl_early_stop_lr_loss:.6g}, "
                                    f"psnr_threshold={ucl_early_stop_lr_psnr:.4f})."
                                )
                                break
                    continue

                self.optimizer.zero_grad()

                if resolution_curriculum:
                    total_loss, res_tag = self._stage2_resolution_curriculum_loss(
                        step,
                        cs2,
                        pan_h,
                        pan_w,
                        ms_h,
                        ms_w,
                        s2_finetune_geo,
                        s2_train_rot,
                        s2_train_scale,
                        s2_hr_w,
                        s2_pan_w,
                        x_prior_w,
                    )
                    total_loss.backward()
                    self.optimizer.step()
                    if s2_train_xy:
                        with torch.no_grad():
                            self.xy.clamp_(0.0, 1.0)
                    if self.step % self.eval_steps == 0:
                        t1 = perf_counter()
                        self.worklog.info(
                            f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                            f"Loss: {total_loss.item():.5f} | Res: {res_tag}"
                        )
                        t0 = t1
                    continue

                g_lr, g_hr = self._render_stage2_pair(
                    pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    train_geom=s2_finetune_geo,
                )
                if use_x_field:
                    ms_lr, ms_hr = self._render_stage2_x_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        train_geom=s2_finetune_geo,
                    )
                else:
                    ms_lr, ms_hr = g_lr, g_hr

                # UCL-style: pure LR_MS supervision via MTF degradation
                ucl_lam = float(getattr(self.args, "stage2_ucl_lam", 0.0))
                if ucl_lam > 0:
                    ms_hr_down = self._ucl_down(self._ucl_blur(ms_hr.unsqueeze(0))).squeeze(0)
                    total_loss = ucl_lam * F.mse_loss(ms_hr_down, self.gt_images)
                else:
                    lr_pred = self._stage2_lr_consistency_image(ms_lr, ms_hr, ms_h, ms_w)
                    total_loss = s2_lr_w * self.l1_loss_ratio * F.l1_loss(
                        lr_pred, self.gt_images
                    )

                if ucl_lam <= 0 and s2_hr_w > 0:
                    ms_hr_loss, gt_hr_loss, _ = self._stage2_frequency_pair(
                        ms_hr, gt_hr, step, cs2
                    )
                    total_loss += s2_hr_w * self.l1_loss_ratio * (
                        band_w * torch.abs(ms_hr_loss - gt_hr_loss)
                    ).mean()

                if s2_pan_w > 0 and step % s2_pan_every == 0 and ucl_lam <= 0:
                    total_loss += s2_pan_w * self._stage2_pan_constraint_loss(
                        ms_hr, step, cs2
                    )

                if (
                    float(getattr(self.args, "stage2_g_lr_ratio_weight", 0.0)) > 0
                    or float(getattr(self.args, "stage2_g_tv_weight", 0.0)) > 0
                ):
                    g_coeff_lr, g_coeff_hr = self._render_stage2_coeff_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        train_geom=s2_finetune_geo,
                        train_rot=s2_train_rot,
                        train_scale=s2_train_scale,
                    )
                    total_loss += self._stage2_g_ratio_prior_loss(
                        g_coeff_lr, g_coeff_hr, ms_h, ms_w
                    )

                if use_x_field and x_prior_w > 0:
                    total_loss += x_prior_w * self.l1_loss_ratio * F.l1_loss(
                        ms_hr, g_hr.detach()
                    )

                total_loss += self._stage2_regularizer_loss(
                    ms_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan
                )

                total_loss.backward()
                self.optimizer.step()
                if s2_train_xy:
                    with torch.no_grad():
                        self.xy.clamp_(0.0, 1.0)

                if self.step % self.eval_steps == 0:
                    t1 = perf_counter()
                    self.worklog.info(
                        f"Step: {self.step:d} | {t1 - t0:.2f}s | Loss: {total_loss.item():.5f}"
                    )
                    t0 = t1

            if freq_global:
                freq_seen_steps = min(
                    freq_seen_steps + cs2, int(self._stage2_freq_total_steps)
                )

            # -- Stage 2 eval --------------------------------------------------

            with torch.no_grad():
                if ucl_ratio:
                    tb_hr = (
                        (pan_w + self.block_w - 1) // self.block_w,
                        (pan_h + self.block_h - 1) // self.block_h, 1,
                    )
                    ms_hr, ratio_hr, product_hr, pan_new = self._render_ucl_current_output(
                        pan_h,
                        pan_w,
                        tb_hr,
                        product_pan_ucl,
                        pan_rendered_s1,
                        render_scale_ratio=1.0,
                    )
                    ms_lr = self._ucl_degrade_once(ms_hr)
                    product_pan_eval = getattr(
                        self, "_stage2_last_product_pan", product_pan_ucl
                    )
                else:
                    g_lr, g_hr = self._render_stage2_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    )
                    if use_x_field:
                        ms_lr, ms_hr = self._render_stage2_x_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        )
                    else:
                        ms_lr, ms_hr = g_lr, g_hr
                self._record_stage2_debug_hr_trace(
                    "stage2_eval",
                    rnd + 1,
                    cs2,
                    ms_hr,
                    stage2_debug_hr_trace_ref,
                    ms_lr=ms_lr,
                )
                fr = full_ref_metrics(ms_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                self.psnr_curr = fr["PSNR"].item()
                self.worklog.info(
                    f"Stage2 Eval | LR_PSNR: {self.psnr_curr:.4f} | "
                    f"LR_SSIM: {fr['SSIM'].item():.4f}"
                )
                if ucl_ratio:
                    self.worklog.info(
                        "Stage2 product PAN eval | "
                        f"mode={self._stage2_product_pan_mode()} "
                        f"l1={float(F.l1_loss(product_pan_eval, self.ref_images).item()):.6f} "
                        f"hf_energy={float(self._highpass(product_pan_eval).abs().mean().item()):.6f}"
                    )
                self._save_pansharpened(ms_hr, "stage2_hr", pan_h, pan_w)

            if (
                rnd == 0
                and hard_patch_threshold > 0
                and self.psnr_curr < hard_patch_threshold
            ):
                self._hard_patch_active = True
                self._hard_patch_start_round = rnd + 1
                gt_hr = self._build_stage2_target()
                _bmu = gt_hr.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                band_w = (1.0 / _bmu) / (1.0 / _bmu).mean()
                self._initialize_stage2_feat_ms(gt_hr, use_bicubic=False)
                self._set_trainable_modules(stage=2)
                self.worklog.info(
                    "Hard patch gate active: "
                    f"stage2_lr_psnr={self.psnr_curr:.4f} < {hard_patch_threshold:.4f}; "
                    f"mtf_gain_scale={self._current_stage2_mtf_gain_scale():.2f}, "
                    f"stage3_output_mix={self._current_stage3_output_mix():.2f}."
                )

                if freq_global:
                    self._stage2_freq_step_offset = freq_seen_steps

                if alternate_xg:
                    x_optimizer, g_optimizer = self._make_stage2_alternating_optimizers(
                        s2_train_xy, s2_train_scale, s2_train_rot
                    )

                for step in range(1, cs2 + 1):
                    self.step = s1_done_steps + step
                    if alternate_xg:
                        with torch.no_grad():
                            _, g_hr_det = self._render_stage2_pair(
                                pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                            )
                        x_optimizer.zero_grad()
                        ms_lr, ms_hr = self._render_stage2_x_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                            train_geom=s2_train_xy,
                            train_rot=s2_train_rot,
                            train_scale=s2_train_scale,
                        )
                        total_loss = s2_lr_w * self.l1_loss_ratio * F.l1_loss(
                            self._stage2_lr_consistency_image(ms_lr, ms_hr, ms_h, ms_w),
                            self.gt_images
                        )
                        if s2_hr_w > 0:
                            ms_hr_loss, gt_hr_loss, _ = self._stage2_frequency_pair(
                                ms_hr, gt_hr, step, cs2
                            )
                            total_loss += s2_hr_w * self.l1_loss_ratio * (
                                band_w * torch.abs(ms_hr_loss - gt_hr_loss)
                            ).mean()
                        if s2_pan_w > 0 and step % s2_pan_every == 0:
                            total_loss += s2_pan_w * self._stage2_pan_constraint_loss(
                                ms_hr, step, cs2
                            )
                        if x_prior_w > 0:
                            total_loss += x_prior_w * self.l1_loss_ratio * F.l1_loss(
                                ms_hr, g_hr_det
                            )
                        total_loss += self._stage2_regularizer_loss(
                            ms_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan
                        )
                        total_loss.backward()
                        x_optimizer.step()
                        if s2_train_xy:
                            with torch.no_grad():
                                self.xy.clamp_(0.0, 1.0)

                        with torch.no_grad():
                            _, x_hr_det = self._render_stage2_x_pair(
                                pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                            )
                        g_optimizer.zero_grad()
                        g_lr, g_hr = self._render_stage2_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms, train_geom=False
                        )
                        g_loss = x_prior_w * self.l1_loss_ratio * F.l1_loss(g_hr, x_hr_det)
                        if alternate_g_lr_w > 0:
                            g_loss += alternate_g_lr_w * self.l1_loss_ratio * F.l1_loss(
                                self._stage2_lr_consistency_image(g_lr, g_hr, ms_h, ms_w),
                                self.gt_images,
                            )
                        if s2_hr_w > 0:
                            g_hr_loss, gt_hr_loss, _ = self._stage2_frequency_pair(
                                g_hr, gt_hr, step, cs2
                            )
                            g_loss += s2_hr_w * self.l1_loss_ratio * (
                                band_w * torch.abs(g_hr_loss - gt_hr_loss)
                            ).mean()
                        if s2_pan_w > 0 and step % s2_pan_every == 0:
                            g_loss += s2_pan_w * self._stage2_pan_constraint_loss(
                                g_hr, step, cs2
                            )
                        g_loss += self._stage2_regularizer_loss(
                            g_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan
                        )
                        g_loss.backward()
                        g_optimizer.step()
                        continue

                    self.optimizer.zero_grad()

                    if resolution_curriculum:
                        total_loss, res_tag = self._stage2_resolution_curriculum_loss(
                            step,
                            cs2,
                            pan_h,
                            pan_w,
                            ms_h,
                            ms_w,
                            s2_train_xy,
                            s2_train_rot,
                            s2_train_scale,
                            s2_hr_w,
                            s2_pan_w,
                            x_prior_w,
                        )
                        total_loss.backward()
                        self.optimizer.step()
                        if s2_train_xy:
                            with torch.no_grad():
                                self.xy.clamp_(0.0, 1.0)
                        if self.step % self.eval_steps == 0:
                            t1 = perf_counter()
                            self.worklog.info(
                                f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                                f"Loss: {total_loss.item():.5f} | Res: {res_tag}"
                            )
                            t0 = t1
                        continue

                    g_lr, g_hr = self._render_stage2_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        train_geom=s2_train_xy,
                        train_rot=s2_train_rot,
                        train_scale=s2_train_scale,
                    )
                    if use_x_field:
                        ms_lr, ms_hr = self._render_stage2_x_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                            train_geom=s2_train_xy,
                            train_rot=s2_train_rot,
                            train_scale=s2_train_scale,
                        )
                    else:
                        ms_lr, ms_hr = g_lr, g_hr
                    total_loss = s2_lr_w * self.l1_loss_ratio * F.l1_loss(
                        self._stage2_lr_consistency_image(ms_lr, ms_hr, ms_h, ms_w),
                        self.gt_images
                    )

                    if s2_hr_w > 0:
                        ms_hr_loss, gt_hr_loss, _ = self._stage2_frequency_pair(
                            ms_hr, gt_hr, step, cs2
                        )
                        total_loss += s2_hr_w * self.l1_loss_ratio * (
                            band_w * torch.abs(ms_hr_loss - gt_hr_loss)
                        ).mean()

                    if s2_pan_w > 0 and step % s2_pan_every == 0:
                        total_loss += s2_pan_w * self._stage2_pan_constraint_loss(
                            ms_hr, step, cs2
                        )

                    if use_x_field and x_prior_w > 0:
                        total_loss += x_prior_w * self.l1_loss_ratio * F.l1_loss(
                            ms_hr, g_hr.detach()
                        )

                    total_loss += self._stage2_regularizer_loss(
                        ms_hr, gt_hr, lms_hr, pan_h, pan_w, tb_pan
                    )

                    total_loss.backward()
                    self.optimizer.step()
                    if s2_train_xy:
                        with torch.no_grad():
                            self.xy.clamp_(0.0, 1.0)

                if freq_global:
                    freq_seen_steps = min(
                        freq_seen_steps + cs2, int(self._stage2_freq_total_steps)
                    )

                with torch.no_grad():
                    g_lr, g_hr = self._render_stage2_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    )
                    if use_x_field:
                        ms_lr, ms_hr = self._render_stage2_x_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        )
                    else:
                        ms_lr, ms_hr = g_lr, g_hr
                    fr = full_ref_metrics(ms_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                    self.psnr_curr = fr["PSNR"].item()
                    self.worklog.info(
                        f"Hard patch Stage2 Eval | LR_PSNR: {self.psnr_curr:.4f} | "
                        f"LR_SSIM: {fr['SSIM'].item():.4f}"
                    )
                    self._save_pansharpened(ms_hr, "stage2_hr", pan_h, pan_w)

            # -- Stage 3: Detail injection -------------------------------------

            with torch.no_grad():
                if ucl_ratio:
                    # UCL ratio branch: render feat_ratio * pan_rendered_s1 at HR
                    tb_hr_ucl = (
                        (pan_w + self.block_w - 1) // self.block_w,
                        (pan_h + self.block_h - 1) // self.block_h, 1,
                    )
                    base_ms_hr, ratio_hr_eval, product_hr_eval, pan_new_eval = (
                        self._render_ucl_current_output(
                            pan_h,
                            pan_w,
                            tb_hr_ucl,
                            product_pan_ucl,
                            pan_rendered_s1,
                            render_scale_ratio=1.0,
                        )
                    )
                elif use_x_field:
                    _, base_ms_hr = self._render_stage2_x_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    )
                else:
                    _, base_ms_hr = self._render_stage2_pair(
                        pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                    )
                if skip_stage3:
                    final_ms_hr = base_ms_hr.detach().clone()
                    # UCL ratio mode outputs HR directly — skip LR projection so
                    # eval_benchmark captures the correct HR image as "stage3".
                    project_ucl = bool(
                        getattr(self.args, "output_lr_projection_apply_ucl", False)
                    )
                    if not stage3_after_loop and (not ucl_ratio or project_ucl):
                        final_ms_hr = self._project_output_to_lrms(final_ms_hr, ms_h, ms_w)
                    final_lr = self._degrade_to_lr(final_ms_hr, ms_h, ms_w)
                    fr = full_ref_metrics(final_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                    self.psnr_curr = fr["PSNR"].item()
                    self._stage3_round_score = self._stage3_proxy_score(
                        final_ms_hr, gt_hr
                    ).item()
                    stage3_status = "deferred" if stage3_after_loop else "skipped"
                    self.worklog.info(
                        f"Stage3 {stage3_status} | Stage2-only LR_PSNR: "
                        f"{self.psnr_curr:.4f} | LR_SSIM: {fr['SSIM'].item():.4f}"
                    )
                    if not stage3_after_loop:
                        self._save_pansharpened(final_ms_hr, "stage3", pan_h, pan_w)

            if skip_stage3:
                prev_final = final_ms_hr.detach().clone()
                round_metric = float(getattr(self, "_stage3_round_score", self.psnr_curr))
                if round_metric > best_round_metric + round_min_delta:
                    best_round_metric = round_metric
                    best_round = rnd + 1
                    best_round_final = prev_final.clone()
                    stale_rounds = 0
                else:
                    stale_rounds += 1
                if n_rounds > 1:
                    self.worklog.info(
                        f">>> Round {rnd+1}/{n_rounds} done. "
                        f"Best round={best_round} ({best_round_metric:.4f})."
                    )
                if (
                    self._hard_patch_active
                    and hard_patch_max_rounds > 0
                    and self._hard_patch_start_round is not None
                    and rnd + 2 - self._hard_patch_start_round >= hard_patch_max_rounds
                ):
                    if best_round_final is not None and best_round != rnd + 1:
                        if not stage3_after_loop:
                            self._save_pansharpened(best_round_final, "stage3", pan_h, pan_w)
                        prev_final = best_round_final
                        self.psnr_curr = best_round_metric
                    self.worklog.info(
                        f"Hard patch stop after {hard_patch_max_rounds:d} round(s); "
                        f"restored round {best_round:d}."
                    )
                    break
                if (
                    round_patience > 0
                    and rnd + 1 >= min_rounds
                    and stale_rounds >= round_patience
                ):
                    if best_round_final is not None and best_round != rnd + 1:
                        if not stage3_after_loop:
                            self._save_pansharpened(best_round_final, "stage3", pan_h, pan_w)
                        prev_final = best_round_final
                        self.psnr_curr = best_round_metric
                    self.worklog.info(
                        f"Iterative early stop at round {rnd+1:d}/{n_rounds:d}; "
                        f"restored round {best_round:d}."
                    )
                    break
                continue

            s3_inject_mode = str(getattr(self.args, "stage3_injection_mode", "gain")).lower()
            s3_use_lig = s3_inject_mode in ("lig", "additive")
            s3_lig_normalize = bool(getattr(self.args, "stage3_lig_normalize", False))
            s3_aux_highpass = bool(getattr(self.args, "stage3_aux_highpass", True))
            self.worklog.info(
                f">>> Stage 3: Detail injection "
                f"(mode={s3_inject_mode}, highpass={s3_aux_highpass}, "
                f"lig_norm={s3_lig_normalize and s3_use_lig})."
            )
            aux_res_clamp = float(getattr(self.args, "stage3_aux_res_clamp", 0.02))
            res_smooth_w = float(getattr(self.args, "stage3_aux_res_smooth_weight", 0.3))
            sam_loss_w = float(getattr(self.args, "stage3_sam_loss_weight", 0.0))
            gain_consistency_w = float(
                getattr(self.args, "stage3_gain_consistency_weight", 0.0)
            )
            lambda_aux = 0.01

            self._init_aux_gaussians(self.args)
            wt = self._pan_weights()
            I_base = (base_ms_hr * wt.view(-1, 1, 1)).sum(dim=0, keepdim=True)

            _, _, res_hint, pan_target = self._build_stage3_pan_targets(I_base)

            # LIG mode: optionally build normalized residual target for direct fitting
            lig_res_target = lig_res_min = lig_res_max = None
            if s3_use_lig and s3_lig_normalize:
                with torch.no_grad():
                    lig_res_target, lig_res_min, lig_res_max = (
                        self._build_residual_target(base_ms_hr, gt_hr)
                    )
                self.worklog.info(
                    f"  LIG residual target: "
                    f"min=[{lig_res_min.squeeze().tolist()}], "
                    f"max=[{lig_res_max.squeeze().tolist()}]"
                )
                self._init_aux_pos_scale_feat_(lig_res_target, residual_hint=None)
            else:
                self._init_aux_pos_scale_feat_(self.ref_images, residual_hint=res_hint)

            with torch.no_grad():
                edge = self._pan_grad_mag(self.ref_images)
                edge = (edge / edge.max().clamp_min(1e-6)).clamp(0, 1)
                smooth_mask = 1.0 - edge

            self._set_trainable_modules(stage=3)
            t0 = perf_counter()

            for step in range(1, cs3 + 1):
                self.step = s1_done_steps + s2_steps + step
                self.optimizer.zero_grad()

                aux_I, _ = self._rasterize_features(
                    self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux,
                    pan_h, pan_w, tb_pan,
                )
                if s3_aux_highpass:
                    aux_I = self._smooth_stage3_aux(self._highpass(aux_I))
                else:
                    aux_I = self._smooth_stage3_aux(aux_I)

                aux_res, _ = self._rasterize_features(
                    self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux_res,
                    pan_h, pan_w, tb_pan,
                )
                if s3_aux_highpass:
                    aux_res = self._smooth_stage3_aux(self._highpass(aux_res))
                else:
                    aux_res = self._smooth_stage3_aux(aux_res)
                if aux_res_clamp > 0:
                    aux_res = aux_res.clamp(-aux_res_clamp, aux_res_clamp)

                # Compute final fused image
                if s3_use_lig and s3_lig_normalize and lig_res_target is not None:
                    # LIG branch: aux Gaussians render normalized residual;
                    # denorm → actual residual → add to base
                    aux_combined = self._aux_to_ms(aux_I, aux_res)
                    aux_residual = self._denorm_residual(
                        aux_combined, lig_res_min, lig_res_max
                    )
                    # Direct residual fitting loss
                    loss_res_fit = self.l1_loss_ratio * F.l1_loss(
                        aux_combined, lig_res_target
                    )
                    raw_final_ms_hr = (base_ms_hr + aux_residual).clamp(0.0, 1.0)
                    raw_final_I = (
                        raw_final_ms_hr * wt.view(-1, 1, 1)
                    ).sum(dim=0, keepdim=True)
                else:
                    loss_res_fit = None
                    aux_ms = self._aux_to_ms(aux_I, aux_res)
                    raw_final_ms_hr, raw_final_I = self._inject_stage3_detail(
                        base_ms_hr, I_base, aux_ms
                    )

                train_mix = (
                    1.0
                    if bool(getattr(self.args, "stage3_adaptive_output_mix", False))
                    else self._current_stage3_output_mix()
                )
                final_ms_hr, final_I = self._mix_stage3_output_with_ratio(
                    base_ms_hr, I_base, raw_final_ms_hr, raw_final_I, train_mix
                )

                loss_pan = self.l1_loss_ratio * F.l1_loss(self._highpass(final_I), pan_target)
                loss_ms = self.l1_loss_ratio * F.l1_loss(final_ms_hr, gt_hr)
                loss_sam = self._spectral_angle_loss(final_ms_hr, gt_hr)
                loss_gain = self._gain_consistency_loss(base_ms_hr, final_ms_hr)
                loss_aux = torch.mean(torch.abs(aux_I)) + torch.mean(torch.abs(aux_res))

                loss_tv = aux_I.new_tensor(0.0)
                if res_smooth_w > 0:
                    if aux_res.dim() == 4:
                        dh = torch.abs(aux_res[:, :, 1:, :] - aux_res[:, :, :-1, :])
                        dw = torch.abs(aux_res[:, :, :, 1:] - aux_res[:, :, :, :-1])
                    else:
                        dh = torch.abs(aux_res[:, 1:, :] - aux_res[:, :-1, :])
                        dw = torch.abs(aux_res[:, :, 1:] - aux_res[:, :, :-1])
                    loss_tv = (
                        torch.mean(dh * smooth_mask[..., :-1, :])
                        + torch.mean(dw * smooth_mask[..., :, :-1])
                    )

                total_loss = (
                    loss_pan
                    + lambda_ms * loss_ms
                    + sam_loss_w * loss_sam
                    + gain_consistency_w * loss_gain
                    + lambda_aux * loss_aux
                    + res_smooth_w * loss_tv
                )
                if loss_res_fit is not None:
                    total_loss = total_loss + loss_res_fit
                total_loss.backward()
                self.optimizer.step()

                if (not self.disable_prog_optim
                    and step % self.add_steps == 0
                    and self.aux_num_gaussians < self.total_num_gaussians):
                    self._add_aux_gaussians(self.max_add_num, final_I, self.ref_images)

                if self.step % self.eval_steps == 0:
                    with torch.no_grad():
                        t1 = perf_counter()
                        _eval_lr = self._degrade_to_lr(final_ms_hr, ms_h, ms_w)
                        fr = full_ref_metrics(_eval_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                        self.psnr_curr = fr["PSNR"].item()
                        self.worklog.info(
                            f"Step: {self.step:d} | {t1 - t0:.2f}s | "
                            f"Loss: {total_loss.item():.5f} | LR_PSNR: {self.psnr_curr:.4f}"
                        )
                        t0 = t1

            # -- Stage 3 eval --------------------------------------------------

            with torch.no_grad():
                aux_I, _ = self._rasterize_features(
                    self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux,
                    pan_h, pan_w, tb_pan,
                )
                aux_I = self._smooth_stage3_aux(
                    self._highpass(aux_I) if s3_aux_highpass else aux_I
                )
                aux_res, _ = self._rasterize_features(
                    self.xy_aux, self._get_scale_(1.0), self.rot_aux, self.feat_aux_res,
                    pan_h, pan_w, tb_pan,
                )
                aux_res = self._smooth_stage3_aux(
                    self._highpass(aux_res) if s3_aux_highpass else aux_res
                )
                if aux_res_clamp > 0:
                    aux_res = aux_res.clamp(-aux_res_clamp, aux_res_clamp)

                if s3_use_lig and s3_lig_normalize and lig_res_target is not None:
                    aux_combined = self._aux_to_ms(aux_I, aux_res)
                    aux_residual = self._denorm_residual(
                        aux_combined, lig_res_min, lig_res_max
                    )
                    raw_final_ms_hr = (base_ms_hr + aux_residual).clamp(0.0, 1.0)
                    raw_final_I = (
                        raw_final_ms_hr * wt.view(-1, 1, 1)
                    ).sum(dim=0, keepdim=True)
                else:
                    aux_ms = self._aux_to_ms(aux_I, aux_res)
                    raw_final_ms_hr, raw_final_I = self._inject_stage3_detail(
                        base_ms_hr, I_base, aux_ms
                    )
                final_ms_hr, final_I, final_lr, selected_mix = self._select_stage3_output(
                    base_ms_hr, I_base, raw_final_ms_hr, raw_final_I, ms_h, ms_w,
                    gt_hr=gt_hr,
                )
                fr = full_ref_metrics(final_lr.unsqueeze(0), self.gt_images.unsqueeze(0))
                nr = no_ref_metrics(
                    self.gt_images.unsqueeze(0), self.ref_images.unsqueeze(0),
                    final_ms_hr.unsqueeze(0), fused_lr_override=final_lr.unsqueeze(0),
                )
                self.psnr_curr = fr["PSNR"].item()
                self._stage3_round_score = self._stage3_proxy_score(final_ms_hr, gt_hr).item()
                self.worklog.info(
                    f"Stage3 Eval | mix={selected_mix:.2f} | LR_PSNR: {self.psnr_curr:.4f} | "
                    f"LR_SSIM: {fr['SSIM'].item():.4f} | LR_SAM: {fr['SAM'].item():.4f} | "
                    f"ERGAS: {fr['ERGAS'].item():.4f} | QNR: {nr['QNR'].item():.4f} | "
                    f"SCC: {nr['SCC'].item():.4f}"
                )
                self._save_pansharpened(final_ms_hr, "stage3", pan_h, pan_w)

            prev_final = final_ms_hr.detach().clone()
            round_metric = float(getattr(self, "_stage3_round_score", self.psnr_curr))
            if round_metric > best_round_metric + round_min_delta:
                best_round_metric = round_metric
                best_round = rnd + 1
                best_round_final = prev_final.clone()
                stale_rounds = 0
            else:
                stale_rounds += 1
            if n_rounds > 1:
                self.worklog.info(
                    f">>> Round {rnd+1}/{n_rounds} done. "
                    f"Best round={best_round} ({best_round_metric:.4f})."
                )
            if (
                self._hard_patch_active
                and hard_patch_max_rounds > 0
                and self._hard_patch_start_round is not None
                and rnd + 2 - self._hard_patch_start_round >= hard_patch_max_rounds
            ):
                if best_round_final is not None and best_round != rnd + 1:
                    self._save_pansharpened(best_round_final, "stage3", pan_h, pan_w)
                    prev_final = best_round_final
                    self.psnr_curr = best_round_metric
                self.worklog.info(
                    f"Hard patch stop after {hard_patch_max_rounds:d} round(s); "
                    f"restored round {best_round:d}."
                )
                break
            if (
                round_patience > 0
                and rnd + 1 >= min_rounds
                and stale_rounds >= round_patience
            ):
                if best_round_final is not None and best_round != rnd + 1:
                    self._save_pansharpened(best_round_final, "stage3", pan_h, pan_w)
                    prev_final = best_round_final
                    self.psnr_curr = best_round_metric
                self.worklog.info(
                    f"Iterative early stop at round {rnd+1:d}/{n_rounds:d}; "
                    f"restored round {best_round:d}."
                )
                break

        if stage3_after_loop:
            if prev_final is None:
                with torch.no_grad():
                    if use_x_field:
                        _, prev_final = self._render_stage2_x_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        )
                    else:
                        _, prev_final = self._render_stage2_pair(
                            pan_h, pan_w, ms_h, ms_w, tb_pan, tb_ms,
                        )
                    prev_final = prev_final.detach().clone()

            target_mode = str(
                getattr(self.args, "stage3_after_loop_target_mode", "mtf_glp")
            ).lower()
            if target_mode in ("current", "last", "iterative"):
                stage3_target = gt_hr.detach().clone()
            else:
                stage3_target = mtf_glp_target.detach().clone()

            self._run_stage3_detail_once(
                prev_final,
                stage3_target,
                pan_h,
                pan_w,
                ms_h,
                ms_w,
                tb_pan,
                s1_done_steps,
                s2_steps * max(1, best_round),
                max(1, s3_steps),
                lambda_ms,
            )

        return self.psnr_curr, self.ssim_curr
