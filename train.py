"""
PanSplat: Pansharpening via 2D Gaussian Splatting
=================================================
Entry point for training on a single MS+PAN pair.

Usage:
    python train.py --device cuda:0 --input_path ms.tif --ref_path pan.tif
"""

import argparse
import torch

from utils.misc_utils import load_cfg


def get_log_dir(args):
    s = f"num-{args.num_gaussians:d}"
    s += f"_inv-scale-{args.init_scale:.1f}" if not args.disable_inverse_scale else f"_scale-{args.init_scale:.1f}"
    if not args.quantize:
        args.pos_bits = args.scale_bits = args.rot_bits = args.feat_bits = 32
    s += f"_bits-{args.pos_bits}-{args.scale_bits}-{args.rot_bits}-{args.feat_bits}"
    if not args.disable_topk_norm:
        s += f"_top-{args.topk}"
    s += f"_{args.init_mode[0]}-{args.init_random_ratio:.1f}"
    lcfg = f"l1-{args.l1_loss_ratio:.1f}_l2-{args.l2_loss_ratio:.1f}_ssim-{args.ssim_loss_ratio:.1f}"
    folder = f"{s}_{lcfg}"
    if not args.disable_lr_schedule:
        folder += f"_decay-{args.max_decay_times}-{args.decay_ratio:.1f}"
    if not args.disable_prog_optim:
        folder += "_prog"
    return f"{args.log_root}/{args.exp_name}/{folder}"


def main(args):
    args.log_dir = get_log_dir(args)
    from model import GaussianSplatting2D
    model = GaussianSplatting2D(args)
    model.optimize_three_stage_complete()


if __name__ == "__main__":
    torch.hub.set_dir("models/torch")
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--cfg", default="cfgs/default.yaml")
    cfg_args, remaining = cfg_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[cfg_parser])
    parser = load_cfg(cfg_path=cfg_args.cfg, parser=parser)
    arguments = parser.parse_args(remaining)
    arguments.cfg = cfg_args.cfg
    main(arguments)
