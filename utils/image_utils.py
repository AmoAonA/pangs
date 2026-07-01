import os

import cv2
try:
    import flip_evaluator
except ImportError:
    flip_evaluator = None
import imageio.v3 as iio
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from numpy.linalg import norm
from PIL import Image
from scipy.ndimage import sobel

import scipy.io as sio
import h5py

FONT_PATH = "assets/fonts/linux_libertine/LinLibertine_R.ttf"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    FONT_PROP = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams['font.family'] = FONT_PROP
    plt.rcParams['text.usetex'] = True
else:
    FONT_PROP = 'sans-serif'
    plt.rcParams['text.usetex'] = False
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['legend.title_fontsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

ALLOWED_IMAGE_FILE_FORMATS = [".jpeg", ".jpg", ".png", ".tiff", ".tif", ".exr",".mat"]

PLOT_DPI = 72.0
GAUSSIAN_ZOOM = 5
GAUSSIAN_COLOR = "#80ed99"


def get_psnr(image1, image2, max_value=1.0):
    mse = torch.mean((image1-image2)**2)
    if mse.item() <= 1e-7:
        return float('inf')
    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr


def get_grid(h, w, x_lim=np.asarray([0, 1]), y_lim=np.asarray([0, 1])):
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1)[:-1] + 0.5 / w
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1)[:-1] + 0.5 / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


def compute_image_gradients(image):
    gy, gx = [], []
    for image_channel in image:
        gy.append(sobel(image_channel, 0))
        gx.append(sobel(image_channel, 1))
    gy = norm(np.stack(gy, axis=0), ord=2, axis=0).astype(np.float32)
    gx = norm(np.stack(gx, axis=0), ord=2, axis=0).astype(np.float32)
    return gy, gx


def load_images(load_path, downsample_ratio=None, gamma=None, mat_key=None):
    """
    Load target images or textures from a directory or a single file.
    Supports JPEG, PNG, TIFF, EXR, and MAT (v7/v7.3) formats.
    """
    image_list = []
    image_path_list = []
    image_fname_list = []
    num_channels_list = []
    bit_depth_list = []

    # Collect valid image/mat files
    if os.path.isfile(load_path) and os.path.splitext(load_path)[1].lower() in ALLOWED_IMAGE_FILE_FORMATS:
        image_path_list.append(load_path)
    elif os.path.isdir(load_path):
        for file in sorted(os.listdir(load_path), key=str.lower):
            if os.path.splitext(file)[1].lower() in ALLOWED_IMAGE_FILE_FORMATS:
                image_path_list.append(os.path.join(load_path, file))

    if len(image_path_list) == 0:
        raise FileNotFoundError(f"No supported image or MAT file found at '{load_path}'")

    for image_path in image_path_list:
        ext = os.path.splitext(image_path)[1].lower()
        image_fname_list.append(os.path.splitext(os.path.basename(image_path))[0])

        if ext == '.mat':
            # Try scipy first (for old MAT files)
            try:
                mat_data = sio.loadmat(image_path)
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if len(keys) == 0:
                    raise ValueError(f"No valid variable found in {image_path}")
                key = mat_key or keys[0]
                image = mat_data[key]
                # import pdb;pdb.set_trace()
            except NotImplementedError:
                # For HDF5-based .mat (v7.3)
                with h5py.File(image_path, 'r') as f:
                    keys = list(f.keys())
                    if len(keys) == 0:
                        raise ValueError(f"No valid dataset found in {image_path}")
                    key = mat_key or keys[0]
                    image = np.array(f[key]).T  # MATLAB存储是列主序
            # Normalize
            image = np.array(image, dtype=np.float32)
            # import pdb; pdb.set_trace()
            if image.ndim == 2:
                image = np.expand_dims(image, axis=0)
            elif image.ndim == 3 and image.shape[2] <= 4:
                image = np.transpose(image, (2, 0, 1))
            else:
                raise ValueError(f"Unsupported MAT image shape: {image.shape}")
            num_channels = image.shape[0]
            num_channels_list.append(num_channels)
            bit_depth_list.append(32)
        else:
            # Use imageio for TIFF files as it's more robust, otherwise use OpenCV
            if ext in ['.tif', '.tiff']:
                image = iio.imread(image_path)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                raise IOError(f"Failed to load image at {image_path}")

            if image.ndim not in [2, 3]:
                raise ValueError(f"Invalid image file ({os.path.basename(image_path)}) with shape {image.shape}")
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)
            elif ext not in ['.tif', '.tiff']: # OpenCV loads as BGR, needs conversion
                image = image[..., ::-1]

            # For non-TIF images, assume 4th channel is alpha and drop it. For TIF, keep all channels.
            if image.shape[-1] == 4 and ext in ['.tif', '.tiff']:
                image = image[..., :4]

            num_channels = image.shape[-1]
            num_channels_list.append(num_channels)
            # Normalize based on bit depth
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
                bit_depth_list.append(8)
            elif image.dtype == np.uint16:
                if image.max()<2048:
                    image = image.astype(np.float32) / 2047.0
                else:
                    image = image.astype(np.float32) / 65535.0  # 或者 / 255.0
                bit_depth_list.append(16)
            elif image.dtype == np.float32:
                bit_depth_list.append(32)
            else:
                raise ValueError(f"Unsupported image dtype: {image.dtype}")

        # Resize if requested
        if downsample_ratio is not None:
            height, width = image.shape[:2]
            if downsample_ratio > 0:
                new_w = round(width / downsample_ratio)
                new_h = round(height / downsample_ratio)
            else:
                factor = abs(downsample_ratio)
                new_w = round(width * factor)
                new_h = round(height * factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        if gamma is not None:
            image = np.power(image, gamma)
        image = image.transpose(2, 0, 1)
        image_list.append(image)
    return np.concatenate(image_list, axis=0), num_channels_list, image_fname_list, bit_depth_list


def load_mat_images(load_path, keys=('HRMS', 'LRMS', 'PAN'), downsample_ratio=None, gamma=None):
    """
    Load data from a .mat file.
    The file should contain one or more of the keys: ['HRMS', 'LRMS', 'PAN'].
    You can control which to load via the `keys` argument.
    """
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"File not found: {load_path}")
    if not load_path.lower().endswith('.mat'):
        raise ValueError(f"Expected a .mat file but got: {load_path}")

    data = sio.loadmat(load_path)
    loaded_data = {}

    for key in keys:
        if key not in data:
            print(f"[Warning] Key '{key}' not found in {load_path}, skipped.")
            continue

        img = data[key]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # 灰度图扩维
        elif img.ndim > 3:
            raise ValueError(f"Invalid shape for key '{key}': {img.shape}")

        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > 1.0:  # 归一化到 [0,1]
            img = img / 255.0 if img_max > 255 else img / img_max

        # Downsample or Upsample if specified
        if downsample_ratio is not None:
            import cv2
            h, w = img.shape[:2]
            if downsample_ratio > 0:
                new_w, new_h = round(w / downsample_ratio), round(h / downsample_ratio)
            else:
                factor = abs(downsample_ratio)
                new_w, new_h = round(w * factor), round(h * factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Gamma correction
        if gamma is not None:
            img = np.power(img, gamma)

        img = img.transpose(2, 0, 1)  # (C, H, W)
        loaded_data[key] = img

    return loaded_data

def to_output_format(image, image_format, gamma):
    if image_format not in ALLOWED_IMAGE_FILE_FORMATS:
        raise ValueError(f"Invalid image format: {image_format}")
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Invalid image format: shape = {image.shape}")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        image = image.transpose(1, 2, 0)
        if image.shape[2] not in [1, 3]:
            raise ValueError(f"Invalid image format: shape = {image.shape}")
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    image = image.astype(np.float32)
    if gamma is not None:
        image = np.power(image, 1.0/gamma)
    if image_format in [".jpeg", ".jpg"]:
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image_format in [".png"]:
        image = np.clip(image, 0.0, 1.0)
        image = (65535.0 * image).astype(np.uint16)
    return image


def save_image(image, save_path, gamma=None, zoom=None):
    """
    统一保存函数：
    1. 自动识别多波段卫星图像(>3通道)，进行百分比拉伸处理。
    2. 普通图像进行标准 Gamma 校正。
    3. 统一支持 Zoom 和 OpenCV 保存。
    """
    # 1. 统一转为 Numpy CPU 格式
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # 2. 核心分支处理
    # 判断是否为多波段卫星图像 (假设输入是 C, H, W 格式)
    # 条件：是3维数组 且 第一维(通道)大于3
    if image.ndim == 3 and image.shape[0] > 3:
        # === 分支 A: 卫星图像特殊处理 ===

        # A.1 波段选择: 取前3个通道并重组为 RGB
        # QuickBird 标准顺序通常是 B, G, R, NIR。我们要 R, G, B -> 索引 [2, 1, 0]
        # image 形状: (C, H, W) -> 取出后转置为 (H, W, C)
        rgb_img = image[[2, 1, 0], :, :].transpose(1, 2, 0)

        # A.2 动态拉伸 (Percentile Stretching) 转 uint8
        img_uint8 = np.zeros_like(rgb_img, dtype=np.uint8)

        for i in range(3):
            band = rgb_img[:, :, i]
            # 过滤背景 0 值
            valid_mask = band > 0

            if valid_mask.sum() > 0:
                # 计算 2% 和 98% 分位数 (去除极亮极暗)
                vmin = np.percentile(band[valid_mask], 2)
                vmax = np.percentile(band[valid_mask], 98)

                # 防止除零
                div = vmax - vmin
                if div < 1e-6: div = 1e-6

                # 拉伸截断并归一化到 0-1
                band_stretched = np.clip((band - vmin) / div, 0, 1)
            else:
                band_stretched = band  # 保持原样(全黑)

            # 转为 0-255
            img_uint8[:, :, i] = (band_stretched * 255).astype(np.uint8)

        # 处理完成，image 变成标准的 HWC uint8 RGB 图
        image = img_uint8

    else:
        # === 分支 B: 普通图像处理 ===
        # 如果是 (C, H, W) 但只有 1 或 3 通道，先转置为 (H, W, C)
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)

        # 获取后缀名
        image_format = os.path.splitext(save_path)[1].lower()
        # 调用你原有的格式转换 (float -> uint8 + gamma)
        # 假设 to_output_format 定义在外部，如果没有，参考下方补充代码
        image = to_output_format(image, image_format, gamma)

    # === 3. 统一后处理 (Zoom + Save) ===

    # 缩放处理
    if zoom is not None and zoom > 0.0 and zoom != 1.0:
        height, width = image.shape[:2]
        new_size = (int(round(width * zoom)), int(round(height * zoom)))
        # INTER_NEAREST 适合这种像素级比较，INTER_LINEAR 适合普通观感
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

    # OpenCV 颜色空间转换 (RGB -> BGR)
    # 我们上面处理的(包括卫星和普通)都是 RGB 顺序
    # cv2.imwrite 默认期望 BGR，所以必须翻转颜色通道
    if image.ndim == 3 and image.shape[2] == 3:
        image = image[..., ::-1]

    # 目录检查与保存
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, image)


def separate_image_channels(images, input_channels):
    # if len(images) != sum(input_channels):
    #     raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_list = []
    curr_channel = 0
    for num_channels in input_channels:
        image_list.append(images[curr_channel:curr_channel+num_channels])
        curr_channel += num_channels
    return image_list


def visualize_gaussian_footprint(filepath, xy, scale, rot, feat, img_h, img_w, input_channels, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the footprint of Gaussians using colored elliptical disks.
    """
    if feat.shape[1] != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {feat.shape[1]:d} vs {sum(input_channels):d}")
    xy = xy.detach().cpu().clone().numpy()
    y, x = xy[:, 1] * img_h, xy[:, 0] * img_w
    scale = GAUSSIAN_ZOOM * scale.detach().cpu().clone().numpy()
    rot = rot.detach().cpu().clone().numpy()
    if gamma is not None:
        feat = torch.pow(feat, 1.0/gamma)
    feat = np.clip(feat.detach().cpu().clone().numpy(), 0.0, 1.0)

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        curr_feat = feat[:, curr_channel:curr_channel+num_channels]
        if curr_feat.shape[1] == 1:
            curr_feat = np.repeat(curr_feat, 3, axis=1)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=img_w/PLOT_DPI, h=img_h/PLOT_DPI, forward=False)
        ax = plt.gca()
        for gid in range(len(xy)):
            ellipse = Ellipse(xy=(x[gid], y[gid]), width=scale[gid, 0], height=scale[gid, 1],
                              angle=rot[gid, 0]*180/np.pi, alpha=alpha, ec=None, fc=curr_feat[gid], lw=None)
            ax.add_patch(ellipse)
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def visualize_gaussian_position(filepath, images, xy, input_channels, color="#7bf1a8", size=700, every_n=10, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the position of Gaussians using dots.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    xy = xy.detach().cpu().clone().numpy()[::every_n]
    x, y = xy[:, 0] * image_width, xy[:, 1] * image_height

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_output_format(image, f".{save_image_format}", gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=255)
        plt.scatter(x, y, s=size, c=color, marker='o', alpha=alpha)
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def visualize_added_gaussians(filepath, images, old_xy, new_xy, input_channels, size=500, every_n=5, alpha=0.8, gamma=None, save_image_format="jpg"):
    """
    Visualize the positions of added Gaussians during error-guided progressive optimization.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    old_xy = old_xy.detach().cpu().clone().numpy()[::every_n]
    new_xy = new_xy.detach().cpu().clone().numpy()[::every_n]
    old_x, old_y = old_xy[:, 0] * image_width, old_xy[:, 1] * image_height
    new_x, new_y = new_xy[:, 0] * image_width, new_xy[:, 1] * image_height

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_output_format(image, f".{save_image_format}", gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=255)
        plt.scatter(old_x, old_y, s=size, c="#ef476f", marker='o', alpha=alpha)  # red
        plt.scatter(new_x, new_y, s=size, c="#06d6a0", marker='o', alpha=alpha)  # green
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.{save_image_format}", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def save_error_maps(path, images, gt_images, channels, gamma, save_image_format="jpg"):
    images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/gamma)
    gt_images = torch.pow(gt_images, 1.0/gamma)
    images_sep = separate_image_channels(images, channels)
    gt_images_sep = separate_image_channels(gt_images, channels)
    for idx, (image, gt_image) in enumerate(zip(images_sep, gt_images_sep), 1):
        gt_image, image = gt_image.detach().cpu().clone().numpy(), image.detach().cpu().clone().numpy()
        if gt_image.shape[0] == 1:
            gt_image = np.repeat(gt_image, 3, axis=0)
            image = np.repeat(image, 3, axis=0)
        gt_image, image = gt_image.transpose(1, 2, 0), image.transpose(1, 2, 0)
        suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
        flip_error_map, _, _ = flip_evaluator.evaluate(reference=gt_image, test=image, dynamicRangeString="LDR", inputsRGB=True, applyMagma=True)
        save_image(flip_error_map, f"{path}{suffix}.{save_image_format}")
