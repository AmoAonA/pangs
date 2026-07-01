"""
Simple wrapper for fused_ssim using pytorch's native SSIM
"""
import torch
import torch.nn.functional as F

def fused_ssim(pred, target, window_size=11):
    """Compute SSIM between two images. Equivalent to the CUDA version.
    pred/target: [B, C, H, W] or [B, 1, H, W]
    """
    if pred.shape[1] == 1:
        pred = pred.expand(-1, 3, -1, -1)
        target = target.expand(-1, 3, -1, -1)

    # Convert to grayscale for SSIM
    pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([0.153, 0.222, 0.254, 0.222, 0.153])
    window = gauss / gauss.sum()
    window = window.view(1, 1, 1, 5).expand(1, 1, 5, 5)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(pred_gray, window, padding=2)
    mu2 = F.conv2d(target_gray, window, padding=2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_gray ** 2, window, padding=2) - mu1_sq
    sigma2_sq = F.conv2d(target_gray ** 2, window, padding=2) - mu2_sq
    sigma12 = F.conv2d(pred_gray * target_gray, window, padding=2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()