# ============================================================
# test_fused.py
#   - Works with your dataloader_fuse exactly as provided:
#       returns: rgb, nir_up, nir_gt  (NO filename)
#   - Saves outputs as 000001.png, 000002.png, ...
#   - Reports PSNR/SSIM: mean, min, max, std
# ============================================================

import os
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure

from arch.fused_arch import MIRNetFused
from dataloader_test import create_dataloaders


# ============================================================
# Metrics
# ============================================================
def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=False):
    """
    img1/img2: BxCxHxW in [0,1]
    """
    if gt_mean:
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)

    mse = F.mse_loss(img1, img2, reduction="mean")
    if mse.item() == 0:
        return float("inf")

    psnr = 20 * torch.log10(
        torch.tensor(max_pixel_value, device=img1.device) / torch.sqrt(mse)
    )
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=False):
    """
    torchmetrics SSIM, expects BxCxHxW
    """
    if gt_mean:
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / (mean_restored + 1e-12)), 0, 1)

    return structural_similarity_index_measure(img1, img2, data_range=max_pixel_value).item()


# ============================================================
# Validation / Test (save outputs + stats)
# ============================================================
def validate_and_save(model, dataloader, device, result_dir, force_png=True, gt_mean=False):
    model.eval()

    psnr_list = []
    ssim_list = []

    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch in dataloader:
            rgb, nir_up, nir_gt, fname = batch

            rgb = rgb.to(device, non_blocking=True)
            nir_up = nir_up.to(device, non_blocking=True)
            nir_gt = nir_gt.to(device, non_blocking=True)

            out = model(rgb, nir_up)
            out = torch.clamp(out, 0, 1)

            # fname comes as list when batch_size=1
            if isinstance(fname, (list, tuple)):
                fname = fname[0]

            save_image(out, os.path.join(result_dir, fname))

            psnr = calculate_psnr(out, nir_gt, gt_mean=gt_mean)
            ssim = calculate_ssim(out, nir_gt, gt_mean=gt_mean)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

    # ---- stats
    psnr_arr = np.array(psnr_list, dtype=np.float64)
    ssim_arr = np.array(ssim_list, dtype=np.float64)

    stats = {
        "psnr": {
            "mean": float(psnr_arr.mean()) if len(psnr_arr) else float("nan"),
            "min": float(psnr_arr.min()) if len(psnr_arr) else float("nan"),
            "max": float(psnr_arr.max()) if len(psnr_arr) else float("nan"),
            "std": float(psnr_arr.std()) if len(psnr_arr) else float("nan"),
        },
        "ssim": {
            "mean": float(ssim_arr.mean()) if len(ssim_arr) else float("nan"),
            "min": float(ssim_arr.min()) if len(ssim_arr) else float("nan"),
            "max": float(ssim_arr.max()) if len(ssim_arr) else float("nan"),
            "std": float(ssim_arr.std()) if len(ssim_arr) else float("nan"),
        }
    }

    return stats


# ============================================================
# Main
# ============================================================
def main():
    # ---- Paths
    test_rgb = 'RGB-NIR-Fusion-Dataset/Wheat/Test/RGB'
    test_nir_up = 'RGB-NIR-Fusion-Dataset/Wheat/Test/upscaled_images_32x'
    test_nir_gt = 'RGB-NIR-Fusion-Dataset/Wheat/Test/NIR'

    weights_path = 'trained_weights/fused_model_wheat_32x.pth'

    # ---- Results directory
    dataset_name = os.path.basename(test_rgb.rstrip("/"))
    result_dir = os.path.join("results_fused", dataset_name)

    # ---- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Dataloader (reuse your dataloader_fuse)
    _, test_loader = create_dataloaders(
        train_rgb=None,
        train_nir_up=None,
        train_nir_gt=None,
        test_rgb=test_rgb,
        test_nir_up=test_nir_up,
        test_nir_gt=test_nir_gt,
        crop_size=None,   # unused for test in your loader
        batch_size=1      # test loader is hardcoded batch_size=1 anyway
    )
    print("Test loader size:", len(test_loader))

    # ---- Model
    model = MIRNetFused().to(device).eval()

    # ---- Load checkpoint (supports either {"state_dict": ...} or raw state_dict)
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded weights: {weights_path}")

    # ---- Run test
    stats = validate_and_save(
        model=model,
        dataloader=test_loader,
        device=device,
        result_dir=result_dir,
        force_png=True,   # save as 000001.png, ...
        gt_mean=False
    )

    print(
        f"PSNR | mean: {stats['psnr']['mean']:.2f} | "
        f"min: {stats['psnr']['min']:.2f} | "
        f"max: {stats['psnr']['max']:.2f} | "
        f"std: {stats['psnr']['std']:.2f}"
    )
    print(
        f"SSIM | mean: {stats['ssim']['mean']:.2f} | "
        f"min: {stats['ssim']['min']:.2f} | "
        f"max: {stats['ssim']['max']:.2f} | "
        f"std: {stats['ssim']['std']:.2f}"
    )

    print("Formatted Print:")
    print(
    f"PSNR : "
    f"{stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} "
    f"({stats['psnr']['min']:.2f}–{stats['psnr']['max']:.2f})"
    )

    print(
        f"SSIM : "
        f"{stats['ssim']['mean']:.2f} ± {stats['ssim']['std']:.2f} "
        f"({stats['ssim']['min']:.2f}–{stats['ssim']['max']:.2f})"
    )
    print("Saved outputs to:", result_dir)


if __name__ == "__main__":
    main()
