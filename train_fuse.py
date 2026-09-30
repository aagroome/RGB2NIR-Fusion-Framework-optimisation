import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchmetrics.functional import structural_similarity_index_measure
import os

from arch.new_fused_arch import MIRNetFused
from losses import CombinedLoss
from dataloader_fuse import create_dataloaders
import numpy as np
import logging
import sys


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=False):
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=False):
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for rgb, nir_up, nir_gt in dataloader:
            rgb = rgb.to(device)
            nir_up = nir_up.to(device)
            nir_gt = nir_gt.to(device)

            output = model(rgb, nir_up)

            psnr = calculate_psnr(output, nir_gt)
            ssim = calculate_ssim(output, nir_gt)
            total_psnr += psnr
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim


def main():
    # === Logging setup ===
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler("training_log.txt", mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Ensures clean logging setup
    )

    # === Your 3 input paths ===
    train_rgb = 'RGB-NIR-Fusion-Dataset/Drybean/Train/RGB'
    train_nir_up = 'RGB-NIR-Fusion-Dataset/Drybean/Train/upscaled_images_8x'
    train_nir_gt = 'RGB-NIR-Fusion-Dataset/Drybean/Train/NIR'
    test_rgb = 'RGB-NIR-Fusion-Dataset/Drybean/Test/RGB'
    test_nir_up = 'RGB-NIR-Fusion-Dataset/Drybean/Test/upscaled_images_8x'
    test_nir_gt = 'RGB-NIR-Fusion-Dataset/Drybean/Test/NIR'

    learning_rate = float(os.environ.get('LEARNING_RATE', '2e-4'))
    num_epochs = int(os.environ.get('NUM_EPOCHS', '500'))
    batch_size = int(os.environ.get('BATCH_SIZE', '4'))
    crop_size = int(os.environ.get('CROP_SIZE', '256'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'LR: {learning_rate}; Epochs: {num_epochs}; Device: {device}')

    train_loader, test_loader = create_dataloaders(
        train_rgb, train_nir_up, train_nir_gt,
        test_rgb, test_nir_up, test_nir_gt,
        crop_size=crop_size, batch_size=batch_size
    )
    logging.info(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    model = MIRNetFused().to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    os.makedirs('trained_weights', exist_ok=True)
    logging.info('Training started.')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for rgb, nir_up, nir_gt in train_loader:
            rgb = rgb.to(device)
            nir_up = nir_up.to(device)
            nir_gt = nir_gt.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(rgb, nir_up)
                loss = criterion(output, nir_gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            model_path = 'trained_weights/spectral_fused_drybean_updated_8x.pth'
            torch.save(model.state_dict(), model_path)
            logging.info(f'[SAVED] Model saved to {model_path} with PSNR: {best_psnr:.4f}')

            # Flush logs to make sure message appears in the file immediately
            for handler in logging.getLogger().handlers:
                handler.flush()


if __name__ == '__main__':
    main()