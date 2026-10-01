import json
import os
import time

import torch
from thop import profile
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.data import DataLoader
from torchvision import transforms

from arch.new_fused_arch import MIRNetFused
from arch.fused_arch import MIRNetFused as LegacyMIRNetFused
from dataloader_test import TriplePairedDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HEIGHT = int(os.environ.get('BENCH_HEIGHT', '512'))
WIDTH = int(os.environ.get('BENCH_WIDTH', '512'))
WARMUP = int(os.environ.get('BENCH_WARMUP', '10'))
ITERATIONS = int(os.environ.get('BENCH_ITERATIONS', '30'))
TEST_RGB = os.environ.get('TEST_RGB', 'RGB-NIR-Fusion-Dataset/Drybean/Test/RGB')
TEST_NIR_UP = os.environ.get('TEST_NIR_UP', 'RGB-NIR-Fusion-Dataset/Drybean/Test/upscaled_images_8x')
TEST_NIR_GT = os.environ.get('TEST_NIR_GT', 'RGB-NIR-Fusion-Dataset/Drybean/Test/NIR')


def measure(name, model, rgb, nir, checkpoint, model_description, nir_channels):
    model = model.to(DEVICE).eval()
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state, strict=True)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(rgb, nir)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            output = model(rgb, nir)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / ITERATIONS
    peak_memory_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

    with torch.no_grad():
        flops, params = profile(model, inputs=(rgb, nir), verbose=False)

    result = {
        'model': model_description,
        'checkpoint': checkpoint,
        'rgb_channels': int(rgb.shape[1]),
        'nir_channels': nir_channels,
        'parameters': int(params),
        'flops': int(flops),
        'flops_g': flops / 1e9,
        'parameters_m': params / 1e6,
        'inference_ms': elapsed_ms,
        'inference_fps': 1000.0 / elapsed_ms,
        'peak_memory_mb': peak_memory_mb,
        'input_shape_rgb': list(rgb.shape),
        'input_shape_nir': list(nir.shape),
    }
    print(name, json.dumps(result), flush=True)
    return result


def evaluate_quality(model, checkpoint, legacy):
    model = model.to(DEVICE).eval()
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state, strict=True)

    dataset = TriplePairedDataset(
        TEST_RGB,
        TEST_NIR_UP,
        TEST_NIR_GT,
        transform=transforms.ToTensor(),
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for rgb, nir_up, nir_gt, _ in loader:
            rgb = rgb.to(DEVICE, non_blocking=True)
            nir_up = nir_up.to(DEVICE, non_blocking=True)
            nir_gt = nir_gt.to(DEVICE, non_blocking=True)

            if legacy:
                nir_up = nir_up.repeat(1, 3, 1, 1)
                nir_gt = nir_gt.repeat(1, 3, 1, 1)

            output = model(rgb, nir_up).clamp(0, 1)
            mse = torch.mean((output - nir_gt) ** 2)
            psnr_values.append((20 * torch.log10(1.0 / torch.sqrt(mse))).item())
            ssim_values.append(
                structural_similarity_index_measure(
                    output, nir_gt, data_range=1.0
                ).item()
            )

    def summarize(values):
        values = torch.tensor(values, dtype=torch.float64)
        return {
            'mean': values.mean().item(),
            'min': values.min().item(),
            'max': values.max().item(),
            'std': values.std(unbiased=False).item(),
        }

    return {
        'test_images': len(dataset),
        'psnr_db': summarize(psnr_values),
        'ssim': summarize(ssim_values),
    }


def main():
    if DEVICE.type != 'cuda':
        raise RuntimeError('This benchmark requires CUDA and must run through Slurm.')

    rgb = torch.randn(1, 3, HEIGHT, WIDTH, device=DEVICE)
    nir_1 = torch.randn(1, 1, HEIGHT, WIDTH, device=DEVICE)
    nir_3 = nir_1.repeat(1, 3, 1, 1)

    results = {
        'experiment': {
            'dataset': 'Drybean',
            'split': 'Test',
            'nir_mode': 'upscaled',
            'nir_scale': '8x',
            'test_rgb': TEST_RGB,
            'test_nir_up': TEST_NIR_UP,
            'test_nir_gt': TEST_NIR_GT,
            'input_resolution': [HEIGHT, WIDTH],
            'batch_size': 1,
            'warmup_iterations': WARMUP,
            'timed_iterations': ITERATIONS,
        },
        'legacy_mirnet': {},
        'spectral_lpienet': {},
    }
    results['legacy_mirnet'] = measure(
        'legacy_mirnet', LegacyMIRNetFused(), rgb, nir_3,
        'trained_weights/fused_model_drybean_8x.pth',
        'legacy MIRNet fusion',
        3)
    results['legacy_mirnet']['quality'] = evaluate_quality(
        LegacyMIRNetFused(),
        'trained_weights/fused_model_drybean_8x.pth',
        legacy=True,
    )
    results['spectral_lpienet'] = measure(
        'spectral_lpienet', MIRNetFused(), rgb, nir_1,
        'trained_weights/spectral_fused_drybean_updated_8x.pth',
        'SpectralLPIENet-style lightweight fusion',
        1)
    results['spectral_lpienet']['quality'] = evaluate_quality(
        MIRNetFused(),
        'trained_weights/spectral_fused_drybean_updated_8x.pth',
        legacy=False,
    )
    with open('benchmark_results.json', 'w') as handle:
        json.dump(results, handle, indent=2)
    print('Saved benchmark_results.json', flush=True)


if __name__ == '__main__':
    main()
