import json
import os
import time

import torch
from thop import profile

from arch.new_fused_arch import MIRNetFused
from arch.fused_arch import MIRNetFused as LegacyMIRNetFused


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HEIGHT = int(os.environ.get('BENCH_HEIGHT', '512'))
WIDTH = int(os.environ.get('BENCH_WIDTH', '512'))
WARMUP = int(os.environ.get('BENCH_WARMUP', '10'))
ITERATIONS = int(os.environ.get('BENCH_ITERATIONS', '30'))


def measure(name, model, rgb, nir, checkpoint):
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


def main():
    if DEVICE.type != 'cuda':
        raise RuntimeError('This benchmark requires CUDA and must run through Slurm.')

    rgb = torch.randn(1, 3, HEIGHT, WIDTH, device=DEVICE)
    nir_1 = torch.randn(1, 1, HEIGHT, WIDTH, device=DEVICE)
    nir_3 = nir_1.repeat(1, 3, 1, 1)

    results = {
        'legacy_mirnet': measure(
            'legacy_mirnet', LegacyMIRNetFused(), rgb, nir_3,
            'trained_weights/fused_model_drybean_8x.pth'),
        'spectral_lpienet': measure(
            'spectral_lpienet', MIRNetFused(), rgb, nir_1,
            'trained_weights/spectral_fused_drybean_updated_8x.pth'),
    }
    with open('benchmark_results.json', 'w') as handle:
        json.dump(results, handle, indent=2)
    print('Saved benchmark_results.json', flush=True)


if __name__ == '__main__':
    main()
