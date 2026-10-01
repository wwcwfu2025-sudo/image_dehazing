"""Run the released fixed ODCR-Mamba model on one RGB image."""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from odcr_mamba import StandaloneODCRMamba


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()


def rgb8(output):
    """RGB float tensor to 8-bit pixels: clip, multiply by 255, round-to-even."""
    pixels = output[0].detach().float().cpu().permute(1, 2, 0).numpy()
    if not np.isfinite(pixels).all():
        raise ValueError('Non-finite output; refusing to write an image.')
    return np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='One input image; decoded as RGB without resizing.')
    parser.add_argument('--output', type=Path, required=True, help='New output PNG; existing files are never overwritten.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='CPU by default; CUDA requires a compatible PyTorch installation.')
    parser.add_argument('--threads', type=int, default=2, help='Number of CPU threads (positive integer).')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not args.input.is_file():
        parser.error('Input image does not exist')
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        parser.error('Output must be a new file, different from the input')
    if args.output.suffix.lower() != '.png':
        parser.error('Output extension must be .png')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu or install a compatible PyTorch build')

    identity = json.loads((ROOT / 'provenance/model_identity_2026100109.json').read_text(encoding='utf-8'))
    configuration = json.loads((ROOT / 'configs/fixed_model_2026100109.json').read_text(encoding='utf-8'))['model']
    weights = ROOT / identity['weights_file']
    if sha256(weights) != identity['weights_sha256']:
        raise RuntimeError('Released weight file hash mismatch; refusing to deserialize it.')
    load_options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_options['weights_only'] = True
    # Older PyTorch versions deserialize only the hash-verified weight file shipped in this release.
    state = torch.load(str(weights), **load_options)
    model = StandaloneODCRMamba(**configuration)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    with Image.open(str(args.input)) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Use a single-frame image.')
        image = image.convert('RGB')
        if min(image.size) < 16:
            raise ValueError('Both image dimensions must be at least 16 pixels; no automatic resizing is applied.')
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().unsqueeze(0).to(args.device)
    with torch.no_grad():
        result = model(tensor)['output']
    output = Image.fromarray(rgb8(result), mode='RGB')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        output.save(stream, format='PNG')
    print(json.dumps({'output': str(args.output), 'width': output.width, 'height': output.height,
                      'weights_sha256': identity['weights_sha256'], 'device': args.device,
                      'precision': 'float32', 'resized': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()

# Generated: 2026-10-01 09:00 +08:00
