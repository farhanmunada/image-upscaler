import argparse
import glob
import os
import sys
from PIL import Image
import numpy as np

try:
    import cv2
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError as e:
    # ponytail: allows CLI help/arg testing even without CUDA/PyTorch environment locally
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Image Upscaler with Real-ESRGAN and High-Quality JPG Export")
    parser.add_argument("-i", "--input", type=str, default="inputs", help="Input directory containing images")
    parser.add_argument("-o", "--output", type=str, default="outputs", help="Output directory for upscaled images")
    parser.add_argument("-m", "--model_path", type=str, default="weights/RealESRGAN_x4plus.pth", help="Path to RealESRGAN model weight")
    parser.add_argument("-s", "--outscale", type=float, default=4.0, help="Output scale factor (default: 4.0)")
    parser.add_argument("-t", "--tile", type=int, default=512, help="Tile size to prevent CUDA OOM (0 for no tile)")
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding size")
    parser.add_argument("-q", "--quality", type=int, default=96, help="JPEG quality (1-100, default: 96)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 half-precision for speed")
    return parser.parse_args()


def get_upsampler(model_path: str, scale: float = 4.0, tile: int = 512, tile_pad: int = 10, fp16: bool = True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Download RealESRGAN_x4plus.pth first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=fp16 and (device.type == "cuda"),
        device=device,
    )
    return upsampler


def save_high_quality_jpg(img_np: np.ndarray, out_path: str, quality: int = 96):
    # Convert OpenCV BGR to RGB PIL Image
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(
        out_path,
        format="JPEG",
        quality=quality,
        subsampling=0,  # 4:4:4 chroma, keeps sharp edges and prevents file size drop
        optimize=True,
    )


def process_batch(args):
    os.makedirs(args.output, exist_ok=True)
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No image files found in '{args.input}'. Place your images there first.")
        return

    print(f"Found {len(image_paths)} images to upscale.")
    print(f"Initializing model: {args.model_path} (tile={args.tile}, scale={args.outscale}x)...")
    upsampler = get_upsampler(
        model_path=args.model_path,
        scale=args.outscale,
        tile=args.tile,
        tile_pad=args.tile_pad,
        fp16=args.fp16,
    )

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, f"{filename}_upscaled.jpg")
        print(f"[{idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)} ...", end=" ", flush=True)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print("FAILED (Unable to read image file)")
                continue

            output, _ = upsampler.enhance(img, outscale=args.outscale)
            save_high_quality_jpg(output, out_path, quality=args.quality)
            print(f"DONE -> {os.path.basename(out_path)} ({output.shape[1]}x{output.shape[0]} px)")
        except Exception as err:
            print(f"FAILED ({err})")


if __name__ == "__main__":
    cli_args = parse_args()
    process_batch(cli_args)
