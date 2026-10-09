import argparse
import glob
import math
import os
import shutil
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# --- PyTorch RRDBNet Architecture ---

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# --- Core Helper Functions ---

def load_model(model_path: str = "weights/RealESRGAN_x4plus.pth", device: torch.device = None, fp16: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model.half() if (fp16 and device.type == "cuda") else model


def process_tiles(img_tensor: torch.Tensor, model: nn.Module, tile: int = 1024, tile_pad: int = 10, scale: int = 4):
    batch, channel, height, width = img_tensor.shape
    if tile == 0 or (height <= tile and width <= tile):
        with torch.inference_mode():
            return model(img_tensor)

    out_tensor = torch.zeros((batch, channel, height * scale, width * scale), dtype=img_tensor.dtype, device=img_tensor.device)
    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)

    for y in range(tiles_y):
        for x in range(tiles_x):
            in_x = x * tile
            in_y = y * tile
            in_x_end = min(in_x + tile, width)
            in_y_end = min(in_y + tile, height)

            in_x_pad = max(in_x - tile_pad, 0)
            in_x_end_pad = min(in_x_end + tile_pad, width)
            in_y_pad = max(in_y - tile_pad, 0)
            in_y_end_pad = min(in_y_end + tile_pad, height)

            tile_in = img_tensor[:, :, in_y_pad:in_y_end_pad, in_x_pad:in_x_end_pad]
            with torch.inference_mode():
                tile_out = model(tile_in)

            out_x = in_x * scale
            out_x_end = in_x_end * scale
            out_y = in_y * scale
            out_y_end = in_y_end * scale

            t_out_x = (in_x - in_x_pad) * scale
            t_out_x_end = t_out_x + (in_x_end - in_x) * scale
            t_out_y = (in_y - in_y_pad) * scale
            t_out_y_end = t_out_y + (in_y_end - in_y) * scale

            out_tensor[:, :, out_y:out_y_end, out_x:out_x_end] = tile_out[:, :, t_out_y:t_out_y_end, t_out_x:t_out_x_end]

    return out_tensor


def upscale_image(pil_img: Image.Image, model: nn.Module, device: torch.device, tile: int = 1024, fp16: bool = True, outscale: float = 4.0):
    orig_w, orig_h = pil_img.size
    img = pil_img.convert("RGB")
    np_img = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(device)
    if fp16 and device.type == "cuda":
        tensor = tensor.half()

    out_tensor = process_tiles(tensor, model, tile=tile, tile_pad=10, scale=4)
    out_tensor = out_tensor.squeeze(0).float().clamp(0.0, 1.0)
    out_np = (out_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    res_img = Image.fromarray(out_np)

    if outscale != 4.0:
        target_w = int(round(orig_w * outscale))
        target_h = int(round(orig_h * outscale))
        res_img = res_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return res_img


def get_image_files(input_dir: str = "inputs"):
    os.makedirs(input_dir, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    if os.path.exists("/content"):
        try:
            for fname in os.listdir("/content"):
                fpath = os.path.join("/content", fname)
                if os.path.isfile(fpath) and fname.lower().endswith(valid_exts):
                    dest = os.path.join(input_dir, fname)
                    if not os.path.exists(dest):
                        shutil.move(fpath, dest)
        except Exception:
            pass

    files = []
    if os.path.exists(input_dir):
        for fname in os.listdir(input_dir):
            fpath = os.path.join(input_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(valid_exts):
                files.append(fpath)

    return sorted(files)


def clean_folders(input_dir="inputs", output_dir="outputs"):
    for folder in [input_dir, output_dir]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                p = os.path.join(folder, f)
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
    if os.path.exists("hasil_upscale.zip"):
        os.remove("hasil_upscale.zip")


# --- CLI Entrypoint ---

def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN Batch Upscaler")
    parser.add_argument("-i", "--input", type=str, default="inputs", help="Folder input")
    parser.add_argument("-o", "--output", type=str, default="outputs", help="Folder output")
    parser.add_argument("-m", "--model_path", type=str, default="weights/RealESRGAN_x4plus.pth", help="Model path")
    parser.add_argument("-s", "--scale", type=float, default=4.0, choices=[2.0, 4.0])
    parser.add_argument("-t", "--tile", type=int, default=1024, help="Tile size (1024 Turbo GPU / 512 Standar)")
    parser.add_argument("-q", "--quality", type=int, default=96)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean:
        clean_folders(args.input, args.output)
        print("Folder inputs, outputs, dan zip berhasil dibersihkan.")
        return

    os.makedirs(args.output, exist_ok=True)
    files = get_image_files(args.input)
    if not files:
        print(f"ERROR: Tidak ditemukan file gambar di '{args.input}'.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Memuat model {args.model_path} (Tile: {args.tile})...")
    model = load_model(args.model_path, device, fp16=args.fp16)

    print(f"Ditemukan {len(files)} gambar | Skala: {args.scale}x | Kualitas JPG: {args.quality}")
    for i, path in enumerate(files, 1):
        filename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output, f"{filename}_upscaled_{int(args.scale)}x.jpg")
        print(f"[{i}/{len(files)}] {os.path.basename(path)} ...", end=" ", flush=True)
        try:
            with Image.open(path) as src_img:
                out_img = upscale_image(src_img, model, device, tile=args.tile, fp16=args.fp16, outscale=args.scale)
                out_img.save(out_path, format="JPEG", quality=args.quality, subsampling=0, optimize=True)
                w, h = out_img.size
                print(f"SELESAI ({w}x{h} px)")
        except Exception as e:
            print(f"GAGAL: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
