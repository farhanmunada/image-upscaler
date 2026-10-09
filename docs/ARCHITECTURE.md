# Arsitektur: Colab Batch Image Upscaler

## 1. Diagram Komponen & Aliran Data

```text
[Input Gambar (.png, .jpg, .webp)]
               │
               ▼
       /content/inputs/
               │
               ▼
  [upscale_batch.py (PyTorch + CUDA)]
   - Inisialisasi RRDBNet (RealESRGAN_x4plus)
   - Tiling (512x512, pad 10) -> Anti-OOM
   - Upscale 4x
   - Export via PIL (JPG, Quality 96, Subsampling 0)
               │
               ▼
      /content/outputs/
               │
               ▼
     [Auto-Zip: hasil_upscale.zip]
               │
               ▼
    [Download ke Komputer Lokal]
```

## 2. Struktur File
Projek terdiri dari file-file berikut:
1. `upscale_batch.py` : Skrip mandiri Python untuk batch processing folder input ke folder output dengan konfigurasi PIL lossless/high-bitrate.
2. `upscale_colab.ipynb` : Google Colab Notebook interaktif siap pakai yang mengatur runtime, instalasi dependensi, patch kompatibilitas, eksekusi batch, dan unduh hasil.
3. `README.md` : Panduan operasional langkah-demi-langkah bagi pengguna (cara buka di Colab, aktifkan T4 GPU, upload, dan eksekusi).

## 3. Rincian Teknis & Dependensi

### A. Penanganan Kompatibilitas Runtime Colab
- Dependensi: `torch`, `torchvision`, `realesrgan`, `basicsr`, `Pillow`.
- Issue: PyTorch/torchvision versi baru menghapus modul `torchvision.transforms.functional_tensor`.
- Penanganan di Notebook:
  ```bash
  sed -i 's/functional_tensor/functional/g' /usr/local/lib/python3.*/dist-packages/basicsr/data/degradations.py
  ```

### B. Konfigurasi Inference Engine
- Bobot Model: `RealESRGAN_x4plus.pth` (unduh langsung dari GitHub releases resmi ke `/content/weights/`).
- Skala: 4x.
- Tile Size: `512` (mencegah VRAM Out-of-Memory pada gambar 2000x1000 ke atas).
- Half Precision (FP16): Diaktifkan secara default pada GPU Nvidia untuk kecepatan 2x lipat dan hemat VRAM.

### C. Pipeline Ekspor Gambar
- Pembacaan: OpenCV / PIL.
- Pemrosesan: RGB mode.
- Penyimpanan:
  ```python
  output_img.convert('RGB').save(
      out_path,
      format='JPEG',
      quality=96,
      subsampling=0
  )
  ```
  Menjamin detail tekstur AI tersimpan utuh dan ukuran file tidak terdegradasi.

## 4. Rencana Tugas & Implementasi
- [x] Dokumen Riset & PRD
- [ ] Task 1: Buat skrip eksekusi batch `upscale_batch.py`.
- [ ] Task 2: Buat Google Colab Notebook `upscale_colab.ipynb`.
- [ ] Task 3: Buat dokumentasi pemakaian `README.md`.
- [ ] Task 4: Verifikasi sintaks dan pengujian skrip Python lokal.
