# Product Requirements Document (PRD): Colab Batch Image Upscaler

## 1. Ringkasan Produk
Alat batch image upscaling berbasis notebook Google Colab dengan akselerasi GPU T4, menggunakan engine Real-ESRGAN (skala 4x) untuk menghasilkan gambar resolusi tinggi tanpa penurunan kualitas/ukuran file via ekspor JPG kualitas tinggi (chroma 4:4:4).

## 2. Masalah yang Diselesaikan
- Eksekusi lokal lambat tanpa GPU mandiri.
- Alat upscaler web publik menurunkan kualitas file dan merusak saturasi warna melalui kompresi agresif.
- Risiko crash memori VRAM saat memproses gambar resolusi masukan sedang/besar.

## 3. Sasaran & Metrik Keberhasilan
- **Peningkatan Resolusi**: Minimal perbesaran 4x (contoh: 2000x1000 px menjadi 8000x4000 px).
- **Kualitas File**: Output JPG tidak drop ukuran/detail, memakai parameter `quality=96` dan `subsampling=0`.
- **Kecepatan**: Waktu proses per gambar ~2 hingga 5 detik pada GPU T4 Colab (dengan tiling 512).
- **Kemudahan Penggunaan**: Alur notebook 3 langkah (Setup -> Upload/Taruh Gambar -> Run & Unduh Zip).

## 4. Ruang Lingkup Fitur (In-Scope)
- Notebook Colab (`.ipynb`) dan skrip pendukung Python.
- Setup otomatis dependensi dan model Real-ESRGAN (`RealESRGAN_x4plus.pth`).
- Batch processor yang memindai semua format gambar populer (`.png`, `.jpg`, `.jpeg`, `.webp`).
- Konversi output ke JPG resolusi tinggi dengan chroma 4:4:4.
- Auto-zip folder output dan tombol unduh otomatis.

## 5. Di Luar Lingkup (Out-of-Scope)
- Hosting server backend persisten (projek berjalan on-demand di Google Colab free tier).
- Training / fine-tuning model baru.

## 6. Persyaratan Teknis & Batasan
- **Runtime Target**: Google Colab GPU (T4 / V100 / A100).
- **Engine**: Real-ESRGAN CLI / Python API.
- **Tiling**: Aktif (`tile=512`, `tile_pad=10`) untuk mencegah CUDA OOM.
- **Format Output**: JPG, RGB, Quality 96, Subsampling 0.
