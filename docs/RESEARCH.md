# Riset: Image Upscaler Batch di Google Colab

## 1. Latar Belakang & Masalah
- **Masalah GPU Lokal**: Upscayl lokal lambat tanpa GPU diskrit.
- **Masalah Web Opensource**: Resolusi piksel naik tapi ukuran file drop drastis karena kompresi JPEG agresif (`quality` < 80) dan subsampling warna 4:2:0.
- **Kebutuhan**: Upscale batch file PNG ke JPG beresolusi tinggi (4x, tembus >4 MP hingga 32 MP), detail tajam, kualitas warna utuh (4:4:4), eksekusi cepat menggunakan GPU T4 Google Colab.

## 2. Pilihan Model & Arsitektur
- **Model**: `RealESRGAN_x4plus` (arsitektur RRDBNet).
- **Keunggulan**: Model yang sama digunakan oleh Upscayl untuk foto/generik, restorasi blur dan tekstur sangat baik.
- **Tiling**: Wajib `tile=512` dengan `tile_pad=10`. Mencegah CUDA Out of Memory (OOM) pada gambar resolusi masukan 2000x1000 px yang membesar jadi 8000x4000 px.

## 3. Kompatibilitas Runtime Colab (PyTorch & BasicSR)
- Colab modern menggunakan Python 3.10+ dan PyTorch 2.x / Torchvision baru.
- Library `basicsr` memiliki isu impor bawaan pada torchvision modern (`cannot import name 'functional_tensor' from 'torchvision.transforms'`).
- **Solusi**: 
  - Patch otomatis file `degradations.py` pada basicsr sebelum eksekusi atau instalasi via wheel kompatibel.
  - Skrip standalone mandiri berbasis PyTorch / ONNX Runtime / Real-ESRGAN repo resmi dengan patch 1-baris.

## 4. Preservasi Kualitas & Ukuran File JPG
- Gambar PNG diubah ke JPG sering kehilangan ukuran karena kompresi lossy.
- **Solusi PIL**:
  - `img.convert('RGB').save(out_path, 'JPEG', quality=96, subsampling=0)`
  - `subsampling=0` memaksa chroma 4:4:4 (tanpa downsampling warna).
  - Menghasilkan detail piksel tajam, minim artifak kompresi, dan ukuran file padat proporsional terhadap detail baru.

## 5. Strategi I/O & Runtime
- Hindari membaca/menulis langsung secara kontinu ke Google Drive mount (`/content/drive`) karena latency FUSE tinggi.
- Gambar ditaruh di folder lokal Colab (`/content/inputs/`).
- Eksekusi batch langsung ke `/content/outputs/` di disk NVMe Colab.
- Paket hasil akhir dikompresi ke `/content/hasil_upscale.zip` untuk unduhan cepat 1-klik.
