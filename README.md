# Batch Image Upscaler 4x (Google Colab GPU)

Tools batch image upscaler berbasis Real-ESRGAN dengan akselerasi GPU Google Colab. Menghasilkan gambar beresolusi tinggi (4x) dengan ekspor JPEG kualitas tinggi (chroma subsampling 4:4:4) sehingga detail tajam dan ukuran file tidak terkompresi berlebih.

---

## Fitur Utama
- **Model**: `RealESRGAN_x4plus` (peningkatan ketajaman, restorasi tekstur, reduksi blur).
- **Resolusi**: Perbesaran 4x (misal: 2000x1000 px -> 8000x4000 px / 32 MP).
- **Anti-OOM (Tiling)**: Mode tile 512x512 mencegah crash memori GPU VRAM.
- **High-Quality Output**: Ekspor JPEG `quality=96`, `subsampling=0` (4:4:4), menjaga kepadatan warna dan ukuran file.
- **Batch Processing & Auto-Zip**: Sekali klik untuk memproses satu folder penuh dan mengunduh arsip `.zip`.

---

## Cara Penggunaan di Google Colab

### 1. Buka Notebook di Colab
1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih tab **Upload** lalu pilih file `upscale_colab.ipynb`.

### 2. Aktifkan GPU T4
1. Pada menu navigasi Colab, klik **Runtime** -> **Change runtime type**.
2. Pada **Hardware accelerator**, pilih **T4 GPU**.
3. Klik **Save**.

### 3. Eksekusi Cell
1. **Cell 1**: Jalankan untuk memastikan GPU T4 aktif.
2. **Cell 2 & 3**: Jalankan untuk menginstal dependensi (`basicsr`, `realesrgan`) dan mengunduh model.
3. **Cell 4**: Unggah file gambar bahan (atau tarik langsung file gambar ke folder `inputs/` di panel file Colab sebelah kiri).
4. **Cell 5**: Jalankan proses batch upscale.
5. **Cell 6**: Mengompresi folder `outputs/` menjadi `hasil_upscale.zip` dan memicu unduhan otomatis ke komputer lokal.

---

## Menjalankan Secara Mandiri via CLI
Jika dijalankan di mesin lokal atau server dengan GPU Nvidia:
```bash
python upscale_batch.py --input inputs --output outputs --tile 512 --quality 96
```

Argumen opsional:
- `--input` : Direktori gambar masukan (default: `inputs`).
- `--output` : Direktori keluaran (default: `outputs`).
- `--tile` : Ukuran tile (default: `512`, gunakan `0` jika gambar kecil).
- `--quality` : Kualitas simpan JPEG 1-100 (default: `96`).
- `--outscale` : Rasio perbesaran (default: `4.0`).
