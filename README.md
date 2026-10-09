# Batch Image Upscaler 2x / 4x (Google Colab GPU)

Tools batch image upscaler berbasis Real-ESRGAN dengan akselerasi GPU Google Colab. Dilengkapi **Tombol GUI Interaktif**, pilihan skala perbesaran (2x atau 4x), dan ekspor JPEG kualitas tinggi (subsampling 4:4:4) sehingga warna dan detail tajam serta ukuran file tidak terkompresi berlebih.

---

## Fitur Utama
- **Tombol GUI Interaktif di Notebook**:
  - Tombol **Upload Gambar Bahan**
  - Toggle Pilihan Skala **2x** atau **4x**
  - Slider Kualitas JPEG (85 - 100, default 96)
  - Tombol **🚀 Mulai Upscale**
  - Tombol **📦 Unduh Hasil (ZIP)**
  - Tombol **🧹 Bersihkan Folder (Reset)** agar file lama tidak tercampur ke batch baru.
- **Model**: `RealESRGAN_x4plus` (restorasi tekstur, reduksi blur, penajaman detail).
- **Anti-OOM (Tiling)**: Mode tile 512x512 mencegah crash memori GPU VRAM.
- **High-Quality Output**: Ekspor JPEG `quality=96`, `subsampling=0` (4:4:4), menjaga kepadatan warna dan ukuran file.
- **Engine Pure PyTorch**: Bawaan Colab, tanpa library luar yang rentan konflik build.

---

## Cara Penggunaan di Google Colab

### 1. Buka Notebook di Colab
Buka link berikut:  
[Buka di Google Colab](https://colab.research.google.com/github/farhanmunada/image-upscaler/blob/main/upscale_colab.ipynb)

### 2. Aktifkan GPU T4
1. Pada menu navigasi Colab, klik **Runtime** -> **Change runtime type**.
2. Pada **Hardware accelerator**, pilih **T4 GPU** -> klik **Save**.

### 3. Eksekusi Cell
1. **Cell 1**: Cek GPU T4 aktif.
2. **Cell 2**: Unduh bobot model `RealESRGAN_x4plus.pth`.
3. **Cell 3**: Buat runner script.
4. **Cell 4**: Muncul **Dashboard GUI**:
   - Pilih Skala: `2x` atau `4x`.
   - Klik `📁 Upload Gambar Bahan` (atau seret gambar ke folder `inputs/`).
   - Klik `🚀 Mulai Upscale`.
   - Klik `📦 Unduh Hasil (ZIP)`.
   - Klik `🧹 Bersihkan Folder (Reset)` jika ingin memproses batch baru berikutnya.

---

## Eksekusi via CLI (Bila di Mesin Mandiri)
```bash
# Upscale 2x
python upscale_batch.py --input inputs --output outputs --scale 2.0 --quality 96

# Upscale 4x
python upscale_batch.py --input inputs --output outputs --scale 4.0 --quality 96

# Reset folder
python upscale_batch.py --clean
```
