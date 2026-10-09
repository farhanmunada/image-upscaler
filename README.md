# Batch Image Upscaler 2x / 4x (Google Colab GPU)

Tools batch image upscaler berbasis Real-ESRGAN dengan akselerasi GPU Google Colab. Menghasilkan gambar beresolusi tinggi dengan ekspor JPEG kualitas tinggi (chroma subsampling 4:4:4) sehingga warna dan detail tajam serta ukuran file tidak terkompresi berlebih.

---

## Fitur Utama
- **Pilihan Skala**: 2x (tajam dan proporsional) atau 4x (detail maksimal).
- **Anti-OOM**: Tiling 512x512 mencegah crash memori GPU VRAM.
- **Ekspor Berkualitas**: JPEG Quality 96, Subsampling 0 (4:4:4).
- **Panel Kontrol Windows Theme**:
  - Deteksi file instan.
  - Progress bar per gambar secara real-time.
  - Banner notifikasi selesai 100%.
  - Tombol unduh arsip ZIP 1-klik.
  - Tombol reset/bersihkan folder batch.
- **Pure PyTorch**: Tanpa dependensi eksternal yang rentan konflik environment.

---

## Cara Penggunaan di Google Colab

### 1. Buka Notebook di Colab
Buka tautan ini:  
[Buka di Google Colab](https://colab.research.google.com/github/farhanmunada/image-upscaler/blob/main/upscale_colab.ipynb)

### 2. Aktifkan GPU T4
1. Pada menu Colab, klik **Runtime** -> **Change runtime type**.
2. Pada **Hardware accelerator**, pilih **T4 GPU** -> klik **Save**.

### 3. Eksekusi Cell Secara Berurutan (Total 4 Cell)
1. **Cell 1**: Verifikasi status GPU (NVIDIA Tesla T4).
2. **Cell 2**: Inisialisasi Model & Engine Real-ESRGAN (Unduh bobot model dan muat arsitektur ke memori).
3. **Cell 3**: Unggah Gambar Bahan (Klik tombol `Choose Files` bawaan Colab, pilih gambar PNG/JPG).
4. **Cell 4**: Panel Kontrol:
   - Pilih Skala: `2x` atau `4x`.
   - Atur Kualitas JPG (default 96).
   - Klik **Mulai Proses Upscale**.
   - Pantau progress bar sampai status `PROSES SELESAI 100%`.
   - Klik **Unduh Arsip ZIP**.
   - Klik **Bersihkan Folder Batch** jika ingin memproses kumpulan gambar baru berikutnya.

---

## Eksekusi via CLI
```bash
# Upscale 2x
python upscale_batch.py --input inputs --output outputs --scale 2.0 --quality 96

# Upscale 4x
python upscale_batch.py --input inputs --output outputs --scale 4.0 --quality 96

# Reset folder
python upscale_batch.py --clean
```
