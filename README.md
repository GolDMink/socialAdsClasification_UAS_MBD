---
title: Prediksi Pembelian Iklan Jejaring Sosial
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.31.1"
app_file: streamlit_app.py
pinned: false
---

# Prediksi Pembelian Iklan Jejaring Sosial

Aplikasi ini menggunakan machine learning untuk memprediksi apakah pelanggan akan membeli produk berdasarkan informasi demografis mereka.

## Fitur

- Prediksi individual untuk satu pelanggan
- Prediksi batch menggunakan file CSV
- Visualisasi data interaktif
- Analisis performa model
- Analisis kepentingan fitur

## Dataset

Dataset berisi informasi pelanggan termasuk:
- Usia
- Estimasi Gaji
- Jenis Kelamin
- Status Pembelian (target)

## Model

- Algoritma: Random Forest Classifier
- Preprocessing: StandardScaler
- Evaluasi: Validasi silang dengan metrik akurasi, presisi, recall, dan AUC

## Teknologi

- Frontend: Streamlit
- Machine Learning: Scikit-learn
- Data Processing: Pandas, NumPy
- Visualisasi: Plotly, Matplotlib, Seaborn

## Deployment ke Hugging Face Spaces

1. Fork repository ini ke akun GitHub Anda

2. Buat Space baru di Hugging Face:
   - Kunjungi https://huggingface.co/spaces
   - Klik "Create new Space"
   - Pilih nama untuk Space
   - Pilih "Streamlit" sebagai SDK
   - Pilih "Docker" sebagai template

3. Clone Space repository:
   ```bash
   git clone https://huggingface.co/spaces/USERNAME/SPACE_NAME
   ```

4. Salin file-file berikut ke repository Space:
   - `streamlit_app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `.gitattributes`
   - Semua file model (*.pkl)
   - Dataset (`Social_Network_Ads.csv`)

5. Push ke Hugging Face:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

6. Space akan otomatis di-build dan di-deploy

## Menjalankan Aplikasi Lokal

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:
```bash
streamlit run streamlit_app.py
```

## Menjalankan dengan Docker

1. Build image:
```bash
docker build -t social-ads-predictor .
```

2. Jalankan container:
```bash
docker run -p 7860:7860 social-ads-predictor
```

## Penggunaan

1. Pilih halaman yang ingin diakses dari sidebar
2. Untuk prediksi individual:
   - Masukkan usia
   - Masukkan estimasi gaji
   - Pilih jenis kelamin
   - Klik tombol prediksi
3. Untuk prediksi batch:
   - Unggah file CSV dengan format yang sesuai
   - Lihat hasil prediksi
   - Unduh hasil dalam format CSV

## Kontribusi

Silakan berkontribusi dengan membuat issue atau pull request!
