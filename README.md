# 🌿 Klasifikasi Spesies Daun Singkong Jepang

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Proyek Machine Learning** untuk membandingkan performa algoritma **K-Nearest Neighbors (KNN)** dan **Naive Bayes** dalam mengklasifikasikan jenis daun singkong Jepang berdasarkan ciri morfologi.

---

## 📖 Tentang Proyek

Aplikasi ini dibangun menggunakan **Streamlit** untuk memberikan antarmuka interaktif dalam menganalisis data daun singkong. Sistem ini mempelajari pola dari dataset untuk memprediksi spesies daun: **Kasetsart**, **Thailand**, atau **Gajah**.

### Mengapa Proyek Ini Dibuat?
* Untuk memahami perbedaan kinerja algoritma **KNN** (berbasis jarak) vs **Naive Bayes** (berbasis probabilitas).
* Membantu identifikasi varietas singkong secara digital berdasarkan ukuran dan tekstur daun.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
| :--- | :--- |
| 🎛️ **Kontrol Parameter** | Atur nilai **K** (untuk KNN) dan rasio **Data Train/Test** secara *real-time* melalui Sidebar. |
| 📊 **Visualisasi Data** | Menampilkan sebaran data (Scatter Plot) dan korelasi antar fitur secara otomatis. |
| 📈 **Komparasi Akurasi** | Membandingkan akurasi, Confusion Matrix, dan Classification Report kedua model secara berdampingan. |
| 🤖 **Prediksi Langsung** | Masukkan data baru (Panjang, Lebar, Warna) dan lihat hasil prediksi seketika. |
| 📝 **Kesimpulan Otomatis** | Sistem akan otomatis memberitahu algoritma mana yang lebih unggul berdasarkan parameter yang dipilih. |

---

## 📂 Penjelasan Dataset

Dataset yang digunakan (`data_daun_singkong_bersih.csv`) memiliki atribut berikut:

| Nama Kolom | Tipe Data | Keterangan |
| :--- | :--- | :--- |
| `Panjang_Daun` | Float (cm) | Panjang helai daun utama. |
| `Lebar_Daun` | Float (cm) | Lebar helai daun di titik terlebar. |
| `Warna_Hijau` | Integer | Tingkat kehijauan daun (Skala RGB/Index warna). |
| `Tekstur` | String | Tekstur permukaan daun (**Halus** atau **Kasar**). |
| `Spesies` | String | **Target Label**: *Kasetsart, Thailand, Gajah*. |

---

## 🛠️ Instalasi & Cara Menjalankan

Ikuti langkah ini untuk menjalankan aplikasi di komputer lokal kamu.

### 1. Clone Repository
```bash
git clone https://github.com/yeszi/klasifikasi-daun-singkong 

cd klasifikasi-daun-singkong

//pilih salah satu

//untuk windows
python -m venv venv
.\venv\Scripts\activate

//untuk mac atau linux
python3 -m venv venv
source venv/bin/activate

//install library
pip install streamlit pandas numpy scikit-learn matplotlib seaborn

//jalankan code
streamlit run singkong.py
