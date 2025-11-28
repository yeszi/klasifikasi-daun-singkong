# ============================================================
# APLIKASI STREAMLIT: PERBANDINGAN KNN VS NAIVE BAYES
# DALAM KLASIFIKASI SPESIES DAUN SINGKONG JEPANG
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------
st.set_page_config(page_title="KNN vs Naive Bayes - Daun Singkong Jepang", layout="wide")
sns.set(style="whitegrid")

# ------------------------------------------------------------
# 🟢 HEADER
# ------------------------------------------------------------
st.title("🌿 Perbandingan Kinerja KNN dan Naive Bayes")
st.subheader("Klasifikasi Spesies Daun Singkong Jepang")

st.write("""
Aplikasi ini membandingkan dua algoritma Machine Learning: **K-Nearest Neighbors (KNN)** dan **Naive Bayes**, 
untuk mengklasifikasikan **spesies daun singkong Jepang** berdasarkan ciri morfologi seperti:
- Panjang daun
- Lebar daun
- Warna hijau daun
- Tekstur permukaan
""")

st.markdown("---")

# ------------------------------------------------------------
# 📦 1. DATA COLLECTION
# ------------------------------------------------------------
st.header("📦 1. Data Collection")

st.write("""
Data yang digunakan adalah **data sekunder** hasil pengamatan daun singkong Jepang.  
Dataset berisi fitur: `Panjang_Daun`, `Lebar_Daun`, `Warna_Hijau`, `Tekstur`, dan `Spesies`.
""")

uploaded_file = st.file_uploader("📂 Unggah file CSV daun singkong (gunakan pemisah ';')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.success("✅ Data berhasil dimuat!")

    st.subheader("📊 Informasi Dataset")
    st.write(f"Jumlah data: **{df.shape[0]} baris**, {df.shape[1]} kolom")
    st.write("Nama kolom:", list(df.columns))
    jumlah_tampil = st.slider("Tampilkan berapa baris data?", 5, len(df), 10)
    st.dataframe(df.head(jumlah_tampil))

    st.markdown("### Statistik Deskriptif")
    st.dataframe(df.describe(include='all'))

    st.markdown("---")

    # ------------------------------------------------------------
    # 🧹 2. DATA CLEANING
    # ------------------------------------------------------------
    st.header("🧹 2. Data Cleaning")

    st.write("""
    Proses data cleaning dilakukan untuk memastikan kualitas data:
    - Menghapus **data duplikat**
    - Menghapus **data kosong (missing values)**
    - Mengubah kolom kategorikal (`Tekstur`) menjadi numerik (Halus=1, Kasar=0)
    - Mengubah kolom `Spesies` menjadi label numerik menggunakan **LabelEncoder**
    """)

    df = df.drop_duplicates().dropna()
    if df['Tekstur'].dtype == object:
        df['Tekstur'] = df['Tekstur'].map({'Halus': 1, 'Kasar': 0})

    encoder = LabelEncoder()
    df['Spesies'] = encoder.fit_transform(df['Spesies'])
    st.success("✅ Data berhasil dibersihkan dan dikonversi!")

    st.dataframe(df.head(10))

    st.markdown("---")

    # ------------------------------------------------------------
    # ⚙️ 3. NORMALISASI
    # ------------------------------------------------------------
    st.header("⚙️ 3. Normalisasi Data")

    st.write("""
    Normalisasi digunakan agar setiap fitur numerik memiliki skala yang sama sehingga tidak mendominasi model.  
    Rumus **Min-Max Normalization**:
    """)
    st.latex(r"X' = \frac{X - X_{min}}{X_{max} - X_{min}}")

    scaler = MinMaxScaler()
    df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']] = scaler.fit_transform(
        df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']]
    )

    st.dataframe(df.head(10))

    st.markdown("---")

    # ------------------------------------------------------------
    # 📈 4. ANALISIS DATA & VISUALISASI
    # ------------------------------------------------------------
    st.header("📈 4. Analisis dan Visualisasi Data")

    st.write("### 🔹 Korelasi antar fitur")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("### 🔹 Hubungan antar fitur dan spesies")
    fig = sns.pairplot(df, hue='Spesies', diag_kind='kde', palette='viridis')
    st.pyplot(fig)

    st.markdown("---")

    # ------------------------------------------------------------
    # 🧠 5. PEMBENTUKAN MODEL
    # ------------------------------------------------------------
    st.header("🧠 5. Pemodelan dan Rumus Algoritma")

    st.write("### 🔹 K-Nearest Neighbors (KNN)")
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
    st.write("Model memprediksiknn = KNeighborsClassifier(n_neighbors=k_value) kelas mayoritas dari **k tetangga terdekat**.")

    st.write("### 🔹 Naive Bayes")
    st.latex(r"P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}")

    st.markdown("---")

    # ------------------------------------------------------------
    # 🧩 6. PEMBENTUKAN MODEL DAN PREDIKSI
    # ------------------------------------------------------------
    X = df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau', 'Tekstur']]
    y = df['Spesies']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k_value = st.slider("🔧 Pilih jumlah tetangga (K) untuk KNN:", 1, 15, 5)
    knn = KNeighborsClassifier(n_neighbors=k_value)
    nb = GaussianNB()

    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_nb = nb.predict(X_test)

    # ------------------------------------------------------------
    # 📊 7. EVALUASI KINERJA MODEL
    # ------------------------------------------------------------
    st.header("📊 7. Evaluasi Kinerja Model")

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi KNN", f"{acc_knn*100:.2f}%")
    with col2:
        st.metric("Akurasi Naive Bayes", f"{acc_nb*100:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=['KNN', 'Naive Bayes'], y=[acc_knn, acc_nb], palette='viridis', ax=ax)
    ax.set_title('📊 Perbandingan Akurasi Model')
    for i, v in enumerate([acc_knn, acc_nb]):
        ax.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', fontsize=12)
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'KNN (K={k_value})')
    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Naive Bayes')
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")
    st.write("**KNN**")
    st.text(classification_report(y_test, y_pred_knn, target_names=encoder.classes_))
    st.write("**Naive Bayes**")
    st.text(classification_report(y_test, y_pred_nb, target_names=encoder.classes_))

    st.markdown("---")

    # ------------------------------------------------------------
    # 🌱 8. PREDIKSI MANUAL
    # ------------------------------------------------------------
    st.header("🌱 8. Prediksi Spesies Daun Baru")
    st.write("Masukkan nilai ciri-ciri daun di bawah ini untuk memprediksi spesiesnya:")

    col1, col2 = st.columns(2)
    with col1:
        panjang = st.number_input("🌿 Panjang Daun", min_value=0.0, max_value=50.0, value=10.0)
        lebar = st.number_input("🌿 Lebar Daun", min_value=0.0, max_value=50.0, value=5.0)
    with col2:
        warna = st.number_input("🎨 Warna Hijau Daun (0-100)", min_value=0.0, max_value=100.0, value=50.0)
        tekstur = st.selectbox("🪶 Tekstur Permukaan", ["Halus", "Kasar"])

    if st.button("🔍 Prediksi Sekarang"):
        new_data = pd.DataFrame({
            'Panjang_Daun': [panjang],
            'Lebar_Daun': [lebar],
            'Warna_Hijau': [warna],
            'Tekstur': [1 if tekstur == "Halus" else 0]
        })
        new_data[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']] = scaler.transform(
            new_data[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']]
        )
        pred_knn = knn.predict(new_data)[0]
        pred_nb = nb.predict(new_data)[0]
        prob_knn = knn.predict_proba(new_data).max()
        prob_nb = nb.predict_proba(new_data).max()

        st.success(f"🌿 **KNN memprediksi:** {encoder.inverse_transform([pred_knn])[0]} (Prob: {prob_knn:.2f})")
        st.info(f"🧠 **Naive Bayes memprediksi:** {encoder.inverse_transform([pred_nb])[0]} (Prob: {prob_nb:.2f})")

    st.markdown("---")

    # ------------------------------------------------------------
    # 📚 9. KESIMPULAN OTOMATIS
    # ------------------------------------------------------------
    st.header("📚 9. Kesimpulan Otomatis")
    if acc_knn > acc_nb:
        st.success(f"✅ **KNN memiliki performa lebih baik** ({acc_knn*100:.2f}% > {acc_nb*100:.2f}%) dalam mengklasifikasi spesies daun singkong Jepang.")
    elif acc_knn < acc_nb:
        st.success(f"✅ **Naive Bayes lebih unggul** ({acc_nb*100:.2f}% > {acc_knn*100:.2f}%) untuk dataset ini.")
    else:
        st.info("⚖️ Keduanya memiliki performa yang setara pada dataset ini.")

    st.write("""
    - **KNN** cocok untuk data dengan pola yang jelas dan jarak antar kelas yang terpisah.
    - **Naive Bayes** bekerja baik pada data yang memiliki distribusi probabilistik yang stabil.
    - Untuk data dengan fitur sederhana seperti daun singkong ini, keduanya cukup efektif.
    """)

else:
    st.warning("⬆️ Silakan unggah file CSV terlebih dahulu untuk memulai analisis.")

# ============================================================
# END OF APP
# ============================================================
