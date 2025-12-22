import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Library Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Library Deep Learning (TensorFlow/Keras) untuk LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Daun Singkong (SVM, BNN, LSTM)", 
    layout="wide", 
    page_icon="🌿"
)

# --- CSS CUSTOM ---
st.markdown(r"""
    <style>
    .main-header {
        background-color: #2e7d32;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-top: 5px solid #2e7d32;
        margin-bottom: 10px;
    }
    .result-box {
        padding: 15px;
        background-color: #e8f5e9;
        border-radius: 8px;
        border-left: 5px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🌿 Klasifikasi Daun Singkong Jepang</h1>
    <p>Komparasi Model: <b>SVM vs Backpropagation (BNN) vs LSTM</b></p>
</div>
""", unsafe_allow_html=True)

# --- DATA CONTOH ---
DEFAULT_DATA = """Panjang_Daun;Lebar_Daun;Warna_Hijau;Spesies
15.2;5.3;120;Manihot_esculenta
18.5;6.1;130;Manihot_esculenta
12.8;4.7;110;Manihot_palmata
14.3;5.0;115;Manihot_palmata
20.1;7.2;140;Manihot_glaziovii
22.3;7.8;150;Manihot_glaziovii
16.5;5.8;125;Manihot_esculenta
13.7;4.9;112;Manihot_palmata
19.8;7.0;145;Manihot_glaziovii
11.9;4.3;105;Manihot_palmata
13.2;4.5;108;Manihot_palmata
17.8;6.3;128;Manihot_esculenta
21.5;7.5;148;Manihot_glaziovii
14.8;5.2;118;Manihot_esculenta
16.0;5.5;122;Manihot_esculenta
13.0;4.8;109;Manihot_palmata
21.0;7.4;142;Manihot_glaziovii"""

# --- INISIALISASI SESSION STATE ---
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}

# --- FUNGSI TRAINING ---
def train_all_models(df):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. Preprocessing
    status_text.text("Menyiapkan data...")
    le = LabelEncoder()
    df['Spesies_Enc'] = le.fit_transform(df['Spesies'])
    mapping = dict(zip(range(len(le.classes_)), le.classes_))
    
    X = df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']].values
    y = df['Spesies_Enc'].values
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling (PENTING untuk semua model)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    progress_bar.progress(20)
    
    # 2. Train SVM
    status_text.text("Melatih SVM (Support Vector Machine)...")
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    progress_bar.progress(50)
    
    # 3. Train BNN (MLP)
    status_text.text("Melatih BNN (Backpropagation)...")
    mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    
    progress_bar.progress(70)
    
    # 4. Train LSTM
    status_text.text("Melatih LSTM (Deep Learning)...")
    
    # Reshape data untuk LSTM [Samples, Timesteps, Features]
    # Kita anggap 1 baris data sebagai 1 timestep
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # One-hot encoding untuk target LSTM
    y_train_ohe = to_categorical(y_train)
    
    # Build Model LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(1, 3), activation='relu')) # Layer LSTM
    lstm_model.add(Dense(32, activation='relu'))                    # Hidden Layer biasa
    lstm_model.add(Dense(len(mapping), activation='softmax'))       # Output Layer
    
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training LSTM (verbose=0 supaya tidak menuhin log)
    lstm_model.fit(X_train_lstm, y_train_ohe, epochs=100, batch_size=4, verbose=0)
    
    # Prediksi LSTM
    y_pred_prob_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm = np.argmax(y_pred_prob_lstm, axis=1)
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    
    progress_bar.progress(100)
    status_text.text("Selesai!")
    
    # Simpan ke Session State
    st.session_state.models = {
        'svm': svm_model,
        'mlp': mlp_model,
        'lstm': lstm_model,
        'scaler': scaler,
        'mapping': mapping,
        'le': le
    }
    
    st.session_state.data = {
        'y_test': y_test,
        'y_pred_svm': y_pred_svm,
        'y_pred_mlp': y_pred_mlp,
        'y_pred_lstm': y_pred_lstm,
        'acc_svm': acc_svm,
        'acc_mlp': acc_mlp,
        'acc_lstm': acc_lstm,
        'df': df
    }
    
    st.session_state.trained = True

# --- SIDEBAR (UPLOAD) ---
st.sidebar.title("📁 Panel Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.sidebar.success(f"Data dimuat: {len(df)} baris")
else:
    st.sidebar.info("Menggunakan Data Contoh")
    df = pd.read_csv(io.StringIO(DEFAULT_DATA), sep=';')

# --- TOMBOL TRAIN ---
if st.sidebar.button("🚀 Latih Semua Model"):
    train_all_models(df)

# --- MAIN DISPLAY ---
if not st.session_state.trained:
    st.info("Silakan klik tombol **'Latih Semua Model'** di sidebar untuk memulai.")
    st.write("### Preview Data:")
    st.dataframe(df.head())
else:
    # Buat Tab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Perbandingan Akurasi", "🔍 Prediksi Baru", "🧮 Hitungan Manual (SVM)", "📈 Visualisasi"])
    
    # --- TAB 1: AKURASI ---
    with tab1:
        st.subheader("Hasil Evaluasi Model")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>SVM</h3>
                <h1 style="color:#2e7d32">{st.session_state.data['acc_svm']:.1%}</h1>
                <p>Support Vector Machine</p>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>BNN</h3>
                <h1 style="color:#1565c0">{st.session_state.data['acc_mlp']:.1%}</h1>
                <p>Backpropagation Neural Network</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>LSTM</h3>
                <h1 style="color:#c62828">{st.session_state.data['acc_lstm']:.1%}</h1>
                <p>Long Short-Term Memory</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("---")
        st.write("### Detail Laporan Klasifikasi")
        
        model_choice = st.selectbox("Pilih Model untuk melihat detail:", ["SVM", "BNN (MLP)", "LSTM"])
        mapping = st.session_state.models['mapping']
        y_test = st.session_state.data['y_test']
        
        if model_choice == "SVM":
            y_pred = st.session_state.data['y_pred_svm']
        elif model_choice == "BNN (MLP)":
            y_pred = st.session_state.data['y_pred_mlp']
        else:
            y_pred = st.session_state.data['y_pred_lstm']
            
        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.text(classification_report(y_test, y_pred, target_names=list(mapping.values())))
        
        with col_rep2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=mapping.values(), yticklabels=mapping.values())
            plt.title(f"Confusion Matrix - {model_choice}")
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            st.pyplot(fig)

    # --- TAB 2: PREDIKSI ---
    with tab2:
        st.subheader("Uji Coba Prediksi")
        
        c1, c2, c3 = st.columns(3)
        p_in = c1.number_input("Panjang Daun", 10.0, 30.0, 15.0)
        l_in = c2.number_input("Lebar Daun", 1.0, 15.0, 5.0)
        w_in = c3.number_input("Warna Hijau (RGB)", 0, 255, 120)
        
        if st.button("🔍 Prediksi Sekarang"):
            # Prepare data
            input_raw = np.array([[p_in, l_in, w_in]])
            scaler = st.session_state.models['scaler']
            mapping = st.session_state.models['mapping']
            
            # Scale data
            input_scaled = scaler.transform(input_raw)
            
            # 1. Prediksi SVM
            svm_pred_idx = st.session_state.models['svm'].predict(input_scaled)[0]
            svm_label = mapping[svm_pred_idx]
            
            # 2. Prediksi BNN
            mlp_pred_idx = st.session_state.models['mlp'].predict(input_scaled)[0]
            mlp_label = mapping[mlp_pred_idx]
            
            # 3. Prediksi LSTM (Butuh Reshape ke 3D)
            input_lstm = input_scaled.reshape((1, 1, 3))
            lstm_prob = st.session_state.models['lstm'].predict(input_lstm)
            lstm_pred_idx = np.argmax(lstm_prob)
            lstm_label = mapping[lstm_pred_idx]
            
            # Tampilkan Hasil
            st.markdown(f"""
            <div class="result-box">
                <h3>Hasil Keputusan Model:</h3>
                <ul>
                    <li><b>SVM:</b> {svm_label}</li>
                    <li><b>BNN:</b> {mlp_label}</li>
                    <li><b>LSTM:</b> {lstm_label}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 3: HITUNGAN MANUAL (SVM) ---
    with tab3:
        st.subheader("Simulasi Perhitungan Manual SVM")
        st.info("Fitur ini mensimulasikan perhitungan matematika di balik layar untuk model SVM (Linear). LSTM & BNN terlalu kompleks untuk dihitung manual di sini.")
        
        if st.button("🧮 Hitung Manual (Ambil data dari Tab Prediksi)"):
            svm_model = st.session_state.models['svm']
            scaler = st.session_state.models['scaler']
            mapping = st.session_state.models['mapping']
            
            # 1. Scaling
            input_raw = np.array([[p_in, l_in, w_in]])
            input_scaled = scaler.transform(input_raw)[0]
            
            st.write("#### 1. Normalisasi Data (Z-Score)")
            st.write(f"Input Asli: {input_raw.flatten()}")
            st.write(f"Mean (μ): {scaler.mean_}")
            st.write(f"Scale (σ): {scaler.scale_}")
            st.code(f"Rumus: z = (x - μ) / σ\nHasil Scaled: {input_scaled}")
            
            # 2. Dot Product
            st.write("#### 2. Fungsi Keputusan (Decision Function)")
            st.write("Rumus: `f(x) = (Weights • Input) + Bias`")
            
            weights = svm_model.coef_
            biases = svm_model.intercept_
            
            results = []
            for i, class_name in mapping.items():
                w = weights[i]
                b = biases[i]
                score = np.dot(w, input_scaled) + b
                results.append((class_name, score))
                
                with st.expander(f"Perhitungan Kelas: {class_name}"):
                    st.write(f"**Bobot (W):** {w}")
                    st.write(f"**Bias (b):** {b}")
                    st.write(f"**Score:** ({w[0]:.2f}*{input_scaled[0]:.2f}) + ({w[1]:.2f}*{input_scaled[1]:.2f}) + ({w[2]:.2f}*{input_scaled[2]:.2f}) + {b:.2f}")
                    st.write(f"**Total = {score:.4f}**")
            
            # 3. Keputusan
            st.write("#### 3. Hasil Akhir")
            df_res = pd.DataFrame(results, columns=["Kelas", "Skor"])
            st.table(df_res)
            best_class = df_res.loc[df_res['Skor'].idxmax()]['Kelas']
            st.success(f"Skor tertinggi ada pada kelas **{best_class}**, maka data diklasifikasikan sebagai **{best_class}**.")

    # --- TAB 4: VISUALISASI ---
    with tab4:
        st.subheader("Sebaran Data Training")
        df_vis = st.session_state.data['df']
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_vis, x='Panjang_Daun', y='Lebar_Daun', hue='Spesies', style='Spesies', s=100)
        plt.title("Panjang vs Lebar Daun")
        st.pyplot(fig)
        
        st.write("Boxplot Distribusi Fitur")
        fig2, ax2 = plt.subplots(1, 3, figsize=(15,5))
        sns.boxplot(data=df_vis, x='Spesies', y='Panjang_Daun', ax=ax2[0])
        sns.boxplot(data=df_vis, x='Spesies', y='Lebar_Daun', ax=ax2[1])
        sns.boxplot(data=df_vis, x='Spesies', y='Warna_Hijau', ax=ax2[2])
        st.pyplot(fig2)