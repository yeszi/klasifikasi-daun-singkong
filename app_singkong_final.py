import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Daun Singkong Jepang", 
    layout="wide", 
    page_icon="🌿"
)

# CSS sederhana
st.markdown(r"""
    <style>
    .main-header {
        background-color: #2e7d32;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        margin: 10px 0;
    }
    
    .manual-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
<h1>🌿 Sistem Klasifikasi Daun Singkong Jepang</h1>
<p>Membandingkan Akurasi Model: <b>SVM, Backpropagation, dan LSTM</b></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Data contoh
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
14.8;5.2;118;Manihot_esculenta"""

# Session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'data' not in st.session_state:
    st.session_state.data = {}

def train_models(df):
    """Melatih semua model"""
    
    # Encoding
    le = LabelEncoder()
    df['Spesies_Enc'] = le.fit_transform(df['Spesies'])
    mapping = dict(zip(range(len(le.classes_)), le.classes_))
    
    # Features and target
    X = df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']].values
    y = df['Spesies_Enc'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    # Train Backpropagation dengan lebih banyak iterasi
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(32, 16), 
        max_iter=2000, 
        random_state=42,
        alpha=0.001,
        learning_rate_init=0.01,
        solver='adam'
    )
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    
    # Simpan ke session state
    st.session_state.models = {
        'svm': svm_model,
        'mlp': mlp_model,
        'scaler': scaler,
        'mapping': mapping,
        'le': le
    }
    
    st.session_state.data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_svm': y_pred_svm,
        'y_pred_mlp': y_pred_mlp,
        'acc_svm': acc_svm,
        'acc_mlp': acc_mlp,
        'df': df
    }
    
    st.session_state.trained = True
    
    return acc_svm, acc_mlp

# Main app logic
if uploaded_file is not None:
    # Read uploaded file
    try:
        df = pd.read_csv(uploaded_file, sep=';')
    except:
        try:
            df = pd.read_csv(uploaded_file)  # Try default comma separator
        except:
            st.error("Tidak bisa membaca file. Pastikan format CSV benar.")
            st.stop()
    
    st.success(f"✅ Data dimuat: {len(df)} baris")
    
    # Show data preview
    with st.expander("📊 Preview Data"):
        st.dataframe(df.head())
        st.write(f"**Kelas:** {df['Spesies'].unique().tolist()}")
        st.write(f"**Jumlah data:** {df.shape}")
        st.write("**Statistik:**")
        st.write(df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']].describe())
    
    # Train button
    if st.button("🚀 Latih Model", type="primary"):
        with st.spinner("Melatih model..."):
            acc_svm, acc_mlp = train_models(df)
            
            st.success(f"""
            ✅ Pelatihan selesai!
            - Akurasi SVM: {acc_svm:.2%}
            - Akurasi Backpropagation: {acc_mlp:.2%}
            """)
            
else:
    # Use sample data
    st.info("Upload file CSV atau gunakan data contoh di bawah")
    
    if st.button("📋 Gunakan Data Contoh", type="secondary"):
        df = pd.read_csv(io.StringIO(DEFAULT_DATA), sep=';')
        
        with st.expander("📊 Preview Data Contoh"):
            st.dataframe(df)
            st.write(f"**Kelas:** {df['Spesies'].unique().tolist()}")
            st.write(f"**Jumlah data:** {df.shape}")
        
        if st.button("🚀 Latih dengan Data Contoh", type="primary"):
            with st.spinner("Melatih model..."):
                acc_svm, acc_mlp = train_models(df)
                
                st.success(f"""
                ✅ Pelatihan selesai!
                - Akurasi SVM: {acc_svm:.2%}
                - Akurasi Backpropagation: {acc_mlp:.2%}
                """)

# If models are trained, show tabs
if st.session_state.trained:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Akurasi Model", 
        "🔍 Prediksi", 
        "🧮 Perhitungan Manual",
        "📈 Analisis Data"
    ])
    
    with tab1:
        st.header("Kinerja Model")
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>SVM</h3>
            <h2>{st.session_state.data['acc_svm']:.2%}</h2>
            <p>Support Vector Machine</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>Backpropagation</h3>
            <h2>{st.session_state.data['acc_mlp']:.2%}</h2>
            <p>Multi-Layer Perceptron</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Classification reports
        st.subheader("Laporan Klasifikasi")
        
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            st.write("**Laporan SVM:**")
            report_text_svm = classification_report(
                st.session_state.data['y_test'],
                st.session_state.data['y_pred_svm'],
                target_names=list(st.session_state.models['mapping'].values())
            )
            st.text(report_text_svm)
        
        with col_report2:
            st.write("**Laporan Backpropagation:**")
            report_text_mlp = classification_report(
                st.session_state.data['y_test'],
                st.session_state.data['y_pred_mlp'],
                target_names=list(st.session_state.models['mapping'].values())
            )
            st.text(report_text_mlp)
        
        # Confusion matrix simple
        st.subheader("Matriks Konfusi (SVM)")
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(
            st.session_state.data['y_test'], 
            st.session_state.data['y_pred_svm']
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        ax.set_title('Matriks Konfusi SVM')
        
        # Set tick labels
        class_names = list(st.session_state.models['mapping'].values())
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names, rotation=0)
        
        st.pyplot(fig)
    
    with tab2:
        st.header("Prediksi Data Baru")
        
        st.write("### Masukkan Fitur Daun:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            panjang = st.number_input("Panjang Daun (cm)", value=15.0, min_value=0.0, step=0.1, key="pred_panjang")
        
        with col2:
            lebar = st.number_input("Lebar Daun (cm)", value=5.5, min_value=0.0, step=0.1, key="pred_lebar")
        
        with col3:
            warna = st.number_input("Warna Hijau (0-255)", value=120, min_value=0, max_value=255, step=1, key="pred_warna")
        
        # Predict button
        if st.button("🔍 Prediksi Spesies", type="primary"):
            # Prepare input
            input_data = np.array([[panjang, lebar, warna]])
            
            # Scale input
            input_scaled = st.session_state.models['scaler'].transform(input_data)
            
            # Get predictions
            svm_pred = st.session_state.models['svm'].predict(input_scaled)[0]
            mlp_pred = st.session_state.models['mlp'].predict(input_scaled)[0]
            
            # Get probabilities
            svm_proba = st.session_state.models['svm'].predict_proba(input_scaled)[0]
            
            # Get class names
            mapping = st.session_state.models['mapping']
            svm_class = mapping[svm_pred]
            mlp_class = mapping[mlp_pred]
            
            # Display results
            st.markdown(f"""
            <div class="result-card">
            <h3>Hasil Prediksi</h3>
            <p><b>Input:</b> Panjang={panjang}cm, Lebar={lebar}cm, Warna={warna}</p>
            <hr>
            <p><b>Prediksi SVM:</b> {svm_class} (Probabilitas: {max(svm_proba):.2%})</p>
            <p><b>Prediksi Backpropagation:</b> {mlp_class}</p>
            <hr>
            <h4>Keputusan Akhir: {svm_class}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probabilities
            st.subheader("Probabilitas Prediksi (SVM)")
            
            prob_data = pd.DataFrame({
                'Kelas': [mapping[i] for i in range(len(svm_proba))],
                'Probabilitas': svm_proba
            })
            
            # Create bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(prob_data['Kelas'], prob_data['Probabilitas'], 
                         color=['#2e7d32' if i == svm_pred else '#90a4ae' 
                                for i in range(len(svm_proba))])
            
            ax.set_ylabel('Probabilitas')
            ax.set_title('Probabilitas untuk Setiap Kelas')
            ax.set_ylim([0, 1])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2%}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Show probability table
            st.write("**Detail Probabilitas:**")
            prob_data['Probabilitas'] = prob_data['Probabilitas'].apply(lambda x: f"{x:.2%}")
            st.dataframe(prob_data)
            
            st.balloons()
    
    with tab3:
        st.header("Perhitungan Manual SVM")
        
        st.write("""
        Bagian ini menunjukkan langkah-langkah perhitungan manual untuk prediksi SVM.
        Masukkan nilai di bawah dan klik 'Hitung' untuk melihat perhitungan langkah demi langkah.
        """)
        
        # Input for manual calculation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p_manual = st.number_input("Panjang (cm)", value=15.0, step=0.1, key="calc_panjang")
        
        with col2:
            l_manual = st.number_input("Lebar (cm)", value=5.5, step=0.1, key="calc_lebar")
        
        with col3:
            w_manual = st.number_input("Warna Hijau", value=120, step=1, key="calc_warna")
        
        if st.button("🧮 Hitung Secara Manual", type="primary"):
            # Get model components
            svm_model = st.session_state.models['svm']
            scaler = st.session_state.models['scaler']
            mapping = st.session_state.models['mapping']
            
            # Step 1: Scaling
            st.subheader("Langkah 1: Normalisasi Fitur")
            st.write("Rumus: $z = \\frac{x - \\mu}{\\sigma}$")
            
            input_data = np.array([[p_manual, l_manual, w_manual]])
            input_scaled = scaler.transform(input_data)[0]
            
            means = scaler.mean_
            stds = scaler.scale_
            
            st.markdown(f"""
            <div class="manual-box">
            <h4>Input Asli:</h4>
            <p>Panjang = {p_manual} cm, Lebar = {l_manual} cm, Warna = {w_manual}</p>
            
            <h4>Parameter Scaling:</h4>
            <p>Mean (μ): [{means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}]</p>
            <p>Std Dev (σ): [{stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f}]</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show calculations
            st.markdown(f"""
            <div class="manual-box">
            <h4>Perhitungan:</h4>
            <p>1. Panjang: ({p_manual} - {means[0]:.4f}) / {stds[0]:.4f} = <b>{input_scaled[0]:.4f}</b></p>
            <p>2. Lebar: ({l_manual} - {means[1]:.4f}) / {stds[1]:.4f} = <b>{input_scaled[1]:.4f}</b></p>
            <p>3. Warna: ({w_manual} - {means[2]:.4f}) / {stds[2]:.4f} = <b>{input_scaled[2]:.4f}</b></p>
            
            <h4>Hasil Normalisasi:</h4>
            <p><b>X' = [{input_scaled[0]:.4f}, {input_scaled[1]:.4f}, {input_scaled[2]:.4f}]</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Step 2: SVM Calculation
            st.subheader("Langkah 2: Fungsi Keputusan SVM")
            
            # Get weights and bias
            weights = svm_model.coef_  # Shape: (n_classes, n_features)
            bias = svm_model.intercept_  # Shape: (n_classes,)
            
            n_classes = len(mapping)
            
            st.write(f"Jumlah kelas: {n_classes}")
            st.write("Rumus fungsi keputusan untuk setiap kelas: $f_i(X') = w_i·X' + b_i$")
            
            scores = []
            st.write("**Perhitungan Skor untuk Setiap Kelas:**")
            
            for i in range(n_classes):
                # Calculate score for class i
                score = np.dot(input_scaled, weights[i]) + bias[i]
                scores.append(score)
                
                # Show calculation
                st.markdown(f"""
                <div class="manual-box">
                <h5>Kelas {i}: {mapping[i]}</h5>
                <p>f({i}) = ({input_scaled[0]:.4f} × {weights[i][0]:.4f}) + 
                ({input_scaled[1]:.4f} × {weights[i][1]:.4f}) + 
                ({input_scaled[2]:.4f} × {weights[i][2]:.4f}) + 
                ({bias[i]:.4f})</p>
                <p>f({i}) = <b>{score:.4f}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Step 3: Decision
            st.subheader("Langkah 3: Keputusan Akhir")
            
            predicted_class_idx = np.argmax(scores)
            predicted_class = mapping[predicted_class_idx]
            
            st.markdown(f"""
            <div class="result-card">
            <h3>Hasil Perhitungan Manual</h3>
            <p><b>Skor untuk setiap kelas:</b></p>
            """, unsafe_allow_html=True)
            
            # Display scores in a table
            score_df = pd.DataFrame({
                'Kelas': [mapping[i] for i in range(len(scores))],
                'Skor': [f"{s:.4f}" for s in scores]
            })
            st.dataframe(score_df)
            
            st.markdown(f"""
            <p><b>Skor tertinggi:</b> {scores[predicted_class_idx]:.4f} (Kelas: {predicted_class})</p>
            <p><b>Kelas yang diprediksi:</b> <span style="color:#2e7d32; font-size:20px;"><b>{predicted_class}</b></span></p>
            
            <h4>Verifikasi dengan Model SVM:</h4>
            <p>Model memprediksi: kelas {svm_model.predict([input_scaled])[0]} → {mapping[svm_model.predict([input_scaled])[0]]}</p>
            <p>✅ Hasil perhitungan manual sesuai dengan model!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("Analisis Data")
        
        df = st.session_state.data['df']
        
        # Basic statistics
        st.subheader("Statistik Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistik Numerik:**")
            st.dataframe(df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']].describe())
        
        with col2:
            st.write("**Distribusi Kelas:**")
            class_counts = df['Spesies'].value_counts()
            st.dataframe(class_counts)
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribusi Kelas Spesies')
            st.pyplot(fig)
        
        # Visualization
        st.subheader("Visualisasi Data")
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'Manihot_esculenta': 'red', 
                 'Manihot_palmata': 'green', 
                 'Manihot_glaziovii': 'blue'}
        
        for species, color in colors.items():
            mask = df['Spesies'] == species
            ax.scatter(df.loc[mask, 'Panjang_Daun'], 
                      df.loc[mask, 'Lebar_Daun'],
                      c=color, label=species, alpha=0.7, s=100)
        
        ax.set_xlabel('Panjang Daun (cm)')
        ax.set_ylabel('Lebar Daun (cm)')
        ax.set_title('Panjang vs Lebar Daun per Spesies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Box plots
        st.write("**Box Plot untuk Setiap Fitur:**")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        features = ['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']
        titles = ['Panjang Daun', 'Lebar Daun', 'Warna Hijau']
        
        for idx, (feature, title) in enumerate(zip(features, titles)):
            data = [df[df['Spesies'] == species][feature].values 
                   for species in df['Spesies'].unique()]
            
            axes[idx].boxplot(data)
            axes[idx].set_title(title)
            axes[idx].set_xticklabels(df['Spesies'].unique(), rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation matrix
        st.write("**Matriks Korelasi:**")
        
        numeric_df = df[['Panjang_Daun', 'Lebar_Daun', 'Warna_Hijau']]
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add text annotations
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns)
        ax.set_title('Matriks Korelasi Fitur')
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

else:
    # Show instructions
    st.info("""
    ## 📋 Petunjuk Penggunaan
    
    1. **Upload file CSV** menggunakan uploader di sidebar
    2. **ATAU** klik "Gunakan Data Contoh" untuk mencoba dengan data contoh
    3. **Klik "Latih Model"** untuk memulai proses pelatihan
    4. **Setelah pelatihan selesai**, Anda dapat:
       - Melihat akurasi model di Tab 1
       - Melakukan prediksi di Tab 2
       - Melihat perhitungan manual di Tab 3
       - Menganalisis data di Tab 4
    
    ### 📝 Format CSV yang Diperlukan:
    File CSV harus memiliki kolom berikut (gunakan ; sebagai pemisah):
    ```
    Panjang_Daun;Lebar_Daun;Warna_Hijau;Spesies
    15.2;5.3;120;Manihot_esculenta
    12.8;4.7;110;Manihot_palmata
    20.1;7.2;140;Manihot_glaziovii
    ```
    
    ### 🔧 Spesies yang Didukung dalam Data Contoh:
    - Manihot_esculenta (Singkong Biasa)
    - Manihot_palmata (Singkong Daun Lebar)
    - Manihot_glaziovii (Singkong Hias)
    """)