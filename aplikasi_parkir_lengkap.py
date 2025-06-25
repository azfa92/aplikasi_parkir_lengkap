import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib # Untuk menyimpan/memuat model
import warnings
import plotly.express as px # Import Plotly

# Abaikan warnings untuk tampilan yang lebih bersih
warnings.filterwarnings('ignore')

# --- 1. Definisi Peta Konversi ---
BULAN_MAP = {
    "Januari": 1,
    "Februari": 2,
    "Maret": 3,
    "April": 4,
    "Mei": 5,
    "Juni": 6,
    "Juli": 7,
    "Agustus": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Desember": 12
}

HARI_MAP = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}

# --- Fungsi untuk Memuat dan Memproses Data (dengan caching) ---
@st.cache_data # Cache data agar tidak di-load ulang setiap kali
def load_and_preprocess_data(data_path='DataParkir.csv'):
    try:
        df = pd.read_csv(data_path, sep=',')
    except Exception:
        df = pd.read_csv(data_path, sep=';')

    print(f"Data '{data_path}' berhasil dimuat untuk pelatihan. Jumlah baris: {len(df)}")

    # Memastikan kolom-kolom yang dibutuhkan ada
    required_columns = ['Bulan', 'Hari', 'Jam', 'Suhu', 'Libur', 'Event', 'Jumlah']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Kolom '{col}' tidak ditemukan di {data_path}. Harap periksa nama kolom.")
            return None

    # Feature Engineering dan Konversi Tipe Data
    if df['Hari'].dtype == 'object':
        df['Hari'] = df['Hari'].map(HARI_MAP)
    df['Libur'] = df['Libur'].astype(int)
    df['Event'] = df['Event'].astype(int)
    df['Jumlah'] = df['Jumlah'].astype(int)
    df['Weekend'] = df['Hari'].apply(lambda x: 1 if x in [5, 6] else 0)

    return df

# --- 2. Fungsi untuk Melatih & Memuat Model SVR (mengembalikan model, X_test, y_test) ---
@st.cache_resource # Cache resource agar model hanya dimuat/dilatih sekali
def get_trained_svr_model(data_path='DataParkir.csv', model_save_path='model_parkir_svr_terbaik_gabungan.joblib'):
    """
    Melatih model SVR terbaik atau memuatnya jika sudah ada.
    Mengembalikan model SVR terbaik, X_test, dan y_test.
    """
    print(f"Mencoba memuat model dari '{model_save_path}'...")
    try:
        # Coba muat model jika sudah ada
        model = joblib.load(model_save_path)
        print(f"Model berhasil dimuat dari '{model_save_path}'.")

        # Untuk mendapatkan X_test dan y_test, kita perlu memuat dan memproses data lagi
        # agar konsisten dengan proses pelatihan, tapi cukup sekali karena di-cache.
        df = load_and_preprocess_data(data_path)
        if df is None:
            return None, None, None

        features = ['Bulan', 'Hari', 'Jam', 'Suhu', 'Libur', 'Event', 'Weekend']
        target = 'Jumlah'
        X = df[features]
        y = df[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return model, X_test, y_test
    except FileNotFoundError:
        print(f"Model '{model_save_path}' tidak ditemukan. Melatih model baru...")
        pass # Lanjutkan untuk melatih model jika file tidak ditemukan
    except Exception as e:
        print(f"Error saat memuat model: {e}. Melatih model baru...")
        pass # Lanjutkan untuk melatih model jika terjadi error lain

    # Jika model tidak ditemukan atau ada error saat memuat, latih model baru
    df = load_and_preprocess_data(data_path)
    if df is None:
        return None, None, None

    print(f"Data '{data_path}' berhasil dimuat untuk pelatihan. Jumlah baris: {len(df)}")

    features = ['Bulan', 'Hari', 'Jam', 'Suhu', 'Libur', 'Event', 'Weekend']
    target = 'Jumlah'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Bulan', 'Hari', 'Jam', 'Suhu']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )

    kernels = ['linear', 'rbf', 'poly']
    best_model_pipeline = None
    best_mae = float('inf')
    best_kernel_name = ""

    print("\n--- Membandingkan Kernel SVR ---")
    for kernel_name in kernels:
        print(f"Melatih SVR dengan kernel: '{kernel_name}'")
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(kernel=kernel_name))
        ])
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        if mae < best_mae:
            best_mae = mae
            best_model_pipeline = model_pipeline
            best_kernel_name = kernel_name

        print(f"  Kernel '{kernel_name}' MAE: {mae:.2f}")

    print(f"\nModel SVR Terbaik menggunakan kernel: '{best_kernel_name}' dengan MAE: {best_mae:.2f}")

    # Simpan model yang baru dilatih
    joblib.dump(best_model_pipeline, model_save_path)
    print(f"Model baru berhasil disimpan sebagai '{model_save_path}'.")
    return best_model_pipeline, X_test, y_test

# --- 3. Fungsi Prediksi ---
# (Tidak ada perubahan di sini)
def prediksi_jumlah_parkir_svr(model, bulan_nama, hari_nama, jam, suhu, libur_kuliah, ada_event):
    """
    Membuat prediksi jumlah kendaraan parkir menggunakan model SVR yang sudah dilatih.
    """
    if model is None:
        return "Model belum dimuat. Tidak dapat membuat prediksi."

    bulan_numeric = BULAN_MAP.get(bulan_nama)
    hari_numeric = HARI_MAP.get(hari_nama)

    if bulan_numeric is None:
        raise ValueError(f"Nama bulan '{bulan_nama}' tidak valid.")
    if hari_numeric is None:
        raise ValueError(f"Nama hari '{hari_nama}' tidak valid. Gunakan Senin-Minggu.")

    weekend = 1 if hari_numeric >= 5 else 0
    libur_numeric = 1 if libur_kuliah else 0
    event_numeric = 1 if ada_event else 0

    input_data = pd.DataFrame({
        'Bulan': [bulan_numeric], 'Hari': [hari_numeric], 'Jam': [jam], 'Suhu': [suhu],
        'Libur': [libur_numeric], 'Event': [event_numeric], 'Weekend': [weekend]
    })

    prediction = model.predict(input_data)
    jumlah_prediksi = int(round(prediction[0]))
    return max(0, jumlah_prediksi)


# --- 4. Aplikasi Streamlit Utama ---
st.set_page_config(page_title="Prediksi Parkir UNPAM (SVR)", page_icon="🚗", layout="wide")
st.title("🏍️ Aplikasi Prediksi Jumlah Parkir (SVR)")
st.write("Aplikasi ini menggunakan model **Support Vector Regression (SVR)** untuk memprediksi estimasi jumlah kendaraan yang terparkir berdasarkan kondisi yang Anda masukkan.")
st.markdown("---")

# Muat atau latih model saat aplikasi dimulai
# Sekarang kita mendapatkan model, X_test, dan y_test
model_svr_loaded, X_test_data, y_test_data = get_trained_svr_model()

# --- Antarmuka Input untuk Prediksi ---
st.header("Masukkan Kondisi untuk Prediksi:")

col1, col2 = st.columns(2)

with col1:
    bulan_display_options = list(BULAN_MAP.keys())
    bulan_selected = st.selectbox("Bulan", options=bulan_display_options, index=5)

    hari_options = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    hari_selected = st.selectbox("Hari dalam Seminggu", hari_options)

with col2:
    jam_selected = st.slider("Jam (0-23)", 0, 23, 10)
    suhu_selected = st.slider("Suhu (°C)", 20.0, 40.0, 28.0, step=0.1)

st.markdown("---")
col_cb1, col_cb2 = st.columns(2)
with col_cb1:
    libur_kuliah_selected = st.checkbox("Apakah Hari Libur Perkuliahan?")
with col_cb2:
    ada_event_selected = st.checkbox("Apakah Ada Event Khusus?")

st.markdown("---")

# --- Tombol Prediksi ---
if model_svr_loaded:
    if st.button("📈 Buat Prediksi Jumlah Parkir", use_container_width=True, type="primary"):
        with st.spinner("Membuat prediksi..."):
            try:
                jumlah_prediksi = prediksi_jumlah_parkir_svr(
                    model=model_svr_loaded,
                    bulan_nama=bulan_selected,
                    hari_nama=hari_selected,
                    jam=jam_selected,
                    suhu=suhu_selected,
                    libur_kuliah=libur_kuliah_selected,
                    ada_event=ada_event_selected
                )

                st.subheader("🎉 Hasil Prediksi:")
                st.metric(label="Estimasi Jumlah Kendaraan Terparkir", value=f"~ {jumlah_prediksi} motor")

                st.success("Prediksi berhasil dibuat!")
                st.info("Estimasi ini didasarkan pada pola data historis yang telah dipelajari model SVR terbaik.")

            except ValueError as e:
                st.error(f"Kesalahan input: {e}")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
else:
    st.warning("Model tidak dapat dimuat atau dilatih. Harap periksa log di terminal.")

# --- Bagian Baru: Visualisasi Prediksi vs. Aktual ---
st.markdown("---")
st.header("📊 Evaluasi Model: Prediksi vs. Aktual")

if model_svr_loaded and X_test_data is not None and y_test_data is not None:
    # Lakukan prediksi pada data test yang dikembalikan
    y_pred_test = model_svr_loaded.predict(X_test_data)

    # Buat DataFrame untuk visualisasi
    df_results = pd.DataFrame({
        'Aktual': y_test_data,
        'Prediksi': y_pred_test.round().astype(int) # Bulatkan ke bilangan bulat
    })
    df_results['Error'] = df_results['Aktual'] - df_results['Prediksi']

    # Hitung metrik evaluasi
    mae = mean_absolute_error(df_results['Aktual'], df_results['Prediksi'])
    rmse = np.sqrt(mean_squared_error(df_results['Aktual'], df_results['Prediksi']))
    r2 = r2_score(df_results['Aktual'], df_results['Prediksi'])

    st.subheader("Performa Model pada Data Uji:")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    col_metrics1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    col_metrics2.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    col_metrics3.metric("R² Score", f"{r2:.2f}")
    st.write("MAE menunjukkan rata-rata selisih absolut antara prediksi dan nilai aktual.")
    st.write("RMSE mengukur rata-rata magnitudo error, memberikan bobot lebih pada error yang lebih besar.")
    st.write("R² Score menunjukkan proporsi variasi target yang dapat dijelaskan oleh model (semakin mendekati 1 semakin baik).")

    st.subheader("Grafik Prediksi vs. Aktual")
    fig = px.scatter(
        df_results,
        x='Aktual',
        y='Prediksi',
        title='Prediksi vs. Aktual Jumlah Parkir',
        labels={'Aktual': 'Jumlah Aktual', 'Prediksi': 'Jumlah Prediksi'},
        hover_data=['Aktual', 'Prediksi', 'Error'],
        template="plotly_white" # Pilihan tema
    )
    # Tambahkan garis diagonal untuk menunjukkan prediksi sempurna
    max_val = max(df_results['Aktual'].max(), df_results['Prediksi'].max())
    min_val = min(df_results['Aktual'].min(), df_results['Prediksi'].min())
    fig.add_shape(
        type="line", line=dict(dash='dash', color='red'),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    st.info("Setiap titik mewakili satu observasi dari data uji. Semakin dekat titik ke garis merah, semakin akurat prediksinya.")
else:
    st.info("Data uji atau model belum tersedia untuk visualisasi performa.")


# --- Footer ---
st.markdown("---")
