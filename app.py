import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deteksi Dini Risiko Kesehatan Pasien", page_icon="🩺")


@st.cache_resource
def load_model():
    kmeans = joblib.load("kmeans_stroke.joblib")
    scaler = joblib.load("scaler_stroke.joblib")
    return kmeans, scaler


kmeans, scaler = load_model()

FEATURES = ["age", "avg_glucose_level"]

# Deskripsi umum tiap cluster berdasarkan urutan rata-rata risiko dari hasil
# training. Sesuaikan teks ini dengan hasil interpretasi cluster_profile pada notebook.
CLUSTER_DESC = {
    0: "Risiko Rendah — usia normal dan kadar glukosa normal.",
    1: "Risiko Sedang — salah satu indikator (usia/glukosa) mulai meningkat.",
    2: "Risiko Tinggi — usia lebih tua dengan kadar glukosa tinggi.",
}

st.title("🩺 Segmentasi Risiko Kesehatan Pasien")
st.write(
    "Aplikasi ini mengelompokkan profil kesehatan pasien ke dalam segmen "
    "risiko berdasarkan usia dan kadar glukosa, menggunakan model "
    "**K-Means Clustering** yang dilatih pada Stroke Prediction Dataset (Kaggle)."
)

st.header("Masukkan Data Pasien")

age = st.number_input("Usia", min_value=0, max_value=120, value=45)
avg_glucose_level = st.number_input(
    "Rata-rata Kadar Glukosa (mg/dL)", min_value=40.0, max_value=350.0, value=100.0
)

if st.button("Prediksi Segmen"):
    input_df = pd.DataFrame([[age, avg_glucose_level]], columns=FEATURES)

    input_scaled = scaler.transform(input_df)
    cluster = int(kmeans.predict(input_scaled)[0])

    st.success(f"Pasien termasuk ke dalam **Cluster {cluster}**")
    st.write(CLUSTER_DESC.get(cluster, "Deskripsi cluster belum didefinisikan."))

    st.caption(
        "Catatan: hasil ini merupakan segmentasi eksploratif dari model "
        "unsupervised, bukan diagnosis medis. Selalu konsultasikan kondisi "
        "kesehatan aktual ke tenaga medis profesional."
    )

st.divider()
st.caption(
    "Model: K-Means Clustering · Dataset: Stroke Prediction Dataset "
    "(fedesoriano, Kaggle) · Dibuat untuk Tugas Mandiri Pertemuan 4 - "
    "Implementasi Clustering dengan CRISP-DM"
)
