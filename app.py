import streamlit as st

# WAJIB: Konfigurasi halaman di paling atas
st.set_page_config(
    page_title="📱 Fraud Message Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
import base64
from PIL import Image

# =====================
# Load Model & Vectorizer
# =====================
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
    return None, None

model, vectorizer = load_model()

# =====================
# Sidebar Menu
# =====================
menu = st.sidebar.radio("📂 Menu", [
    "Home",
    "Data Visualization",
    "Project Overview",
    "Paper",
    "Poster",
    "Train Model"
])

# =====================
# Mapping Label
# =====================
label_map = {
    0: "✅ Normal",
    1: "⚠️ Fraud / Penipuan",
    2: "📢 Promo / Iklan"
}

# =====================
# HOME
# =====================
if menu == "Home":
    st.title("🛡️ Fraud Message Detection")
    st.markdown("Masukkan pesan yang ingin diperiksa apakah termasuk **penipuan**, **promo**, atau **normal**.")

    user_input = st.text_area("✉️ Masukkan Pesan", height=150)

    if st.button("🔍 Deteksi"):
        if not user_input.strip():
            st.warning("Silakan masukkan pesan terlebih dahulu.")
        else:
            if model and vectorizer:
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0]
                st.subheader("Hasil Prediksi:")
                st.success(f"{label_map[prediction]}")
            else:
                st.error("Model belum tersedia. Jalankan `train_model.py` terlebih dahulu.")

# =====================
# DATA VISUALIZATION
# =====================
elif menu == "Data Visualization":
    st.title("📊 Visualisasi Data")
    data_path = "data/dataset_sms_spam_v1.csv"

    if not os.path.exists(data_path):
        st.warning("File data belum tersedia di `data/dataset_sms_spam_v1.csv`.")
    else:
        df = pd.read_csv(data_path)
        df = df.dropna(subset=["Teks", "label"])
        df["label"] = df["label"].astype(int)
        df["label_name"] = df["label"].map({0: "Normal", 1: "Fraud", 2: "Promo"})

        # Distribusi kelas
        st.subheader("🔢 Distribusi Kelas (Pie Chart)")
        label_counts = df["label_name"].value_counts()
        labels = label_counts.index
        sizes = label_counts.values
        total = sizes.sum()
        explode = [0.05] * len(labels)

        fig1, ax1 = plt.subplots()
        ax1.pie(
            sizes,
            labels=[f"{label} ({size} | {size/total:.1%})" for label, size in zip(labels, sizes)],
            explode=explode,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )
        ax1.axis('equal')
        st.pyplot(fig1)

        # Contoh data berdasarkan label
        st.subheader("🔍 Contoh Data Berdasarkan Label")
        label_options = {
            "Normal (0)": 0,
            "Fraud (1)": 1,
            "Promo (2)": 2
        }
        selected_label_name = st.selectbox("Pilih Label untuk Ditampilkan:", list(label_options.keys()))
        selected_label = label_options[selected_label_name]
        filtered_df = df[df["label"] == selected_label]
        st.markdown(f"Menampilkan pesan dengan label **{selected_label_name}**")
        st.dataframe(filtered_df[["Teks", "label"]].reset_index(drop=True).head(10))

# =====================
# PROJECT OVERVIEW
# =====================
elif menu == "Project Overview":
    st.title("📘 Project Overview")
    st.markdown("""
    Proyek ini bertujuan untuk mendeteksi apakah suatu pesan chat termasuk:
    - `Normal`: pesan biasa sehari-hari
    - `Fraud`: pesan penipuan, undian palsu, permintaan transfer
    - `Promo`: pesan iklan, penawaran produk, dll

    ### ✅ Fitur:
    - Deteksi dari input teks
    - Visualisasi distribusi data dan WordCloud
    - Model klasifikasi 3 kelas

    ### 📊 Dataset:
    - Nama: `dataset_sms_spam_v1.csv`
    - Label: 0 = Normal, 1 = Fraud, 2 = Promo
    - Format: CSV, kolom `Teks` dan `label`

    ### 🧠 Teknologi:
    - Python, Streamlit
    - TF-IDF Vectorizer
    - Logistic Regression (Scikit-Learn)

    ### 📂 Struktur Folder:
    ```
    /app.py
    /model/model.pkl
    /model/vectorizer.pkl
    /data/dataset_sms_spam_v1.csv
    /train_model.py
    /requirements.txt
    ```
    """)

# =====================
# PAPER
# =====================
elif menu == "Paper":
    st.title("📄 Paper Ilmiah")
    paper_path = "assets/paper.pdf"

    if os.path.exists(paper_path):
        with open(paper_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>'

            st.download_button(
                label="📥 Download Paper",
                data=base64.b64decode(base64_pdf),
                file_name="FraudDetection_Paper.pdf",
                mime="application/pdf"
            )

            st.markdown("### 📄 Preview Paper")
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.warning("File paper belum tersedia. Simpan file `paper.pdf` di folder `assets/`.")

# =====================
# POSTER
# =====================
elif menu == "Poster":
    st.title("🖼️ Poster Proyek")
    st.markdown("Berikut adalah poster visualisasi proyek deteksi pesan penipuan:")

    poster_path = "assets/poster.png"

    if os.path.exists(poster_path):
        try:
            img = Image.open(poster_path)
            st.image(img, caption="Poster Proyek Deteksi Penipuan", width=700)
        except Exception as e:
            st.error(f"❌ Gagal menampilkan poster. Error: {e}")
    else:
        st.info("🖼️ Poster belum tersedia. Simpan sebagai `assets/poster.png` untuk ditampilkan di sini.")


# =====================
# TRAIN MODEL
# =====================
elif menu == "Train Model":
    st.title("🧠 Train the Classification Model")

    if st.button("🔁 Jalankan Training"):
        with st.spinner("Sedang melatih model..."):
            import subprocess
            result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)

            st.success("✅ Model berhasil dilatih ulang.")
            st.code(result.stdout)
            if result.stderr:
                st.error("❌ Error:")
                st.code(result.stderr)
