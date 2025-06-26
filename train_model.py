"""
train_model.py  – Fraud Message Detection (Bahasa Indonesia, 3 kelas)
--------------------------------------------------------------------
Dataset : data/dataset_sms_spam_v1.csv
Kolom   : Teks  |  label  (0=Normal, 1=Fraud, 2=Promo)

Hasil   : model/model.pkl
          model/vectorizer.pkl
"""

import os
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. Unduh & siapkan stop-word Bahasa Indonesia
# ------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
stopwords_ind = stopwords.words("indonesian")

# Jika perlu menambah kata khusus:
# stopwords_ind += ["aja", "kok", "deh"]

# ------------------------------------------------------------------
# 2. Baca dataset
# ------------------------------------------------------------------
DATA_PATH = "data/dataset_sms_spam_v1.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌  Dataset tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH).dropna(subset=["Teks", "label"])
df["label"] = df["label"].astype(int)

print("✅ Dataset dimuat  :", len(df), "baris")
print("Distribusi label  :\n", df["label"].value_counts())

# ------------------------------------------------------------------
# 3. Split train / test
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["Teks"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42,
)

# ------------------------------------------------------------------
# 4. TF-IDF Vectorizer
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words=stopwords_ind,
    lowercase=True,
    ngram_range=(1, 2),          # unigram + bigram
    max_features=5_000,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ------------------------------------------------------------------
# 5. Train Logistic Regression (multi-class)
# ------------------------------------------------------------------
model = LogisticRegression(
    max_iter=1_000,
    multi_class="ovr",           # one-vs-rest
    class_weight="balanced",
)
model.fit(X_train_vec, y_train)

# ------------------------------------------------------------------
# 6. Evaluasi
# ------------------------------------------------------------------
y_pred = model.predict(X_test_vec)
label_names = ["Normal (0)", "Fraud (1)", "Promo (2)"]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------
# 7. Simpan model & vectorizer
# ------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model & vectorizer tersimpan di folder  model/")
