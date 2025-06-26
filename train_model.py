import os
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Download stopwords jika belum ada
nltk.download('stopwords')

# === 1. Load Data === #
DATA_PATH = "data/dataset_sms_spam_v1.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["Teks", "label"])

# === 2. Preprocessing Manual Stopword Removal === #
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df["Teks"] = df["Teks"].apply(clean_text)

# === 3. Split Data === #
X_train, X_test, y_train, y_test = train_test_split(
    df["Teks"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# === 4. TF-IDF Vectorization === #
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 5. Model Training === #
model = OneVsRestClassifier(LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
model.fit(X_train_vec, y_train)

# === 6. Evaluation === #
y_pred = model.predict(X_test_vec)
print("=== Classification Report ===")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal (0)", "Fraud (1)", "Promo (2)"]
))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 7. Save Model & Vectorizer === #
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model training completed and saved.")
