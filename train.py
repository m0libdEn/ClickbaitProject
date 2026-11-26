import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from clickbait import analyze_headline, embedder

# 1. Load dataset
df = pd.read_csv("clickbait_data.csv")
df = df.dropna(subset=["headline", "clickbait"])

print(f"Loaded {len(df)} headlines. Starting feature extraction...")

# 2. Vectorize all texts in a batch
headlines = df["headline"].astype(str).tolist()
embeddings = embedder.encode(headlines, show_progress_bar=True)

# 3. Extract linguistic and syntactic features via clickbait.py
X_features = []
for i, title in enumerate(tqdm(headlines, desc="Linguistic & syntactic features")):
    feats = analyze_headline(title)
    lex = feats["lexical"]
    syn = feats["syntactic"]
    sent = feats["sentiment"]

    numeric_row = [
        lex["word_count"],
        lex["superlatives"],
        lex["forward_references"],
        lex["caps_ratio"],
        lex["exclamation_marks"],
        lex["question_marks"],
        syn["pronoun_density"],
        syn["modal_verb_density"],
        sent["intensity"]
    ]
    # Concatenate 9 handcrafted features + 384 embedding dimensions = 393 features
    full_row = np.concatenate([numeric_row, embeddings[i]])
    X_features.append(full_row)

X = np.array(X_features)
y = df["clickbait"].astype(int).values

# 4. Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train classifier
print("Training LogisticRegression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate performance
y_pred = model.predict(X_test)
print("\n=== Test Set Results ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 7. Save trained model
joblib.dump(model, "clickbait_model.joblib")
print("Model saved to clickbait_model.joblib")