import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("datasets/Resume/Resume.csv")

# -----------------------------
# CHECK COLUMNS
# -----------------------------
print(df.columns)

# -----------------------------
# INPUT AND OUTPUT
# -----------------------------
X = df['Resume_str']

y = df['Category']

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english')

X_tfidf = tfidf.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# ACCURACY
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(model, open("models/model.pkl", "wb"))

pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))

print("Model and Vectorizer Saved Successfully!")