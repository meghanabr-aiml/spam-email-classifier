import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 1. Load Dataset
# ===============================

# Download SMS Spam Collection dataset from:
# https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# Save as spam.csv with columns: label, message

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ===============================
# 2. Text Cleaning Function
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['message'] = df['message'].apply(clean_text)

# ===============================
# 3. Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# ===============================
# 4. TF-IDF Vectorization
# ===============================

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 5. Model Training
# ===============================

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ===============================
# 6. Prediction
# ===============================

y_pred = model.predict(X_test_vec)

# ===============================
# 7. Evaluation
# ===============================

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
