# train_text.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset (adjust the path if necessary)
df = pd.read_csv("data/offensive_language.csv")

# Ensure the text column is string type
df['tweet'] = df['tweet'].astype(str)

# Define vectorizer and features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet'])
y = df['class']  # Adjust based on your label definition

# Split into training and test sets (for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Optional: Print evaluation metrics
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("Text model and vectorizer saved.")
