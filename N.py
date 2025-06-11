# train_model.py
import pandas as pd
import numpy as np
import re
import contractions
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if pd.isnull(text):
        return ""
    text = contractions.fix(text.lower())
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and prepare data
df = pd.read_csv("test.csv")
label_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
df['Text'] = df['Title'] + " " + df['Description']
df['Category'] = df['Class Index'].map(label_map)
df = df[['Text', 'Category']].dropna()

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category'])

# Preprocess
train_df['clean_text'] = train_df['Text'].apply(preprocess)
test_df['clean_text'] = test_df['Text'].apply(preprocess)

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])
y_train = train_df['Category']
y_test = test_df['Category']

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
test_preds = model.predict(X_test)
acc = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {acc:.4f}")

# Save artifacts
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
test_df.to_csv('test_processed.csv', index=False)
joblib.dump(test_preds, 'test_predictions.pkl')

# app.py
import streamlit as st
import pandas as pd
import re
import contractions
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Setup NLTK
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# Preprocess function
def preprocess(text):
    if pd.isnull(text):
        return ""
    text = contractions.fix(text.lower())
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Load processed test data and predictions
@st.cache_data
def load_test_data():
    test_df = pd.read_csv("test_processed.csv")
    preds = joblib.load('test_predictions.pkl')
    return test_df, preds

model, vectorizer = load_model()
test_df, test_preds = load_test_data()

# Streamlit UI
st.title("📰 AG News Topic Classifier")
st.write("Classify AG news articles into **World**, **Sports**, **Business**, or **Sci/Tech**.")

# Accuracy (optional, based on precomputed preds)
st.markdown("### 📊 Model Accuracy: Already computed and shown in terminal during training")

# User input
user_input = st.text_area("Enter article text here:")

if st.button("Classify"):
    processed = preprocess(user_input)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    st.success(f"Predicted Category: **{prediction}**")

# Show predictions
if st.checkbox("Show predictions on test set"):
    st.write(test_df[['Text', 'Category']].assign(Predicted=test_preds))

# Classification report
if st.checkbox("Show classification report"):
    report = classification_report(test_df['Category'], test_preds, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

# Confusion matrix
if st.checkbox("Show confusion matrix"):
    cm = confusion_matrix(test_df['Category'], test_preds, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
