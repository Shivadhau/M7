import streamlit as st
import pandas as pd
import numpy as np
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing
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

# Load and preprocess data
@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("test.csv")
    
    # Mapping AG News numeric class labels to names
    label_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    
    # Combine title and description as the input text
    df['Text'] = df['Title'] + " " + df['Description']
    
    # Map 'Class Index' to category names
    df['Category'] = df['Class Index'].map(label_map)

    df = df[['Text', 'Category']].dropna()

    return train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category'])



train_df, test_df = load_data()

# Apply preprocessing
train_df['clean_text'] = train_df['Text'].apply(preprocess)
test_df['clean_text'] = test_df['Text'].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
y_train = train_df['Category']
X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['Category']

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
test_preds = model.predict(X_test)
accuracy = accuracy_score(y_test, test_preds)

# Streamlit App
st.title("📰 AG News Topic Classifier")
st.write("Classify AG news articles into **World**, **Sports**, **Business**, or **Sci/Tech**.")

st.markdown(f"### 📊 Model Accuracy on Test Set: **{accuracy * 100:.2f}%**")

user_input = st.text_area("Enter article text here:")

if st.button("Classify"):
    processed = preprocess(user_input)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    st.success(f"Predicted Category: **{prediction}**")

# Show predictions
if st.checkbox("Show model predictions on test set"):
    st.write(test_df[['Text', 'Category']].assign(Predicted=test_preds))

# Show classification report
if st.checkbox("Show classification report"):
    report = classification_report(y_test, test_preds, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

# Show confusion matrix
if st.checkbox("Show confusion matrix"):
    cm = confusion_matrix(y_test, test_preds, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

