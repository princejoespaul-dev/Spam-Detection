import streamlit as st
import joblib
import re
import nltk
import os

# Download NLTK data (works better on Streamlit Cloud)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True, download_dir='/root/nltk_data')
    nltk.download('punkt', quiet=True, download_dir='/root/nltk_data')
    nltk.download('wordnet', quiet=True, download_dir='/root/nltk_data')

download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Preprocess function
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# ====================== UI ======================
st.title("📧 Spam Email Detector")
st.markdown("### Check if an email is **Spam** or **Ham**")

email_text = st.text_area("Paste Email Text Here:", height=300)

if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter email text!")
    else:
        cleaned = preprocess(email_text)
        vectorized = vectorizer.transform([cleaned])
        
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        
        if prediction == "spam":
            st.error(f"🚨 **SPAM DETECTED!** ({prob[1]:.1%} confidence)")
        else:
            st.success(f"✅ **Ham (Normal Email)** ({prob[0]:.1%} confidence)")

        with st.expander("Debug: Preprocessed Text"):
            st.write(cleaned)
