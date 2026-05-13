import streamlit as st
import joblib
import re
import nltk
import subprocess
import sys

# ====================== Force Install (Safety) ======================
subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib", "scikit-learn", "nltk"])

# ====================== NLTK Setup (Fixed) ======================
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)   # ← This is the important one
    nltk.download('wordnet', quiet=True)
    return True

setup_nltk()

# ====================== Load Model ======================
@st.cache_resource
def load_model():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ====================== Preprocess ======================
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

email_text = st.text_area("Paste the full email here:", height=300)

if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text!")
    else:
        cleaned = preprocess(email_text)
        vectorized = vectorizer.transform([cleaned])
        
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        
        if prediction == "spam":
            st.error(f"🚨 **SPAM DETECTED!** ({prob[1]:.1%} confidence)")
        else:
            st.success(f"✅ **This is a Normal Email (Ham)** ({prob[0]:.1%} confidence)")

        with st.expander("Debug: Preprocessed Text"):
            st.write(cleaned)
