import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ====================== Load Model & Vectorizer ======================
@st.cache_resource
def load_model():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ====================== Preprocess Function ======================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)      # Keep only letters and spaces
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# ====================== Streamlit App ======================
st.title("📧 Spam Email Detector")
st.markdown("### Paste an email to check if it's **Spam** or **Ham**")

email_text = st.text_area("Enter Email Text Here:", height=250)

if st.button("🔍 Check Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text!")
    else:
        # Preprocess
        cleaned_text = preprocess(email_text)
        
        # Transform using vectorizer
        vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        # Show Result
        if prediction == "spam":
            st.error("🚨 **SPAM EMAIL DETECTED!**")
            st.write(f"Confidence: **{probability[1]:.1%}**")
        else:
            st.success("✅ **This is a Normal Email (Ham)**")
            st.write(f"Confidence: **{probability[0]:.1%}**")

        # Optional: Show cleaned text for debugging
        with st.expander("See Preprocessed Text (for debugging)"):
            st.write(cleaned_text)