import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download punkt and stopwords if they are not already available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {e}")

# Test with sample text
text = st.text_area("Enter text to tokenize:")
if text:
    sentences = sent_tokenize(text)
    st.write("Tokenized Sentences:")
    st.write(sentences)

    # Use stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [word for word in sentences if word not in stop_words]
    st.write("Filtered Sentences (without stopwords):")
    st.write(filtered_sentences)
