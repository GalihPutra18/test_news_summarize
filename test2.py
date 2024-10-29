import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Ensure punkt tokenizer is downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {e}")

# Now you can use sent_tokenize and stopwords without issues
text = "Your sample text goes here."
sentences = sent_tokenize(text)

# Use stopwords
stop_words = set(stopwords.words('english'))
