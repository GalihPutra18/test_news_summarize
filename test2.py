# Import libraries
import streamlit as st
import requests
import matplotlib.pyplot as plt
import io
import re
from bs4 import BeautifulSoup
from transformers import BertTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from googletrans import Translator

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Download stopwords for nltk
stopwords_set = stopwords.words('english')  # Using NLTK for English stopwords
stop_words = {
    'en': set(stopwords_set).union({'said', 'will', 'also', 'one', 'new', 'make'}),
    'id': set(stopwords.words('indonesian')).union({'dan', 'yang', 'di', 'dari', 'pada', 'untuk', 'dengan', 'ke', 'dalam', 'adalah'}),
    'es': set(stopwords.words('spanish')).union({'y', 'el', 'en', 'con', 'para', 'de'}),
    'fr': set(stopwords.words('french')).union({'et', 'le', 'est', 'dans', 'sur', 'avec'})
}

# Function to fetch and parse article from the given URL
def fetch_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No Title Found'
        paragraphs = soup.find_all('p')
        article = ' '.join([para.get_text() for para in paragraphs])

        # Remove common ad phrases
        ad_patterns = re.compile(r"(Advertisement|Scroll to Continue|Baca Juga|Lanjutkan dengan Konten)", re.IGNORECASE)
        article_cleaned = ad_patterns.sub('', article)

        return title, article_cleaned.strip()
    else:
        return None, None

# Function to summarize the article
def summarize_article_flexible(article, num_clusters=2):
    # Tokenize the article into sentences
    sentences = tokenizer.tokenize(article)

    # Create TF-IDF vector
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

    # Get key points summary from each cluster
    point_summary = []
    for i in range(num_clusters):
        cluster_sentences = [sentences[j] for j in range(len(sentences)) if kmeans.labels_[j] == i]
        if cluster_sentences:
            point_summary.append(max(cluster_sentences, key=len))  # Longest sentence as key point

    # Short paragraph summary
    paragraph_summary = ' '.join(point_summary)

    # Infographic: Visualize word count
    sentence_lengths = [len(sentence.split()) for sentence in point_summary]
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(point_summary) + 1), sentence_lengths, color='skyblue')
    plt.xlabel('Point Number')
    plt.ylabel('Word Count')
    plt.title('Word Count per Key Point')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return point_summary, paragraph_summary, buf

# Function for generating a longer summary
def long_summary(article):
    sentences = tokenizer.tokenize(article)
    return ' '.join(sentences)

# Function to translate the article into a specific language
def translate_article(article, dest_language='en'):
    translator = Translator()
    try:
        detected_lang = translator.detect(article).lang
        if detected_lang != dest_language:
            translated = translator.translate(article, dest=dest_language)
            return translated.text
        else:
            return article
    except Exception as e:
        st.error(f'Translation failed: {e}')
        return None

# Function to generate hashtags from the title and content
def generate_hashtags(title, content, lang='en', num_hashtags=5):
    stop_words_set = stop_words.get(lang, set())
    title_words = [word for word in tokenizer.tokenize(title.lower()) if word.isalnum() and len(word) > 3 and word not in stop_words_set]
    content_words = [word for word in tokenizer.tokenize(content.lower()) if word.isalnum() and len(word) > 3 and word not in stop_words_set]

    # Combine keywords from title and content
    keywords = title_words * 2 + content_words  # Duplicate title words to increase weight

    # Generate TF-IDF scores
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(keywords)])

    tfidf_scores = X.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    scored_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

    top_keywords = [f"#{keyword.capitalize()}" for keyword, score in scored_keywords[:num_hashtags]]
    return top_keywords

# Main function to run the Streamlit app
def main():
    st.title('News Article Summary & Hashtag Generator')

    # Store URL and selected language in session state
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'lang' not in st.session_state:
        st.session_state.lang = "en"

    st.session_state.url = st.text_input('Enter news article URL:', st.session_state.url)
    st.session_state.lang = st.selectbox('Select language for translation:', ['en', 'id', 'es', 'fr'], index=['en', 'id', 'es', 'fr'].index(st.session_state.lang))

    if st.button('Summarize and Generate Hashtags'):
        if st.session_state.url:
            if not (st.session_state.url.startswith('http://') or st.session_state.url.startswith('https://')):
                st.error('Please enter a valid URL starting with http:// or https://')
                return

            title, article = fetch_article(st.session_state.url)
            if article:
                if not article.strip():
                    st.error('Article is empty or could not be fetched.')
                    return

                # Translate title if necessary
                translated_title = translate_article(title, st.session_state.lang)
                st.subheader('Article Title:')
                st.write(translated_title)

                # Translate article if necessary
                translated_article = translate_article(article, st.session_state.lang)
                if translated_article is None:
                    return

                num_clusters = st.slider('Select number of clusters for summary:', 1, 5, 2)
                point_summary, paragraph_summary, infographic_buf = summarize_article_flexible(translated_article, num_clusters)

                # Display flexible summary options
                st.subheader('Flexible Summary Options:')
                
                # Key Points
                st.write("### Key Points:")
                for idx, point in enumerate(point_summary, 1):
                    st.write(f"{idx}. {point}")

                # Short Paragraph
                st.write("### Short Paragraph:")
                st.write(paragraph_summary)

                # Detailed Summary
                detailed_summary = long_summary(translated_article)
                st.write("### Detailed Summary:")
                st.write(detailed_summary)

                # Infographic
                st.write("### Infographic:")
                st.image(infographic_buf)

                # Generate hashtags
                hashtags = generate_hashtags(translated_title, translated_article, st.session_state.lang)
                st.write("### Generated Hashtags:")
                st.write(", ".join(hashtags))

if __name__ == "__main__":
    main()

