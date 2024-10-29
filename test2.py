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

# Inisialisasi tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Unduh stopwords untuk nltk
stopwords_set = stopwords.words('english')  # Menggunakan NLTK untuk stopwords bahasa Inggris
stop_words = {
    'en': set(stopwords_set).union({'said', 'will', 'also', 'one', 'new', 'make'}),
    'id': set(stopwords.words('indonesian')).union({'dan', 'yang', 'di', 'dari', 'pada', 'untuk', 'dengan', 'ke', 'dalam', 'adalah'}),
    'es': set(stopwords.words('spanish')).union({'y', 'el', 'en', 'con', 'para', 'de'}),
    'fr': set(stopwords.words('french')).union({'et', 'le', 'est', 'dans', 'sur', 'avec'})
}

# Fungsi untuk mengambil dan memparsing artikel dari URL yang diberikan
def fetch_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No Title Found'
        paragraphs = soup.find_all('p')
        article = ' '.join([para.get_text() for para in paragraphs])
        
        # Hapus frasa iklan yang umum
        ad_patterns = re.compile(r"(Advertisement|Scroll to Continue|Baca Juga|Lanjutkan dengan Konten)", re.IGNORECASE)
        article_cleaned = ad_patterns.sub('', article)
        
        return title, article_cleaned.strip()
    else:
        return None, None

# Fungsi untuk meringkas artikel
def summarize_article_flexible(article, num_clusters=2):
    # Tokenisasi artikel menjadi kalimat dan menghilangkan karakter khusus
    sentences = re.split(r'(?<=[.!?]) +', article)
    sentences = [sentence for sentence in sentences if sentence.strip()]

    # Buat vektor TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

    # Dapatkan ringkasan poin kunci dari setiap cluster
    point_summary = []
    for i in range(num_clusters):
        cluster_sentences = [sentences[j] for j in range(len(sentences)) if kmeans.labels_[j] == i]
        if cluster_sentences:
            point_summary.append(max(cluster_sentences, key=len))  # Kalimat terpanjang sebagai poin kunci

    # Ringkasan paragraf pendek
    paragraph_summary = ' '.join(point_summary)

    # Infografik: Visualisasi jumlah kata
    sentence_lengths = [len(sentence.split()) for sentence in point_summary]
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(point_summary) + 1), sentence_lengths, color='skyblue')
    plt.xlabel('Nomor Poin')
    plt.ylabel('Jumlah Kata')
    plt.title('Jumlah Kata per Poin Kunci')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return point_summary, paragraph_summary, buf

# Fungsi untuk menghasilkan ringkasan yang lebih panjang
def long_summary(article):
    sentences = re.split(r'(?<=[.!?]) +', article)
    return ' '.join(sentences)

# Fungsi untuk menerjemahkan artikel ke bahasa tertentu
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
        st.error(f'Terjemahan gagal: {e}')
        return None

# Fungsi untuk menghasilkan hashtag dari judul dan konten
def generate_hashtags(title, content, lang='en', num_hashtags=5):
    stop_words_set = stop_words.get(lang, set())
    title_words = [word for word in tokenizer.tokenize(title.lower()) if word.isalnum() and len(word) > 3 and word not in stop_words_set]
    content_words = [word for word in tokenizer.tokenize(content.lower()) if word.isalnum() and len(word) > 3 and word not in stop_words_set]
    
    # Gabungkan kata kunci dari judul dan konten
    keywords = title_words * 2 + content_words  # Menggandakan kata judul untuk meningkatkan bobot

    # Menghasilkan skor TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(keywords)])
    
    tfidf_scores = X.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    scored_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    top_keywords = [f"#{keyword.capitalize()}" for keyword, score in scored_keywords[:num_hashtags]]
    return top_keywords

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title('Aplikasi Ringkasan Berita & Pembuat Hashtag')

    # Menyimpan URL dan bahasa yang dipilih dalam state sesi
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'lang' not in st.session_state:
        st.session_state.lang = "en"

    st.session_state.url = st.text_input('Masukkan URL artikel berita:', st.session_state.url)
    st.session_state.lang = st.selectbox('Pilih bahasa untuk terjemahan:', ['en', 'id', 'es', 'fr'], index=['en', 'id', 'es', 'fr'].index(st.session_state.lang))

    if st.button('Ringkas dan Hasilkan Hashtag'):
        if st.session_state.url:
            if not (st.session_state.url.startswith('http://') or st.session_state.url.startswith('https://')):
                st.error('Silakan masukkan URL yang valid dimulai dengan http:// atau https://')
                return
            
            title, article = fetch_article(st.session_state.url)
            if article:
                if not article.strip():
                    st.error('Artikel kosong atau tidak dapat diambil.')
                    return

                # Terjemahkan judul jika perlu
                translated_title = translate_article(title, st.session_state.lang)
                st.subheader('Judul Artikel:')
                st.write(translated_title)

                # Terjemahkan artikel jika perlu
                translated_article = translate_article(article, st.session_state.lang)
                if translated_article is None:
                    return

                num_clusters = st.slider('Pilih jumlah cluster untuk ringkasan:', 1, 5, 2)
                point_summary, paragraph_summary, infographic_buf = summarize_article_flexible(translated_article, num_clusters)

                # Tampilkan opsi ringkasan fleksibel
                st.subheader('Opsi Ringkasan Fleksibel:')
                
                # Poin Kunci
                st.write("### Poin Kunci:")
                for idx, point in enumerate(point_summary, 1):
                    st.write(f"{idx}. {point}")

                # Paragraf Pendek
                st.write("### Paragraf Pendek:")
                st.write(paragraph_summary)

                # Ringkasan yang lebih panjang
                detailed_summary = long_summary(translated_article)
                st.write("### Ringkasan Detail:")
                st.write(detailed_summary)

                # Infografik
                st.write("### Infografik:")
                st.image(infographic_buf)

                # Hasilkan hashtag
                hashtags = generate_hashtags(translated_title, translated_article, st.session_state.lang)
                st.write("### Hashtag yang Dihasilkan:")
                st.write(", ".join(hashtags))

if __name__ == "__main__":
    main()
