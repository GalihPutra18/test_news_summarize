from nltk.tokenize import sent_tokenize

text = """galih suka

Artikel ini telah tayang di Kompas.com dengan judul "Presiden Prabowo Panggil Menkeu, Menko Perekonomian, dan Menaker, Bahas Sritex?", Klik untuk baca: https://nasional.kompas.com/read/2024/10/29/14020721/presiden-prabowo-panggil-menkeu-menko-perekonomian-dan-menaker-bahas-sritex.

Kompascom+ baca berita tanpa iklan: https://kmp.im/plus6
Download aplikasi: https://kmp.im/app6
"""

# Tokenisasi dengan bahasa Indonesia
sentences = sent_tokenize(text, language='indonesian')
print(sentences)
