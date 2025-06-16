import re
import string
import unicodedata
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup Sastrawi (stopword + stemmer)
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

def clean_text(text, remove_stopwords=True):
    # Lowercase
    text = text.lower()

    # Normalisasi unicode (hapus karakter aneh)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # Hapus HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Hapus angka dan tanda baca
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Hapus stopwords (opsional)
    if remove_stopwords:
        text = stopword_remover.remove(text)

    return text
