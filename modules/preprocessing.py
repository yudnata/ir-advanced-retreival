import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

sw_factory = StopWordRemoverFactory()
stopwords = sw_factory.get_stop_words()


def preprocess(text):
    """Preprocessing: lowercase -> hapus tanda baca -> tokenisasi -> stopword removal -> stemming"""
    text = text.lower()

    text = text.replace("persia", "persiax")

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]

    tokens = [t.replace("persiax", "persia") for t in tokens]
    tokens = [t for t in tokens if t.strip()]
    return tokens
