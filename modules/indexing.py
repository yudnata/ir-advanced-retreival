import os
import re
import math
import numpy as np
from modules.preprocessing import preprocess


def natural_sort_key(filename):
    """Key untuk sorting natural: D1, D2, ..., D10."""
    parts = re.findall(r"\d+", filename)
    return int(parts[0]) if parts else 0


def load_documents(data_path):
    """Memuat semua dokumen .txt dari folder data."""
    files = sorted(
        [f for f in os.listdir(data_path) if f.endswith(".txt")], key=natural_sort_key
    )
    documents = {}
    for fname in files:
        filepath = os.path.join(data_path, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            documents[fname] = f.read().strip()
    return documents


def build_tf(documents):
    """Menghitung TF untuk setiap dokumen."""
    doc_tokens = {}
    tf_raw = {}
    tf_normalized = {}
    vocab = set()

    for doc_id, text in documents.items():
        tokens = preprocess(text)
        doc_tokens[doc_id] = tokens

        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        tf_raw[doc_id] = freq

        total_terms = len(tokens)
        tf_norm = {}
        for term, count in freq.items():
            tf_norm[term] = count / total_terms if total_terms > 0 else 0
        tf_normalized[doc_id] = tf_norm
        vocab.update(tokens)

    vocab_list = sorted(list(vocab))
    return doc_tokens, tf_raw, tf_normalized, vocab_list


def build_df(doc_tokens, vocab):
    """Menghitung Document Frequency (DF)."""
    df = {}
    for term in vocab:
        count = sum(1 for tokens in doc_tokens.values() if term in set(tokens))
        df[term] = count
    return df


def build_idf(df, total_docs):
    """Menghitung IDF(t) = log10(N / DF(t))."""
    idf = {}
    for term, df_val in df.items():
        idf[term] = math.log10(total_docs / df_val) if df_val > 0 else 0
    return idf


def build_tfidf_matrix(tf_normalized, idf, vocab, doc_ids):
    """Membuat matrix TF-IDF [len(vocab) x len(doc_ids)]."""
    matrix = np.zeros((len(vocab), len(doc_ids)))
    for i, term in enumerate(vocab):
        for j, doc_id in enumerate(doc_ids):
            tf_val = tf_normalized[doc_id].get(term, 0)
            idf_val = idf.get(term, 0)
            matrix[i][j] = round(tf_val * idf_val, 6)
    return matrix


def build_all_index(documents):
    """
    Membangun semua komponen index dari dokumen.
    Return: vocab, doc_ids, doc_tokens, tf_raw, tf_normalized, df, idf, tfidf_matrix
    """
    doc_tokens, tf_raw, tf_normalized, vocab = build_tf(documents)
    doc_ids = sorted(list(documents.keys()), key=natural_sort_key)
    total_docs = len(doc_ids)

    df = build_df(doc_tokens, vocab)
    idf = build_idf(df, total_docs)
    tfidf_matrix = build_tfidf_matrix(tf_normalized, idf, vocab, doc_ids)

    return vocab, doc_ids, doc_tokens, tf_raw, tf_normalized, df, idf, tfidf_matrix
