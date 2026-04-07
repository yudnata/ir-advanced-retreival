import numpy as np
from modules.preprocessing import preprocess


def build_gvsm_term_vectors(tfidf_matrix, vocab, doc_ids):
    """
    Membangun representasi GVSM dengan dekomposisi term correlation.

    Dalam GVSM, term tidak lagi dianggap orthogonal satu sama lain.
    Korelasi antar term dihitung dari matrix TF-IDF (Minterm vectors).

    Minterm vector untuk setiap term diperoleh dari baris TF-IDF matrix
    yang dinormalisasi.
    """
    term_vectors = {}
    for i, term in enumerate(vocab):
        row = tfidf_matrix[i].copy()
        norm = np.linalg.norm(row)
        if norm > 0:
            term_vectors[term] = row / norm
        else:
            term_vectors[term] = row
    return term_vectors


def gvsm_query(query_str, vocab, doc_ids, idf, tfidf_matrix, term_vectors):
    """
    Melakukan pencarian menggunakan Generalized Vector Space Model (GVSM).

    Langkah-langkah:
    1. Preprocessing query
    2. Hitung vektor query di ruang term
    3. Hitung vektor dokumen di ruang minterm
    4. Hitung similarity menggunakan dot product antar vektor
    5. Ranking berdasarkan similarity score
    """
    query_tokens = preprocess(query_str)
    if not query_tokens:
        return [], {}, {}, [], "Query tidak menghasilkan token setelah preprocessing."

    n_docs = len(doc_ids)

    query_vec = np.zeros(n_docs)
    query_term_weights = {}
    for term in query_tokens:
        if term in vocab:
            idx = vocab.index(term)
            idf_val = idf.get(term, 0)
            query_term_weights[term] = idf_val
            query_vec += idf_val * term_vectors[term]
        else:
            query_term_weights[term] = 0.0

    q_norm = np.linalg.norm(query_vec)
    query_vec_normalized = query_vec / q_norm if q_norm > 0 else query_vec

    doc_vectors = {}
    for j, doc_id in enumerate(doc_ids):
        dv = np.zeros(n_docs)
        for i, term in enumerate(vocab):
            tfidf_val = tfidf_matrix[i, j]
            if tfidf_val > 0:
                dv += tfidf_val * term_vectors[term]
        norm = np.linalg.norm(dv)
        doc_vectors[doc_id] = dv / norm if norm > 0 else dv

    doc_scores = {}
    for j, doc_id in enumerate(doc_ids):
        sim = float(np.dot(query_vec_normalized, doc_vectors[doc_id]))
        doc_scores[doc_id] = round(max(sim, 0.0), 6)

    ranked_docs = sorted(
        [(doc_id, score) for doc_id, score in doc_scores.items() if score > 0],
        key=lambda x: -x[1],
    )

    term_contrib = {}
    for term in query_tokens:
        term_contrib[term] = {}
        if term in vocab:
            idf_val = idf.get(term, 0)
            idx = vocab.index(term)
            for j, doc_id in enumerate(doc_ids):
                tfidf_val = tfidf_matrix[idx][j]
                term_contrib[term][doc_id] = round(float(idf_val * tfidf_val), 6)
        else:
            for doc_id in doc_ids:
                term_contrib[term][doc_id] = 0.0

    detail_lines = []
    detail_lines.append("=" * 60)
    detail_lines.append("GENERALIZED VECTOR SPACE MODEL - STEP BY STEP")
    detail_lines.append("=" * 60)
    detail_lines.append(f'\nQuery Input  : "{query_str}"')
    detail_lines.append(f"Query Tokens : {query_tokens}")
    detail_lines.append("")
    detail_lines.append("─" * 60)
    detail_lines.append("STEP 1: BOBOT QUERY TERM (IDF-weighted)")
    detail_lines.append("─" * 60)
    detail_lines.append("Dalam GVSM, bobot query = IDF dari setiap term query")
    detail_lines.append("")
    for term in query_tokens:
        w = query_term_weights.get(term, 0)
        status = "✓ ada di vocab" if term in vocab else "✗ tidak ada di vocab"
        detail_lines.append(f'  w("{term}") = IDF = {w:.6f}   [{status}]')

    detail_lines.append("")
    detail_lines.append("─" * 60)
    detail_lines.append("STEP 2: TERM VECTORS (Minterm dalam ruang dokumen)")
    detail_lines.append("─" * 60)
    detail_lines.append(
        "Setiap term direpresentasikan sebagai vektor di ruang dokumen."
    )
    detail_lines.append("tv(term) = normalize(TF-IDF row of term)")
    detail_lines.append("")
    for term in query_tokens:
        if term in vocab:
            tv = term_vectors[term]
            tv_str = ", ".join([f"{v:.4f}" for v in tv])
            detail_lines.append(f'  tv("{term}") = [{tv_str}]')
        else:
            detail_lines.append(f'  tv("{term}") = [0, 0, ...] (tidak ada di vocab)')
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 3: QUERY VECTOR (agregasi term vectors)")
    detail_lines.append("─" * 60)
    detail_lines.append("Rumus: q = Σ (IDF(t) × tv(t)) untuk setiap term query")
    qv_str = ", ".join([f"{v:.4f}" for v in query_vec])
    qvn_str = ", ".join([f"{v:.4f}" for v in query_vec_normalized])
    detail_lines.append(f"  q (raw)        = [{qv_str}]")
    detail_lines.append(f"  ||q||          = {q_norm:.6f}")
    detail_lines.append(f"  q (normalized) = [{qvn_str}]")
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 4: DOCUMENT VECTORS (agregasi term vectors dokumen)")
    detail_lines.append("─" * 60)
    detail_lines.append("Rumus: dv = Σ (TF-IDF(t, d) × tv(t)) untuk setiap term di d")
    for doc_id in doc_ids:
        label = doc_id.replace(".txt", "")
        dv = doc_vectors[doc_id]
        dv_str = ", ".join([f"{v:.4f}" for v in dv[:8]])
        suffix = ", ..." if len(dv) > 8 else ""
        detail_lines.append(f"  dv({label}) = [{dv_str}{suffix}]")
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 5: SIMILARITY = dot(q_normalized, dv(doc))")
    detail_lines.append("─" * 60)
    detail_lines.append("")
    for doc_id in doc_ids:
        label = doc_id.replace(".txt", "")
        score = doc_scores[doc_id]
        detail_lines.append(f"  sim(q, {label}) = {score:.6f}")

    detail_lines.append("")
    detail_lines.append("─" * 60)
    detail_lines.append("STEP 6: RANKING DOKUMEN")
    detail_lines.append("─" * 60)
    detail_lines.append("")
    if ranked_docs:
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            label = doc_id.replace(".txt", "")
            detail_lines.append(f"  #{rank}  {label}  →  Similarity = {score:.6f}")
    else:
        detail_lines.append("  Tidak ada dokumen yang relevan ditemukan.")

    return query_tokens, term_contrib, doc_scores, ranked_docs, "\n".join(detail_lines)
