import numpy as np
from modules.preprocessing import preprocess


def build_lsi_model(tfidf_matrix, vocab, doc_ids, k=None):
    """
    Membangun model LSI menggunakan Singular Value Decomposition (SVD).

    TF-IDF matrix A (m x n) didekomposisi menjadi:
    A ≈ U_k × Σ_k × V_k^T

    Di mana:
    - U_k: matrix term-concept (m x k)
    - Σ_k: matrix singular value diagonal (k x k)
    - V_k^T: matrix concept-document (k x n)
    - k: jumlah dimensi laten (topik)
    """
    A = tfidf_matrix  # shape: (m_terms, n_docs)

    min_dim = min(A.shape)
    if k is None:
        k = min(min_dim, max(2, min(10, min_dim // 2)))
    k = max(1, min(k, min_dim))

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    S_k = np.diag(s_k)

    total_var = np.sum(s**2)
    explained_var = np.sum(s_k**2)
    explained_ratio = explained_var / total_var if total_var > 0 else 0

    return U_k, s_k, Vt_k, S_k, k, explained_ratio, s


def lsi_query(query_str, vocab, doc_ids, idf, tfidf_matrix, U_k, s_k, Vt_k, k):
    """
    Melakukan pencarian menggunakan Latent Semantic Indexing (LSI).

    Langkah-langkah:
    1. Preprocessing query
    2. Representasikan query dalam ruang TF-IDF (query vector)
    3. Proyeksikan query ke ruang laten k-dimensi: q_k = q^T × U_k × Σ_k^{-1}
    4. Hitung cosine similarity antara query_k dengan setiap dokumen di ruang laten (Vt_k)
    5. Ranking berdasarkan cosine similarity
    """
    query_tokens = preprocess(query_str)
    if not query_tokens:
        return [], {}, {}, [], "Query tidak menghasilkan token setelah preprocessing."

    n_terms = len(vocab)
    n_docs = len(doc_ids)

    q_vec = np.zeros(n_terms)
    query_term_weights = {}
    for term in query_tokens:
        if term in vocab:
            idx = vocab.index(term)
            idf_val = idf.get(term, 0)
            q_vec[idx] += idf_val
            query_term_weights[term] = idf_val
        else:
            query_term_weights[term] = 0.0

    s_inv = np.where(s_k > 1e-10, 1.0 / s_k, 0.0)
    q_k = q_vec @ U_k * s_inv

    doc_vectors_laten = Vt_k.T

    doc_scores = {}
    q_norm = np.linalg.norm(q_k)
    for j, doc_id in enumerate(doc_ids):
        dv = doc_vectors_laten[j]
        d_norm = np.linalg.norm(dv)
        if q_norm > 0 and d_norm > 0:
            sim = float(np.dot(q_k, dv) / (q_norm * d_norm))
        else:
            sim = 0.0
        doc_scores[doc_id] = round(max(sim, 0.0), 6)

    ranked_docs = sorted(
        [(doc_id, score) for doc_id, score in doc_scores.items() if score > 0],
        key=lambda x: -x[1],
    )

    detail_lines = []
    detail_lines.append("=" * 60)
    detail_lines.append("LATENT SEMANTIC INDEXING (LSI) - STEP BY STEP")
    detail_lines.append("=" * 60)
    detail_lines.append(f'\nQuery Input  : "{query_str}"')
    detail_lines.append(f"Query Tokens : {query_tokens}")
    detail_lines.append(f"Dimensi k    : {k}")
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 1: SINGULAR VALUE DECOMPOSITION (SVD)")
    detail_lines.append("─" * 60)
    detail_lines.append(
        f"Matrix A (TF-IDF) berukuran: {n_terms} term × {n_docs} dokumen"
    )
    detail_lines.append(f"Didekomposisi menjadi: A = U × Σ × Vt")
    detail_lines.append(f"  U  : matrix term-concept  ({n_terms} × {k})")
    detail_lines.append(f"  Σ  : singular values      ({k} × {k})")
    detail_lines.append(f"  Vt : concept-document     ({k} × {n_docs})")
    detail_lines.append("")

    s_k_str = ", ".join([f"{v:.4f}" for v in s_k])
    detail_lines.append(f"Singular Values (k={k}): [{s_k_str}]")
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 2: QUERY VECTOR DI RUANG TERM")
    detail_lines.append("─" * 60)
    detail_lines.append("Bobot query menggunakan IDF:")
    detail_lines.append("")
    for term in query_tokens:
        w = query_term_weights.get(term, 0)
        status = "✓" if term in vocab else "✗ tidak ada di vocab"
        detail_lines.append(f'  w("{term}") = {w:.6f}   {status}')

    q_nonzero = [(vocab[i], q_vec[i]) for i in range(n_terms) if q_vec[i] > 0]
    detail_lines.append("")
    detail_lines.append("  Query vector non-zero entries:")
    for t, v in q_nonzero:
        detail_lines.append(f'    "{t}": {v:.6f}')
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 3: PROYEKSI QUERY KE RUANG LATEN")
    detail_lines.append("─" * 60)
    detail_lines.append("Rumus: q_k = q^T × U_k × Σ_k^{-1}")
    detail_lines.append("")
    q_k_str = ", ".join([f"{v:.4f}" for v in q_k])
    detail_lines.append(f"  q_k = [{q_k_str}]")
    detail_lines.append(f"  ||q_k|| = {q_norm:.6f}")
    detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 4: COSINE SIMILARITY Q_K vs DOCUMENT VECTORS")
    detail_lines.append("─" * 60)
    detail_lines.append("Rumus: sim(q, d) = (q_k · dv_k) / (||q_k|| × ||dv_k||)")
    detail_lines.append("")
    for j, doc_id in enumerate(doc_ids):
        label = doc_id.replace(".txt", "")
        dv = doc_vectors_laten[j]
        dv_str = ", ".join([f"{v:.4f}" for v in dv])
        d_norm = np.linalg.norm(dv)
        score = doc_scores[doc_id]
        detail_lines.append(f"  dv({label}) = [{dv_str}]")
        detail_lines.append(f"  ||dv({label})|| = {d_norm:.6f}")
        detail_lines.append(f"  sim(q, {label}) = {score:.6f}")
        detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 5: RANKING DOKUMEN")
    detail_lines.append("─" * 60)
    detail_lines.append("")
    if ranked_docs:
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            label = doc_id.replace(".txt", "")
            detail_lines.append(
                f"  #{rank}  {label}  →  Cosine Similarity = {score:.6f}"
            )
    else:
        detail_lines.append("  Tidak ada dokumen yang relevan ditemukan.")

    return (
        query_tokens,
        query_term_weights,
        doc_scores,
        ranked_docs,
        "\n".join(detail_lines),
    )
