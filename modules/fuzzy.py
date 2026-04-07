import numpy as np
from modules.preprocessing import preprocess


def build_fuzzy_membership(tfidf_matrix, vocab, doc_ids):
    """
    Membangun matrix keanggotaan fuzzy dari TF-IDF.
    Membership µ(t, d) = TF-IDF(t,d) / max TF-IDF(t) di semua dokumen
    Nilai membership berkisar [0, 1].
    """
    membership = np.zeros_like(tfidf_matrix)
    for i in range(len(vocab)):
        row = tfidf_matrix[i]
        max_val = row.max()
        if max_val > 0:
            membership[i] = row / max_val
        else:
            membership[i] = row  # tetap 0
    return membership


def fuzzy_query(
    query_str, vocab, doc_ids, idf, tfidf_matrix, membership_matrix, threshold=0.0
):
    """
    Melakukan pencarian menggunakan Fuzzy Information Retrieval.

    Langkah-langkah:
    1. Preprocessing query
    2. Hitung membership query term di setiap dokumen
    3. Agregasi dengan operator fuzzy (MAX-MIN / rata-rata)
    4. Ranking berdasarkan skor keanggotaan

    Args:
        threshold: batas minimum membership (default 0.0 = tampilkan semua)
    """
    query_tokens = preprocess(query_str)
    if not query_tokens:
        return [], {}, {}, [], "Query tidak menghasilkan token setelah preprocessing."

    # Step 1: Hitung membership setiap term query di setiap dokumen
    term_memberships = {}
    for term in query_tokens:
        term_memberships[term] = {}
        if term in vocab:
            idx = vocab.index(term)
            for j, doc_id in enumerate(doc_ids):
                mu = membership_matrix[idx][j]
                term_memberships[term][doc_id] = round(float(mu), 6)
        else:
            for doc_id in doc_ids:
                term_memberships[term][doc_id] = 0.0

    doc_scores = {}
    for doc_id in doc_ids:
        values = [term_memberships[term].get(doc_id, 0.0) for term in query_tokens]
        # Fuzzy aggregation: rata-rata dari membership (soft AND)
        doc_scores[doc_id] = round(float(np.mean(values)), 6)

    ranked_docs = sorted(
        [(doc_id, score) for doc_id, score in doc_scores.items() if score > threshold],
        key=lambda x: -x[1],
    )

    detail_lines = []
    detail_lines.append("=" * 60)
    detail_lines.append("FUZZY INFORMATION RETRIEVAL - STEP BY STEP")
    detail_lines.append("=" * 60)
    detail_lines.append(f'\nQuery Input    : "{query_str}"')
    detail_lines.append(f"Query Tokens   : {query_tokens}")
    detail_lines.append(f"Threshold (µ)  : {threshold}")
    detail_lines.append("")
    detail_lines.append("─" * 60)
    detail_lines.append("STEP 1: HITUNG FUZZY MEMBERSHIP PER TERM")
    detail_lines.append("─" * 60)
    detail_lines.append("Rumus: µ(term, doc) = TF-IDF(term,doc) / max(TF-IDF(term))")
    detail_lines.append("")

    for term in query_tokens:
        if term in vocab:
            idx = vocab.index(term)
            tfidf_row = tfidf_matrix[idx]
            max_val = tfidf_row.max()
            detail_lines.append(f'  Term: "{term}"')
            detail_lines.append(f"    max TF-IDF = {max_val:.6f}")
            for j, doc_id in enumerate(doc_ids):
                tfidf_val = tfidf_matrix[idx][j]
                mu_val = term_memberships[term][doc_id]
                label = doc_id.replace(".txt", "")
                detail_lines.append(
                    f'    µ("{term}", {label}) = {tfidf_val:.6f} / {max_val:.6f} = {mu_val:.6f}'
                )
        else:
            detail_lines.append(
                f'  Term: "{term}" → TIDAK ADA dalam vocabulary (µ = 0)'
            )
        detail_lines.append("")

    detail_lines.append("─" * 60)
    detail_lines.append("STEP 2: FUZZY AGGREGATION (Rata-rata membership)")
    detail_lines.append("─" * 60)
    detail_lines.append("Rumus: Score(doc) = MEAN(µ(t1,doc), µ(t2,doc), ...)")
    detail_lines.append("")

    for doc_id in doc_ids:
        label = doc_id.replace(".txt", "")
        values = [term_memberships[t].get(doc_id, 0.0) for t in query_tokens]
        parts = " + ".join([f"{v:.6f}" for v in values])
        n = len(query_tokens)
        score = doc_scores[doc_id]
        detail_lines.append(f"  Score({label}) = ({parts}) / {n} = {score:.6f}")

    detail_lines.append("")
    detail_lines.append("─" * 60)
    detail_lines.append("STEP 3: RANKING DOKUMEN (Score > threshold)")
    detail_lines.append("─" * 60)
    detail_lines.append("")

    if ranked_docs:
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            label = doc_id.replace(".txt", "")
            detail_lines.append(f"  #{rank}  {label}  →  Score = {score:.6f}")
    else:
        detail_lines.append("  Tidak ada dokumen yang relevan (score = 0 semua).")

    return (
        query_tokens,
        term_memberships,
        doc_scores,
        ranked_docs,
        "\n".join(detail_lines),
    )
