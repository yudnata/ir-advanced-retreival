import tkinter as tk
from tkinter import ttk
import re
import numpy as np

from modules.fuzzy import fuzzy_query
from modules.gvsm import gvsm_query
from modules.lsi import lsi_query


def create_stat_card(parent, title, value, icon, colors):
    card = tk.Frame(
        parent,
        bg=colors["bg_card"],
        padx=25,
        pady=25,
        highlightbackground=colors["border"],
        highlightthickness=1,
    )
    card.pack(side=tk.LEFT, padx=(0, 20), expand=True, fill=tk.BOTH)
    tk.Label(card, text=icon, font=("Inter", 24), bg=colors["bg_card"]).pack(anchor="w")
    tk.Label(
        card,
        text=value,
        font=("Inter", 28, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_card"],
    ).pack(anchor="w", pady=(10, 0))
    tk.Label(
        card,
        text=title,
        font=("Inter", 11, "bold"),
        fg=colors["text_secondary"],
        bg=colors["bg_card"],
    ).pack(anchor="w")


def create_feature_card(parent, title, description, color, command, colors):
    card = tk.Frame(
        parent,
        bg=colors["bg_card"],
        padx=30,
        pady=30,
        highlightbackground=colors["border"],
        highlightthickness=1,
    )
    card.pack(fill=tk.X, pady=(0, 20))
    text_frame = tk.Frame(card, bg=colors["bg_card"])
    text_frame.pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(
        text_frame,
        text=title,
        font=("Inter", 16, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_card"],
    ).pack(anchor="w")
    tk.Label(
        text_frame,
        text=description,
        font=("Inter", 11),
        fg=colors["text_secondary"],
        bg=colors["bg_card"],
        wraplength=500,
        justify=tk.LEFT,
    ).pack(anchor="w", pady=(5, 0))
    btn = tk.Button(
        card,
        text="Buka →",
        font=("Inter", 11, "bold"),
        fg=colors["white"],
        bg=color,
        activebackground=colors["accent_hover"],
        activeforeground=colors["white"],
        bd=0,
        padx=20,
        pady=10,
        cursor="hand2",
        command=command,
    )
    btn.pack(side=tk.RIGHT, padx=10)


def scrollable_canvas(parent, colors):
    """Helper: buat canvas + scrollbar + frame scrollable."""
    canvas = tk.Canvas(parent, bg=colors["bg_main"], highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas, bg=colors["bg_main"])
    frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    win_id = canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return frame


def build_dashboard(
    main_area, colors, doc_ids, vocab, cmd_fuzzy, cmd_gvsm, cmd_lsi, cmd_docs
):
    frame = scrollable_canvas(main_area, colors)
    container = tk.Frame(frame, bg=colors["bg_main"], padx=60, pady=50)
    container.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        container,
        text="Selamat Datang!",
        font=("Inter", 32, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_main"],
    ).pack(anchor="w")
    tk.Label(
        container,
        text="Advanced Information Retrieval — Fuzzy, Generalized Vector Space Model, dan Latent Semantic Indexing.",
        font=("Inter", 13),
        fg=colors["text_secondary"],
        bg=colors["bg_main"],
    ).pack(anchor="w", pady=(5, 30))

    stats_frame = tk.Frame(container, bg=colors["bg_main"])
    stats_frame.pack(fill=tk.X, pady=(0, 40))
    create_stat_card(stats_frame, "Total Dokumen", f"{len(doc_ids)}", "📄", colors)
    create_stat_card(stats_frame, "Total Terms (Unik)", f"{len(vocab)}", "🔤", colors)

    tk.Label(
        container,
        text="Metode Information Retrieval",
        font=("Inter", 18, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_main"],
    ).pack(anchor="w", pady=(0, 20))

    card_container = tk.Frame(container, bg=colors["bg_main"])
    card_container.pack(fill=tk.X)

    create_feature_card(
        card_container,
        "🌫️Fuzzy IR",
        "Pencarian berbasis teori himpunan fuzzy. Dokumen memiliki derajat keanggotaan [0,1] "
        "terhadap query, bukan sekadar relevan/tidak relevan.",
        colors["accent_blue"],
        cmd_fuzzy,
        colors,
    )
    create_feature_card(
        card_container,
        "📐  Generalized Vector Space Model",
        "Perluasan VSM klasik yang mempertimbangkan korelasi antar term. "
        "Term tidak lagi dianggap orthogonal sehingga hubungan semantik antar kata ikut diperhitungkan.",
        colors["accent_green"],
        cmd_gvsm,
        colors,
    )
    create_feature_card(
        card_container,
        "🧠  Latent Semantic Indexing",
        "Menggunakan Singular Value Decomposition (SVD) untuk menemukan topik laten. "
        "Mampu menangani sinonim dan polisemi dengan mereduksi dimensi ruang vektor.",
        colors["accent_purple"],
        cmd_lsi,
        colors,
    )
    create_feature_card(
        card_container,
        "📂  Daftar Dokumen",
        "Lihat isi lengkap semua dokumen yang tersimpan dalam corpus.",
        colors["text_secondary"],
        cmd_docs,
        colors,
    )


def build_documents_view(main_area, colors, doc_ids, documents):
    container = tk.Frame(main_area, bg=colors["bg_main"], padx=30, pady=20)
    container.pack(fill=tk.BOTH, expand=True)

    list_frame = tk.Frame(
        container,
        bg=colors["white"],
        padx=15,
        pady=15,
        highlightbackground=colors["border"],
        highlightthickness=1,
    )
    list_frame.pack(side=tk.LEFT, fill=tk.Y)
    list_frame.config(width=260)
    list_frame.pack_propagate(False)

    tk.Label(
        list_frame,
        text="📑 Pilih Dokumen:",
        font=("Inter", 12, "bold"),
        bg=colors["white"],
        fg=colors["text_primary"],
    ).pack(anchor="w", pady=(0, 15))

    list_scroll = tk.Scrollbar(list_frame)
    list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    listbox = tk.Listbox(
        list_frame,
        font=("Inter", 11),
        bd=0,
        highlightthickness=0,
        selectbackground=colors["accent_blue"],
        selectforeground=colors["white"],
        activestyle="none",
        yscrollcommand=list_scroll.set,
    )
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    list_scroll.config(command=listbox.yview)

    for doc_name in doc_ids:
        listbox.insert(tk.END, f" 📄 {doc_name}")

    content_frame = tk.Frame(
        container,
        bg=colors["white"],
        padx=30,
        pady=30,
        highlightbackground=colors["border"],
        highlightthickness=1,
    )
    content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

    header_frame = tk.Frame(content_frame, bg=colors["white"])
    header_frame.pack(fill=tk.X, pady=(0, 20))
    doc_title_lbl = tk.Label(
        header_frame,
        text="Isi Dokumen",
        font=("Inter", 16, "bold"),
        bg=colors["white"],
        fg=colors["text_primary"],
    )
    doc_title_lbl.pack(side=tk.LEFT)
    doc_meta_lbl = tk.Label(
        header_frame,
        text="0 Kata",
        font=("Inter", 11, "bold"),
        bg=colors["bg_main"],
        fg=colors["text_secondary"],
        padx=12,
        pady=5,
    )
    doc_meta_lbl.pack(side=tk.RIGHT)

    text_scroll = tk.Scrollbar(content_frame)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    doc_text_area = tk.Text(
        content_frame,
        font=("Consolas", 12),
        bd=0,
        padx=20,
        pady=20,
        bg="#F8FAFC",
        fg=colors["text_primary"],
        wrap=tk.WORD,
        state=tk.DISABLED,
        yscrollcommand=text_scroll.set,
        spacing1=5,
        spacing2=5,
        spacing3=5,
    )
    doc_text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text_scroll.config(command=doc_text_area.yview)

    def on_select(evt):
        if not listbox.curselection():
            return
        index = int(listbox.curselection()[0])
        doc_name = doc_ids[index]
        content = documents[doc_name]
        doc_title_lbl.config(text=f"Isi: {doc_name}")
        doc_meta_lbl.config(text=f" {len(content.split())} Kata ")
        doc_text_area.config(state=tk.NORMAL)
        doc_text_area.delete("1.0", tk.END)
        doc_text_area.insert(tk.END, content)
        doc_text_area.config(state=tk.DISABLED)

    listbox.bind("<<ListboxSelect>>", on_select)
    if doc_ids:
        listbox.selection_set(0)
        on_select(None)


def _make_query_header(parent, colors, method_name, method_desc, header_color):
    header = tk.Frame(parent, bg=header_color, pady=12, padx=20)
    header.pack(fill=tk.X)
    tk.Label(
        header,
        text=f"🔍 {method_name}",
        font=("Inter", 14, "bold"),
        fg="white",
        bg=header_color,
    ).pack(anchor="w")
    tk.Label(
        header, text=method_desc, font=("Inter", 10), fg="#B3B9D1", bg=header_color
    ).pack(anchor="w")


def _make_result_tabs(
    parent,
    colors,
    query_tokens,
    term_data,
    doc_scores,
    ranked_docs,
    documents,
    doc_ids,
    detail_str,
    term_col_label="Bobot per Term",
):
    """
    Membuat 3 tab hasil:
    1. Detail Perhitungan (step by step)
    2. Skor per Term per Dokumen
    3. Dokumen Relevan
    """
    result_notebook = ttk.Notebook(parent)
    result_notebook.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

    tab_detail = tk.Frame(result_notebook, bg=colors["white"])
    result_notebook.add(tab_detail, text="📋 Detail Perhitungan")

    detail_scroll = tk.Scrollbar(tab_detail, orient=tk.VERTICAL)
    detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    detail_text = tk.Text(
        tab_detail,
        wrap=tk.WORD,
        font=("Consolas", 10),
        yscrollcommand=detail_scroll.set,
        state="normal",
        bg="#F5F5F5",
        padx=15,
        pady=15,
    )
    detail_scroll.config(command=detail_text.yview)
    detail_text.insert(tk.END, detail_str)
    detail_text.config(state="disabled")
    detail_text.pack(fill=tk.BOTH, expand=True)

    tab_scores = tk.Frame(result_notebook, bg=colors["white"])
    result_notebook.add(tab_scores, text=f"📊 {term_col_label}")

    info_frame = tk.Frame(tab_scores, bg="#E8F4FD", pady=6, padx=15)
    info_frame.pack(fill=tk.X)
    tk.Label(
        info_frame,
        text="Nilai bobot setiap kata query di setiap dokumen",
        font=("Inter", 9, "bold"),
        bg="#E8F4FD",
        fg="#1565C0",
    ).pack(anchor="w")

    scores_frame = tk.Frame(tab_scores, bg=colors["white"], padx=10, pady=10)
    scores_frame.pack(fill=tk.BOTH, expand=True)
    sy = tk.Scrollbar(scores_frame, orient=tk.VERTICAL)
    sy.pack(side=tk.RIGHT, fill=tk.Y)
    sx = tk.Scrollbar(scores_frame, orient=tk.HORIZONTAL)
    sx.pack(side=tk.BOTTOM, fill=tk.X)

    doc_labels = [d.replace(".txt", "") for d in doc_ids]
    cols = ["term"] + doc_labels + ["skor_total"]
    tree = ttk.Treeview(
        scores_frame,
        columns=cols,
        show="headings",
        yscrollcommand=sy.set,
        xscrollcommand=sx.set,
    )
    sy.config(command=tree.yview)
    sx.config(command=tree.xview)

    tree.heading("term", text="Query Term", anchor="center")
    tree.column("term", width=130, anchor="center")
    for dl in doc_labels:
        tree.heading(dl, text=dl, anchor="center")
        tree.column(dl, width=80, anchor="center")
    tree.heading("skor_total", text="Skor Total", anchor="center")
    tree.column("skor_total", width=100, anchor="center")

    tree.tag_configure("even", background="#FFFFFF")
    tree.tag_configure("odd", background="#F8FAFC")
    tree.tag_configure("total_row", background="#E8F5E9", font=("Inter", 10, "bold"))

    for i, term in enumerate(query_tokens):
        row = [term]
        for doc_id in doc_ids:
            val = term_data.get(term, {}).get(doc_id, 0.0)
            row.append(f"{val:.4f}" if val > 0 else "0")
        row.append("")
        tag = "even" if i % 2 == 0 else "odd"
        tree.insert("", tk.END, values=row, tags=(tag,))

    total_row = ["TOTAL SKOR"]
    for doc_id in doc_ids:
        total_row.append(f"{doc_scores.get(doc_id, 0):.4f}")
    if ranked_docs:
        total_row.append(f"🏆 {ranked_docs[0][0].replace('.txt','')}")
    else:
        total_row.append("-")
    tree.insert("", tk.END, values=total_row, tags=("total_row",))
    tree.pack(fill=tk.BOTH, expand=True)

    tab_docs = tk.Frame(result_notebook, bg=colors["white"])
    result_notebook.add(tab_docs, text=f"📄 Dokumen Relevan ({len(ranked_docs)})")

    doc_scroll = tk.Scrollbar(tab_docs, orient=tk.VERTICAL)
    doc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    doc_text = tk.Text(
        tab_docs,
        wrap=tk.WORD,
        font=("Inter", 11),
        yscrollcommand=doc_scroll.set,
        state="normal",
        bg="#FAFAFA",
        padx=15,
        pady=15,
    )
    doc_scroll.config(command=doc_text.yview)

    doc_text.tag_configure("doc_id", font=("Inter", 13, "bold"), foreground="#1A237E")
    doc_text.tag_configure("score", font=("Consolas", 11, "bold"), foreground="#2E7D32")
    doc_text.tag_configure(
        "highlight",
        font=("Inter", 11, "bold"),
        background="#FFFF00",
        foreground="black",
    )
    doc_text.tag_configure("separator", foreground="#CCCCCC")
    doc_text.tag_configure("rank", font=("Inter", 12, "bold"), foreground="#D84315")

    if ranked_docs:
        escaped = [re.escape(w) for w in query_tokens if w]
        pattern = (
            re.compile(f"({'|'.join(escaped)})", re.IGNORECASE) if escaped else None
        )

        for rank_num, (doc_id, score) in enumerate(ranked_docs, 1):
            content = documents[doc_id]
            doc_text.insert(tk.END, f"  #{rank_num}  ", "rank")
            doc_text.insert(tk.END, f"📄 {doc_id}", "doc_id")
            doc_text.insert(tk.END, f"   [Skor: {score:.6f}]\n", "score")

            start_idx = doc_text.index(tk.INSERT)
            doc_text.insert(tk.END, f"{content}\n")
            if pattern:
                for m in pattern.finditer(content):
                    doc_text.tag_add(
                        "highlight",
                        f"{start_idx}+{m.start()}c",
                        f"{start_idx}+{m.end()}c",
                    )
            doc_text.insert(tk.END, "\n" + "─" * 100 + "\n\n", "separator")
    else:
        doc_text.insert(tk.END, "Tidak ada dokumen yang relevan ditemukan.\n")

    doc_text.config(state="disabled")
    doc_text.pack(fill=tk.BOTH, expand=True)

    return result_notebook


def build_fuzzy_view(
    parent, colors, vocab, doc_ids, idf, tfidf_matrix, membership_matrix, documents
):
    _make_query_header(
        parent,
        colors,
        "FUZZY INFORMATION RETRIEVAL",
        "Pencarian berbasis membership function — setiap dokumen memiliki derajat keanggotaan [0,1]",
        "#1A237E",
    )

    input_frame = tk.Frame(parent, padx=20, pady=15, bg=colors["bg_main"])
    input_frame.pack(fill=tk.X)

    tk.Label(
        input_frame,
        text="Masukkan Query:",
        font=("Inter", 12, "bold"),
        bg=colors["bg_main"],
        fg=colors["text_primary"],
    ).pack(anchor="w")
    tk.Label(
        input_frame,
        text='Contoh: "kucing persia", "bulu lebat", "vaksin kesehatan", "pemburu insting"',
        font=("Inter", 9),
        fg="#888888",
        bg=colors["bg_main"],
    ).pack(anchor="w", pady=(0, 8))

    entry_frame = tk.Frame(input_frame, bg=colors["bg_main"])
    entry_frame.pack(fill=tk.X)

    query_entry = tk.Entry(entry_frame, font=("Consolas", 14), width=50)
    query_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=6)

    tk.Label(
        entry_frame,
        text="Threshold µ:",
        font=("Inter", 10),
        bg=colors["bg_main"],
        fg=colors["text_primary"],
    ).pack(side=tk.LEFT, padx=(10, 5))
    threshold_var = tk.DoubleVar(value=0.0)
    threshold_slider = tk.Scale(
        entry_frame,
        from_=0.0,
        to=0.5,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=threshold_var,
        bg=colors["bg_main"],
        length=120,
    )
    threshold_slider.pack(side=tk.LEFT)

    result_area = tk.Frame(parent, bg=colors["bg_main"], padx=20, pady=5)
    result_area.pack(fill=tk.BOTH, expand=True)

    result_info = tk.Label(
        result_area,
        text="",
        font=("Inter", 12, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_main"],
    )
    result_info.pack(anchor="w", pady=(0, 5))

    result_content = tk.Frame(result_area, bg=colors["bg_main"])
    result_content.pack(fill=tk.BOTH, expand=True)

    def execute_search(event=None):
        for w in result_content.winfo_children():
            w.destroy()
        query = query_entry.get().strip()
        if not query:
            result_info.config(text="⚠️ Query kosong!", fg="red")
            return
        threshold = threshold_var.get()
        tokens, term_mems, doc_scores, ranked, detail = fuzzy_query(
            query, vocab, doc_ids, idf, tfidf_matrix, membership_matrix, threshold
        )
        if not tokens:
            result_info.config(
                text="⚠️ Tidak ada token setelah preprocessing.", fg="red"
            )
            return
        if ranked:
            result_info.config(
                text=f'✅ {len(ranked)} dokumen relevan untuk "{query}" | threshold={threshold}',
                fg="#2E7D32",
            )
        else:
            result_info.config(
                text=f'❌ Tidak ada dokumen dengan skor > {threshold} untuk "{query}"',
                fg="red",
            )
        _make_result_tabs(
            result_content,
            colors,
            tokens,
            term_mems,
            doc_scores,
            ranked,
            documents,
            doc_ids,
            detail,
            "Membership (µ)",
        )

    btn = tk.Button(
        entry_frame,
        text="🔍 Cari",
        font=("Inter", 12, "bold"),
        bg="#1A237E",
        fg="white",
        padx=20,
        pady=5,
        cursor="hand2",
        command=execute_search,
    )
    btn.pack(side=tk.LEFT, padx=(10, 0))
    query_entry.bind("<Return>", execute_search)


def build_gvsm_view(
    parent, colors, vocab, doc_ids, idf, tfidf_matrix, term_vectors, documents
):
    _make_query_header(
        parent,
        colors,
        "GENERALIZED VECTOR SPACE MODEL",
        "Perluasan VSM — term tidak orthogonal, korelasi antar kata ikut diperhitungkan",
        "#1B5E20",
    )

    input_frame = tk.Frame(parent, padx=20, pady=15, bg=colors["bg_main"])
    input_frame.pack(fill=tk.X)

    tk.Label(
        input_frame,
        text="Masukkan Query:",
        font=("Inter", 12, "bold"),
        bg=colors["bg_main"],
        fg=colors["text_primary"],
    ).pack(anchor="w")
    tk.Label(
        input_frame,
        text='Contoh: "kucing persia", "bulu lebat", "vaksin kesehatan", "mendengkur pemburu"',
        font=("Inter", 9),
        fg="#888888",
        bg=colors["bg_main"],
    ).pack(anchor="w", pady=(0, 8))

    entry_frame = tk.Frame(input_frame, bg=colors["bg_main"])
    entry_frame.pack(fill=tk.X)

    query_entry = tk.Entry(entry_frame, font=("Consolas", 14), width=55)
    query_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=6)

    result_area = tk.Frame(parent, bg=colors["bg_main"], padx=20, pady=5)
    result_area.pack(fill=tk.BOTH, expand=True)

    result_info = tk.Label(
        result_area,
        text="",
        font=("Inter", 12, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_main"],
    )
    result_info.pack(anchor="w", pady=(0, 5))

    result_content = tk.Frame(result_area, bg=colors["bg_main"])
    result_content.pack(fill=tk.BOTH, expand=True)

    def execute_search(event=None):
        for w in result_content.winfo_children():
            w.destroy()
        query = query_entry.get().strip()
        if not query:
            result_info.config(text="⚠️ Query kosong!", fg="red")
            return
        tokens, term_contrib, doc_scores, ranked, detail = gvsm_query(
            query, vocab, doc_ids, idf, tfidf_matrix, term_vectors
        )
        if not tokens:
            result_info.config(
                text="⚠️ Tidak ada token setelah preprocessing.", fg="red"
            )
            return
        if ranked:
            result_info.config(
                text=f'✅ {len(ranked)} dokumen relevan untuk "{query}" (GVSM Similarity)',
                fg="#2E7D32",
            )
        else:
            result_info.config(
                text=f'❌ Tidak ada dokumen relevan untuk "{query}"', fg="red"
            )
        _make_result_tabs(
            result_content,
            colors,
            tokens,
            term_contrib,
            doc_scores,
            ranked,
            documents,
            doc_ids,
            detail,
            "Kontribusi Term (GVSM)",
        )

    btn = tk.Button(
        entry_frame,
        text="🔍 Cari",
        font=("Inter", 12, "bold"),
        bg="#1B5E20",
        fg="white",
        padx=20,
        pady=5,
        cursor="hand2",
        command=execute_search,
    )
    btn.pack(side=tk.LEFT)
    query_entry.bind("<Return>", execute_search)


def build_lsi_view(
    parent,
    colors,
    vocab,
    doc_ids,
    idf,
    tfidf_matrix,
    U_k,
    s_k,
    Vt_k,
    k,
    explained_ratio,
    s_all,
    documents,
):
    _make_query_header(
        parent,
        colors,
        "LATENT SEMANTIC INDEXING (LSI)",
        f"Menggunakan SVD dengan k={k} dimensi laten · Explained variance: {explained_ratio*100:.1f}%",
        "#4A148C",
    )

    input_frame = tk.Frame(parent, padx=20, pady=15, bg=colors["bg_main"])
    input_frame.pack(fill=tk.X)

    info_bar = tk.Frame(
        input_frame,
        bg="#EDE7F6",
        pady=8,
        padx=15,
        highlightbackground="#7B1FA2",
        highlightthickness=1,
    )
    info_bar.pack(fill=tk.X, pady=(0, 12))

    sv_str = " | ".join(
        [f"σ{i+1}={v:.3f}" for i, v in enumerate(s_all[: min(5, len(s_all))])]
    )
    tk.Label(
        info_bar,
        text=f"SVD Info:  k={k} dimensi  |  Explained variance={explained_ratio*100:.1f}%  |  {sv_str}",
        font=("Inter", 9, "bold"),
        bg="#EDE7F6",
        fg="#4A148C",
    ).pack(anchor="w")

    tk.Label(
        input_frame,
        text="Masukkan Query:",
        font=("Inter", 12, "bold"),
        bg=colors["bg_main"],
        fg=colors["text_primary"],
    ).pack(anchor="w")
    tk.Label(
        input_frame,
        text='Contoh: "kucing persia", "bulu lebat", "vaksin kesehatan", "mendengkur insting"',
        font=("Inter", 9),
        fg="#888888",
        bg=colors["bg_main"],
    ).pack(anchor="w", pady=(0, 8))

    entry_frame = tk.Frame(input_frame, bg=colors["bg_main"])
    entry_frame.pack(fill=tk.X)

    query_entry = tk.Entry(entry_frame, font=("Consolas", 14), width=50)
    query_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=6)

    tk.Label(
        entry_frame,
        text="k dimensi:",
        font=("Inter", 10),
        bg=colors["bg_main"],
        fg=colors["text_primary"],
    ).pack(side=tk.LEFT, padx=(10, 5))
    k_var = tk.IntVar(value=k)
    max_k = min(20, min(tfidf_matrix.shape))
    k_slider = tk.Scale(
        entry_frame,
        from_=1,
        to=max_k,
        resolution=1,
        orient=tk.HORIZONTAL,
        variable=k_var,
        bg=colors["bg_main"],
        length=120,
    )
    k_slider.pack(side=tk.LEFT)

    result_area = tk.Frame(parent, bg=colors["bg_main"], padx=20, pady=5)
    result_area.pack(fill=tk.BOTH, expand=True)

    result_info = tk.Label(
        result_area,
        text="",
        font=("Inter", 12, "bold"),
        fg=colors["text_primary"],
        bg=colors["bg_main"],
    )
    result_info.pack(anchor="w", pady=(0, 5))

    result_content = tk.Frame(result_area, bg=colors["bg_main"])
    result_content.pack(fill=tk.BOTH, expand=True)

    def execute_search(event=None):
        for w in result_content.winfo_children():
            w.destroy()
        query = query_entry.get().strip()
        if not query:
            result_info.config(text="⚠️ Query kosong!", fg="red")
            return

        from modules.lsi import build_lsi_model

        U_k_cur, s_k_cur, Vt_k_cur, S_k_cur, k_cur, exp_cur, _ = build_lsi_model(
            tfidf_matrix, vocab, doc_ids, k=k_var.get()
        )

        tokens, term_weights, doc_scores, ranked, detail = lsi_query(
            query, vocab, doc_ids, idf, tfidf_matrix, U_k_cur, s_k_cur, Vt_k_cur, k_cur
        )
        if not tokens:
            result_info.config(
                text="⚠️ Tidak ada token setelah preprocessing.", fg="red"
            )
            return
        if ranked:
            result_info.config(
                text=f'✅ {len(ranked)} dokumen relevan untuk "{query}" '
                f"(LSI k={k_cur}, variance={exp_cur*100:.1f}%)",
                fg="#2E7D32",
            )
        else:
            result_info.config(
                text=f'❌ Tidak ada dokumen relevan untuk "{query}" (k={k_cur})',
                fg="red",
            )

        term_data = {}
        for term in tokens:
            term_data[term] = {}
            idf_val = idf.get(term, 0)
            if term in vocab:
                idx = vocab.index(term)
                for j, doc_id in enumerate(doc_ids):
                    term_data[term][doc_id] = round(
                        float(tfidf_matrix[idx][j] * idf_val), 6
                    )
            else:
                for doc_id in doc_ids:
                    term_data[term][doc_id] = 0.0

        _make_result_tabs(
            result_content,
            colors,
            tokens,
            term_data,
            doc_scores,
            ranked,
            documents,
            doc_ids,
            detail,
            "Bobot Laten (LSI)",
        )

    btn = tk.Button(
        entry_frame,
        text="🔍 Cari",
        font=("Inter", 12, "bold"),
        bg="#4A148C",
        fg="white",
        padx=20,
        pady=5,
        cursor="hand2",
        command=execute_search,
    )
    btn.pack(side=tk.LEFT, padx=(10, 0))
    query_entry.bind("<Return>", execute_search)
