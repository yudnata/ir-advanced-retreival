import tkinter as tk
from tkinter import ttk
import os

from modules.indexing import load_documents, build_all_index
from modules.fuzzy import build_fuzzy_membership
from modules.gvsm import build_gvsm_term_vectors
from modules.lsi import build_lsi_model

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Information Retrieval System")
        self.root.geometry("1100x750")
        self.root.state("zoomed")

        self.colors = {
            "bg_sidebar": "#1E293B",
            "bg_main": "#F8FAFC",
            "bg_card": "#FFFFFF",
            "text_primary": "#1E293B",
            "text_secondary": "#64748B",
            "accent_blue": "#2563EB",
            "accent_green": "#059669",
            "accent_purple": "#7C3AED",
            "accent_hover": "#334155",
            "white": "#FFFFFF",
            "border": "#E2E8F0",
        }

        self.documents = load_documents(DATA_PATH)
        (
            self.vocab,
            self.doc_ids,
            self.doc_tokens,
            self.tf_raw,
            self.tf_normalized,
            self.df,
            self.idf,
            self.tfidf_matrix,
        ) = build_all_index(self.documents)

        self.membership_matrix = build_fuzzy_membership(
            self.tfidf_matrix, self.vocab, self.doc_ids
        )

        self.term_vectors = build_gvsm_term_vectors(
            self.tfidf_matrix, self.vocab, self.doc_ids
        )

        (
            self.U_k,
            self.s_k,
            self.Vt_k,
            self.S_k,
            self.k_lsi,
            self.explained_ratio,
            self.s_all,
        ) = build_lsi_model(self.tfidf_matrix, self.vocab, self.doc_ids)

        self.sidebar = tk.Frame(self.root, bg=self.colors["bg_sidebar"], width=280)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.main_area = tk.Frame(self.root, bg=self.colors["bg_main"])
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_sidebar()
        self.show_home()

    def setup_sidebar(self):
        logo_frame = tk.Frame(self.sidebar, bg=self.colors["bg_sidebar"], pady=35)
        logo_frame.pack(fill=tk.X)

        tk.Label(
            logo_frame,
            text="Advanced IR",
            font=("Inter", 22, "bold"),
            fg=self.colors["white"],
            bg=self.colors["bg_sidebar"],
        ).pack()
        tk.Label(
            logo_frame,
            text="Fuzzy  ·  GVSM  ·  LSI",
            font=("Inter", 10),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_sidebar"],
        ).pack(pady=(4, 0))

        tk.Frame(self.sidebar, bg="#334155", height=1).pack(fill=tk.X, padx=20, pady=10)

        self.menu_items = []
        self.add_menu_item("🏠      Dashboard", self.show_home)
        self.add_menu_item("📂      Daftar Dokumen", self.open_documents)
        self.add_menu_item("🌫️ Fuzzy IR", self.open_fuzzy)
        self.add_menu_item("📐      GVSM", self.open_gvsm)
        self.add_menu_item("🧠      LSI", self.open_lsi)

        footer = tk.Frame(self.sidebar, bg=self.colors["bg_sidebar"], pady=30)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(
            footer,
            text="oleh",
            font=("Inter", 9),
            fg="#94A3B8",
            bg=self.colors["bg_sidebar"],
        ).pack()
        tk.Label(
            footer,
            text="Gede Yudhi Adinata",
            font=("Inter", 11, "bold"),
            fg=self.colors["white"],
            bg=self.colors["bg_sidebar"],
        ).pack()
        tk.Label(
            footer,
            text="2305551142",
            font=("Inter", 10),
            fg="#94A3B8",
            bg=self.colors["bg_sidebar"],
        ).pack()

    def add_menu_item(self, text, command):
        btn = tk.Button(
            self.sidebar,
            text=text,
            font=("Inter", 11),
            fg=self.colors["white"],
            bg=self.colors["bg_sidebar"],
            bd=0,
            padx=30,
            pady=15,
            anchor="w",
            cursor="hand2",
            activebackground=self.colors["accent_hover"],
            activeforeground=self.colors["white"],
            command=command,
        )
        btn.pack(fill=tk.X)
        self.menu_items.append(btn)
        btn.bind("<Enter>", lambda e: btn.config(bg=self.colors["accent_hover"]))
        btn.bind("<Leave>", lambda e: btn.config(bg=self.colors["bg_sidebar"]))

    def clear_main_area(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()

    def inner_view_header(self, title):
        header = tk.Frame(
            self.main_area,
            bg=self.colors["white"],
            pady=20,
            padx=30,
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=title,
            font=("Inter", 18, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["white"],
        ).pack(side=tk.LEFT)

    def show_home(self):
        self.clear_main_area()
        from modules.interface import build_dashboard

        build_dashboard(
            self.main_area,
            self.colors,
            self.doc_ids,
            self.vocab,
            self.open_fuzzy,
            self.open_gvsm,
            self.open_lsi,
            self.open_documents,
        )

    def open_documents(self):
        self.clear_main_area()
        self.inner_view_header("📂 Daftar Dokumen Sumber")
        from modules.interface import build_documents_view

        build_documents_view(self.main_area, self.colors, self.doc_ids, self.documents)

    def open_fuzzy(self):
        self.clear_main_area()
        self.inner_view_header("🌫️ Fuzzy Information Retrieval")
        inner = tk.Frame(self.main_area, bg=self.colors["bg_main"], padx=10, pady=5)
        inner.pack(fill=tk.BOTH, expand=True)
        from modules.interface import build_fuzzy_view

        build_fuzzy_view(
            inner,
            self.colors,
            self.vocab,
            self.doc_ids,
            self.idf,
            self.tfidf_matrix,
            self.membership_matrix,
            self.documents,
        )

    def open_gvsm(self):
        self.clear_main_area()
        self.inner_view_header("📐 Generalized Vector Space Model")
        inner = tk.Frame(self.main_area, bg=self.colors["bg_main"], padx=10, pady=5)
        inner.pack(fill=tk.BOTH, expand=True)
        from modules.interface import build_gvsm_view

        build_gvsm_view(
            inner,
            self.colors,
            self.vocab,
            self.doc_ids,
            self.idf,
            self.tfidf_matrix,
            self.term_vectors,
            self.documents,
        )

    def open_lsi(self):
        self.clear_main_area()
        self.inner_view_header("🧠 Latent Semantic Indexing (LSI)")
        inner = tk.Frame(self.main_area, bg=self.colors["bg_main"], padx=10, pady=5)
        inner.pack(fill=tk.BOTH, expand=True)
        from modules.interface import build_lsi_view

        build_lsi_view(
            inner,
            self.colors,
            self.vocab,
            self.doc_ids,
            self.idf,
            self.tfidf_matrix,
            self.U_k,
            self.s_k,
            self.Vt_k,
            self.k_lsi,
            self.explained_ratio,
            self.s_all,
            self.documents,
        )


if __name__ == "__main__":
    root = tk.Tk()
    try:
        from tkinter import font

        font.nametofont("TkDefaultFont").configure(family="Inter", size=10)
    except Exception:
        pass
    app = MainApplication(root)
    root.mainloop()
