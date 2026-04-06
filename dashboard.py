"""
dashboard.py
============
Dashboard GUI (Tkinter) yang menampilkan 5 output terpenting dari
eksperimen link prediction, dengan dukungan multi-skenario.

Fitur:
- Dropdown untuk memilih skenario (confidence score threshold)
- 5 tab: Tabel Evaluasi, Kurva ROC, Perbandingan Metrik,
  Feature Importance, Prediksi Baru

Penggunaan:
    python dashboard.py
"""

import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ============================================================================
# UTILITAS
# ============================================================================

def load_image(filepath: str, max_width: int = 900, max_height: int = 520):
    """Load dan resize gambar agar muat di panel."""
    img = Image.open(filepath)
    ratio = min(max_width / img.width, max_height / img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def discover_scenarios() -> list:
    """
    Deteksi otomatis folder skenario yang ada di results/.

    Returns
    -------
    list of dict
        Daftar skenario yang ditemukan, masing-masing berisi
        'name' (nama folder) dan 'path' (path lengkap).
    """
    scenarios = []
    if not os.path.exists(RESULTS_DIR):
        return scenarios

    for entry in sorted(os.listdir(RESULTS_DIR)):
        scenario_path = os.path.join(RESULTS_DIR, entry)
        eval_csv = os.path.join(scenario_path, "evaluation_results.csv")
        # Folder valid jika berisi evaluation_results.csv
        if os.path.isdir(scenario_path) and os.path.exists(eval_csv):
            scenarios.append({
                "name": entry,
                "path": scenario_path,
            })

    return scenarios


# ============================================================================
# DASHBOARD UTAMA
# ============================================================================

class Dashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dashboard - Link Prediction PPI Kanker Paru-Paru")
        self.root.geometry("1100x780")
        self.root.configure(bg="#0f172a")
        self.root.minsize(900, 600)

        # Deteksi skenario
        self.scenarios = discover_scenarios()
        if not self.scenarios:
            raise FileNotFoundError(
                "Tidak ada skenario ditemukan di folder results/.\n"
                "Jalankan main.py terlebih dahulu."
            )

        self.current_scenario = None
        self.image_refs = []  # keep references agar tidak di-garbage-collect

        # Styling
        self.setup_style()

        # Build UI
        self.build_header()
        self.build_scenario_selector()
        self.summary_frame = tk.Frame(self.root, bg="#0f172a")
        self.summary_frame.pack(fill="x")
        self.notebook_frame = tk.Frame(self.root, bg="#0f172a")
        self.notebook_frame.pack(fill="both", expand=True)

        # Load skenario pertama
        self.load_scenario(self.scenarios[0])

    def setup_style(self):
        """Konfigurasi tema dark untuk ttk widgets."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background="#0f172a", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#1e293b", foreground="#94a3b8",
                        padding=[20, 10], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                   background=[("selected", "#0f172a")],
                   foreground=[("selected", "#38bdf8")])

        style.configure("Treeview",
                        background="#1e293b", foreground="#e2e8f0",
                        fieldbackground="#1e293b", borderwidth=0,
                        font=("Segoe UI", 10), rowheight=28)
        style.configure("Treeview.Heading",
                        background="#334155", foreground="#e2e8f0",
                        font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Treeview",
                   background=[("selected", "#334155")],
                   foreground=[("selected", "#38bdf8")])

        style.configure("Vertical.TScrollbar",
                        background="#334155", troughcolor="#1e293b",
                        borderwidth=0, arrowsize=14)
        style.configure("Horizontal.TScrollbar",
                        background="#334155", troughcolor="#1e293b",
                        borderwidth=0, arrowsize=14)

        # Combobox style
        style.configure("Scenario.TCombobox",
                        fieldbackground="#1e293b", background="#334155",
                        foreground="#e2e8f0", arrowsize=16)
        style.map("Scenario.TCombobox",
                   fieldbackground=[("readonly", "#1e293b")],
                   foreground=[("readonly", "#e2e8f0")])

    def build_header(self):
        """Header bar."""
        header = tk.Frame(self.root, bg="#1e293b", padx=30, pady=16)
        header.pack(fill="x")

        tk.Label(
            header,
            text="Link Prediction pada Jaringan PPI Kanker Paru-Paru",
            bg="#1e293b", fg="#f1f5f9",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        ).pack(fill="x")

        tk.Label(
            header,
            text="Menggunakan Fitur Topologi Jaringan dan Random Forest Classifier",
            bg="#1e293b", fg="#64748b",
            font=("Segoe UI", 10),
            anchor="w"
        ).pack(fill="x")

    def build_scenario_selector(self):
        """Dropdown untuk memilih skenario."""
        selector_frame = tk.Frame(self.root, bg="#0f172a", padx=30, pady=10)
        selector_frame.pack(fill="x")

        tk.Label(
            selector_frame,
            text="Pilih Skenario:",
            bg="#0f172a", fg="#94a3b8",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=(0, 10))

        scenario_names = [s["name"] for s in self.scenarios]
        self.scenario_var = tk.StringVar(value=scenario_names[0])

        combo = ttk.Combobox(
            selector_frame,
            textvariable=self.scenario_var,
            values=scenario_names,
            state="readonly",
            style="Scenario.TCombobox",
            font=("Segoe UI", 11),
            width=30,
        )
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self.on_scenario_change)

        # Label info
        self.scenario_info_label = tk.Label(
            selector_frame,
            text="",
            bg="#0f172a", fg="#64748b",
            font=("Segoe UI", 10),
        )
        self.scenario_info_label.pack(side="left", padx=20)

    def on_scenario_change(self, event=None):
        """Handler ketika user memilih skenario lain."""
        selected_name = self.scenario_var.get()
        scenario = next(s for s in self.scenarios if s["name"] == selected_name)
        self.load_scenario(scenario)

    def load_scenario(self, scenario: dict):
        """Load dan tampilkan data dari skenario yang dipilih."""
        self.current_scenario = scenario
        scenario_path = scenario["path"]

        # Cek file yang dibutuhkan
        required = {
            "eval_csv": os.path.join(scenario_path, "evaluation_results.csv"),
            "pred_csv": os.path.join(scenario_path, "new_interaction_predictions.csv"),
            "roc_png": os.path.join(scenario_path, "roc_curves.png"),
            "metrics_png": os.path.join(scenario_path, "metrics_comparison.png"),
            "fi_png": os.path.join(scenario_path, "feature_importance.png"),
            "cm_png": os.path.join(scenario_path, "confusion_matrices.png"),
        }

        for name, path in required.items():
            if not os.path.exists(path):
                self.scenario_info_label.config(
                    text=f"File tidak lengkap: {os.path.basename(path)}",
                    fg="#ef4444"
                )
                return

        # Load data
        self.df_eval = pd.read_csv(required["eval_csv"])
        self.df_pred = pd.read_csv(required["pred_csv"])
        self.files = required

        # Update info label
        best = self.df_eval.iloc[0]
        self.scenario_info_label.config(
            text=f"Protein: {self.df_eval.shape[0]} metode  |  "
                 f"Terbaik: {best['Method']} (AUC-ROC: {best['AUC-ROC']:.4f})",
            fg="#94a3b8"
        )

        # Rebuild summary dan tabs
        self.rebuild_summary()
        self.rebuild_tabs()

    def rebuild_summary(self):
        """Rebuild summary cards."""
        # Bersihkan frame lama
        for widget in self.summary_frame.winfo_children():
            widget.destroy()

        bar = tk.Frame(self.summary_frame, bg="#0f172a", padx=30, pady=8)
        bar.pack(fill="x")

        best = self.df_eval.iloc[0]

        cards_data = [
            ("METODE TERBAIK", str(best["Method"]), "#4ade80", ""),
            ("AUC-ROC", f"{best['AUC-ROC']:.4f}", "#38bdf8", ""),
            ("AUPR", f"{best['AUPR']:.4f}", "#38bdf8", ""),
            ("TOTAL METODE", str(len(self.df_eval)), "#38bdf8",
             "7 topologi + 1 ML"),
            ("PREDIKSI BARU", str(len(self.df_pred)), "#38bdf8",
             "kandidat interaksi"),
        ]

        for label, value, color, sub in cards_data:
            card = tk.Frame(bar, bg="#1e293b", padx=16, pady=10,
                            highlightbackground="#334155",
                            highlightthickness=1)
            card.pack(side="left", fill="x", expand=True, padx=4)

            tk.Label(card, text=label, bg="#1e293b", fg="#64748b",
                     font=("Segoe UI", 8, "bold")).pack()
            tk.Label(card, text=value, bg="#1e293b", fg=color,
                     font=("Segoe UI", 16, "bold")).pack()
            if sub:
                tk.Label(card, text=sub, bg="#1e293b", fg="#64748b",
                         font=("Segoe UI", 8)).pack()

    def rebuild_tabs(self):
        """Rebuild notebook tabs dengan data skenario baru."""
        # Bersihkan frame lama
        for widget in self.notebook_frame.winfo_children():
            widget.destroy()

        # Bersihkan image references
        self.image_refs.clear()

        notebook = ttk.Notebook(self.notebook_frame)
        notebook.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # --- Tab 1: Tabel Evaluasi ---
        tab1 = tk.Frame(notebook, bg="#0f172a")
        notebook.add(tab1, text="  Tabel Evaluasi  ")

        tk.Label(
            tab1,
            text="Baris hijau = metode terbaik. Semua metode dievaluasi pada "
                 "test set yang sama.",
            bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 10),
            wraplength=900, justify="left", anchor="w", padx=20, pady=10
        ).pack(fill="x")

        df_display = self.df_eval.drop(
            columns=[c for c in self.df_eval.columns if "Unnamed" in c],
            errors="ignore"
        )
        self._create_table(tab1, df_display, highlight_first=True)

        # --- Tab 2: Kurva ROC ---
        tab2 = tk.Frame(notebook, bg="#0f172a")
        notebook.add(tab2, text="  Kurva ROC  ")
        self._create_image_tab(
            tab2, self.files["roc_png"],
            "Kurva ROC menunjukkan trade-off antara True Positive Rate dan "
            "False Positive Rate. Semakin dekat kurva ke pojok kiri atas, "
            "semakin baik. Garis putus-putus = random classifier (AUC = 0.5)."
        )

        # --- Tab 3: Perbandingan Metrik ---
        tab3 = tk.Frame(notebook, bg="#0f172a")
        notebook.add(tab3, text="  Perbandingan Metrik  ")
        self._create_image_tab(
            tab3, self.files["metrics_png"],
            "Bar chart horizontal menunjukkan perbandingan AUC-ROC dan AUPR "
            "antar metode. AUC-ROC mengukur kemampuan diskriminasi, "
            "AUPR fokus pada ketepatan menemukan interaksi yang benar."
        )

        # --- Tab 4: Feature Importance ---
        tab4 = tk.Frame(notebook, bg="#0f172a")
        notebook.add(tab4, text="  Feature Importance  ")
        self._create_image_tab(
            tab4, self.files["fi_png"],
            "Kontribusi relatif tiap metode topologi dalam Random Forest. "
            "Dihitung menggunakan Gini impurity. Semakin tinggi nilai, "
            "semakin besar pengaruh metode tersebut terhadap prediksi."
        )

        # --- Tab 5: Confusion Matrix ---
        tab5 = tk.Frame(notebook, bg="#0f172a")
        notebook.add(tab5, text="  Confusion Matrix  ")
        self._create_image_tab(
            tab5, self.files["cm_png"],
            "Confusion Matrix menunjukkan distribusi prediksi benar dan salah "
            "untuk setiap metode. TN = True Negative, FP = False Positive, "
            "FN = False Negative, TP = True Positive. Threshold optimal "
            "ditentukan menggunakan Youden's J statistic dari kurva ROC."
        )

        # --- Tab 6: Prediksi Baru ---
        tab6 = tk.Frame(notebook, bg="#0f172a")
        notebook.add(tab6, text="  Prediksi Baru  ")

        tk.Label(
            tab6,
            text="Pasangan protein yang diprediksi berinteraksi oleh Random "
                 "Forest. Skor mendekati 1.0 = confidence tinggi.",
            bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 10),
            wraplength=900, justify="left", anchor="w", padx=20, pady=10
        ).pack(fill="x")

        pred_cols = ["protein1", "protein2", "rf_score"]
        extra_cols = [c for c in self.df_pred.columns
                      if c not in pred_cols and c not in ["Rank", "Unnamed: 0"]]
        df_pred_display = self.df_pred[pred_cols + extra_cols].head(30)
        self._create_table(tab6, df_pred_display)

    def _create_table(self, parent, df: pd.DataFrame,
                      highlight_first: bool = False):
        """Buat treeview table dari DataFrame."""
        frame = tk.Frame(parent, bg="#1e293b")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrolly = ttk.Scrollbar(frame, orient="vertical")
        scrolly.pack(side="right", fill="y")

        scrollx = ttk.Scrollbar(frame, orient="horizontal")
        scrollx.pack(side="bottom", fill="x")

        columns = list(df.columns)
        tree = ttk.Treeview(
            frame, columns=columns, show="headings",
            yscrollcommand=scrolly.set, xscrollcommand=scrollx.set,
            height=min(20, len(df))
        )
        scrolly.config(command=tree.yview)
        scrollx.config(command=tree.xview)

        for col in columns:
            tree.heading(col, text=col, anchor="center")
            max_len = max(len(str(col)), df[col].astype(str).str.len().max())
            width = min(max(max_len * 9, 80), 200)
            tree.column(col, width=width, anchor="center")

        for i, (_, row) in enumerate(df.iterrows()):
            values = [f"{v:.4f}" if isinstance(v, float) else str(v)
                      for v in row]
            tag = "best" if (i == 0 and highlight_first) else (
                "even" if i % 2 == 0 else "odd")
            tree.insert("", "end", values=values, tags=(tag,))

        tree.tag_configure("best", background="#1a3a2a", foreground="#4ade80")
        tree.tag_configure("even", background="#1e293b", foreground="#e2e8f0")
        tree.tag_configure("odd", background="#0f172a", foreground="#e2e8f0")

        tree.pack(fill="both", expand=True)

    def _create_image_tab(self, parent, img_path: str, description: str):
        """Buat tab dengan gambar dan deskripsi."""
        desc_label = tk.Label(
            parent, text=description, bg="#0f172a", fg="#94a3b8",
            font=("Segoe UI", 10), wraplength=900, justify="left",
            anchor="w", padx=20, pady=10
        )
        desc_label.pack(fill="x")

        img_frame = tk.Frame(parent, bg="#1e293b", padx=10, pady=10)
        img_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        photo = load_image(img_path)
        self.image_refs.append(photo)  # keep reference
        img_label = tk.Label(img_frame, image=photo, bg="#1e293b")
        img_label.pack(expand=True)

    def run(self):
        """Jalankan aplikasi."""
        self.root.mainloop()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        app = Dashboard()
        print("[Dashboard] Aplikasi berhasil dibuka.")
        app.run()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except ImportError:
        print("[ERROR] Modul Pillow belum terinstall.")
        print("        Jalankan: pip install Pillow")
