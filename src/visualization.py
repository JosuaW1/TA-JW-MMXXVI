"""
visualization.py
================
Modul visualisasi untuk hasil eksperimen link prediction pada jaringan PPI.

Visualisasi yang dihasilkan:
1. Kurva ROC untuk semua metode
2. Kurva Precision-Recall untuk semua metode
3. Bar chart perbandingan AUC-ROC dan AUPR
4. Heatmap perbandingan Top-k Precision
5. Feature importance dari Random Forest
6. Visualisasi jaringan PPI dengan prediksi baru
7. Distribusi degree jaringan
8. Distribusi skor prediksi

Semua visualisasi disimpan dalam format PNG resolusi tinggi (300 DPI)
yang sesuai untuk publikasi/skripsi.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: simpan file tanpa pop-up
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi matplotlib untuk output berkualitas tinggi
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (8, 6),
})


# ---------------------------------------------------------------------------
# 1. KURVA ROC
# ---------------------------------------------------------------------------

def plot_roc_curves(roc_data: dict, save_path: str = None):
    """
    Menggambar kurva ROC untuk semua metode dalam satu plot.

    Parameters
    ----------
    roc_data : dict
        Dictionary {method: {fpr, tpr, auc}} dari evaluation.get_roc_curve_data().
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Warna berbeda untuk setiap metode
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for (method, data), color in zip(roc_data.items(), colors):
        label = f"{method} (AUC = {data['auc']:.4f})"
        ax.plot(data["fpr"], data["tpr"], color=color, lw=2, label=label)

    # Garis diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve - Perbandingan Metode Link Prediction")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] ROC curve disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. KURVA PRECISION-RECALL
# ---------------------------------------------------------------------------

def plot_pr_curves(pr_data: dict, save_path: str = None):
    """
    Menggambar kurva Precision-Recall untuk semua metode.

    Parameters
    ----------
    pr_data : dict
        Dictionary {method: {precision, recall, aupr}} dari
        evaluation.get_pr_curve_data().
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))

    for (method, data), color in zip(pr_data.items(), colors):
        label = f"{method} (AUPR = {data['aupr']:.4f})"
        ax.plot(data["recall"], data["precision"],
                color=color, lw=2, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Perbandingan Metode")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] PR curve disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. BAR CHART PERBANDINGAN AUC-ROC & AUPR
# ---------------------------------------------------------------------------

def plot_metrics_comparison(df_results: pd.DataFrame, save_path: str = None):
    """
    Menggambar bar chart perbandingan AUC-ROC dan AUPR semua metode.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame hasil evaluasi dari evaluate_all_methods().
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot AUC-ROC
    df_sorted = df_results.sort_values("AUC-ROC", ascending=True)
    colors_roc = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, len(df_sorted))
    )
    axes[0].barh(df_sorted["Method"], df_sorted["AUC-ROC"], color=colors_roc)
    axes[0].set_xlabel("AUC-ROC")
    axes[0].set_title("Perbandingan AUC-ROC")
    axes[0].set_xlim([0, 1])
    # Tambahkan nilai pada bar
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        axes[0].text(row["AUC-ROC"] + 0.01, i, f'{row["AUC-ROC"]:.4f}',
                     va="center", fontsize=9)

    # Plot AUPR
    df_sorted = df_results.sort_values("AUPR", ascending=True)
    colors_aupr = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, len(df_sorted))
    )
    axes[1].barh(df_sorted["Method"], df_sorted["AUPR"], color=colors_aupr)
    axes[1].set_xlabel("AUPR")
    axes[1].set_title("Perbandingan AUPR")
    axes[1].set_xlim([0, 1])
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        axes[1].text(row["AUPR"] + 0.01, i, f'{row["AUPR"]:.4f}',
                     va="center", fontsize=9)

    plt.suptitle("Perbandingan Performa Metode Link Prediction",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Metrics comparison disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. HEATMAP TOP-K PRECISION
# ---------------------------------------------------------------------------

def plot_topk_heatmap(df_results: pd.DataFrame, save_path: str = None):
    """
    Menggambar heatmap Top-k Precision untuk semua metode dan nilai k.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame hasil evaluasi.
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    # Ambil kolom Top-k
    topk_cols = [col for col in df_results.columns if col.startswith("Top-")]
    if not topk_cols:
        print("[Viz] Tidak ada data Top-k Precision untuk divisualisasikan.")
        return

    heatmap_data = df_results.set_index("Method")[topk_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Top-k Precision - Perbandingan Metode", fontweight="bold")
    ax.set_ylabel("Metode")
    ax.set_xlabel("Metrik")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Top-k heatmap disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE RANDOM FOREST
# ---------------------------------------------------------------------------

def plot_feature_importance(fi_df: pd.DataFrame, save_path: str = None):
    """
    Menggambar bar chart feature importance dari Random Forest.

    Parameters
    ----------
    fi_df : pd.DataFrame
        DataFrame feature importance dari random_forest.train_random_forest().
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    fi_sorted = fi_df.sort_values("Importance", ascending=True)
    colors = plt.cm.viridis(
        np.linspace(0.3, 0.9, len(fi_sorted))
    )
    bars = ax.barh(fi_sorted["Feature"], fi_sorted["Importance"], color=colors)

    # Tambahkan nilai pada bar
    for bar, val in zip(bars, fi_sorted["Importance"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Feature Importance - Random Forest\n"
                 "(Kontribusi Tiap Metode Topologi)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Feature importance disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. VISUALISASI JARINGAN PPI
# ---------------------------------------------------------------------------

def plot_ppi_network(G: nx.Graph,
                     predicted_edges: list = None,
                     top_n_nodes: int = 100,
                     save_path: str = None):
    """
    Menggambar visualisasi jaringan PPI dengan highlight pada prediksi baru.

    Karena jaringan PPI bisa sangat besar, hanya subset node yang
    ditampilkan (berdasarkan degree tertinggi).

    Parameters
    ----------
    G : nx.Graph
        Graf PPI.
    predicted_edges : list of tuple, optional
        Daftar edge prediksi baru yang akan di-highlight.
    top_n_nodes : int, default=100
        Jumlah node teratas (berdasarkan degree) yang ditampilkan.
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    # Ambil subgraph dari node berdegree tertinggi
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get,
                       reverse=True)[:top_n_nodes]
    G_sub = G.subgraph(top_nodes).copy()

    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout
    pos = nx.spring_layout(G_sub, seed=42, k=1.5 / np.sqrt(len(G_sub)))

    # Ukuran node berdasarkan degree
    degrees = [G_sub.degree(n) for n in G_sub.nodes()]
    node_sizes = [max(d * 5, 30) for d in degrees]

    # Gambar edge yang sudah ada
    nx.draw_networkx_edges(G_sub, pos, alpha=0.15, width=0.5,
                           edge_color="gray", ax=ax)

    # Gambar node
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes,
                           node_color=degrees, cmap=plt.cm.YlOrRd,
                           alpha=0.8, ax=ax)

    # Highlight prediksi baru
    if predicted_edges:
        pred_in_sub = [(u, v) for u, v in predicted_edges
                       if u in G_sub.nodes() and v in G_sub.nodes()]
        if pred_in_sub:
            nx.draw_networkx_edges(G_sub, pos, edgelist=pred_in_sub,
                                   edge_color="blue", width=2.0,
                                   alpha=0.7, style="dashed", ax=ax)
            ax.plot([], [], color="blue", linestyle="--", linewidth=2,
                    label=f"Prediksi baru ({len(pred_in_sub)} edges)")
            ax.legend(loc="upper left", fontsize=10)

    # Label untuk top-10 node
    top_10 = sorted(degree_dict, key=degree_dict.get, reverse=True)[:10]
    labels_top10 = {n: n.split(".")[-1] if "." in n else n
                    for n in top_10 if n in G_sub.nodes()}
    nx.draw_networkx_labels(G_sub, pos, labels=labels_top10,
                            font_size=7, font_weight="bold", ax=ax)

    ax.set_title(f"Jaringan PPI - Top {top_n_nodes} Protein\n"
                 f"(berdasarkan degree)", fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Network disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. DISTRIBUSI DEGREE
# ---------------------------------------------------------------------------

def plot_degree_distribution(G: nx.Graph, save_path: str = None):
    """
    Menggambar distribusi degree jaringan PPI (linear dan log-log).

    Jaringan PPI umumnya memiliki distribusi degree scale-free (power-law),
    di mana sedikit protein (hub) memiliki banyak koneksi.

    Parameters
    ----------
    G : nx.Graph
        Graf PPI.
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    degrees = [d for _, d in G.degree()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram linear
    axes[0].hist(degrees, bins=50, color="steelblue", edgecolor="white",
                 alpha=0.8)
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frekuensi")
    axes[0].set_title("Distribusi Degree (Linear)")
    axes[0].axvline(np.mean(degrees), color="red", linestyle="--",
                    label=f"Mean = {np.mean(degrees):.1f}")
    axes[0].legend()

    # Log-log plot
    degree_count = {}
    for d in degrees:
        degree_count[d] = degree_count.get(d, 0) + 1
    deg_vals = sorted(degree_count.keys())
    freq_vals = [degree_count[d] for d in deg_vals]

    axes[1].scatter(deg_vals, freq_vals, s=15, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Frekuensi (log)")
    axes[1].set_title("Distribusi Degree (Log-Log)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Distribusi Degree Jaringan PPI", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Degree distribution disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. DISTRIBUSI SKOR PREDIKSI
# ---------------------------------------------------------------------------

def plot_score_distributions(scores_positive: dict,
                             scores_negative: dict,
                             save_path: str = None):
    """
    Menggambar distribusi skor prediksi untuk positif vs negatif
    pada setiap metode.

    Parameters
    ----------
    scores_positive : dict
        Skor untuk pasangan positif.
    scores_negative : dict
        Skor untuk pasangan negatif.
    save_path : str, optional
        Path untuk menyimpan gambar.
    """
    methods = list(scores_positive.keys())
    n_methods = len(methods)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        pos = scores_positive[method]
        neg = scores_negative[method]

        ax.hist(pos, bins=30, alpha=0.6, label="Positif", color="green",
                density=True)
        ax.hist(neg, bins=30, alpha=0.6, label="Negatif", color="red",
                density=True)
        ax.set_title(method, fontweight="bold")
        ax.set_xlabel("Skor")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Sembunyikan axes kosong
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribusi Skor Prediksi (Positif vs Negatif)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Score distributions disimpan: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FUNGSI UTILITAS: SIMPAN SEMUA VISUALISASI
# ---------------------------------------------------------------------------

def generate_all_visualizations(results: dict, output_dir: str):
    """
    Menghasilkan dan menyimpan semua visualisasi.

    Parameters
    ----------
    results : dict
        Dictionary lengkap hasil dari pipeline utama.
    output_dir : str
        Direktori untuk menyimpan file gambar.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 60)

    # 1. ROC Curves
    if "roc_data" in results:
        print("\n[1/8] ROC Curves...")
        plot_roc_curves(
            results["roc_data"],
            save_path=os.path.join(output_dir, "roc_curves.png")
        )

    # 2. PR Curves
    if "pr_data" in results:
        print("[2/8] Precision-Recall Curves...")
        plot_pr_curves(
            results["pr_data"],
            save_path=os.path.join(output_dir, "pr_curves.png")
        )

    # 3. Metrics Comparison
    if "df_results" in results:
        print("[3/8] Metrics Comparison...")
        plot_metrics_comparison(
            results["df_results"],
            save_path=os.path.join(output_dir, "metrics_comparison.png")
        )

    # 4. Top-k Heatmap
    if "df_results" in results:
        print("[4/8] Top-k Heatmap...")
        plot_topk_heatmap(
            results["df_results"],
            save_path=os.path.join(output_dir, "topk_heatmap.png")
        )

    # 5. Feature Importance
    if "feature_importance" in results:
        print("[5/8] Feature Importance...")
        plot_feature_importance(
            results["feature_importance"],
            save_path=os.path.join(output_dir, "feature_importance.png")
        )

    # 6. Network Visualization
    if "G_full" in results:
        print("[6/8] Network Visualization...")
        plot_ppi_network(
            results["G_full"],
            predicted_edges=results.get("top_predictions", None),
            save_path=os.path.join(output_dir, "ppi_network.png")
        )

    # 7. Degree Distribution
    if "G_full" in results:
        print("[7/8] Degree Distribution...")
        plot_degree_distribution(
            results["G_full"],
            save_path=os.path.join(output_dir, "degree_distribution.png")
        )

    # 8. Score Distributions
    if "scores_positive" in results and "scores_negative" in results:
        print("[8/8] Score Distributions...")
        plot_score_distributions(
            results["scores_positive"],
            results["scores_negative"],
            save_path=os.path.join(output_dir, "score_distributions.png")
        )

    print(f"\n[Viz] Semua visualisasi disimpan di: {output_dir}")
