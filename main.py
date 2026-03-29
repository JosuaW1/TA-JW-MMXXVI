"""
main.py
=======
Pipeline utama untuk eksperimen Link Prediction pada Jaringan Interaksi
Protein (PPI) Kanker Paru-Paru Menggunakan Fitur Topologi Jaringan.

Script ini mengorkestrasi seluruh alur eksperimen:
1. Data Preparation     -> load, filter, split, negative sampling
2. Topology Methods     -> hitung skor 7 metode link prediction
3. Random Forest        -> model ML sebagai metode ke-8
4. Evaluation           -> AUC-ROC, AUPR, Top-k Precision
5. Visualization        -> grafik perbandingan dan jaringan PPI
6. Output               -> kandidat interaksi baru

Mendukung multi-skenario: setiap confidence score threshold dijalankan
secara terpisah dan hasilnya disimpan ke folder masing-masing.

Penggunaan:
    python main.py

Pastikan file data STRING sudah ditempatkan di folder data/.
Data dapat diunduh dari: https://string-db.org/

Konfigurasi parameter ada di bagian CONFIG dan SCENARIOS di bawah.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import networkx as nx

# Tambahkan path src ke sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from data_preparation import prepare_data
from topology_methods import (
    compute_all_scores,
    build_feature_matrix,
    METHOD_REGISTRY,
)
from random_forest import (
    prepare_ml_data,
    train_random_forest,
    predict_scores,
)
from evaluation import (
    evaluate_all_methods,
    format_results_table,
    get_roc_curve_data,
    get_pr_curve_data,
)
from visualization import generate_all_visualizations


# ============================================================================
# KONFIGURASI
# ============================================================================

# Daftar skenario confidence score yang akan dijalankan
# Tambah/hapus skenario cukup edit list ini
SCENARIOS = [
    {"name": "skenario_04", "min_score": 0.4, "label": "Medium Confidence (>= 0.4)"},
    {"name": "skenario_07", "min_score": 0.7, "label": "High Confidence (>= 0.7)"},
]

CONFIG = {
    # Path data STRING (subset protein kanker paru-paru dari UniProt)
    # Format TSV: #node1, node2, ..., combined_score (skala 0-1)
    "data_path": os.path.join(BASE_DIR, "data",
                              "string_interactions.tsv"),

    # Proporsi testing edges
    "test_ratio": 0.10,

    # Random seed untuk reprodusibilitas
    "random_state": 42,

    # Nilai k untuk Top-k Precision
    "k_values": [10, 50, 100, 200, 500],

    # Random Forest hyperparameters
    "rf_n_estimators": 100,
    "rf_max_depth": None,

    # Output
    "results_dir": os.path.join(BASE_DIR, "results"),
    "top_n_predictions": 50,  # jumlah kandidat interaksi baru yang dilaporkan
}


# ============================================================================
# FUNGSI UTILITAS
# ============================================================================

def print_header(title: str):
    """Cetak header section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def save_results_to_csv(df: pd.DataFrame, filepath: str):
    """Simpan DataFrame ke CSV."""
    df.to_csv(filepath, index=False)
    print(f"[Save] Hasil disimpan: {filepath}")


def predict_new_interactions(G_train: nx.Graph,
                             G_full: nx.Graph,
                             model,
                             random_state: int = 42,
                             top_n: int = 50) -> pd.DataFrame:
    """
    Memprediksi kandidat interaksi protein baru menggunakan metode terbaik.

    Mengambil pasangan node yang TIDAK ada di graf penuh, menghitung skor
    dari semua metode, dan menggunakan Random Forest untuk menghasilkan
    skor prediksi gabungan.

    Parameters
    ----------
    G_train : nx.Graph
        Graf training.
    G_full : nx.Graph
        Graf penuh (termasuk testing edges).
    model : RandomForestClassifier
        Model Random Forest yang sudah dilatih.
    random_state : int
        Seed untuk reprodusibilitas.
    top_n : int
        Jumlah kandidat teratas yang dikembalikan.

    Returns
    -------
    pd.DataFrame
        DataFrame kandidat interaksi baru dengan skor prediksi.
    """
    print_header("PREDIKSI INTERAKSI PROTEIN BARU")

    nodes = list(G_train.nodes())
    existing_edges = set(G_full.edges())
    edge_set = set()
    for u, v in existing_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Sampling kandidat pasangan (sampling acak dari non-edge)
    import random
    rng = random.Random(random_state)
    candidates = []
    max_candidates = min(50000, len(nodes) * 10)

    print(f"[Prediksi] Sampling kandidat pasangan protein...")
    attempts = 0
    seen = set()
    while len(candidates) < max_candidates and attempts < max_candidates * 5:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        attempts += 1
        if u == v or (u, v) in edge_set:
            continue
        pair = tuple(sorted([u, v]))
        if pair not in seen:
            seen.add(pair)
            candidates.append((pair[0], pair[1]))

    print(f"[Prediksi] Jumlah kandidat: {len(candidates)}")

    # Hitung skor topologi
    print("[Prediksi] Menghitung skor topologi untuk kandidat...")
    candidate_scores = compute_all_scores(
        G_train, candidates, verbose=False
    )

    # Prediksi dengan Random Forest
    print("[Prediksi] Menghitung skor Random Forest...")
    rf_scores = predict_scores(model, candidate_scores)

    # Buat DataFrame hasil
    results = pd.DataFrame({
        "protein1": [c[0] for c in candidates],
        "protein2": [c[1] for c in candidates],
        "rf_score": rf_scores,
    })

    # Tambahkan skor individual
    for method_name, scores in candidate_scores.items():
        col_name = method_name.replace(" ", "_").lower()
        results[col_name] = scores

    # Urutkan dan ambil top-N
    results = results.sort_values("rf_score", ascending=False).head(top_n)
    results = results.reset_index(drop=True)
    results.index = results.index + 1  # mulai dari 1
    results.index.name = "Rank"

    print(f"\n[Prediksi] Top-{top_n} kandidat interaksi baru:")
    print(results[["protein1", "protein2", "rf_score"]].to_string())

    return results


# ============================================================================
# PIPELINE SATU SKENARIO
# ============================================================================

def run_scenario(scenario: dict) -> dict:
    """
    Menjalankan seluruh pipeline untuk satu skenario confidence score.

    Parameters
    ----------
    scenario : dict
        Dictionary berisi 'name', 'min_score', dan 'label'.

    Returns
    -------
    dict
        Hasil lengkap skenario.
    """
    scenario_name = scenario["name"]
    min_score = scenario["min_score"]
    label = scenario["label"]

    # Folder output per skenario
    scenario_dir = os.path.join(CONFIG["results_dir"], scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    start_time = time.time()

    print_header(f"SKENARIO: {label}")
    print(f"  Confidence threshold : {min_score}")
    print(f"  Output folder        : {scenario_dir}")

    # ------------------------------------------------------------------
    # TAHAP 1: DATA PREPARATION
    # ------------------------------------------------------------------
    print_header("TAHAP 1: DATA PREPARATION")

    data = prepare_data(
        filepath=CONFIG["data_path"],
        min_score=min_score,
        test_ratio=CONFIG["test_ratio"],
        random_state=CONFIG["random_state"],
    )

    G_full = data["G_full"]
    G_train = data["G_train"]
    test_edges = data["test_edges"]
    negative_samples = data["negative_samples"]

    # ------------------------------------------------------------------
    # TAHAP 2: HITUNG SKOR TOPOLOGI
    # ------------------------------------------------------------------
    print_header("TAHAP 2: HITUNG SKOR 7 METODE TOPOLOGI")

    print("\n--- Skor untuk Test Edges (Positif) ---")
    scores_positive = compute_all_scores(G_train, test_edges)

    print("\n--- Skor untuk Negative Samples ---")
    scores_negative = compute_all_scores(G_train, negative_samples)

    # ------------------------------------------------------------------
    # TAHAP 3: RANDOM FOREST (METODE KE-8)
    # ------------------------------------------------------------------
    print_header("TAHAP 3: RANDOM FOREST CLASSIFIER")

    # Siapkan data ML
    X, y, feature_names = prepare_ml_data(scores_positive, scores_negative)

    # Latih model
    rf_result = train_random_forest(
        X, y, feature_names,
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        random_state=CONFIG["random_state"],
    )

    # Prediksi skor RF untuk test edges dan negative samples
    rf_pred_positive = predict_scores(rf_result["model"], scores_positive)
    rf_pred_negative = predict_scores(rf_result["model"], scores_negative)

    # ------------------------------------------------------------------
    # TAHAP 4: EVALUASI
    # ------------------------------------------------------------------
    print_header("TAHAP 4: EVALUASI SEMUA METODE")

    df_results, curve_data = evaluate_all_methods(
        scores_positive=scores_positive,
        scores_negative=scores_negative,
        rf_pred_positive=rf_pred_positive,
        rf_pred_negative=rf_pred_negative,
        k_values=CONFIG["k_values"],
    )

    # Tampilkan tabel hasil
    print("\n--- TABEL PERBANDINGAN LENGKAP ---")
    print(format_results_table(df_results))

    # Simpan tabel hasil
    save_results_to_csv(
        df_results,
        os.path.join(scenario_dir, "evaluation_results.csv")
    )

    # Hitung data kurva
    roc_data = get_roc_curve_data(curve_data)
    pr_data = get_pr_curve_data(curve_data)

    # ------------------------------------------------------------------
    # TAHAP 5: PREDIKSI INTERAKSI BARU
    # ------------------------------------------------------------------
    print_header("TAHAP 5: PREDIKSI INTERAKSI BARU")

    new_predictions = predict_new_interactions(
        G_train, G_full, rf_result["model"],
        random_state=CONFIG["random_state"],
        top_n=CONFIG["top_n_predictions"]
    )

    save_results_to_csv(
        new_predictions,
        os.path.join(scenario_dir, "new_interaction_predictions.csv")
    )

    # ------------------------------------------------------------------
    # TAHAP 6: VISUALISASI
    # ------------------------------------------------------------------
    print_header("TAHAP 6: VISUALISASI")

    viz_results = {
        "roc_data": roc_data,
        "pr_data": pr_data,
        "df_results": df_results,
        "feature_importance": rf_result["feature_importance"],
        "G_full": G_full,
        "top_predictions": list(zip(
            new_predictions["protein1"],
            new_predictions["protein2"]
        )),
        "scores_positive": scores_positive,
        "scores_negative": scores_negative,
    }

    generate_all_visualizations(
        viz_results,
        output_dir=scenario_dir
    )

    # ------------------------------------------------------------------
    # RINGKASAN SKENARIO
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    best = df_results.iloc[0]

    print_header(f"RINGKASAN SKENARIO: {label}")
    print(f"  Data:")
    print(f"    Protein (node)       : {G_full.number_of_nodes()}")
    print(f"    Interaksi (edge)     : {G_full.number_of_edges()}")
    print(f"    Training edges       : {G_train.number_of_edges()}")
    print(f"    Testing edges        : {len(test_edges)}")
    print(f"    Negative samples     : {len(negative_samples)}")
    print(f"\n  Metode terbaik:")
    print(f"    Nama    : {best['Method']}")
    print(f"    AUC-ROC : {best['AUC-ROC']:.4f}")
    print(f"    AUPR    : {best['AUPR']:.4f}")
    print(f"\n  Random Forest CV AUC-ROC:")
    print(f"    Mean    : {rf_result['cv_scores'].mean():.4f}")
    print(f"    Std     : {rf_result['cv_scores'].std():.4f}")
    print(f"\n  Output disimpan di: {scenario_dir}")
    print(f"  Waktu eksekusi: {elapsed:.1f} detik")
    print("=" * 70)

    return {
        "scenario": scenario,
        "data": data,
        "scores_positive": scores_positive,
        "scores_negative": scores_negative,
        "rf_result": rf_result,
        "df_results": df_results,
        "new_predictions": new_predictions,
    }


# ============================================================================
# PIPELINE UTAMA (MULTI-SKENARIO)
# ============================================================================

def main():
    """Menjalankan pipeline untuk semua skenario."""

    total_start = time.time()

    print_header("LINK PREDICTION PADA JARINGAN PPI KANKER PARU-PARU")
    print(f"  Jumlah skenario: {len(SCENARIOS)}")
    for i, s in enumerate(SCENARIOS, 1):
        print(f"    {i}. {s['label']} (threshold = {s['min_score']})")

    # Buat direktori output utama
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # Jalankan setiap skenario
    all_results = {}
    for i, scenario in enumerate(SCENARIOS, 1):
        print("\n")
        print("#" * 70)
        print(f"#  MENJALANKAN SKENARIO {i}/{len(SCENARIOS)}: {scenario['label']}")
        print("#" * 70)

        result = run_scenario(scenario)
        all_results[scenario["name"]] = result

    # ------------------------------------------------------------------
    # RINGKASAN PERBANDINGAN ANTAR SKENARIO
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start

    print("\n\n")
    print_header("PERBANDINGAN ANTAR SKENARIO")

    comparison_rows = []
    for name, result in all_results.items():
        scenario = result["scenario"]
        df = result["df_results"]
        best = df.iloc[0]
        data = result["data"]

        row = {
            "Skenario": scenario["label"],
            "Threshold": scenario["min_score"],
            "Protein": data["G_full"].number_of_nodes(),
            "Interaksi": data["G_full"].number_of_edges(),
            "Metode Terbaik": best["Method"],
            "AUC-ROC": best["AUC-ROC"],
            "AUPR": best["AUPR"],
        }
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)
    print(df_comparison.to_string(index=False))

    # Simpan perbandingan
    save_results_to_csv(
        df_comparison,
        os.path.join(CONFIG["results_dir"], "scenario_comparison.csv")
    )

    print(f"\n  Total waktu eksekusi: {total_elapsed:.1f} detik")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()

    # Otomatis buka dashboard setelah pipeline selesai
    print("\n[Info] Membuka dashboard...")
    import subprocess
    subprocess.Popen([
        sys.executable,
        os.path.join(BASE_DIR, "dashboard.py")
    ])
