"""
evaluation.py
=============
Modul evaluasi untuk membandingkan performa semua metode link prediction.

Metrik evaluasi yang digunakan:
1. AUC-ROC (Area Under the ROC Curve)
   - Mengukur kemampuan model membedakan antara pasangan positif dan negatif
   - Range [0, 1], nilai 0.5 = random, 1.0 = sempurna

2. AUPR (Area Under the Precision-Recall Curve)
   - Lebih informatif daripada AUC-ROC ketika data tidak seimbang
   - Fokus pada performa untuk kelas positif
   - Range [0, 1]

3. Top-k Precision
   - Mengukur proporsi prediksi benar dalam k prediksi teratas
   - Relevan untuk aplikasi praktis: dari sekian kandidat teratas,
     berapa banyak yang benar-benar interaksi?
   - Dihitung untuk k = 10, 50, 100, 200, 500

Referensi:
- Yang, Y., et al. (2015). Evaluating link prediction methods.
  Knowledge and Information Systems, 45(3), 751-782.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# METRIK EVALUASI INDIVIDUAL
# ---------------------------------------------------------------------------

def compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Menghitung AUC-ROC (Area Under the Receiver Operating Characteristic Curve).

    Parameters
    ----------
    y_true : np.ndarray
        Label sebenarnya (0 atau 1).
    y_scores : np.ndarray
        Skor prediksi (nilai kontinu).

    Returns
    -------
    float
        Nilai AUC-ROC.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_scores)


def compute_aupr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Menghitung AUPR (Area Under the Precision-Recall Curve).

    AUPR lebih cocok untuk evaluasi link prediction karena:
    - Dataset biasanya sangat sparse (banyak negatif)
    - Fokus pada kemampuan menemukan interaksi yang benar (positif)

    Parameters
    ----------
    y_true : np.ndarray
        Label sebenarnya (0 atau 1).
    y_scores : np.ndarray
        Skor prediksi (nilai kontinu).

    Returns
    -------
    float
        Nilai AUPR.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_scores)


def compute_top_k_precision(y_true: np.ndarray,
                            y_scores: np.ndarray,
                            k: int) -> float:
    """
    Menghitung Top-k Precision.

    Mengurutkan pasangan berdasarkan skor prediksi dari tertinggi ke
    terendah, lalu menghitung proporsi label positif dalam k teratas.

    Parameters
    ----------
    y_true : np.ndarray
        Label sebenarnya (0 atau 1).
    y_scores : np.ndarray
        Skor prediksi (nilai kontinu).
    k : int
        Jumlah prediksi teratas yang dievaluasi.

    Returns
    -------
    float
        Precision pada top-k prediksi.
    """
    if k <= 0 or k > len(y_true):
        k = min(k, len(y_true))
    if k == 0:
        return 0.0

    # Urutkan berdasarkan skor (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k_labels = y_true[sorted_indices[:k]]

    return np.sum(top_k_labels) / k


def compute_accuracy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Menghitung Accuracy menggunakan threshold optimal dari kurva ROC.

    Threshold dipilih sebagai titik pada kurva ROC yang memaksimalkan
    Youden's J statistic (TPR - FPR), sehingga menghasilkan keseimbangan
    terbaik antara sensitivitas dan spesifisitas.

    Parameters
    ----------
    y_true : np.ndarray
        Label sebenarnya (0 atau 1).
    y_scores : np.ndarray
        Skor prediksi (nilai kontinu).

    Returns
    -------
    float
        Nilai accuracy pada threshold optimal.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Youden's J statistic: optimal threshold = max(TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred = (y_scores >= best_threshold).astype(int)
    return accuracy_score(y_true, y_pred)


def compute_confusion_matrix(y_true: np.ndarray,
                             y_scores: np.ndarray) -> np.ndarray:
    """
    Menghitung Confusion Matrix menggunakan threshold optimal dari kurva ROC.

    Parameters
    ----------
    y_true : np.ndarray
        Label sebenarnya (0 atau 1).
    y_scores : np.ndarray
        Skor prediksi (nilai kontinu).

    Returns
    -------
    np.ndarray
        Confusion matrix 2x2: [[TN, FP], [FN, TP]].
    """
    if len(np.unique(y_true)) < 2:
        return np.zeros((2, 2), dtype=int)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred = (y_scores >= best_threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# EVALUASI SATU METODE
# ---------------------------------------------------------------------------

def evaluate_method(method_name: str,
                    scores_positive: np.ndarray,
                    scores_negative: np.ndarray,
                    k_values: list = None) -> dict:
    """
    Evaluasi lengkap untuk satu metode link prediction.

    Parameters
    ----------
    method_name : str
        Nama metode.
    scores_positive : np.ndarray
        Skor untuk pasangan positif (interaksi ada).
    scores_negative : np.ndarray
        Skor untuk pasangan negatif (interaksi tidak ada).
    k_values : list of int, optional
        Daftar nilai k untuk Top-k Precision.
        Default: [10, 50, 100, 200, 500].

    Returns
    -------
    dict
        Dictionary berisi semua metrik evaluasi.
    """
    if k_values is None:
        k_values = [10, 50, 100, 200, 500]

    # Gabungkan skor dan label
    y_scores = np.concatenate([scores_positive, scores_negative])
    y_true = np.concatenate([
        np.ones(len(scores_positive)),
        np.zeros(len(scores_negative))
    ])

    # Hitung metrik
    auc_roc = compute_auc_roc(y_true, y_scores)
    aupr = compute_aupr(y_true, y_scores)
    acc = compute_accuracy(y_true, y_scores)
    cm = compute_confusion_matrix(y_true, y_scores)

    result = {
        "Method": method_name,
        "AUC-ROC": auc_roc,
        "AUPR": aupr,
        "Accuracy": acc,
    }

    # Top-k Precision
    for k in k_values:
        if k <= len(y_true):
            result[f"Top-{k} Prec"] = compute_top_k_precision(
                y_true, y_scores, k
            )

    # Simpan data mentah untuk kurva dan confusion matrix
    result["_y_true"] = y_true
    result["_y_scores"] = y_scores
    result["_confusion_matrix"] = cm

    return result


# ---------------------------------------------------------------------------
# EVALUASI SEMUA METODE
# ---------------------------------------------------------------------------

def evaluate_all_methods(scores_positive: dict,
                         scores_negative: dict,
                         rf_pred_positive: np.ndarray = None,
                         rf_pred_negative: np.ndarray = None,
                         k_values: list = None) -> pd.DataFrame:
    """
    Evaluasi dan bandingkan semua 8 metode link prediction.

    Parameters
    ----------
    scores_positive : dict
        Dictionary skor 7 metode untuk pasangan positif.
    scores_negative : dict
        Dictionary skor 7 metode untuk pasangan negatif.
    rf_pred_positive : np.ndarray, optional
        Prediksi Random Forest untuk pasangan positif.
    rf_pred_negative : np.ndarray, optional
        Prediksi Random Forest untuk pasangan negatif.
    k_values : list of int, optional
        Daftar nilai k untuk Top-k Precision.

    Returns
    -------
    pd.DataFrame
        Tabel perbandingan performa semua metode.
    """
    print("=" * 60)
    print("EVALUASI SEMUA METODE LINK PREDICTION")
    print("=" * 60)

    results = []
    curve_data = {}

    # Evaluasi 7 metode topologi
    for method_name in scores_positive.keys():
        print(f"\n  Evaluasi: {method_name}")
        result = evaluate_method(
            method_name,
            scores_positive[method_name],
            scores_negative[method_name],
            k_values=k_values
        )
        curve_data[method_name] = {
            "y_true": result.pop("_y_true"),
            "y_scores": result.pop("_y_scores"),
            "confusion_matrix": result.pop("_confusion_matrix"),
        }
        results.append(result)
        print(f"    AUC-ROC: {result['AUC-ROC']:.4f}, "
              f"AUPR: {result['AUPR']:.4f}, "
              f"Accuracy: {result['Accuracy']:.4f}")

    # Evaluasi Random Forest (metode ke-8)
    if rf_pred_positive is not None and rf_pred_negative is not None:
        print(f"\n  Evaluasi: Random Forest (Combined)")
        result = evaluate_method(
            "Random Forest",
            rf_pred_positive,
            rf_pred_negative,
            k_values=k_values
        )
        curve_data["Random Forest"] = {
            "y_true": result.pop("_y_true"),
            "y_scores": result.pop("_y_scores"),
            "confusion_matrix": result.pop("_confusion_matrix"),
        }
        results.append(result)
        print(f"    AUC-ROC: {result['AUC-ROC']:.4f}, "
              f"AUPR: {result['AUPR']:.4f}, "
              f"Accuracy: {result['Accuracy']:.4f}")

    # Buat DataFrame hasil
    df_results = pd.DataFrame(results)

    # Urutkan berdasarkan AUC-ROC
    df_results = df_results.sort_values(
        "AUC-ROC", ascending=False
    ).reset_index(drop=True)

    # Tentukan metode terbaik
    best = df_results.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"METODE TERBAIK: {best['Method']}")
    print(f"  AUC-ROC: {best['AUC-ROC']:.4f}")
    print(f"  AUPR: {best['AUPR']:.4f}")
    print(f"{'=' * 60}")

    return df_results, curve_data


# ---------------------------------------------------------------------------
# TABEL HASIL LENGKAP
# ---------------------------------------------------------------------------

def format_results_table(df_results: pd.DataFrame) -> str:
    """
    Memformat tabel hasil evaluasi untuk ditampilkan di console/laporan.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame hasil dari evaluate_all_methods().

    Returns
    -------
    str
        Tabel terformat sebagai string.
    """
    # Buat salinan tanpa kolom internal
    display_df = df_results.copy()

    # Format angka
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    return display_df.to_string(index=False)


def get_roc_curve_data(curve_data: dict) -> dict:
    """
    Menghitung data kurva ROC untuk setiap metode.

    Parameters
    ----------
    curve_data : dict
        Dictionary {method: {y_true, y_scores}} dari evaluate_all_methods().

    Returns
    -------
    dict
        Dictionary {method: {fpr, tpr, auc}}.
    """
    roc_data = {}
    for method, data in curve_data.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_scores"])
        auc = roc_auc_score(data["y_true"], data["y_scores"])
        roc_data[method] = {"fpr": fpr, "tpr": tpr, "auc": auc}
    return roc_data


def get_pr_curve_data(curve_data: dict) -> dict:
    """
    Menghitung data kurva Precision-Recall untuk setiap metode.

    Parameters
    ----------
    curve_data : dict
        Dictionary {method: {y_true, y_scores}} dari evaluate_all_methods().

    Returns
    -------
    dict
        Dictionary {method: {precision, recall, aupr}}.
    """
    pr_data = {}
    for method, data in curve_data.items():
        precision, recall, _ = precision_recall_curve(
            data["y_true"], data["y_scores"]
        )
        aupr = average_precision_score(data["y_true"], data["y_scores"])
        pr_data[method] = {
            "precision": precision,
            "recall": recall,
            "aupr": aupr,
        }
    return pr_data
