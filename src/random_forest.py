"""
random_forest.py
================
Model Machine Learning (Random Forest) sebagai metode link prediction ke-8.

Modul ini menggunakan skor dari 7 metode topologi sebagai fitur input
untuk melatih Random Forest Classifier. Pendekatan ini menggabungkan
informasi dari berbagai metode topologi untuk menghasilkan prediksi
yang lebih akurat.

Alasan pemilihan Random Forest:
1. Mampu menangkap hubungan non-linear antar fitur topologi
2. Robust terhadap ketidakseimbangan kelas (class imbalance)
3. Menghasilkan feature importance untuk analisis kontribusi tiap metode
4. Tidak memerlukan feature scaling
5. Resisten terhadap overfitting dengan ensemble of trees

Referensi:
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PERSIAPAN DATA UNTUK ML
# ---------------------------------------------------------------------------

def prepare_ml_data(scores_positive: dict,
                    scores_negative: dict) -> tuple:
    """
    Menyiapkan data training untuk model Random Forest.

    Menggabungkan skor dari pasangan positif (interaksi ada) dan negatif
    (interaksi tidak ada) menjadi feature matrix X dan label vector y.

    Parameters
    ----------
    scores_positive : dict
        Dictionary skor 7 metode untuk test edges (label=1).
    scores_negative : dict
        Dictionary skor 7 metode untuk negative samples (label=0).

    Returns
    -------
    tuple
        (X, y, feature_names)
        - X: np.ndarray feature matrix (n_samples, 7)
        - y: np.ndarray label vector (n_samples,)
        - feature_names: list of str, nama-nama fitur
    """
    feature_names = list(scores_positive.keys())

    # Bangun feature matrix untuk positif dan negatif
    X_pos = np.column_stack(
        [scores_positive[name] for name in feature_names]
    )
    X_neg = np.column_stack(
        [scores_negative[name] for name in feature_names]
    )

    # Gabungkan dan buat label
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.ones(len(X_pos)),
        np.zeros(len(X_neg))
    ])

    print(f"[ML Data] Jumlah sampel positif: {len(X_pos)}")
    print(f"[ML Data] Jumlah sampel negatif: {len(X_neg)}")
    print(f"[ML Data] Total sampel: {len(X)}")
    print(f"[ML Data] Jumlah fitur: {len(feature_names)}")
    print(f"[ML Data] Nama fitur: {feature_names}")

    return X, y, feature_names


# ---------------------------------------------------------------------------
# TRAINING MODEL
# ---------------------------------------------------------------------------

def train_random_forest(X: np.ndarray,
                        y: np.ndarray,
                        feature_names: list,
                        n_estimators: int = 100,
                        max_depth: int = None,
                        random_state: int = 42) -> dict:
    """
    Melatih model Random Forest Classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Label vector (0 atau 1).
    feature_names : list of str
        Nama-nama fitur untuk analisis feature importance.
    n_estimators : int, default=100
        Jumlah decision tree dalam ensemble.
    max_depth : int, default=None
        Kedalaman maksimum tree. None berarti node akan terus split
        sampai semua leaf murni.
    random_state : int, default=42
        Seed untuk reprodusibilitas.

    Returns
    -------
    dict
        Dictionary berisi:
        - 'model': trained RandomForestClassifier
        - 'feature_importance': pd.DataFrame dengan importance tiap fitur
        - 'cv_scores': skor cross-validation
    """
    print("=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    # Inisialisasi model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Cross-validation untuk estimasi performa
    print("\n--- Cross-Validation (5-Fold) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"  AUC-ROC per fold: {cv_auc}")
    print(f"  Mean AUC-ROC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    # Training pada seluruh data
    print("\n--- Training pada seluruh data ---")
    rf.fit(X, y)

    # Feature importance
    importance = rf.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\n--- Feature Importance ---")
    for _, row in fi_df.iterrows():
        bar = "#" * int(row["Importance"] * 50)
        print(f"  {row['Feature']:25s} {row['Importance']:.4f} {bar}")

    return {
        "model": rf,
        "feature_importance": fi_df,
        "cv_scores": cv_auc,
    }


# ---------------------------------------------------------------------------
# PREDIKSI
# ---------------------------------------------------------------------------

def predict_scores(model: RandomForestClassifier,
                   scores_dict: dict) -> np.ndarray:
    """
    Menghasilkan skor prediksi dari model Random Forest.

    Menggunakan probabilitas kelas positif (P(y=1|X)) sebagai skor
    prediksi, yang memberikan nilai kontinu antara 0 dan 1.

    Parameters
    ----------
    model : RandomForestClassifier
        Model yang sudah dilatih.
    scores_dict : dict
        Dictionary skor 7 metode untuk pasangan yang akan diprediksi.

    Returns
    -------
    np.ndarray
        Array probabilitas prediksi (skor antara 0 dan 1).
    """
    feature_names = list(scores_dict.keys())
    X = np.column_stack([scores_dict[name] for name in feature_names])
    proba = model.predict_proba(X)[:, 1]
    return proba


# ---------------------------------------------------------------------------
# PIPELINE RANDOM FOREST
# ---------------------------------------------------------------------------

def run_random_forest_pipeline(G_train,
                               test_edges: list,
                               negative_samples: list,
                               compute_all_scores_func,
                               random_state: int = 42) -> dict:
    """
    Pipeline lengkap untuk metode Random Forest.

    Tahapan:
    1. Hitung skor 7 metode topologi untuk test edges dan negative samples
    2. Siapkan data ML (feature matrix + labels)
    3. Latih Random Forest
    4. Hasilkan prediksi

    Parameters
    ----------
    G_train : nx.Graph
        Graf training.
    test_edges : list of tuple
        Daftar test edges (positif).
    negative_samples : list of tuple
        Daftar negative samples.
    compute_all_scores_func : callable
        Fungsi compute_all_scores dari topology_methods.
    random_state : int, default=42
        Seed untuk reprodusibilitas.

    Returns
    -------
    dict
        Dictionary berisi model, feature importance, prediksi, dan data ML.
    """
    # 1. Hitung skor untuk positive (test edges)
    print("\n--- Hitung skor topologi: Test Edges (Positif) ---")
    scores_positive = compute_all_scores_func(G_train, test_edges)

    # 2. Hitung skor untuk negative samples
    print("\n--- Hitung skor topologi: Negative Samples ---")
    scores_negative = compute_all_scores_func(G_train, negative_samples)

    # 3. Persiapan data ML
    print("\n--- Persiapan Data ML ---")
    X, y, feature_names = prepare_ml_data(scores_positive, scores_negative)

    # 4. Training
    rf_result = train_random_forest(
        X, y, feature_names, random_state=random_state
    )

    # 5. Prediksi skor untuk semua pasangan (positif + negatif)
    all_pairs = test_edges + negative_samples
    all_scores = compute_all_scores_func(G_train, all_pairs, verbose=False)
    rf_predictions = predict_scores(rf_result["model"], all_scores)

    # Pisahkan prediksi kembali
    n_pos = len(test_edges)
    pred_positive = rf_predictions[:n_pos]
    pred_negative = rf_predictions[n_pos:]

    return {
        "model": rf_result["model"],
        "feature_importance": rf_result["feature_importance"],
        "cv_scores": rf_result["cv_scores"],
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "scores_positive": scores_positive,
        "scores_negative": scores_negative,
        "rf_pred_positive": pred_positive,
        "rf_pred_negative": pred_negative,
    }
