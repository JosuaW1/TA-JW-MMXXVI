"""
data_preparation.py
===================
Modul untuk memuat, membersihkan, dan menyiapkan data PPI (Protein-Protein
Interaction) dari database STRING untuk eksperimen link prediction.

Tahapan utama:
1. Load data mentah dari file STRING (format TSV)
2. Filtering berdasarkan combined confidence score
3. Pembangunan graf NetworkX (undirected, unweighted)
4. Split edges menjadi training (90%) dan testing (10%)
5. Negative sampling untuk pasangan protein non-interaksi
6. Validasi konektivitas graf training

Referensi:
- STRING database: https://string-db.org/
- Format data STRING: protein1, protein2, combined_score (0-1000)
"""

import random
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. LOAD & CLEANING DATA
# ---------------------------------------------------------------------------

def load_string_data(filepath: str) -> pd.DataFrame:
    """
    Memuat data interaksi protein dari file STRING.

    Mendukung dua format:
    1. Format STRING standar (TXT): protein1, protein2, combined_score (0-1000)
    2. Format STRING TSV (dari web export): #node1, node2, ..., combined_score (0-1)

    Parameters
    ----------
    filepath : str
        Path ke file data STRING (.txt atau .tsv).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom [protein1, protein2, combined_score].
    """
    # Deteksi format berdasarkan ekstensi
    if filepath.endswith(".tsv"):
        df = pd.read_csv(filepath, sep="\t", header=0)
    else:
        df = pd.read_csv(filepath, sep=r"\s+", header=0)

    # Mapping kolom dari format STRING web export ke format standar
    column_mapping = {
        "#node1": "protein1",
        "node1": "protein1",
        "node2": "protein2",
    }
    df = df.rename(columns=column_mapping)

    # Validasi kolom yang dibutuhkan
    required_cols = {"protein1", "protein2", "combined_score"}
    actual_cols = set(df.columns)

    if not required_cols.issubset(actual_cols):
        raise ValueError(
            f"Kolom yang dibutuhkan: {required_cols}, "
            f"kolom yang ditemukan: {actual_cols}. "
            "Pastikan file memiliki kolom protein1/node1, protein2/node2, "
            "dan combined_score."
        )

    # Ambil hanya kolom yang dibutuhkan
    df = df[["protein1", "protein2", "combined_score"]].copy()

    # Hapus duplikat (STRING kadang menyimpan A-B dan B-A)
    # Normalisasi agar pasangan (A,B) dan (B,A) dianggap sama
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["protein1"], r["protein2"]])), axis=1
    )
    df = df.drop_duplicates(subset=["pair"])
    df = df.drop(columns=["pair"])

    # Hapus self-loop
    df = df[df["protein1"] != df["protein2"]].reset_index(drop=True)

    print(f"[Load] Total interaksi setelah cleaning: {len(df)}")
    print(f"[Load] Jumlah protein unik: "
          f"{len(set(df['protein1']) | set(df['protein2']))}")
    print(f"[Load] Range combined_score: "
          f"{df['combined_score'].min():.3f} - {df['combined_score'].max():.3f}")

    return df


# ---------------------------------------------------------------------------
# 2. FILTERING BERDASARKAN CONFIDENCE SCORE
# ---------------------------------------------------------------------------

def filter_by_confidence(df: pd.DataFrame,
                         min_score: float = 0.700) -> pd.DataFrame:
    """
    Filter interaksi berdasarkan combined confidence score dari STRING.

    Kategori confidence score STRING (skala 0-1):
    - Low confidence    : < 0.400
    - Medium confidence : 0.400 - 0.700
    - High confidence   : >= 0.700
    - Highest confidence: >= 0.900

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil load_string_data().
    min_score : float, default=0.700
        Threshold minimum confidence score. Default 0.700 (high confidence).

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah difilter.
    """
    filtered = df[df["combined_score"] >= min_score].reset_index(drop=True)

    print(f"[Filter] Threshold: {min_score}")
    print(f"[Filter] Interaksi sebelum filter: {len(df)}")
    print(f"[Filter] Interaksi setelah filter: {len(filtered)}")
    print(f"[Filter] Protein unik setelah filter: "
          f"{len(set(filtered['protein1']) | set(filtered['protein2']))}")

    return filtered


# ---------------------------------------------------------------------------
# 3. PEMBANGUNAN GRAF
# ---------------------------------------------------------------------------

def build_ppi_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Membangun graf PPI dari DataFrame interaksi.

    Graf bersifat undirected dan unweighted sesuai dengan desain penelitian.
    Setiap node merepresentasikan protein, setiap edge merepresentasikan
    interaksi yang telah tervalidasi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom [protein1, protein2].

    Returns
    -------
    nx.Graph
        Graf PPI undirected dan unweighted.
    """
    G = nx.Graph()
    edges = list(zip(df["protein1"], df["protein2"]))
    G.add_edges_from(edges)

    print(f"[Graf] Jumlah node (protein): {G.number_of_nodes()}")
    print(f"[Graf] Jumlah edge (interaksi): {G.number_of_edges()}")
    print(f"[Graf] Densitas graf: {nx.density(G):.6f}")

    # Cek konektivitas
    if nx.is_connected(G):
        print("[Graf] Graf terhubung (connected)")
    else:
        components = list(nx.connected_components(G))
        print(f"[Graf] Graf TIDAK terhubung - {len(components)} komponen")
        largest = max(components, key=len)
        print(f"[Graf] Komponen terbesar: {len(largest)} node")

    return G


def ensure_connected(G: nx.Graph) -> nx.Graph:
    """
    Mengambil komponen terhubung terbesar (Largest Connected Component / LCC)
    dari graf.

    Langkah ini penting karena metode link prediction berbasis topologi
    membutuhkan graf yang terhubung agar semua pasangan node memiliki
    path yang valid.

    Parameters
    ----------
    G : nx.Graph
        Graf PPI yang mungkin tidak terhubung.

    Returns
    -------
    nx.Graph
        Subgraf dari komponen terhubung terbesar.
    """
    if nx.is_connected(G):
        print("[LCC] Graf sudah terhubung, tidak perlu filtering.")
        return G

    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()

    print(f"[LCC] Node sebelum: {G.number_of_nodes()}, "
          f"sesudah: {G_lcc.number_of_nodes()}")
    print(f"[LCC] Edge sebelum: {G.number_of_edges()}, "
          f"sesudah: {G_lcc.number_of_edges()}")

    return G_lcc


# ---------------------------------------------------------------------------
# 4. SPLIT DATA: TRAINING & TESTING
# ---------------------------------------------------------------------------

def split_train_test(G: nx.Graph,
                     test_ratio: float = 0.1,
                     random_state: int = 42) -> tuple:
    """
    Membagi edges graf menjadi training set dan testing set.

    Proses split menjamin bahwa graf training tetap terhubung (connected)
    dengan cara hanya menghapus edge yang bukan bridge. Edge yang merupakan
    bridge (jembatan) tidak boleh dihapus karena akan memutus konektivitas
    graf.

    Parameters
    ----------
    G : nx.Graph
        Graf PPI yang terhubung (connected).
    test_ratio : float, default=0.1
        Proporsi edges untuk testing (10%).
    random_state : int, default=42
        Seed untuk reprodusibilitas.

    Returns
    -------
    tuple
        (G_train, test_edges)
        - G_train: graf training (nx.Graph)
        - test_edges: list of tuple, edges untuk testing
    """
    rng = random.Random(random_state)
    np.random.seed(random_state)

    edges = list(G.edges())
    num_test = int(len(edges) * test_ratio)

    print(f"[Split] Total edges: {len(edges)}")
    print(f"[Split] Target test edges: {num_test}")

    # Identifikasi bridges (edge yang tidak boleh dihapus)
    bridges = set(nx.bridges(G))
    print(f"[Split] Jumlah bridge edges: {len(bridges)}")

    # Kandidat edge yang bisa dihapus (bukan bridge)
    removable = [e for e in edges if e not in bridges
                 and (e[1], e[0]) not in bridges]
    rng.shuffle(removable)

    if len(removable) < num_test:
        print(f"[Split] WARNING: hanya {len(removable)} edge yang bisa "
              f"dihapus tanpa memutus graf (target: {num_test})")
        num_test = len(removable)

    test_edges = removable[:num_test]

    # Bangun graf training
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    # Validasi konektivitas
    assert nx.is_connected(G_train), \
        "GAGAL: Graf training tidak terhubung setelah split!"

    print(f"[Split] Edge training: {G_train.number_of_edges()}")
    print(f"[Split] Edge testing: {len(test_edges)}")
    print(f"[Split] Graf training terhubung: {nx.is_connected(G_train)}")

    return G_train, test_edges


# ---------------------------------------------------------------------------
# 5. NEGATIVE SAMPLING
# ---------------------------------------------------------------------------

def generate_negative_samples(G: nx.Graph,
                              num_samples: int,
                              random_state: int = 42) -> list:
    """
    Menghasilkan negative samples (pasangan protein yang TIDAK berinteraksi).

    Negative samples dipilih secara acak dari pasangan node yang tidak
    memiliki edge di graf LENGKAP (bukan graf training). Jumlah negative
    samples dibuat sama dengan jumlah positive testing edges agar dataset
    evaluasi seimbang (balanced).

    Parameters
    ----------
    G : nx.Graph
        Graf PPI lengkap (sebelum split).
    num_samples : int
        Jumlah negative samples yang diinginkan (= jumlah test edges).
    random_state : int, default=42
        Seed untuk reprodusibilitas.

    Returns
    -------
    list of tuple
        Daftar pasangan node yang tidak berinteraksi.
    """
    rng = random.Random(random_state)
    nodes = list(G.nodes())
    existing_edges = set(G.edges())
    negative_samples = []

    # Buat set edge yang sudah ada (kedua arah untuk pengecekan cepat)
    edge_set = set()
    for u, v in existing_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    attempts = 0
    max_attempts = num_samples * 20  # batas percobaan

    while len(negative_samples) < num_samples and attempts < max_attempts:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        attempts += 1

        if u == v:
            continue
        if (u, v) in edge_set:
            continue

        # Pastikan tidak ada duplikat
        pair = tuple(sorted([u, v]))
        if pair not in negative_samples:
            negative_samples.append(pair)
            edge_set.add((u, v))
            edge_set.add((v, u))

    # Konversi kembali ke list of tuple biasa
    negative_samples = [(u, v) for u, v in negative_samples]

    print(f"[NegSample] Target: {num_samples}, "
          f"dihasilkan: {len(negative_samples)}")

    return negative_samples


# ---------------------------------------------------------------------------
# 6. PIPELINE UTAMA DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_data(filepath: str,
                 min_score: int = 700,
                 test_ratio: float = 0.1,
                 random_state: int = 42) -> dict:
    """
    Pipeline lengkap data preparation.

    Menjalankan semua tahapan dari load data hingga negative sampling.

    Parameters
    ----------
    filepath : str
        Path ke file data STRING.
    min_score : int, default=700
        Threshold confidence score.
    test_ratio : float, default=0.1
        Proporsi testing edges.
    random_state : int, default=42
        Seed untuk reprodusibilitas.

    Returns
    -------
    dict
        Dictionary berisi:
        - 'G_full': graf PPI lengkap (setelah LCC)
        - 'G_train': graf training
        - 'test_edges': daftar edges testing (positive)
        - 'negative_samples': daftar pasangan non-interaksi (negative)
        - 'df_filtered': DataFrame interaksi setelah filtering
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n--- Tahap 1: Load Data ---")
    df = load_string_data(filepath)

    # 2. Filter confidence
    print("\n--- Tahap 2: Filter Confidence Score ---")
    df_filtered = filter_by_confidence(df, min_score=min_score)

    # 3. Bangun graf
    print("\n--- Tahap 3: Bangun Graf PPI ---")
    G = build_ppi_graph(df_filtered)

    # 4. Pastikan terhubung (ambil LCC)
    print("\n--- Tahap 4: Largest Connected Component ---")
    G_full = ensure_connected(G)

    # 5. Split training/testing
    print("\n--- Tahap 5: Split Training/Testing ---")
    G_train, test_edges = split_train_test(
        G_full, test_ratio=test_ratio, random_state=random_state
    )

    # 6. Negative sampling
    print("\n--- Tahap 6: Negative Sampling ---")
    negative_samples = generate_negative_samples(
        G_full, num_samples=len(test_edges), random_state=random_state
    )

    print("\n" + "=" * 60)
    print("RINGKASAN DATA PREPARATION")
    print("=" * 60)
    print(f"  Protein (node)      : {G_full.number_of_nodes()}")
    print(f"  Interaksi (edge)    : {G_full.number_of_edges()}")
    print(f"  Training edges      : {G_train.number_of_edges()}")
    print(f"  Testing edges (+)   : {len(test_edges)}")
    print(f"  Negative samples (-): {len(negative_samples)}")
    print(f"  Graf training connected: {nx.is_connected(G_train)}")
    print("=" * 60)

    return {
        "G_full": G_full,
        "G_train": G_train,
        "test_edges": test_edges,
        "negative_samples": negative_samples,
        "df_filtered": df_filtered,
    }
