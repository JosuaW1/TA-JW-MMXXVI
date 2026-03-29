"""
topology_methods.py
===================
Implementasi 7 metode link prediction berbasis fitur topologi jaringan.

Setiap metode menghitung skor kedekatan (proximity score) untuk pasangan
node berdasarkan struktur lokal atau semi-lokal dari graf. Skor yang lebih
tinggi mengindikasikan kemungkinan interaksi yang lebih besar.

Metode yang diimplementasikan:
1. Common Neighbors (CN)
2. Jaccard Coefficient (JC)
3. Adamic-Adar Index (AA)
4. Resource Allocation Index (RA)
5. Preferential Attachment (PA)
6. Local Path Index (LP)
7. L3 Index

Referensi utama:
- Lu, L., & Zhou, T. (2011). Link prediction in complex networks: A survey.
  Physica A, 390(6), 1150-1170.
- Kovacs, I.A., et al. (2019). Network-based prediction of protein
  interactions. Nature Communications, 10, 1240.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from tqdm import tqdm


# ---------------------------------------------------------------------------
# UTILITAS
# ---------------------------------------------------------------------------

def _get_adjacency_matrix(G: nx.Graph) -> tuple:
    """
    Mengonversi graf ke adjacency matrix sparse untuk komputasi efisien.

    Returns
    -------
    tuple
        (A, node_list) di mana A adalah scipy sparse matrix dan
        node_list adalah daftar node sesuai urutan indeks.
    """
    node_list = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=node_list)
    return A, node_list


def _node_to_index(node_list: list) -> dict:
    """Membuat mapping dari node ID ke indeks numerik."""
    return {node: idx for idx, node in enumerate(node_list)}


# ---------------------------------------------------------------------------
# 1. COMMON NEIGHBORS (CN)
# ---------------------------------------------------------------------------

def common_neighbors(G: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Menghitung skor Common Neighbors untuk daftar pasangan node.

    CN(u, v) = |N(u) ∩ N(v)|

    Skor CN adalah jumlah tetangga yang dimiliki bersama oleh dua node.
    Intuisi: semakin banyak tetangga bersama, semakin besar kemungkinan
    kedua node akan terhubung.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.

    Returns
    -------
    np.ndarray
        Array skor CN untuk setiap pasangan.
    """
    scores = []
    for u, v in edge_list:
        if G.has_node(u) and G.has_node(v):
            cn = len(set(G.neighbors(u)) & set(G.neighbors(v)))
            scores.append(cn)
        else:
            scores.append(0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. JACCARD COEFFICIENT (JC)
# ---------------------------------------------------------------------------

def jaccard_coefficient(G: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Menghitung skor Jaccard Coefficient untuk daftar pasangan node.

    JC(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    Normalisasi CN dengan ukuran gabungan tetangga. Metode ini mengurangi
    bias terhadap node berdegree tinggi.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.

    Returns
    -------
    np.ndarray
        Array skor JC untuk setiap pasangan (range [0, 1]).
    """
    scores = []
    for u, v in edge_list:
        if G.has_node(u) and G.has_node(v):
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            union = neighbors_u | neighbors_v
            if len(union) == 0:
                scores.append(0.0)
            else:
                scores.append(len(neighbors_u & neighbors_v) / len(union))
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. ADAMIC-ADAR INDEX (AA)
# ---------------------------------------------------------------------------

def adamic_adar(G: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Menghitung skor Adamic-Adar Index untuk daftar pasangan node.

    AA(u, v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / log(|N(w)|)

    Mirip CN, tetapi memberikan bobot lebih besar pada tetangga bersama
    yang memiliki sedikit koneksi (degree rendah). Node berdegree rendah
    yang menjadi tetangga bersama dianggap lebih informatif.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.

    Returns
    -------
    np.ndarray
        Array skor AA untuk setiap pasangan.
    """
    scores = []
    for u, v in edge_list:
        if G.has_node(u) and G.has_node(v):
            common = set(G.neighbors(u)) & set(G.neighbors(v))
            score = 0.0
            for w in common:
                deg_w = G.degree(w)
                if deg_w > 1:
                    score += 1.0 / np.log(deg_w)
            scores.append(score)
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# 4. RESOURCE ALLOCATION INDEX (RA)
# ---------------------------------------------------------------------------

def resource_allocation(G: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Menghitung skor Resource Allocation Index untuk daftar pasangan node.

    RA(u, v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / |N(w)|

    Terinspirasi dari proses alokasi sumber daya dalam jaringan.
    Mirip Adamic-Adar tetapi menggunakan 1/degree langsung (tanpa log).
    Memberikan penalti lebih besar pada hub nodes.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.

    Returns
    -------
    np.ndarray
        Array skor RA untuk setiap pasangan.
    """
    scores = []
    for u, v in edge_list:
        if G.has_node(u) and G.has_node(v):
            common = set(G.neighbors(u)) & set(G.neighbors(v))
            score = 0.0
            for w in common:
                deg_w = G.degree(w)
                if deg_w > 0:
                    score += 1.0 / deg_w
            scores.append(score)
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# 5. PREFERENTIAL ATTACHMENT (PA)
# ---------------------------------------------------------------------------

def preferential_attachment(G: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Menghitung skor Preferential Attachment untuk daftar pasangan node.

    PA(u, v) = |N(u)| × |N(v)|

    Berdasarkan mekanisme "rich-get-richer": node dengan banyak koneksi
    cenderung mendapatkan lebih banyak koneksi baru. Tidak mempertimbangkan
    tetangga bersama, hanya degree masing-masing node.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.

    Returns
    -------
    np.ndarray
        Array skor PA untuk setiap pasangan.
    """
    scores = []
    for u, v in edge_list:
        if G.has_node(u) and G.has_node(v):
            scores.append(G.degree(u) * G.degree(v))
        else:
            scores.append(0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# 6. LOCAL PATH INDEX (LP)
# ---------------------------------------------------------------------------

def local_path_index(G: nx.Graph, edge_list: list,
                     epsilon: float = 0.01) -> np.ndarray:
    """
    Menghitung skor Local Path Index untuk daftar pasangan node.

    LP(u, v) = |paths_length_2(u,v)| + ε × |paths_length_3(u,v)|
             = (A²)_{uv} + ε × (A³)_{uv}

    Memperluas CN dengan mempertimbangkan path dengan panjang 3.
    Parameter epsilon mengontrol kontribusi relatif path panjang 3.
    Memberikan informasi struktural yang lebih kaya dari CN.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.
    epsilon : float, default=0.01
        Parameter bobot untuk path panjang 3.

    Returns
    -------
    np.ndarray
        Array skor LP untuk setiap pasangan.
    """
    A, node_list = _get_adjacency_matrix(G)
    node_idx = _node_to_index(node_list)

    # Hitung A^2 dan A^3
    A2 = A @ A
    A3 = A2 @ A

    scores = []
    for u, v in edge_list:
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            # (A^2)_{ij} = jumlah path panjang 2, (A^3)_{ij} = path panjang 3
            score = A2[i, j] + epsilon * A3[i, j]
            scores.append(score)
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# 7. L3 INDEX
# ---------------------------------------------------------------------------

def l3_index(G: nx.Graph, edge_list: list) -> np.ndarray:
    """
    Menghitung skor L3 (Path of Length 3) Index untuk daftar pasangan node.

    L3(u, v) = Σ_{s,t} 1/(k_s × k_t)
    di mana s ∈ N(u), t ∈ N(v), dan (s,t) ∈ E

    Berdasarkan paper Kovacs et al. (2019) yang menunjukkan bahwa pada
    jaringan PPI, protein cenderung berinteraksi melalui path panjang 3
    (ganjil) daripada panjang 2 (genap). Ini disebut "complementarity"
    dalam interaksi biologis.

    Setiap path panjang 3 dinormalisasi dengan degree dari node perantara
    untuk menghindari bias terhadap hub.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.

    Returns
    -------
    np.ndarray
        Array skor L3 untuk setiap pasangan.
    """
    scores = []
    for u, v in edge_list:
        if G.has_node(u) and G.has_node(v):
            score = 0.0
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            for s in neighbors_u:
                deg_s = G.degree(s)
                for t in neighbors_v:
                    if G.has_edge(s, t):
                        deg_t = G.degree(t)
                        if deg_s > 0 and deg_t > 0:
                            score += 1.0 / (deg_s * deg_t)
            scores.append(score)
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# FUNGSI AGREGAT: HITUNG SEMUA SKOR
# ---------------------------------------------------------------------------

# Registry semua metode dan nama pendeknya
METHOD_REGISTRY = {
    "Common Neighbors": common_neighbors,
    "Jaccard Coefficient": jaccard_coefficient,
    "Adamic-Adar": adamic_adar,
    "Resource Allocation": resource_allocation,
    "Preferential Attachment": preferential_attachment,
    "Local Path": local_path_index,
    "L3 Index": l3_index,
}


def compute_all_scores(G: nx.Graph, edge_list: list,
                       verbose: bool = True) -> dict:
    """
    Menghitung skor dari semua 7 metode link prediction untuk daftar
    pasangan node.

    Parameters
    ----------
    G : nx.Graph
        Graf training.
    edge_list : list of tuple
        Daftar pasangan node yang akan dihitung skornya.
    verbose : bool, default=True
        Tampilkan progress bar.

    Returns
    -------
    dict
        Dictionary {nama_metode: np.ndarray skor}.
    """
    all_scores = {}
    methods = METHOD_REGISTRY.items()
    if verbose:
        methods = tqdm(list(methods), desc="Menghitung skor topologi")

    for name, func in methods:
        all_scores[name] = func(G, edge_list)
        if verbose:
            # Tampilkan statistik ringkas
            s = all_scores[name]
            nonzero = np.count_nonzero(s)
            print(f"  {name}: min={s.min():.4f}, max={s.max():.4f}, "
                  f"mean={s.mean():.4f}, nonzero={nonzero}/{len(s)}")

    return all_scores


def build_feature_matrix(scores_dict: dict) -> np.ndarray:
    """
    Membangun feature matrix dari dictionary skor semua metode.

    Setiap kolom merepresentasikan skor dari satu metode topologi.
    Matrix ini akan digunakan sebagai input untuk model Random Forest.

    Parameters
    ----------
    scores_dict : dict
        Dictionary {nama_metode: np.ndarray skor} dari compute_all_scores().

    Returns
    -------
    np.ndarray
        Feature matrix dengan shape (n_samples, 7).
    """
    feature_names = list(scores_dict.keys())
    features = np.column_stack([scores_dict[name] for name in feature_names])
    return features
