from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
import re
from sklearn.mixture import GaussianMixture
from umap import UMAP
from ale.scdeed import scDEED
from tqdm.notebook import tqdm
from hdbscan import HDBSCAN  # Assicurati di avere installato la libreria hdbscan

class ProgressWriter:
    def write(self, text):
        match = re.search(r"(\d+)/(\d+)", text)
        if match:
            n, total = map(int, match.groups())
            print("custom progress", n, total)
            # custom reporting logic here

    def flush(self):
        pass

tqdm_kwds = {"file": "progress_writer", "disable": False }

#############################################
# Funzioni di clustering e metriche

def calculate_combined_score(ari_mean, ari_std, silhouette_mean, silhouette_std):
    """
    Calcola un combined score complesso basato su ARI e Silhouette, ponderato in base alla deviazione standard
    e con penalità non lineari per punteggi bassi.
    """
    # Step 1: Normalizza gli score tra 0 e 1
    ari_normalized = ari_mean / (ari_mean + 1)
    silhouette_normalized = silhouette_mean / (silhouette_mean + 1)

    # Step 2: Ponderazione dinamica in base alla stabilità (deviazione standard)
    ari_weight = 1 / (1 + np.exp(ari_std))  # Più alta è la deviazione, meno peso ha
    silhouette_weight = 1 / (1 + np.exp(silhouette_std))

    # Step 3: Penalità non lineari se ARI o Silhouette scendono sotto una certa soglia
    penalty_factor_ari = 1 if ari_mean >= 0.7 else np.exp(-(0.7 - ari_mean) * 10)
    penalty_factor_silhouette = 1 if silhouette_mean >= 0.7 else np.exp(-(0.7 - silhouette_mean) * 10)

    # Step 4: Calcola i punteggi ponderati e penalizzati
    ari_adjusted = ari_normalized * ari_weight * penalty_factor_ari
    silhouette_adjusted = silhouette_normalized * silhouette_weight * penalty_factor_silhouette

    # Step 5: Usa una funzione non lineare per combinare i due score
    combined_score = np.sqrt(ari_adjusted ** 2 + silhouette_adjusted ** 2)

    return combined_score

def apply_clustering(X, k, hdbscan_params=None):
    """
    Applica tre algoritmi di clustering (KMeans, Agglomerative, Spectral)
    e l'HDBSCAN (usando i parametri passati se disponibili).
    """
    results = {}

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    results['KMeans'] = kmeans_labels

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=k)
    agglo_labels = agglo.fit_predict(X)
    results['Agglomerative'] = agglo_labels

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral.fit_predict(X)
    results['Spectral'] = spectral_labels

    # HDBSCAN: non è legato a k, ma usiamo i parametri ottimizzati se disponibili
    if hdbscan_params:
        hdbscan_model = HDBSCAN(**hdbscan_params)
    else:
        hdbscan_model = HDBSCAN(min_cluster_size=10)
    hdbscan_labels = hdbscan_model.fit_predict(X)
    results['HDBSCAN'] = hdbscan_labels

    return results

def calculate_ari_consensus(clusterings):
    """
    Calcola la media dell'Adjusted Rand Index (ARI)
    tra ogni coppia di algoritmi di clustering.
    """
    algorithms = list(clusterings.keys())
    ari_scores = []
    
    # Confronta ogni coppia di algoritmi e calcola l'ARI
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            labels1 = clusterings[algorithms[i]]
            labels2 = clusterings[algorithms[j]]
            ari = adjusted_rand_score(labels1, labels2)
            ari_scores.append((algorithms[i], algorithms[j], ari))
    
    return ari_scores

#############################################
# Funzione per la ricerca dei migliori parametri HDBSCAN

def hdbscan_parameter_search(X, param_grid):
    """
    Esplora una griglia di parametri per HDBSCAN e restituisce
    la combinazione che massimizza il silhouette score.
    
    :param X: dati (embedding) su cui eseguire HDBSCAN
    :param param_grid: dizionario con le liste dei valori per 'min_cluster_size' e 'min_samples'
    :return: (best_params, best_labels, best_score)
    """
    best_score = -np.inf
    best_params = None
    best_labels = None
    
    for min_cluster_size in param_grid.get("min_cluster_size", [5, 10, 15, 20]):
        for min_samples in param_grid.get("min_samples", [None, 5, 10]):
            # Se min_samples è None, HDBSCAN usa il valore di default
            if min_samples is None:
                model = HDBSCAN(min_cluster_size=min_cluster_size)
            else:
                model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            
            labels = model.fit_predict(X)
            # Verifica che vengano trovati almeno 2 cluster (escludendo il rumore) 
            if len(set(labels)) < 2 or (set(labels) == {-1}):
                continue
            
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = -np.inf  # Se il calcolo fallisce, ignora questa combinazione
                
            if score > best_score:
                best_score = score
                best_params = {"min_cluster_size": min_cluster_size}
                if min_samples is not None:
                    best_params["min_samples"] = min_samples
                best_labels = labels

    return best_params, best_labels, best_score

#############################################
# Funzione optimizer integrata

def optimizer(X, n_neighbors=[5, 15, 30, 40, 50], min_dist=[0.1, 0.3, 0.0001, 0.01], k_min=15, k_max=60):
    num_rows = X.shape[0]

    if num_rows > k_min and num_rows < k_max:
        k_range = np.linspace(15, num_rows - 1, min(5, num_rows - 15 + 1), dtype=int)
    elif num_rows < k_min:
        k_range = np.arange(1, num_rows - 1)
    else:
        k_range = np.arange(k_min, k_max - 1, 5)
    
    print("Valori di k testati:", k_range)

    best_overall_score = -np.inf
    best_overall_k = None
    best_overall_clustering = None
    silhouette_scores = {}
    ari_combinations = []
    
    k_values = []
    ari_ratios = []
    silhouette_ratios = []
    combined_scores = []

    # Ottieni una lista di possibili parametri UMAP tramite scDEED
    param_list = scDEED(X, K=3, n_neighbors=n_neighbors, min_dist=min_dist)
    best_umap_params = None
    fewest_dubious_cells = np.inf
    highest_trustworthy_cells = -1

    for umap_params, cell_data in param_list:
        dubious_cells_count = len(cell_data['dubious_cells'])
        trustworthy_cells_count = len(cell_data['trustworthy_cells'])

        if dubious_cells_count < fewest_dubious_cells or (dubious_cells_count == fewest_dubious_cells):
            fewest_dubious_cells = dubious_cells_count
            highest_trustworthy_cells = trustworthy_cells_count
            best_umap_params = umap_params

    # Se abbiamo una configurazione UMAP scelta, calcoliamo l'embedding
    if best_umap_params:
        umap = UMAP(n_neighbors=best_umap_params['n_neighbors'], 
                    min_dist=best_umap_params['min_dist'], 
                    n_components=2, metric='cosine')
        umap_embedding = umap.fit_transform(X)
        
        # Ricerca dei migliori parametri per HDBSCAN sull'embedding UMAP
        hdbscan_param_grid = {"min_cluster_size": [5, 10, 15, 20], "min_samples": [None, 5, 10]}
        best_hdbscan_params, best_hdbscan_labels, hdbscan_score = hdbscan_parameter_search(umap_embedding, hdbscan_param_grid)
        print("Migliori parametri HDBSCAN trovati:", best_hdbscan_params, "con silhouette score:", hdbscan_score)

        for k in tqdm(k_range):
            # Applica clustering: i metodi basati su k e HDBSCAN con i parametri ottimizzati
            clusterings = apply_clustering(umap_embedding, k, hdbscan_params=best_hdbscan_params)
            consensus_ari = calculate_ari_consensus(clusterings)

            for alg1, alg2, ari in consensus_ari:
                ari_combinations.append((k, alg1, alg2, ari))

            silhouette_scores[k] = {
                'KMeans': silhouette_score(umap_embedding, clusterings['KMeans']),
                'Agglomerative': silhouette_score(umap_embedding, clusterings['Agglomerative']),
                'Spectral': silhouette_score(umap_embedding, clusterings['Spectral']),
                'HDBSCAN': silhouette_score(umap_embedding, clusterings['HDBSCAN'])
            }

            ari_mean = np.mean([ari for _, _, ari in consensus_ari])
            ari_std = np.std([ari for _, _, ari in consensus_ari])

            silhouette_mean = np.mean(list(silhouette_scores[k].values()))
            silhouette_std = np.std(list(silhouette_scores[k].values()))

            ari_ratio = ari_mean * (1 + np.exp(-ari_std)) * np.log(3)
            silhouette_ratio = silhouette_mean * (1 + np.exp(-silhouette_std)) * np.log(3)

            combined_score = calculate_combined_score(ari_mean, ari_std, silhouette_mean, silhouette_std)

            k_values.append(k)
            ari_ratios.append(ari_ratio)
            silhouette_ratios.append(silhouette_ratio)
            combined_scores.append(combined_score)

            if combined_score > best_overall_score:
                best_overall_score = combined_score
                best_overall_k = k
                best_overall_clustering = clusterings

    return {
        'best_k': best_overall_k,
        'best_clustering': best_overall_clustering,
        'silhouette_scores': silhouette_scores,
        'ari_combinations': ari_combinations,
        'umap_embedding': umap_embedding,
        'best_umap_params': best_umap_params,
        'best_hdbscan_params': best_hdbscan_params,
        'hdbscan_score': hdbscan_score,
        'k_values': k_values,
        'ari_ratios': ari_ratios,
        'silhouette_ratios': silhouette_ratios,
        'combined_scores': combined_scores
    }

#############################################
# Funzioni di visualizzazione (rimangono invariati)

def plot_ari_combinations(result):
    ari_combinations = result['ari_combinations']
    ks = sorted(set([k for k, _, _, _ in ari_combinations]))
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crea un dizionario per memorizzare le ARI per ogni combinazione di algoritmi
    ari_data = {
        'KMeans-Agglomerative': [], 
        'KMeans-Spectral': [], 
        'KMeans-HDBSCAN': [],
        'Agglomerative-Spectral': [], 
        'Agglomerative-HDBSCAN': [],
        'Spectral-HDBSCAN': []
    }
    
    for k in ks:
        for (k_val, alg1, alg2, ari) in ari_combinations:
            if k_val == k:
                # Gestisce entrambi gli ordini
                combo = f'{alg1}-{alg2}' if f'{alg1}-{alg2}' in ari_data else f'{alg2}-{alg1}'
                ari_data[combo].append(ari)
    
    # Plot delle ARI per ogni coppia di algoritmi
    for combo, ari_scores in ari_data.items():
        ax.plot(ks, ari_scores, label=combo)

    ax.set_title("ARI tra le combinazioni di algoritmi per ogni valore di k")
    ax.set_xlabel("Numero di cluster (k)")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.legend()
    plt.show()
    
def plot_ari_combinations_product(result):
    ari_combinations = result['ari_combinations']
    ks = sorted(set([k for k, _, _, _ in ari_combinations]))
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crea una lista per memorizzare il prodotto degli ARI per ogni k
    ari_products = []

    for k in ks:
        ari_product = 1  # Inizializza il prodotto degli ARI a 1 per ogni valore di k
        for (k_val, _, _, ari) in ari_combinations:
            if k_val == k:
                ari_product *= ari  # Moltiplica tutti gli ARI per lo stesso valore di k
        
        ari_products.append(ari_product)
    
    # Plot del prodotto totale degli ARI per ogni valore di k
    ax.plot(ks, ari_products, label='Prodotto degli ARI')
    ax.set_title("Prodotto degli ARI per ogni valore di k")
    ax.set_xlabel("Numero di cluster (k)")
    ax.set_ylabel("Prodotto Adjusted Rand Index (ARI)")
    ax.legend()
    plt.show()
    
def plot_ari_combinations_stability_mean_ratio(result):
    ari_combinations = result['ari_combinations']
    ks = sorted(set([k for k, _, _, _ in ari_combinations]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ari_mean_std_ratio = []

    for k in ks:
        ari_values = [ari for (k_val, _, _, ari) in ari_combinations if k_val == k]
        ari_mean = np.mean(ari_values)
        ari_std = np.std(ari_values)
        
        if ari_std > 0:
            ratio = ari_mean * (1 + np.exp(-ari_std)) * np.log(3)
        else:
            ratio = float('inf')
        
        ari_mean_std_ratio.append(ratio)
    
    ax.plot(ks, ari_mean_std_ratio, label='Media ARI / Deviazione standard ARI', marker='o')
    ax.set_title("Rapporto Media ARI / Deviazione standard ARI per ogni valore di k")
    ax.set_xlabel("Numero di cluster (k)")
    ax.set_ylabel("Rapporto Media ARI / Deviazione standard ARI")
    ax.legend()
    plt.show()

def plot_opt_weight(k_values, ari_ratios, silhouette_ratios, combined_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, combined_scores, label='Combined Score', marker='o', color='r', linestyle='--')
    plt.title('Optimization Function: ARI and Silhouette Ratios')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
