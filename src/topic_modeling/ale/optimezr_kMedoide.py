from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
import re
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids  # Importazione del KMedoids
from umap import UMAP
from ale.scdeed import scDEED
from tqdm.notebook import tqdm

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

# Funzione per calcolare un combined score complesso
def calculate_combined_score(ari_mean, ari_std, silhouette_mean, silhouette_std):
    """
    Calcola un combined score complesso basato su ARI e Silhouette, ponderato in base alla deviazione standard
    e con penalità non lineari per punteggi bassi.
    
    :param ari_mean: ARI medio per un dato valore di k
    :param ari_std: Deviazione standard dell'ARI per un dato valore di k
    :param silhouette_mean: Silhouette medio per un dato valore di k
    :param silhouette_std: Deviazione standard del Silhouette per un dato valore di k
    :return: Combined score calcolato in modo complesso
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

# Funzione per applicare i modelli di clustering (aggiunto KMedoids)
def apply_clustering(X, k):
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
    
    # KMedoids - AGGIUNTO 
    try:
        kmedoids = KMedoids(n_clusters=k, random_state=42, method='pam')
        kmedoids_labels = kmedoids.fit_predict(X)
        results['KMedoids'] = kmedoids_labels
    except Exception as e:
        print(f"KMedoids failed with error: {e}")
        # In caso di errore, assegna etichette casuali (questo è un fallback)
        results['KMedoids'] = np.random.randint(0, k, size=X.shape[0])

    # Gaussian Mixture Clustering (commentato nell'originale)
    #gmm = GaussianMixture(n_components=k, random_state=42)
    #gmm_labels = gmm.fit_predict(X)
    #results['GaussianMixture'] = gmm_labels

    return results

# Funzione per calcolare la media dell'ARI tra i vari algoritmi
def calculate_ari_consensus(clusterings):
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

def optimizer(X, n_neighbors=[5, 15, 30, 40, 50], min_dist=[0.1, 0.3, 0.0001, 0.01], k_min=15, k_max=60):
    num_rows = X.shape[0]

    if num_rows > k_min and num_rows < k_max:
        k_range = np.linspace(15, num_rows -1 , min(5, num_rows-15+1), dtype=int)
    elif num_rows < k_min:
        k_range = np.arange(1, num_rows -1)
    else:
        k_range = np.arange(k_min, k_max -1, 5)
    
    print(k_range)

    best_overall_score = -np.inf
    best_overall_k = None
    best_overall_clustering = None
    silhouette_scores = {}
    ari_combinations = []
    
    k_values = []
    ari_ratios = []
    silhouette_ratios = []
    combined_scores = []
    
    # Best parameters per algorithm
    best_params = {
        'KMeans': None,
        'Agglomerative': None,
        'Spectral': None,
        'KMedoids': None,  # Aggiunto per KMedoids
    }

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

    if best_umap_params:
        umap = UMAP(n_neighbors=best_umap_params['n_neighbors'], min_dist=best_umap_params['min_dist'], n_components=2, metric='cosine')
        umap_embedding = umap.fit_transform(X)

        for k in tqdm(k_range):
            clusterings = apply_clustering(umap_embedding, k)
            consensus_ari = calculate_ari_consensus(clusterings)

            for alg1, alg2, ari in consensus_ari:
                ari_combinations.append((k, alg1, alg2, ari))

            # Calculate silhouette scores for each algorithm
            silhouette_scores[k] = {}
            for alg, labels in clusterings.items():
                # Calcola il silhouette score solo se ci sono almeno 2 cluster unici
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    sil_score = silhouette_score(umap_embedding, labels)
                    silhouette_scores[k][alg] = sil_score
                    
                    # Mantieni traccia dei migliori parametri per ogni algoritmo
                    if not best_params[alg] or sil_score > best_params[alg]['silhouette']:
                        best_params[alg] = {
                            'k': k,
                            'silhouette': sil_score,
                            'labels': labels
                        }
                else:
                    silhouette_scores[k][alg] = -1  # Valore di default negativo se solo 1 cluster

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

    # Trova i migliori k per ogni algoritmo basato sul silhouette score
    best_k_per_algorithm = {}
    for alg in best_params:
        if best_params[alg]:
            best_k_per_algorithm[alg] = best_params[alg]['k']
    
    return {
        'best_k': best_overall_k,  # Miglior k complessivo basato sul punteggio combinato
        'best_k_per_algorithm': best_k_per_algorithm,  # Miglior k per ogni algoritmo basato su silhouette
        'best_clustering': best_overall_clustering,
        'silhouette_scores': silhouette_scores,
        'ari_combinations': ari_combinations,
        'umap_embedding': umap_embedding,
        'best_umap_params': best_umap_params,
        'k_values': k_values,
        'ari_ratios': ari_ratios,
        'silhouette_ratios': silhouette_ratios,
        'combined_scores': combined_scores,
        'best_params': best_params  # Dettagli completi dei migliori parametri per ogni algoritmo
    }


def plot_ari_combinations(result):
    ari_combinations = result['ari_combinations']
    ks = sorted(set([k for k, _, _, _ in ari_combinations]))
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crea un dizionario per memorizzare le ARI per ogni combinazione di algoritmi
    ari_data = {
        'KMeans-Agglomerative': [], 
        'KMeans-Spectral': [], 
        'KMeans-KMedoids': [],  # Aggiunto per KMedoids
        'Agglomerative-Spectral': [], 
        'Agglomerative-KMedoids': [],  # Aggiunto per KMedoids
        'Spectral-KMedoids': []  # Aggiunto per KMedoids
        #'KMeans-GaussianMixture': [],
        #'Agglomerative-GaussianMixture': [],
        #'Spectral-GaussianMixture': []
    }
    
    for k in ks:
        for (k_val, alg1, alg2, ari) in ari_combinations:
            if k_val == k:
                combo = f'{alg1}-{alg2}' if f'{alg1}-{alg2}' in ari_data else f'{alg2}-{alg1}'
                ari_data[combo].append(ari)
    
    # Plot delle ARI per ogni coppia di algoritmi
    for combo, ari_scores in ari_data.items():
        if ari_scores:  # Solo se abbiamo punteggi per questa combinazione
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
        
        # Aggiungi il prodotto calcolato alla lista
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

    # Crea una lista per memorizzare il rapporto media / deviazione standard degli ARI per ogni k
    ari_mean_std_ratio = []

    for k in ks:
        ari_values = [ari for (k_val, _, _, ari) in ari_combinations if k_val == k]
        ari_mean = np.mean(ari_values)  # Media degli ARI per ogni k
        ari_std = np.std(ari_values)    # Deviazione standard degli ARI per ogni k
        
        if ari_std > 0:
            ratio = ari_mean * (1+np.exp(-ari_std))*np.log(3)  # Applico la sigmoide 
        else:
            ratio = float('inf')  # Gestisci il caso in cui la deviazione standard è 0
        
        ari_mean_std_ratio.append(ratio)
    
    # Plot del rapporto media / deviazione standard degli ARI per ogni valore di k
    ax.plot(ks, ari_mean_std_ratio, label='Media ARI / Deviazione standard ARI', marker='o')

    ax.set_title("Rapporto Media ARI / Deviazione standard ARI per ogni valore di k")
    ax.set_xlabel("Numero di cluster (k)")
    ax.set_ylabel("Rapporto Media ARI / Deviazione standard ARI")
    ax.legend()
    plt.show()

def plot_opt_weight(k_values, ari_ratios, silhouette_ratios, combined_scores):
    """
    Plot ARI, Silhouette Ratios, and Combined Score over different values of k.

    :param k_values: List of k values (number of clusters)
    :param ari_ratios: List of ARI ratios for each k
    :param silhouette_ratios: List of silhouette ratios for each k
    :param combined_scores: List of combined scores for each k
    """
    plt.figure(figsize=(10, 6))
    
    # Plot delle ARI Ratios e Silhouette Ratios (opzionale)
    plt.plot(k_values, ari_ratios, label='ARI Ratio', marker='x', alpha=0.7)
    plt.plot(k_values, silhouette_ratios, label='Silhouette Ratio', marker='+', alpha=0.7)

    # Plot Combined Score
    plt.plot(k_values, combined_scores, label='Combined Score', marker='o', color='r', linestyle='--')

    plt.title('Optimization Function: ARI and Silhouette Ratios')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Funzione per visualizzare i migliori parametri di ogni algoritmo
def print_best_params(result):
    """
    Visualizza i migliori parametri per ogni algoritmo.
    
    :param result: Dizionario con i risultati dell'ottimizzazione
    """
    best_params = result['best_params']
    best_umap_params = result['best_umap_params']
    best_k_overall = result['best_k']
    best_k_per_algorithm = result['best_k_per_algorithm']
    
    print(f"Miglior k complessivo (basato sul punteggio combinato): {best_k_overall}")
    print("\nMiglior k per ogni algoritmo (basato sul silhouette score):")
    for alg, k in best_k_per_algorithm.items():
        print(f"{alg}: k={k}")
    
    print("\nMigliori parametri UMAP:")
    print(f"n_neighbors: {best_umap_params['n_neighbors']}, min_dist: {best_umap_params['min_dist']}")
    
    print("\nDettagli completi per algoritmo:")
    for alg, params in best_params.items():
        if params:
            print(f"{alg}: k={params['k']}, silhouette={params['silhouette']:.4f}")
        else:
            print(f"{alg}: No valid parameters found")