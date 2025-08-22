import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial import distance
from sklearn.preprocessing import scale
import itertools
from tqdm import tqdm

# Permutation function equivalent to the Seurat object permutation
def permute_data(input_data, random_state=42):
    np.random.seed(random_state)
    permuted_data = input_data.copy()

    for i in range(input_data.shape[0]):
        np.random.shuffle(permuted_data[i, :])

    return permuted_data

# Function to calculate pre-embedding distances
def calculate_distances(data, K, method='pca'):
    if method == 'pca':
        pca = PCA(n_components=K)
        embeddings = pca.fit_transform(data)
    distances = distance.pdist(embeddings[:, :K], metric='euclidean')
    return distance.squareform(distances)

# Running t-SNE
def run_tsne(data, perplexity=40, random_state=1000):
    tsne = TSNE(perplexity=perplexity, random_state=random_state)
    embeddings = tsne.fit_transform(data)
    tsne_distances = distance.pdist(embeddings, metric='euclidean')
    return distance.squareform(tsne_distances)

# Running UMAP
def run_umap(data, n_neighbors=30, min_dist=0.3, random_state=42):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embeddings = umap_model.fit_transform(data)
    umap_distances = distance.pdist(embeddings, metric='euclidean')
    return distance.squareform(umap_distances)

# Function to calculate similarity scores
def cell_similarity(pre_embedding_distances, reduced_dim_distances, similarity_percent=0.5):
    number_selected = int(pre_embedding_distances.shape[1] * similarity_percent)
    
    rho_original = []
    for i in range(pre_embedding_distances.shape[0]):
        sorted_indices = np.argsort(pre_embedding_distances[i, :])
        selected_original = reduced_dim_distances[i, sorted_indices[:number_selected]]
        sorted_reduced = np.sort(reduced_dim_distances[i, :])
        rho = np.corrcoef(selected_original, sorted_reduced[:number_selected])[0, 1]
        rho_original.append(rho)

    return np.array(rho_original)

# Classify cells based on similarity scores
def classify_cells(rho_original, rho_permuted, dubious_cutoff=0.05, trustworthy_cutoff=0.95):
    rho_trustworthy = np.quantile(rho_permuted, trustworthy_cutoff)
    rho_dubious = np.quantile(rho_permuted, dubious_cutoff)

    dubious_cells = np.where(rho_original < rho_dubious)[0]
    trustworthy_cells = np.where(rho_original > rho_trustworthy)[0]
    intermediate_cells = np.setdiff1d(np.arange(len(rho_original)), np.concatenate([dubious_cells, trustworthy_cells]))

    return {
        "dubious_cells": dubious_cells,
        "trustworthy_cells": trustworthy_cells,
        "intermediate_cells": intermediate_cells
    }

# Function to expand grid of hyperparameters (similar to expand.grid in R)
def expand_grid(params_dict):
    keys, values = zip(*params_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

# Main optimization function handling UMAP and t-SNE
def optimize(input_data, permuted_data, K, pre_embedding, reduction_method, params, similarity_percent, dubious_cutoff, trustworthy_cutoff):
    if reduction_method == 'umap':
        n_neighbors = params.get('n_neighbors')
        min_dist = params.get('min_dist')
        
        reduced_distances = run_umap(input_data, n_neighbors=n_neighbors, min_dist=min_dist)
        reduced_distances_permuted = run_umap(permuted_data, n_neighbors=n_neighbors, min_dist=min_dist)
    
    elif reduction_method == 'tsne':
        perplexity = params.get('perplexity')
        
        reduced_distances = run_tsne(input_data, perplexity=perplexity)
        reduced_distances_permuted = run_tsne(permuted_data, perplexity=perplexity)
    
    rho_original = cell_similarity(calculate_distances(input_data, K, pre_embedding), reduced_distances, similarity_percent)
    rho_permuted = cell_similarity(calculate_distances(permuted_data, K, pre_embedding), reduced_distances_permuted, similarity_percent)
    
    return classify_cells(rho_original, rho_permuted, dubious_cutoff, trustworthy_cutoff)

# Main wrapper function to run the full analysis
def scDEED(input_data, K, n_neighbors=[5, 20, 30, 40, 50,60], min_dist=[0.001,0.01,0.1, 0.4], perplexity=[30, 50, 100], 
           similarity_percent=0.5, reduction_method='umap', pre_embedding='pca', dubious_cutoff=0.05, trustworthy_cutoff=0.95):
    
    # Step 1: Permute the input data
    print("Permuting data...")
    permuted_data = permute_data(input_data)
    
    # Step 2: Calculate pre-embedding distances
    print("Calculating pre-embedding distances...")
    pre_embedding_distances = calculate_distances(input_data, K, method=pre_embedding)
    pre_embedding_distances_permuted = calculate_distances(permuted_data, K, method=pre_embedding)

    # Step 3: Define hyperparameter grid
    if reduction_method == 'umap':
        param_grid = expand_grid({'n_neighbors': n_neighbors, 'min_dist': min_dist})
    elif reduction_method == 'tsne':
        param_grid = expand_grid({'perplexity': perplexity})
    
    # Step 4: Iterate through hyperparameters and run optimization
    print("Running optimization...")
    results = []
    for params in tqdm(param_grid):
        classification_result = optimize(input_data, permuted_data, K, pre_embedding, reduction_method, params, 
                                         similarity_percent, dubious_cutoff, trustworthy_cutoff)
        results.append((params, classification_result))
    
    return results
