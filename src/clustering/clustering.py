import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import umap
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa


class UnlabeledSpeechDataset(Dataset):
    """Dataset class for audio files used in clustering"""
    def __init__(self, dataframe, processor, audio_column='segment_file', max_length=32000):
        self.dataframe = dataframe
        self.processor = processor
        self.audio_column = audio_column
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_file = self.dataframe.iloc[idx][self.audio_column]
        
        speech, sr = librosa.load(audio_file, sr=16000)
        speech = librosa.util.normalize(speech)
        
        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            "input_values": inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0]
        }


def extract_embeddings(model, dataset):
    """Extract embeddings using Wav2Vec2 model """
    dataloader = DataLoader(dataset, batch_size=16)
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings..."):
            inputs = batch["input_values"].to(model.device)
            outputs = model.wav2vec2(inputs)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)  # Temporal pooling
            embeddings.append(pooled.cpu().numpy())
    
    return np.concatenate(embeddings)


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = "/content/speech-emotion-clustering/configs/clustering_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def define_parameter_grids(config):
    """Define parameter grids for all clustering algorithms from config"""
    param_grids = {}
    
    # Common UMAP parameters
    umap_params = {
        'umap__n_neighbors': config['umap']['n_neighbors'],
        'umap__min_dist': config['umap']['min_dist'],
        'umap__n_components': config['umap']['n_components']
    }
    
    # Build parameter grids for enabled algorithms
    algorithms = config['algorithms']
    
    
    param_grids['kmeans'] = {
        **umap_params,
        'kmeans__n_clusters': algorithms['kmeans']['n_clusters'],
        'kmeans__init': algorithms['kmeans']['init'],
        'kmeans__random_state': algorithms['kmeans']['random_state']
    }


    param_grids['spectral'] = {
        **umap_params,
        'spectral__n_clusters': algorithms['spectral']['n_clusters'],
        'spectral__affinity': algorithms['spectral']['affinity'],
        'spectral__n_neighbors': algorithms['spectral']['n_neighbors'],
        'spectral__assign_labels': algorithms['spectral']['assign_labels'],
        'spectral__random_state': algorithms['spectral']['random_state']
    }


    param_grids['agglomerative'] = {
        **umap_params,
        'agg__n_clusters': algorithms['agglomerative']['n_clusters'],
        'agg__linkage': algorithms['agglomerative']['linkage']
    }
    
    return param_grids


def run_algorithm_search(algorithm_name, normalized_embeddings, param_grid):
    """Run hyperparameter search for specific algorithm"""
    grid = list(ParameterGrid(param_grid))
    results = []
    
    print(f"Testing {algorithm_name} with {len(grid)} configurations...")
    
    for cfg in tqdm(grid, desc=f"{algorithm_name} hypreparameter research"):
        # Setup UMAP
        reducer = umap.UMAP(
            n_neighbors=int(cfg['umap__n_neighbors']),  # Converti in int
            min_dist=cfg['umap__min_dist'],
            n_components=int(cfg['umap__n_components']),  # Converti in int
            random_state=42
        )
        X_emb = reducer.fit_transform(normalized_embeddings)

        # Setup clustering algorithm
        if algorithm_name == 'kmeans':
            clusterer = KMeans(
                n_clusters=int(cfg['kmeans__n_clusters']),  # Converti in int
                init=cfg['kmeans__init'],
                random_state=int(cfg['kmeans__random_state'])  # Converti in int
            )
        elif algorithm_name == 'spectral':
            clusterer = SpectralClustering(
                n_clusters=int(cfg['spectral__n_clusters']),  # Converti in int
                affinity=cfg['spectral__affinity'],
                n_neighbors=int(cfg['spectral__n_neighbors']),  # Converti in int
                assign_labels=cfg['spectral__assign_labels'],
                random_state=int(cfg['spectral__random_state'])  # Converti in int
            )
        elif algorithm_name == 'agglomerative':
            clusterer = AgglomerativeClustering(
                n_clusters=int(cfg['agg__n_clusters']),  # Converti in int
                linkage=cfg['agg__linkage']
            )
        
        # Fit and predict
        labels = clusterer.fit_predict(X_emb)

        # Calculate silhouette score
        if len(set(labels)) > 1:
            score = silhouette_score(X_emb, labels)
        else:
            score = -1

        # Store result
        result = {**cfg, 'silhouette': score, 'algorithm': algorithm_name}
        results.append(result)
    
    return results


def display_results_summary(df_results, top_n=5):
    """Display comprehensive results summary"""
    print("\n" + "="*80)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal configurations tested: {len(df_results)}")
    print(f"Algorithms tested: {', '.join(df_results['algorithm'].unique())}")
    
    # Best results by algorithm
    print(f"\nBEST RESULTS BY ALGORITHM:")
    print("-"*50)
    for algorithm in df_results['algorithm'].unique():
        algo_results = df_results[df_results['algorithm'] == algorithm]
        best_algo = algo_results.sort_values('silhouette', ascending=False).iloc[0]
        print(f"{algorithm.upper():>15}: {best_algo['silhouette']:.4f} (best silhouette score)")
    
    # Top configurations overall
    print(f"\nTOP {top_n} CONFIGURATIONS OVERALL:")
    print("-"*50)
    top_configs = df_results.sort_values('silhouette', ascending=False).head(top_n)
    for i, (idx, config) in enumerate(top_configs.iterrows(), 1):
        print(f"{i:>2}. {config['algorithm'].upper():>15}: {config['silhouette']:.4f}")
        # Show key parameters
        if config['algorithm'] == 'kmeans':
            print(f"    {'':>4} Parameters: n_clusters={config['kmeans__n_clusters']}, init={config['kmeans__init']}")
        elif config['algorithm'] == 'spectral':
            print(f"    {'':>4} Parameters: n_clusters={config['spectral__n_clusters']}, affinity={config['spectral__affinity']}")
        elif config['algorithm'] == 'agglomerative':
            print(f"    {'':>4} Parameters: n_clusters={config['agg__n_clusters']}, linkage={config['agg__linkage']}")
    
    return top_configs


def find_best_configuration(all_results, config):
    """
    Three-phase evaluation:
    1. For each algorithm, find optimal k using mean silhouette scores
    2. For each algorithm, select best configuration with optimal k
    3. Compare algorithms and select the best overall
    """
    df_results = pd.DataFrame(all_results)
    
    # Display comprehensive summary
    display_results_summary(df_results, top_n=config['output']['display_top_configs'])
    
    print(f"\nALGORITHM-SPECIFIC K OPTIMIZATION:")
    print("-"*50)
    
    # Phase 1 & 2: For each algorithm, find optimal k and best config
    algorithm_best_configs = {}
    
    for algorithm in df_results['algorithm'].unique():
        algo_results = df_results[df_results['algorithm'] == algorithm]
        
        # Find the k column for this algorithm
        k_column = None
        if algorithm == 'kmeans':
            k_column = 'kmeans__n_clusters'
        elif algorithm == 'spectral':
            k_column = 'spectral__n_clusters'
        elif algorithm == 'agglomerative':
            k_column = 'agg__n_clusters'
        
        if k_column and k_column in algo_results.columns:
            # Calculate mean silhouette score for each k
            k_means = algo_results.groupby(k_column)['silhouette'].mean()
            optimal_k = k_means.idxmax()
            optimal_k_score = k_means.max()
            
            print(f"{algorithm.upper()}:")
            print(f"  Mean silhouette scores by k:")
            for k, score in k_means.sort_index().items():
                marker = " ← OPTIMAL" if k == optimal_k else ""
                print(f"    k={k}: {score:.4f}{marker}")
            
            # Filter configurations with optimal k and select the best one
            optimal_configs = algo_results[algo_results[k_column] == optimal_k]
            best_config_for_algo = optimal_configs.sort_values('silhouette', ascending=False).iloc[0]
            
            print(f"  Best configuration with k={optimal_k}: silhouette={best_config_for_algo['silhouette']:.4f}")
            algorithm_best_configs[algorithm] = best_config_for_algo
        
        print()
    
    # Phase 3: Compare algorithms and select the best overall
    if algorithm_best_configs:
        best_overall = max(algorithm_best_configs.values(), key=lambda x: x['silhouette'])
        
        print(f"ALGORITHM COMPARISON:")
        print("-"*50)
        for algorithm, config in algorithm_best_configs.items():
            marker = " ← SELECTED" if config['silhouette'] == best_overall['silhouette'] else ""
            print(f"{algorithm.upper()}: {config['silhouette']:.4f}{marker}")
        
        print(f"\nFINAL SELECTION:")
        print("-"*50)
        print(f"Best overall configuration: {best_overall['algorithm'].upper()}")
        print(f"Silhouette score: {best_overall['silhouette']:.4f}")
        
        # Show detailed parameters of best config
        if best_overall['algorithm'] == 'kmeans':
            print(f"Parameters: n_clusters={best_overall['kmeans__n_clusters']}, "
                  f"init={best_overall['kmeans__init']}, "
                  f"UMAP(n_neighbors={best_overall['umap__n_neighbors']}, "
                  f"min_dist={best_overall['umap__min_dist']}, "
                  f"n_components={best_overall['umap__n_components']})")
        elif best_overall['algorithm'] == 'spectral':
            print(f"Parameters: n_clusters={best_overall['spectral__n_clusters']}, "
                  f"affinity={best_overall['spectral__affinity']}, "
                  f"UMAP(n_neighbors={best_overall['umap__n_neighbors']}, "
                  f"min_dist={best_overall['umap__min_dist']}, "
                  f"n_components={best_overall['umap__n_components']})")
        elif best_overall['algorithm'] == 'agglomerative':
            print(f"Parameters: n_clusters={best_overall['agg__n_clusters']}, "
                  f"linkage={best_overall['agg__linkage']}, "
                  f"UMAP(n_neighbors={best_overall['umap__n_neighbors']}, "
                  f"min_dist={best_overall['umap__min_dist']}, "
                  f"n_components={best_overall['umap__n_components']})")
        
        print("="*80)
        return best_overall
    else:
        # Fallback to original logic if no algorithm-specific configs found
        best_config = df_results.sort_values('silhouette', ascending=False).iloc[0]
        print(f"Fallback: Best overall configuration: {best_config['algorithm'].upper()}")
        print("="*80)
        return best_config


def apply_final_clustering(best_config, normalized_embeddings, df):
    """Apply best clustering configuration and add results to dataframe"""
    # Reconstruct UMAP with best parameters
    reducer = umap.UMAP(
        n_neighbors=int(best_config['umap__n_neighbors']),
        min_dist=best_config['umap__min_dist'],
        n_components=int(best_config['umap__n_components']),
        random_state=42
    )
    X_best = reducer.fit_transform(normalized_embeddings)
    
    # Reconstruct best clustering algorithm
    algorithm = best_config['algorithm']
    if algorithm == 'kmeans':
        clusterer = KMeans(
            n_clusters=int(best_config['kmeans__n_clusters']),
            init=best_config['kmeans__init'],
            random_state=int(best_config['kmeans__random_state'])
        )
    elif algorithm == 'spectral':
        clusterer = SpectralClustering(
            n_clusters=int(best_config['spectral__n_clusters']),
            affinity=best_config['spectral__affinity'],
            n_neighbors=int(best_config['spectral__n_neighbors']),
            assign_labels=best_config['spectral__assign_labels'],
            random_state=int(best_config['spectral__random_state'])
        )
    elif algorithm == 'agglomerative':
        clusterer = AgglomerativeClustering(
            n_clusters=int(best_config['agg__n_clusters']),
            linkage=best_config['agg__linkage']
        )
    
    # Apply clustering
    cluster_labels = clusterer.fit_predict(X_best)
    
    # Calculate distances from centroids
    if algorithm == 'kmeans':
        # KMeans has built-in centroids
        distances = [np.linalg.norm(point - clusterer.cluster_centers_[label]) 
                    for point, label in zip(X_best, cluster_labels)]
    else:
        # Calculate centroids manually for other algorithms
        cluster_centers = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_points = X_best[cluster_labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            cluster_centers[cluster_id] = centroid
        
        distances = [np.linalg.norm(point - cluster_centers[label]) 
                    for point, label in zip(X_best, cluster_labels)]
    
    # Add results to dataframe
    df = df.copy()
    df['cluster'] = cluster_labels
    df['distance_from_centroid'] = distances
    
    return df


def clustering_analysis(input_csv, output_csv, model_path, config_path=None):
    """
    Args:
        input_csv: CSV with processed features from post-processing step
        output_csv: Output CSV with cluster assignments added
        model_path: Path to Wav2Vec2 model directory
        config_path: Path to YAML config file (optional)
    Returns:
        pd.DataFrame: DataFrame with cluster assignments
    """
    print("Starting clustering analysis...")
    
    # 1. Load configuration
    config = load_config(config_path)
    if model_path:
        config['model']['path'] = model_path  # Override with provided path
    
    # 2. Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} segments from {input_csv}")
    
    # 3. Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['model']['device'] != 'auto':
        device = torch.device(config['model']['device'])
    
    processor = Wav2Vec2Processor.from_pretrained(config['model']['path'], local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config['model']['path'], 
        num_labels=7, 
        local_files_only=True, 
        use_safetensors=True
    )
    model.to(device)
    model.eval()
    
    # 4. Extract embeddings
    dataset = UnlabeledSpeechDataset(
        df, 
        processor, 
        audio_column=config['dataset']['audio_column'],
        max_length=config['model']['max_length']
    )
    embeddings = extract_embeddings(model, dataset)
    
    # 5. Normalize embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # 6. Run clustering algorithms
    param_grids = define_parameter_grids(config)
    all_results = []
    
    for algorithm_name, param_grid in param_grids.items():
        results = run_algorithm_search(algorithm_name, normalized_embeddings, param_grid)
        all_results.extend(results)
    
    # 7. Find best configuration
    best_config = find_best_configuration(all_results, config)
    
    # 8. Apply final clustering
    final_df = apply_final_clustering(best_config, normalized_embeddings, df)
    
    # 9. Save results
    final_df.to_csv(output_csv, index=False)
    print(f"Clustering results saved to {output_csv}")
    
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Clustering analysis pipeline")
    parser.add_argument('--input-csv', required=True, 
                       help='Input CSV with processed audio features')
    parser.add_argument('--output-csv', required=True, 
                       help='Output CSV with cluster assignments')
    parser.add_argument('--model-path', required=True,
                       help='Path to Wav2Vec2 model directory')
    parser.add_argument('--config', 
                       help='Path to YAML config file (default: clustering_config.yaml)')
    args = parser.parse_args()
    
    clustering_analysis(args.input_csv, args.output_csv, args.model_path, args.config)


if __name__ == "__main__":
    main()