#!/usr/bin/env python3
"""
Unified Clustering Hyperparameter Search Module

Combines K-means, Agglomerative Clustering, and Spectral Clustering algorithms 
with comprehensive hyperparameter tuning and automatic best algorithm selection.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yaml
import torch
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import umap
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ClusteringConfig:
    """Configuration for clustering hyperparameter search."""
    
    # Data configuration
    data_file: str
    model_path: str
    output_dir: str
    
    # Embedding extraction
    max_length: int = 32000
    batch_size: int = 16
    sampling_rate: int = 16000
    
    # Clustering parameters
    k_range: List[int] = None
    random_state: int = 42
    
    # Evaluation
    metric: str = "silhouette"  # Currently only silhouette supported
    
    # Visualization
    create_plots: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        if self.k_range is None:
            self.k_range = list(range(5, 11))  # Default: 5-10 clusters
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> 'ClusteringConfig':
        """Load clustering configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                yaml_config = yaml.safe_load(file)
            
            # Extract clustering-specific configuration
            clustering_config = yaml_config.get('clustering', {})
            
            config = cls(
                data_file=clustering_config['data_file'],
                model_path=clustering_config['model_path'],
                output_dir=clustering_config['output_dir'],
                max_length=clustering_config.get('max_length', 32000),
                batch_size=clustering_config.get('batch_size', 16),
                sampling_rate=clustering_config.get('sampling_rate', 16000),
                k_range=clustering_config.get('k_range', list(range(5, 11))),
                random_state=clustering_config.get('random_state', 42),
                metric=clustering_config.get('metric', 'silhouette'),
                create_plots=clustering_config.get('create_plots', True),
                save_results=clustering_config.get('save_results', True)
            )
            
            print(f"Clustering configuration loaded from: {yaml_path}")
            return config
            
        except KeyError as e:
            raise ValueError(f"Missing required key in YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {e}")


class UnlabeledSpeechDataset(Dataset):
    """Dataset for unlabeled speech data for embedding extraction."""
    
    def __init__(self, dataframe: pd.DataFrame, processor, max_length: int = 32000):
        self.dataframe = dataframe
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_file = self.dataframe.iloc[idx]['path']
        
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


class EmbeddingExtractor:
    """Extracts embeddings from audio using Wav2Vec2 model."""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize embedding extractor.
        
        Args:
            model_path: Path to the fine-tuned Wav2Vec2 model
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=7, 
            local_files_only=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
    
    def extract_embeddings(self, dataframe: pd.DataFrame, max_length: int = 32000, 
                          batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from audio files in dataframe.
        
        Args:
            dataframe: DataFrame with 'path' column containing audio file paths
            max_length: Maximum audio length in samples
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings with shape (n_samples, embedding_dim)
        """
        dataset = UnlabeledSpeechDataset(dataframe, self.processor, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                inputs = batch["input_values"].to(self.device)
                outputs = self.model.wav2vec2(inputs)
                hidden_states = outputs.last_hidden_state
                pooled = hidden_states.mean(dim=1)  # Temporal pooling
                embeddings.append(pooled.cpu().numpy())
        
        return np.concatenate(embeddings)


class ClusteringAlgorithms:
    """Unified clustering algorithms with hyperparameter grids."""
    
    @staticmethod
    def get_kmeans_param_grid(k_range: List[int], random_state: int = 42) -> Dict:
        """Get parameter grid for K-means clustering."""
        return {
            'algorithm': 'kmeans',
            'umap__n_neighbors': [5, 15, 30],
            'umap__min_dist': [0.0, 0.1],
            'umap__n_components': [2, 5, 10],
            'kmeans__n_clusters': k_range,
            'kmeans__init': ['k-means++', 'random'],
            'kmeans__random_state': [random_state]
        }
    
    @staticmethod
    def get_agglomerative_param_grid(k_range: List[int]) -> Dict:
        """Get parameter grid for Agglomerative clustering."""
        return {
            'algorithm': 'agglomerative',
            'umap__n_neighbors': [5, 15, 30],
            'umap__min_dist': [0.0, 0.1],
            'umap__n_components': [2, 5, 10],
            'agg__n_clusters': k_range,
            'agg__linkage': ['ward', 'complete', 'average']
        }
    
    @staticmethod
    def get_spectral_param_grid(k_range: List[int], random_state: int = 42) -> Dict:
        """Get parameter grid for Spectral clustering."""
        return {
            'algorithm': 'spectral',
            'umap__n_neighbors': [5, 15, 30],
            'umap__min_dist': [0.0, 0.1],
            'umap__n_components': [2, 5, 10],
            'spec__n_clusters': k_range,
            'spec__affinity': ['nearest_neighbors'],
            'spec__n_neighbors': [15],
            'spec__assign_labels': ['kmeans'],
            'spec__random_state': [random_state]
        }
    
    @staticmethod
    def create_clusterer(algorithm: str, config: Dict):
        """Create clustering algorithm instance from config."""
        if algorithm == 'kmeans':
            return KMeans(
                n_clusters=config['kmeans__n_clusters'],
                init=config['kmeans__init'],
                random_state=config['kmeans__random_state']
            )
        elif algorithm == 'agglomerative':
            return AgglomerativeClustering(
                n_clusters=config['agg__n_clusters'],
                linkage=config['agg__linkage']
            )
        elif algorithm == 'spectral':
            return SpectralClustering(
                n_clusters=config['spec__n_clusters'],
                affinity=config['spec__affinity'],
                n_neighbors=config['spec__n_neighbors'],
                assign_labels=config['spec__assign_labels'],
                random_state=config['spec__random_state']
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


class HyperparameterSearcher:
    """Manages hyperparameter search across multiple clustering algorithms."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.embeddings = None
        self.normalized_embeddings = None
        self.results = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data_and_extract_embeddings(self) -> pd.DataFrame:
        """Load data and extract embeddings."""
        print("Loading data...")
        df = pd.read_csv(self.config.data_file)
        
        # Basic data cleaning
        if 'speaker' in df.columns:
            df = df[df["speaker"] != "unknown"]
        
        if 'segment_file' in df.columns:
            df.rename(columns={"segment_file": "path"}, inplace=True)
        
        df = df.reset_index(drop=True)
        df['row_id'] = np.arange(len(df))
        
        print(f"Loaded {len(df)} samples")
        
        # Extract embeddings
        print("Extracting embeddings...")
        extractor = EmbeddingExtractor(self.config.model_path)
        self.embeddings = extractor.extract_embeddings(
            df, 
            max_length=self.config.max_length,
            batch_size=self.config.batch_size
        )
        
        # Normalize embeddings
        scaler = StandardScaler()
        self.normalized_embeddings = scaler.fit_transform(self.embeddings)
        
        print(f"Embeddings shape: {self.embeddings.shape}")
        return df
    
    def run_hyperparameter_search(self) -> List[Dict]:
        """Run comprehensive hyperparameter search across all algorithms."""
        algorithms = ClusteringAlgorithms()
        
        # Create parameter grids for all algorithms
        param_grids = [
            algorithms.get_kmeans_param_grid(self.config.k_range, self.config.random_state),
            algorithms.get_agglomerative_param_grid(self.config.k_range),
            algorithms.get_spectral_param_grid(self.config.k_range, self.config.random_state)
        ]
        
        all_results = []
        
        for param_dict in param_grids:
            algorithm = param_dict.pop('algorithm')
            grid = list(ParameterGrid(param_dict))
            
            print(f"\n🔄 Testing {algorithm.upper()} - {len(grid)} configurations")
            
            for cfg in tqdm(grid, desc=f"{algorithm} search"):
                try:
                    # UMAP dimensionality reduction
                    reducer = umap.UMAP(
                        n_neighbors=cfg['umap__n_neighbors'],
                        min_dist=cfg['umap__min_dist'],
                        n_components=cfg['umap__n_components'],
                        random_state=self.config.random_state
                    )
                    X_reduced = reducer.fit_transform(self.normalized_embeddings)
                    
                    # Clustering
                    clusterer = algorithms.create_clusterer(algorithm, cfg)
                    labels = clusterer.fit_predict(X_reduced)
                    
                    # Evaluation
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_reduced, labels)
                    else:
                        score = -1
                    
                    # Store result
                    result = {
                        'algorithm': algorithm,
                        **cfg,
                        'silhouette': score
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error in configuration {cfg}: {e}")
                    continue
        
        self.results = all_results
        return all_results
    
    def select_best_configurations(self) -> Tuple[Dict, List[Dict], Dict]:
        """Select best configurations using the two-step process."""
        if not self.results:
            raise ValueError("No results available. Run hyperparameter search first.")
        
        df_results = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("CLUSTERING ALGORITHM SELECTION PROCESS")
        print("="*60)
        
        # Step 1: Find best number of clusters for each algorithm based on mean scores
        print("\n📊 STEP 1: Best cluster numbers by mean silhouette score")
        print("-" * 50)
        
        best_k_per_algorithm = {}
        for algorithm in df_results['algorithm'].unique():
            algo_data = df_results[df_results['algorithm'] == algorithm]
            
            # Get cluster column name for this algorithm
            k_col = None
            for col in algo_data.columns:
                if 'n_clusters' in col:
                    k_col = col
                    break
            
            if k_col:
                mean_scores = algo_data.groupby(k_col)['silhouette'].mean()
                best_k = mean_scores.idxmax()
                best_mean_score = mean_scores.max()
                
                best_k_per_algorithm[algorithm] = {
                    'best_k': best_k,
                    'mean_score': best_mean_score,
                    'k_column': k_col
                }
                
                print(f"{algorithm.upper()}: k={best_k} (mean silhouette: {best_mean_score:.4f})")
        
        # Step 2: Find best absolute configuration for each algorithm's optimal k
        print("\n🎯 STEP 2: Best absolute configurations")
        print("-" * 50)
        
        best_configs_per_algorithm = {}
        for algorithm, k_info in best_k_per_algorithm.items():
            algo_data = df_results[df_results['algorithm'] == algorithm]
            best_k_data = algo_data[algo_data[k_info['k_column']] == k_info['best_k']]
            best_config = best_k_data.loc[best_k_data['silhouette'].idxmax()]
            
            best_configs_per_algorithm[algorithm] = best_config.to_dict()
            print(f"{algorithm.upper()}: silhouette={best_config['silhouette']:.4f}")
        
        # Step 3: Select overall best algorithm
        print("\n🏆 STEP 3: Final algorithm selection")
        print("-" * 50)
        
        best_algorithm = max(best_configs_per_algorithm.keys(), 
                           key=lambda alg: best_configs_per_algorithm[alg]['silhouette'])
        best_overall_config = best_configs_per_algorithm[best_algorithm]
        
        print(f"🥇 WINNER: {best_algorithm.upper()}")
        print(f"   Silhouette Score: {best_overall_config['silhouette']:.4f}")
        print(f"   Number of clusters: {best_overall_config.get(best_k_per_algorithm[best_algorithm]['k_column'])}")
        print("="*60)
        
        return best_overall_config, list(best_configs_per_algorithm.values()), best_k_per_algorithm
    
    def create_visualizations(self, df_results: pd.DataFrame):
        """Create visualizations of the hyperparameter search results."""
        if not self.config.create_plots:
            return
        
        plt.style.use('default')
        
        # 1. Mean silhouette scores by number of clusters for each algorithm
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, algorithm in enumerate(df_results['algorithm'].unique()):
            algo_data = df_results[df_results['algorithm'] == algorithm]
            
            # Find cluster column
            k_col = None
            for col in algo_data.columns:
                if 'n_clusters' in col:
                    k_col = col
                    break
            
            if k_col:
                mean_scores = algo_data.groupby(k_col)['silhouette'].mean()
                axes[i].plot(mean_scores.index, mean_scores.values, marker='o', linewidth=2)
                axes[i].set_title(f'{algorithm.upper()} - Mean Silhouette by K')
                axes[i].set_xlabel('Number of Clusters')
                axes[i].set_ylabel('Mean Silhouette Score')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mean_silhouette_by_k.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Top 5 configurations comparison
        top_configs = df_results.nlargest(15, 'silhouette')
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_configs['algorithm'].unique())))
        color_map = dict(zip(top_configs['algorithm'].unique(), colors))
        
        bars = plt.bar(range(len(top_configs)), top_configs['silhouette'], 
                      color=[color_map[alg] for alg in top_configs['algorithm']])
        
        plt.title('Top 15 Clustering Configurations', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration Rank')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(len(top_configs)), range(1, len(top_configs)+1))
        
        # Add algorithm labels
        for i, (_, row) in enumerate(top_configs.iterrows()):
            plt.text(i, row['silhouette'] + 0.005, row['algorithm'][:4].upper(), 
                    ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[alg], label=alg.upper()) 
                          for alg in color_map.keys()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_configurations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, df_results: pd.DataFrame, best_config: Dict):
        """Save detailed results to files."""
        if not self.config.save_results:
            return
        
        # Save all results
        results_file = self.output_dir / 'clustering_hyperparameter_results.csv'
        df_results.to_csv(results_file, index=False)
        print(f"📁 Results saved to: {results_file}")
        
        # Save best configuration
        best_config_file = self.output_dir / 'best_clustering_config.yaml'
        with open(best_config_file, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        print(f"📁 Best config saved to: {best_config_file}")
        
        # Save summary
        summary_file = self.output_dir / 'clustering_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("CLUSTERING HYPERPARAMETER SEARCH SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total configurations tested: {len(df_results)}\n")
            f.write(f"Best algorithm: {best_config['algorithm'].upper()}\n")
            f.write(f"Best silhouette score: {best_config['silhouette']:.4f}\n")
            f.write(f"Optimal number of clusters: {[v for k, v in best_config.items() if 'n_clusters' in k][0]}\n")
            f.write(f"\nConfiguration details:\n")
            for key, value in best_config.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"📁 Summary saved to: {summary_file}")


def main():
    """Main CLI function for clustering hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Unified Clustering Hyperparameter Search - K-means, Agglomerative, Spectral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic hyperparameter search with YAML config
  python -m src.clustering.hyperparameter_search --config clustering_config.yaml

  # Override data file and output directory
  python -m src.clustering.hyperparameter_search --config clustering_config.yaml --data-file data.csv --output-dir results/

  # Custom cluster range
  python -m src.clustering.hyperparameter_search --config clustering_config.yaml --k-range 3 4 5 6 7 8
        """
    )
    
    # Required arguments
    parser.add_argument("--config", "-c", default="clustering_config.yaml",
                       help="YAML configuration file path (default: clustering_config.yaml)")
    
    # Override options
    parser.add_argument("--data-file", 
                       help="Override data file path from config")
    parser.add_argument("--model-path",
                       help="Override model path from config")
    parser.add_argument("--output-dir", "-o",
                       help="Override output directory from config")
    parser.add_argument("--k-range", nargs='+', type=int,
                       help="Override cluster range (e.g., --k-range 3 4 5 6 7)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    parser.add_argument("--no-save", action="store_true",
                       help="Disable result saving")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ClusteringConfig.load_from_yaml(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading configuration: {e}")
        print(f"💡 Make sure {args.config} exists and contains required configuration parameters.")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.data_file:
        config.data_file = args.data_file
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.k_range:
        config.k_range = args.k_range
    if args.no_plots:
        config.create_plots = False
    if args.no_save:
        config.save_results = False
    
    # Validate required paths
    if not Path(config.data_file).exists():
        print(f"❌ Error: Data file not found: {config.data_file}")
        sys.exit(1)
    
    if not Path(config.model_path).exists():
        print(f"❌ Error: Model path not found: {config.model_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("CLUSTERING HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Data file: {config.data_file}")
    print(f"Model path: {config.model_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Cluster range: {config.k_range}")
    print(f"Algorithms: K-means, Agglomerative, Spectral")
    print("=" * 60)
    
    # Run hyperparameter search
    try:
        searcher = HyperparameterSearcher(config)
        
        # Load data and extract embeddings
        df = searcher.load_data_and_extract_embeddings()
        
        # Run search
        all_results = searcher.run_hyperparameter_search()
        
        # Select best configurations
        best_config, top_configs, k_selection = searcher.select_best_configurations()
        
        # Create visualizations and save results
        df_results = pd.DataFrame(all_results)
        searcher.create_visualizations(df_results)
        searcher.save_results(df_results, best_config)
        
        print(f"\n✅ Hyperparameter search completed successfully!")
        print(f"🎯 Best algorithm: {best_config['algorithm'].upper()}")
        print(f"📊 Best silhouette score: {best_config['silhouette']:.4f}")
        print(f"📁 Results saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Hyperparameter search failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()