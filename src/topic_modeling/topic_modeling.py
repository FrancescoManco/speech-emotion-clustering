import pandas as pd
import numpy as np
import yaml
import pickle
import argparse
import logging
import openai
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from optimaizer_umap_cons_weight import optimizer

def load_config(config_path: str = "configs/topic_modeling_config.yaml") -> Dict[str, Any]:
    """Load topic modeling configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        # Fallback to default configs location
        config_file = Path(__file__).parent.parent.parent / "configs" / "topic_modeling_config.yaml"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    """
    Preprocess the input dataframe according to configuration settings.
    
    Args:
        df: Input dataframe with text segments
        config: Configuration dictionary
        
    Returns:
        Tuple of (texts_list, filtered_dataframe)
    """
    preprocessing_config = config['preprocessing']
    
    # Filter unknown speakers if enabled
    if preprocessing_config.get('filter_unknown_speakers', True):
        original_len = len(df)
        df = df[df['speaker'] != 'unknown'].copy()
        logging.info(f"Filtered out {original_len - len(df)} unknown speaker segments")
    
    # Filter by minimum duration if specified
    min_duration = preprocessing_config.get('min_duration', 1.5)
    if min_duration > 0 and 'duration' in df.columns:
        original_len = len(df)
        df = df[df['duration'] >= min_duration].copy()
        logging.info(f"Filtered out {original_len - len(df)} segments below {min_duration}s duration")
    
    # Extract texts
    texts = df['text'].tolist()
    
    # Clean texts if enabled
    if preprocessing_config.get('clean_text', True):
        texts = [text.replace('\x00', '') for text in texts]
        logging.info("Cleaned text data")
    
    logging.info(f"Preprocessed {len(texts)} text segments")
    return texts, df



def compute_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Compute embeddings for the input texts."""
    logging.info(f"Setup embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    logging.info("Computing embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    logging.info(f"Computed embeddings shape: {embeddings.shape}")
    return embedding_model, embeddings


def optimize_parameters(embeddings: np.ndarray) -> Dict[str, Any]:
    opt = optimizer(embeddings, k_min=15, k_max=100)
    return opt

def dimensionality_reduction(optimization_results: Dict[str,Any], embeddings: np.ndarray) -> np.ndarray:
    best_params = optimization_results['best_umap_params']
    logging.info(f"Setup UMAP with params: {best_params}")
    umap_model = UMAP(
        n_neighbors=best_params['n_neighbors'],
        min_dist=best_params['min_dist'],
        n_components=5,
        random_state=42,
        n_jobs=1,
        metric='cosine'
    )
    logging.info("Performing UMAP dimensionality reduction...")
    embeddings_reduced = umap_model.fit_transform(embeddings)
    logging.info("Reduction completed.")
    return umap_model,embeddings_reduced


def clustering(optimization_results: Dict[str,Any], embeddings_reduced: np.ndarray):
    best_params = optimization_results['best_k']
    logging.info(f"Setup KMeans with {best_params} clusters")
    clustering_model = KMeans(
        n_clusters=best_params,
        random_state=42
    )
    logging.info("Performing clustering...")
    clusters = clustering_model.fit(embeddings_reduced)
    logging.info("Clustering completed.")
    return clustering_model, clusters

def vectorization(language: str = 'italian') -> CountVectorizer:
    stop_words = stopwords.words(language)
    vectorizer_model = CountVectorizer(min_df=2, ngram_range=(1, 2), stop_words=stop_words, analyzer='word')
    return vectorizer_model


def setup_representation_model(base_url: str="http://localhost:11434/v1", api_key: str="ollama"):
    """Setup LLM representation model for topic labeling."""

    client = openai.OpenAI(
        base_url = base_url,
        api_key=api_key, 
    )

    prompt= """ 
    Sei un esperto nell'analisi di conversazioni e nell'estrazione di argomenti tematici. Analizza attentamente le seguenti frasi di conversazione tra un commerciale e un cliente:

    FRASI: [DOCUMENTS]

    PAROLE CHIAVE ASSOCIATE: [KEYWORDS]

    Basandoti sulle frasi e sulle parole chiave fornite, identifica l'argomento principale all'interno di quel frammento di conversazione.

    Estrai un'etichetta tematica molto specifica e descrittiva, di massimo 2 parole, che catturi l'essenza del tema principale. Ignora le parole comuni o di supporto e concentrati sul contenuto semantico.

    Risposta nel formato: topic: <topic label>

    """
    representation_model = OpenAI(client, model='gemma3:12b-it-q4_K_M',exponential_backoff=True, chat=True, prompt=prompt)
    return representation_model


def search_similar_topics(
    topic_model: BERTopic,
    config: Dict[str, Any],
    df_documents: pd.DataFrame,
    original_df: pd.DataFrame,
    output_dir: Path
):
    """Search for topics similar to specified keywords."""
    search_config = config['topic_search']
    
    if not search_config.get('enable_search', False):
        return
    
    keywords = search_config.get('search_keywords', [])
    top_n = search_config.get('top_n', 5)
    
    if not keywords:
        logging.info("No search keywords specified, skipping topic search")
        return
    
    for keyword in keywords:
        try:
            similar_topics, similarity = topic_model.find_topics(keyword, top_n=top_n)
            
            # Log results
            logging.info(f"Similar topics for '{keyword}':")
            for i, topic_id in enumerate(similar_topics):
                if i < len(similarity):
                    topic_info = topic_model.get_topic(topic_id)
                    logging.info(f"  Topic {topic_id}: {topic_info}, Similarity: {similarity[i]:.4f}")
            
            # Filter documents for these topics
            filtered_docs = df_documents[df_documents['Topic'].isin(similar_topics)]
            
            if len(filtered_docs) > 0:
                # Create mapping and filter original dataframe
                document_to_name = dict(zip(filtered_docs['Document'], filtered_docs['Name']))
                matching_rows = original_df[original_df['text'].isin(filtered_docs['Document'])].copy()
                matching_rows['Topic_Name'] = matching_rows['text'].map(document_to_name)
                
                # Save filtered results
                output_file = output_dir / f"{keyword}_filtered.csv"
                matching_rows.to_csv(output_file, index=False)
                logging.info(f"Saved {len(matching_rows)} matching rows for '{keyword}' to {output_file}")
        
        except Exception as e:
            logging.warning(f"Could not search for keyword '{keyword}': {e}")


def topic_modeling_analysis(
    input_csv: str,
    output_dir: str,
    config_path: str = None,
    representation_type: str = None,
    return_filtered_csv: bool = False
) -> str:
    """
    Main topic modeling analysis function.
    
    Args:
        input_csv: Path to input CSV with text segments
        output_dir: Directory to save outputs
        config_path: Path to configuration file
        representation_type: Type of LLM representation to use
        return_filtered_csv: If True, return path to filtered CSV for clustering
        
    Returns:
        Path to output directory or filtered CSV if return_filtered_csv=True
    """
    # Load configuration
    if config_path is None:
        config_path = "configs/topic_modeling_config.yaml"
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} segments from {input_csv}")
    
    texts, filtered_df = preprocess_data(df, config)
    
    # Setup embending models
   
    
    # Compute embeddings
    
    
    # Optimize parameters

    
    # Setup pipeline components
    
    # Create BERTopic model

    
    # Train model

    
    # Get results

    
    # Save main results

    
    # Create filtered CSV for clustering pipeline

    
    # Filter original dataframe and add topic information

    # Save filtered CSV for clustering

    # Save model if enabled

    # Save embeddings if enabled

    # Save topics pickle if enabled

    
    # Create visualizations

    # Search similar topics

    # Create summary

    # Return filtered CSV path if requested (for pipeline use)



def main():
    """CLI interface for topic modeling analysis."""
    parser = argparse.ArgumentParser(description="BERTopic Analysis for Speech Emotion Clustering")
    parser.add_argument("input_csv", help="Input CSV file with text segments")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--representation", 
        choices=['base', 'deep_seek', 'chat', 'claude', 'summarization'],
        help="Type of LLM representation to use"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run topic modeling analysis
        output_path = topic_modeling_analysis(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            config_path=args.config,
            representation_type=args.representation
        )
        
        logging.info(f"Topic modeling analysis completed successfully. Results saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Topic modeling analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()