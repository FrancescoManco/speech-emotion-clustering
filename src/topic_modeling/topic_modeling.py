#!/usr/bin/env python3

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

# Topic modeling imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go

# NLTK for stopwords
import nltk
from nltk.corpus import stopwords


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
    min_duration = preprocessing_config.get('min_duration', 0.0)
    if min_duration > 0 and 'duration' in df.columns:
        original_len = len(df)
        df = df[df['duration'] >= min_duration].copy()
        logging.info(f"Filtered out {original_len - len(df)} segments below {min_duration}s duration")
    
    # Extract texts
    texts = df['text'].tolist()
    
    # Clean texts if enabled
    if preprocessing_config.get('clean_text', True):
        texts = [text.strip() for text in texts]
        texts = [text.replace('\x00', '') for text in texts]
        logging.info("Cleaned text data")
    
    logging.info(f"Preprocessed {len(texts)} text segments")
    return texts, df


def setup_embedding_model(config: Dict[str, Any]) -> SentenceTransformer:
    """Setup the sentence transformer embedding model."""
    model_name = config['models']['embedding_model']
    logging.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def compute_embeddings(texts: List[str], embedding_model: SentenceTransformer) -> np.ndarray:
    """Compute embeddings for the input texts."""
    logging.info("Computing embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    logging.info(f"Computed embeddings shape: {embeddings.shape}")
    return embeddings


def optimize_parameters(embeddings: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize UMAP and clustering parameters.
    
    This is a placeholder for the custom optimizer used in the notebook.
    In practice, you would implement or import the ale.optimaizer_umap_cons_weight.optimizer
    """
    optimization_config = config['optimization']
    
    if not optimization_config.get('enable_optimization', True):
        # Return default parameters from config
        return {
            'best_k': 20,  # Default number of clusters
            'best_umap_params': {
                'n_neighbors': config['umap']['n_neighbors'],
                'min_dist': config['umap']['min_dist']
            }
        }
    
    # Placeholder optimization logic
    # In a real implementation, you would import and use:
    # from ale.optimaizer_umap_cons_weight import optimizer
    # opt = optimizer(embeddings, k_min=optimization_config['k_min'], k_max=optimization_config['k_max'])
    
    logging.warning("Parameter optimization not implemented - using default values")
    logging.info("To enable optimization, implement or import the optimizer from ale.optimaizer_umap_cons_weight")
    
    # Return default values
    return {
        'best_k': min(20, len(embeddings) // 10),  # Heuristic default
        'best_umap_params': {
            'n_neighbors': config['umap']['n_neighbors'],
            'min_dist': config['umap']['min_dist']
        }
    }


def setup_umap_model(config: Dict[str, Any], optimization_results: Dict[str, Any]) -> UMAP:
    """Setup UMAP dimensionality reduction model."""
    umap_config = config['umap']
    best_params = optimization_results['best_umap_params']
    
    umap_model = UMAP(
        n_neighbors=best_params['n_neighbors'],
        min_dist=best_params['min_dist'],
        n_components=umap_config['n_components'],
        random_state=umap_config['random_state'],
        metric=umap_config['metric'],
        n_jobs=1
    )
    
    logging.info(f"Setup UMAP with n_neighbors={best_params['n_neighbors']}, min_dist={best_params['min_dist']}")
    return umap_model


def setup_clustering_model(config: Dict[str, Any], optimization_results: Dict[str, Any]):
    """Setup clustering model based on configuration."""
    clustering_config = config['clustering']
    best_k = optimization_results['best_k']
    
    algorithm = clustering_config.get('algorithm', 'kmeans')
    
    if algorithm == 'kmeans':
        model = KMeans(
            n_clusters=best_k,
            random_state=clustering_config.get('random_state', 42)
        )
    elif algorithm == 'hdbscan':
        model = HDBSCAN(
            min_cluster_size=max(2, best_k // 5),
            metric='euclidean'
        )
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    logging.info(f"Setup {algorithm} clustering with {best_k} clusters")
    return model


def setup_vectorizer(config: Dict[str, Any]) -> CountVectorizer:
    """Setup count vectorizer with stopwords."""
    vectorization_config = config['vectorization']
    language = vectorization_config.get('language', 'italian')
    
    # Download stopwords if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = stopwords.words(language)
    
    vectorizer = CountVectorizer(
        min_df=vectorization_config.get('min_df', 2),
        ngram_range=tuple(vectorization_config.get('ngram_range', [1, 2])),
        stop_words=stop_words,
        analyzer=vectorization_config.get('analyzer', 'word')
    )
    
    logging.info(f"Setup vectorizer with {language} stopwords")
    return vectorizer


def setup_representation_model(config: Dict[str, Any], representation_type: str = None):
    """Setup LLM representation model for topic labeling."""
    if representation_type is None:
        representation_type = config.get('default_representation', 'chat')
    
    ollama_config = config['models']['ollama']
    prompts = config['representation_prompts']
    
    if representation_type not in prompts:
        raise ValueError(f"Unknown representation type: {representation_type}")
    
    # Setup OpenAI client for Ollama
    client = openai.OpenAI(
        base_url=ollama_config['base_url'],
        api_key=ollama_config['api_key']
    )
    
    prompt = prompts[representation_type]
    
    representation_model = OpenAI(
        client,
        model=ollama_config['name'],
        exponential_backoff=True,
        chat=True,
        prompt=prompt
    )
    
    logging.info(f"Setup {representation_type} representation model with {ollama_config['name']}")
    return representation_model


def create_visualizations(
    topic_model: BERTopic,
    texts: List[str],
    embeddings_reduced: np.ndarray,
    topics: List[int],
    config: Dict[str, Any],
    output_dir: Path
):
    """Create and save visualization plots."""
    viz_config = config['visualization']
    
    if not viz_config.get('save_visualizations', True):
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # BERTopic built-in visualizations
    if viz_config.get('enable_topics_plot', True):
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(output_dir / "topics_overview.html")
            logging.info("Saved topics overview plot")
        except Exception as e:
            logging.warning(f"Could not create topics overview plot: {e}")
    
    if viz_config.get('enable_documents_plot', True):
        try:
            fig = topic_model.visualize_documents(texts, reduced_embeddings=embeddings_reduced)
            fig.write_html(output_dir / "documents_plot.html")
            logging.info("Saved documents plot")
        except Exception as e:
            logging.warning(f"Could not create documents plot: {e}")
    
    # Custom 2D plot
    if viz_config.get('enable_2d_plot', True):
        try:
            df_plot = pd.DataFrame({
                'x': embeddings_reduced[:, 0],
                'y': embeddings_reduced[:, 1],
                'topic': topics,
                'text': [doc[:100] + "..." for doc in texts]
            })
            
            fig = px.scatter(
                df_plot, x='x', y='y', color='topic',
                hover_data=['text'],
                title="2D Visualization of Documents by Topic",
                width=viz_config.get('plot_width', 800),
                height=viz_config.get('plot_height', 600)
            )
            fig.write_html(output_dir / "2d_visualization.html")
            logging.info("Saved 2D visualization")
        except Exception as e:
            logging.warning(f"Could not create 2D visualization: {e}")
    
    # Custom 3D plot
    if viz_config.get('enable_3d_plot', True) and embeddings_reduced.shape[1] >= 3:
        try:
            df_plot = pd.DataFrame({
                'x': embeddings_reduced[:, 0],
                'y': embeddings_reduced[:, 1],
                'z': embeddings_reduced[:, 2],
                'topic': topics,
                'text': [doc[:100] + "..." for doc in texts]
            })
            
            fig = px.scatter_3d(
                df_plot, x='x', y='y', z='z', color='topic',
                hover_data=['text'],
                title="3D Visualization of Documents by Topic",
                width=viz_config.get('plot_width', 800),
                height=viz_config.get('plot_height', 600)
            )
            fig.write_html(output_dir / "3d_visualization.html")
            logging.info("Saved 3D visualization")
        except Exception as e:
            logging.warning(f"Could not create 3D visualization: {e}")


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
    
    # Setup models
    embedding_model = setup_embedding_model(config)
    
    # Compute embeddings
    embeddings = compute_embeddings(texts, embedding_model)
    
    # Optimize parameters
    optimization_results = optimize_parameters(embeddings, config)
    logging.info(f"Optimization results: {optimization_results}")
    
    # Setup pipeline components
    umap_model = setup_umap_model(config, optimization_results)
    clustering_model = setup_clustering_model(config, optimization_results)
    vectorizer_model = setup_vectorizer(config)
    representation_model = setup_representation_model(config, representation_type)
    
    # Perform dimensionality reduction
    logging.info("Performing UMAP dimensionality reduction...")
    embeddings_reduced = umap_model.fit_transform(embeddings)
    logging.info(f"Reduced embeddings shape: {embeddings_reduced.shape}")
    
    # Create BERTopic model
    logging.info("Creating BERTopic model...")
    topic_model = BERTopic(
        language="italian",
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True
    )
    
    # Train model
    logging.info("Training BERTopic model...")
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    # Get results
    df_topic_info = topic_model.get_topic_info()
    df_documents = topic_model.get_document_info(texts)
    
    # Save main results
    df_topic_info.to_csv(output_path / "topic_info.csv", index=False)
    df_documents.to_csv(output_path / "document_info.csv", index=False)
    logging.info(f"Saved topic analysis results to {output_path}")
    
    # Create filtered CSV for clustering pipeline
    # Map documents back to original dataframe with topic assignments
    document_to_topic = dict(zip(df_documents['Document'], df_documents['Topic']))
    document_to_name = dict(zip(df_documents['Document'], df_documents['Name']))
    
    # Filter original dataframe and add topic information
    filtered_df_for_clustering = filtered_df[filtered_df['text'].isin(df_documents['Document'])].copy()
    filtered_df_for_clustering['topic'] = filtered_df_for_clustering['text'].map(document_to_topic)
    filtered_df_for_clustering['topic_name'] = filtered_df_for_clustering['text'].map(document_to_name)
    
    # Save filtered CSV for clustering
    filtered_csv_path = output_path / "filtered_for_clustering.csv"
    filtered_df_for_clustering.to_csv(filtered_csv_path, index=False)
    logging.info(f"Saved {len(filtered_df_for_clustering)} filtered segments for clustering to {filtered_csv_path}")
    
    # Save model if enabled
    if config['output'].get('save_model', True):
        try:
            model_path = output_path / "topic_model"
            topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
            logging.info(f"Saved BERTopic model to {model_path}")
        except Exception as e:
            logging.warning(f"Could not save model: {e}")
    
    # Save embeddings if enabled
    if config['output'].get('save_embeddings', True):
        np.save(output_path / "embeddings.npy", embeddings)
        np.save(output_path / "embeddings_reduced.npy", embeddings_reduced)
        logging.info("Saved embeddings")
    
    # Save topics pickle if enabled
    if config['output'].get('save_topics_pickle', False):
        with open(output_path / "topics.pickle", "wb") as f:
            pickle.dump(topics, f)
        logging.info("Saved topics pickle")
    
    # Create visualizations
    create_visualizations(topic_model, texts, embeddings_reduced, topics, config, output_path / "visualizations")
    
    # Search similar topics
    search_similar_topics(topic_model, config, df_documents, filtered_df, output_path)
    
    # Create summary
    summary = {
        'total_documents': len(texts),
        'total_topics': len(df_topic_info),
        'representation_model': representation_type or config.get('default_representation', 'chat'),
        'optimization_results': optimization_results
    }
    
    with open(output_path / "analysis_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    logging.info(f"Topic modeling analysis completed. Results saved to: {output_path}")
    
    # Return filtered CSV path if requested (for pipeline use)
    if return_filtered_csv:
        return str(filtered_csv_path)
    else:
        return str(output_path)


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