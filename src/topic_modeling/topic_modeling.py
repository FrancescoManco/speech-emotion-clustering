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

# Try to import the optimizer, provide fallback if not available
try:
    from optimaizer_umap_cons_weight import optimizer
    USE_OPTIMIZER = True
except ImportError:
    logging.warning("Custom optimizer not found, using default parameters")
    USE_OPTIMIZER = False


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
    Preprocess the input dataframe .
    
    Args:
        df: Input dataframe with text segments
        config: Configuration dictionary
        
    Returns:
        Tuple of (texts_list, filtered_dataframe)
    """
    preprocessing_config = config.get('preprocessing', {})
    
    # Filter unknown speakers if enabled
    if preprocessing_config.get('filter_unknown_speakers', True) and 'speaker' in df.columns:
        original_len = len(df)
        df = df[df['speaker'] != 'unknown'].copy()
        logging.info(f"Filtered out {original_len - len(df)} unknown speaker segments")
    
    # Filter by minimum duration if specified
    min_duration = preprocessing_config.get('min_duration', 0)
    if min_duration > 0 and 'duration' in df.columns:
        original_len = len(df)
        df = df[df['duration'] >= min_duration].copy()
        logging.info(f"Filtered out {original_len - len(df)} segments below {min_duration}s duration")
    
    # Extract texts
    texts = df['text'].tolist()
    
    # Clean texts if enabled
    if preprocessing_config.get('clean_text', True):
        texts = [text.strip().replace('\x00', '') for text in texts]
        logging.info("Cleaned text data")
    
    logging.info(f"Preprocessed {len(texts)} text segments")
    return texts, df


def compute_embeddings(
    texts: List[str], 
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Compute embeddings for the input texts.
    
    Args:
        texts: List of text strings
        model_name: Name of the sentence transformer model
        
    Returns:
        Tuple of (embedding_model, embeddings_array)
    """
    logging.info(f"Setup embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    logging.info("Computing embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    logging.info(f"Computed embeddings shape: {embeddings.shape}")
    
    return embedding_model, embeddings


def optimize_parameters(
    embeddings: np.ndarray, 
    k_min: int = 15, 
    k_max: int = 100
) -> Dict[str, Any]:
    """
    Optimize UMAP and clustering parameters.
    
    Args:
        embeddings: Embedding vectors
        k_min: Minimum number of clusters
        k_max: Maximum number of clusters
        
    Returns:
        Dictionary with optimized parameters
    """
    if USE_OPTIMIZER:
        logging.info(f"Optimizing parameters with k_min={k_min}, k_max={k_max}")
        opt = optimizer(embeddings, k_min=k_min, k_max=k_max)
        logging.info(f"Optimization complete.")
        return opt
    else:
        # Fallback to default parameters
        logging.info("Using default parameters (optimizer not available)")
        return {
            'best_umap_params': {
                'n_neighbors': 15,
                'min_dist': 0.1
            },
            'best_k': min(20, len(embeddings) // 10)
        }


def dimensionality_reduction(
    optimization_results: Dict[str, Any], 
    embeddings: np.ndarray,
    n_components: int = 5
) -> Tuple[UMAP, np.ndarray]:
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        optimization_results: Dictionary with optimized parameters
        embeddings: Input embeddings
        n_components: Number of UMAP components
        
    Returns:
        Tuple of (umap_model, reduced_embeddings)
    """
    best_params = optimization_results['best_umap_params']
    logging.info(f"Setup UMAP with params: {best_params}")
    
    umap_model = UMAP(
        n_neighbors=best_params['n_neighbors'],
        min_dist=best_params['min_dist'],
        n_components=n_components,
        random_state=42,
        n_jobs=1,
        metric='cosine'
    )
    
    logging.info("Performing UMAP dimensionality reduction...")
    embeddings_reduced = umap_model.fit_transform(embeddings)
    logging.info(f"Reduced embeddings shape: {embeddings_reduced.shape}")
    
    return umap_model, embeddings_reduced


def clustering(
    optimization_results: Dict[str, Any], 
    embeddings: np.ndarray,
    clustering_algorithm: str = "kmeans"
) -> Any:
    """
    Perform clustering on reduced embeddings.
    
    Args:
        optimization_results: Dictionary with optimized parameters
        embeddings: Input embeddings (original or reduced)
        clustering_algorithm: Algorithm to use ("kmeans" or "hdbscan")
        
    Returns:
        Clustering model
    """
    if clustering_algorithm.lower() == "kmeans":
        best_k = optimization_results['best_k']
        logging.info(f"Setup KMeans with {best_k} clusters")
        clustering_model = KMeans(
            n_clusters=best_k,
            random_state=42
        )
    elif clustering_algorithm.lower() == "hdbscan":
        logging.info("Setup HDBSCAN clustering")
        clustering_model = HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='euclidean'
        )
    else:
        raise ValueError(f"Unknown clustering algorithm: {clustering_algorithm}")
    
    logging.info("Performing clustering...")
    clustering_model.fit(embeddings)
    logging.info("Clustering completed.")
    
    return clustering_model


def vectorization(language: str = 'italian', min_df: int = 2, ngram_range: Tuple[int, int] = (1, 2)) -> CountVectorizer:
    """
    Create CountVectorizer for term extraction.
    
    Args:
        language: Language for stopwords
        min_df: Minimum document frequency
        ngram_range: Range of n-grams to extract
        
    Returns:
        Configured CountVectorizer
    """
    # Download stopwords if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logging.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    stop_words = stopwords.words(language)
    vectorizer_model = CountVectorizer(
        min_df=min_df, 
        ngram_range=ngram_range, 
        stop_words=stop_words, 
        analyzer='word'
    )
    
    logging.info(f"Created vectorizer for {language} with ngram_range={ngram_range}")
    return vectorizer_model


def setup_representation_model(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    model_name: str = "gemma3:12b-it-q4_K_M",
    ) -> OpenAI:
    """
    Setup LLM representation model for topic labeling.
    
    Args:
        base_url: OpenAI API compatible base URL
        api_key: API key
        model_name: LLM model name
        prompt_type: Type of prompt to use
        
    Returns:
        Configured OpenAI representation model
    """
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


    
    prompt = """
                Sei un esperto nell'analisi di conversazioni e nell'estrazione di argomenti tematici. 
                Analizza attentamente le seguenti frasi di conversazione:

                FRASI: [DOCUMENTS]

                PAROLE CHIAVE ASSOCIATE: [KEYWORDS]

                Basandoti sulle frasi e sulle parole chiave fornite, identifica l'argomento principale all'interno di quel frammento di conversazione.

                Estrai un'etichetta tematica molto specifica e descrittiva, di massimo 2 parole, che catturi l'essenza del tema principale. 
                Ignora le parole comuni o di supporto e concentrati sul contenuto semantico.

                Risposta nel formato: topic: <topic label>
            """
    
    representation_model = OpenAI(
        client, 
        model=model_name,
        exponential_backoff=True, 
        chat=True, 
        prompt=prompt
    )
    
    logging.info(f"Setup representation model with {model_name}.")
    return representation_model


def create_visualizations(
    topic_model: BERTopic,
    texts: List[str],
    embeddings_reduced: np.ndarray,
    topics: List[int],
    output_dir: Path
) -> None:
    """
    Create and save topic visualizations.
    
    Args:
        topic_model: Trained BERTopic model
        texts: Original texts
        embeddings_reduced: Reduced embeddings
        topics: Topic assignments
        output_dir: Directory to save visualizations
    """
    logging.info("Creating visualizations...")
    
    try:
        # Topic visualization
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(output_dir / "topics_visualization.html")
        logging.info("Saved topics visualization")
        
        # Document visualization
        fig_docs = topic_model.visualize_documents(
            texts, 
            reduced_embeddings=embeddings_reduced
        )
        fig_docs.write_html(output_dir / "documents_visualization.html")
        logging.info("Saved documents visualization")
        
        # 2D scatter plot
        if embeddings_reduced.shape[1] >= 2:
            df_viz = pd.DataFrame({
                'x': embeddings_reduced[:, 0],
                'y': embeddings_reduced[:, 1],
                'topic': topics,
                'text': [text[:100] + "..." for text in texts]
            })
            
            fig_2d = px.scatter(
                df_viz,
                x='x', y='y',
                color='topic',
                hover_data=['text'],
                title="Topic Distribution (2D)"
            )
            
            fig_2d.update_traces(
                marker=dict(size=5, opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
            )
            
            fig_2d.write_html(output_dir / "topics_2d.html")
            logging.info("Saved 2D visualization")
            
    except Exception as e:
        logging.warning(f"Could not create some visualizations: {e}")


def search_similar_topics(
    topic_model: BERTopic,
    config: Dict[str, Any],
    df_documents: pd.DataFrame,
    original_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Search for topics similar to specified keywords.
    
    Args:
        topic_model: Trained BERTopic model
        config: Configuration dictionary
        df_documents: Document info from BERTopic
        original_df: Original dataframe
        output_dir: Directory to save outputs
    """
    search_config = config.get('topic_search', {})
    
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
                    if topic_info:
                        logging.info(f"  Topic {topic_id}: {topic_info[0][0]}, Similarity: {similarity[i]:.4f}")
            
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
    
    # Override representation type if provided
    if representation_type:
        config.setdefault('llm', {})['prompt_type'] = representation_type
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_path / "topic_modeling.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Load and preprocess data
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} segments from {input_csv}")
    
    texts, filtered_df = preprocess_data(df, config)
    
    # Get model configurations
    embedding_config = config.get('embedding', {})
    umap_config = config.get('umap', {})
    clustering_config = config.get('clustering', {})
    vectorizer_config = config.get('vectorizer', {})
    llm_config = config.get('llm', {})
    
    # Compute embeddings
    embedding_model, embeddings = compute_embeddings(
        texts, 
        embedding_config.get('model', 'paraphrase-multilingual-MiniLM-L12-v2')
    )
    
    # Optimize parameters
    optimization_results = optimize_parameters(
        embeddings,
        k_min=umap_config.get('k_min', 15),
        k_max=umap_config.get('k_max', 100)
    )
    
    # Dimensionality reduction
    umap_model, embeddings_reduced = dimensionality_reduction(
        optimization_results,
        embeddings,
        n_components=umap_config.get('n_components', 5)
    )
    
    # Clustering
    clustering_model = clustering(
        optimization_results,
        embeddings_reduced,
        clustering_algorithm=clustering_config.get('algorithm', 'kmeans')
    )
    
    # Vectorization
    vectorizer_model = vectorization(
        language=vectorizer_config.get('language', 'italian'),
        min_df=vectorizer_config.get('min_df', 2),
        ngram_range=tuple(vectorizer_config.get('ngram_range', [1, 2]))
    )
    
    # Setup representation model
    representation_model = setup_representation_model(
        base_url=llm_config.get('base_url', 'http://localhost:11434/v1'),
        api_key=llm_config.get('api_key', 'ollama'),
        model_name=llm_config.get('model', 'gemma3:12b-it-q4_K_M'),
        prompt_type=llm_config.get('prompt_type', 'chat')
    )
    
    # Create BERTopic model
    topic_model = BERTopic(
        language=config.get('language', 'italian'),
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
    logging.info(f"Model trained. Found {len(set(topics))} topics")
    
    # Get results
    df_topics = topic_model.get_topic_info()
    df_documents = topic_model.get_document_info(texts)
    
    # Save main results
    df_topics.to_csv(output_path / "topic_info.csv", index=False)
    df_documents.to_csv(output_path / "document_info.csv", index=False)
    logging.info("Saved topic and document information")
    
    # Create filtered CSV for clustering pipeline
    filtered_with_topics = filtered_df.copy()
    filtered_with_topics['topic'] = topics
    filtered_with_topics['topic_name'] = df_documents['Name'].tolist()
    filtered_csv_path = output_path / "filtered_with_topics.csv"
    filtered_with_topics.to_csv(filtered_csv_path, index=False)
    logging.info(f"Saved filtered CSV with topics to {filtered_csv_path}")
    
    
    # Search similar topics
    search_similar_topics(
        topic_model,
        config,
        df_documents,
        filtered_df,
        output_path
    )
    
    # Create summary report
    summary = f"""
Topic Modeling Analysis Summary
================================
Input file: {input_csv}
Total segments: {len(df)}
Processed segments: {len(texts)}
Number of topics: {len(set(topics))}
Output directory: {output_path}

Top 10 Topics:
--------------
{df_topics[['Topic', 'Count', 'Name']].head(10).to_string()}
"""
    
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)
    
    logging.info("Analysis complete")
    logging.info(summary)
    
    # Return filtered CSV path if requested (for pipeline use)
    if return_filtered_csv:
        return str(filtered_csv_path)
    
    return str(output_path)


def main():
    """CLI interface for topic modeling analysis."""
    parser = argparse.ArgumentParser(description="BERTopic Analysis for Speech Emotion Clustering")
    parser.add_argument("input_csv", help="Input CSV file with text segments")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--representation", 
        choices=['base', 'chat', 'summarization'],
        default='chat',
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