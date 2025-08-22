#!/usr/bin/env python3

import pandas as pd
import numpy as np
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_config(config_path: str = "configs/llm_config.yaml") -> Dict[str, Any]:
    """Load LLM analysis configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        # Fallback to default configs location
        config_file = Path(__file__).parent.parent.parent / "configs" / "llm_config.yaml"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)




def extract_representative_segments(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract representative segments from each cluster based on distance and duration.
    
    Args:
        df: Input dataframe with cluster assignments
        config: Configuration dictionary
        
    Returns:
        DataFrame with representative segments for analysis
    """
    analysis_config = config['analysis']
    top_k = analysis_config['top_segments_per_cluster']
    weight_distance = analysis_config['weight_distance']
    weight_duration = analysis_config['weight_duration']
    min_duration = analysis_config['min_duration']
    max_segments = analysis_config['max_segments_total']
    
    representative_segments = []
    
    # Process each cluster
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id].copy()
        
        # Filter by minimum duration
        cluster_df = cluster_df[cluster_df['duration'] >= min_duration]
        
        if len(cluster_df) == 0:
            continue
            
        # Calculate composite score (lower distance + higher duration = higher score)
        if 'distance_to_centroid' in cluster_df.columns:
            # Normalize distance (invert so lower distance = higher score)
            max_distance = cluster_df['distance_to_centroid'].max()
            min_distance = cluster_df['distance_to_centroid'].min()
            if max_distance != min_distance:
                distance_score = 1 - (cluster_df['distance_to_centroid'] - min_distance) / (max_distance - min_distance)
            else:
                distance_score = 1.0
        else:
            distance_score = 1.0
            
        # Normalize duration
        max_duration = cluster_df['duration'].max()
        min_duration_val = cluster_df['duration'].min()
        if max_duration != min_duration_val:
            duration_score = (cluster_df['duration'] - min_duration_val) / (max_duration - min_duration_val)
        else:
            duration_score = 1.0
            
        # Composite score
        cluster_df['selection_score'] = (weight_distance * distance_score + 
                                       weight_duration * duration_score)
        
        # Select top segments
        top_segments = cluster_df.nlargest(top_k, 'selection_score')
        representative_segments.append(top_segments)
    
    # Combine all representative segments
    result_df = pd.concat(representative_segments, ignore_index=True)
    
    # Limit total segments if needed
    if len(result_df) > max_segments:
        result_df = result_df.nlargest(max_segments, 'selection_score')
    
    logging.info(f"Selected {len(result_df)} representative segments from {df['cluster'].nunique()} clusters")
    return result_df


def call_ollama_api(text: str, config: Dict[str, Any], prompt_type: str = 'text_only') -> Dict[str, Any]:
    """
    Call Ollama API for text-based analysis.
    
    Args:
        text: Text to analyze
        config: Configuration dictionary
        prompt_type: Type of prompt to use ('text_only' or 'text_audio')
        
    Returns:
        API response as dictionary
    """
    model_config = config['models']['ollama']
    prompts = config['prompts']
    
    # Construct full prompt
    system_prompt = prompts['system_base']
    user_prompt = prompts[prompt_type]
    
    if prompt_type == 'text_audio':
        # For text+audio, we'll need to format with placeholders
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
    else:
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nTESTO: {text}"
    
    payload = {
        "model": model_config['name'],
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(
            f"{model_config['base_url']}/api/generate",
            json=payload,
            timeout=model_config['timeout']
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Ollama API call failed: {e}")
        return {"response": f"Error: {str(e)}"}


def analyze_text_only(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyze segments using text-only prompting with Ollama.
    
    Args:
        df: Input dataframe with representative segments
        config: Configuration dictionary
        
    Returns:
        DataFrame with analysis results
    """
    logging.info("Starting text-only analysis with Ollama...")
    
    results = []
    
    # Group by cluster for analysis
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Format texts with speakers like in the original method
        formatted_texts = ""
        for idx, row in cluster_df.iterrows():
            speaker = row.get('speaker', 'Unknown')
            text = row.get('text', '')
            formatted_texts += f'"{speaker}": {text}\n'
        
        # Call Ollama API with cluster analysis
        response = call_ollama_api(formatted_texts, config, 'text_only')
        
        # Create result for this cluster
        result = {
            'cluster': cluster_id,
            'num_segments': len(cluster_df),
            'formatted_texts': formatted_texts,
            'analysis_method': 'text_only',
            'llm_response': response.get('response', ''),
            'model_used': config['models']['ollama']['name']
        }
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_text_with_audio_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyze segments using text + audio features prompting with Ollama.
    
    Args:
        df: Input dataframe with representative segments
        config: Configuration dictionary
        
    Returns:
        DataFrame with analysis results
    """
    logging.info("Starting text+audio features analysis with Ollama...")
    
    results = []
    
    # Group by cluster for analysis
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Format texts with speakers and impressions like in the original method
        formatted_texts = ""
        impressions_summary = []
        
        for idx, row in cluster_df.iterrows():
            speaker = row.get('speaker', 'Unknown')
            text = row.get('text', '')
            impression = row.get('impression', 'Audio features description not available')
            formatted_texts += f'"{speaker}": {text} | Descrizione feature acustiche: {impression}\n'
            impressions_summary.append(impression)
        
        # Combine impressions for the cluster
        combined_impressions = "; ".join(impressions_summary)
        
        # Format prompt with placeholders
        prompt_template = config['prompts']['text_audio']
        formatted_prompt = prompt_template.format(
            text=formatted_texts,
            audio_features=combined_impressions
        )
        
        # Call Ollama API with formatted prompt
        system_prompt = config['prompts']['system_base']
        full_prompt = f"{system_prompt}\n\n{formatted_prompt}"
        
        payload = {
            "model": config['models']['ollama']['name'],
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9}
        }
        
        try:
            response = requests.post(
                f"{config['models']['ollama']['base_url']}/api/generate",
                json=payload,
                timeout=config['models']['ollama']['timeout']
            )
            response.raise_for_status()
            api_response = response.json()
        except Exception as e:
            logging.error(f"Ollama API call failed: {e}")
            api_response = {"response": f"Error: {str(e)}"}
        
        # Create result for this cluster
        result = {
            'cluster': cluster_id,
            'num_segments': len(cluster_df),
            'formatted_texts': formatted_texts,
            'combined_impressions': combined_impressions,
            'analysis_method': 'text_audio',
            'llm_response': api_response.get('response', ''),
            'model_used': config['models']['ollama']['name']
        }
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_pure_audio(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Placeholder for pure audio analysis using multimodal model.
    
    This would require implementing audio processing with Qwen2.5-Omni or similar.
    For now, returns a placeholder implementation.
    
    Args:
        df: Input dataframe with representative segments
        config: Configuration dictionary
        
    Returns:
        DataFrame with analysis results
    """
    logging.info("Starting pure audio analysis...")
    logging.warning("Pure audio analysis is not yet implemented - returning placeholder results")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing audio segments"):
        result = {
            'segment_id': row.get('segment_id', idx),
            'cluster': row['cluster'],
            'audio_path': row.get('audio_path', ''),
            'analysis_method': 'pure_audio',
            'llm_response': "Pure audio analysis not yet implemented",
            'model_used': config['models']['multimodal']['name']
        }
        results.append(result)
    
    return pd.DataFrame(results)


def llm_analysis(
    input_csv: str, 
    output_csv: str, 
    analysis_methods: List[str] = None,
    config_path: str = None
) -> str:
    """
    Main LLM analysis function for pipeline integration.
    
    Args:
        input_csv: Path to input CSV with clustered segments
        output_csv: Path to output CSV with LLM analysis results
        analysis_methods: List of analysis methods to use
        config_path: Path to configuration file
        
    Returns:
        Path to output CSV file
    """
    # Load configuration
    if config_path is None:
        config_path = "configs/llm_config.yaml"
    config = load_config(config_path)
    
    # Default analysis methods
    if analysis_methods is None:
        analysis_methods = ['text_only']
    
    # Load input data
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} segments from {input_csv}")
    
    # Extract representative segments
    representative_df = extract_representative_segments(df, config)
    
    all_results = []
    
    # Run each analysis method
    for method in analysis_methods:
        logging.info(f"Running analysis method: {method}")
        
        if method == 'text_only':
            results = analyze_text_only(representative_df, config)
        elif method == 'text_audio':
            results = analyze_text_with_audio_features(representative_df, config)
        elif method == 'pure_audio':
            results = analyze_pure_audio(representative_df, config)
        else:
            logging.warning(f"Unknown analysis method: {method}")
            continue
            
        all_results.append(results)
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
    else:
        final_results = pd.DataFrame()
    
    # Save results
    final_results.to_csv(output_csv, index=False)
    logging.info(f"Saved {len(final_results)} analysis results to {output_csv}")
    
    return output_csv


def main():
    """CLI interface for LLM analysis."""
    parser = argparse.ArgumentParser(description="LLM Analysis for Speech Emotion Clustering")
    parser.add_argument("input_csv", help="Input CSV file with clustered segments")
    parser.add_argument("output_csv", help="Output CSV file for analysis results")
    parser.add_argument(
        "--methods", 
        nargs="+", 
        choices=['text_only', 'text_audio', 'pure_audio'],
        default=['text_only'],
        help="Analysis methods to use"
    )
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        # Run LLM analysis
        output_path = llm_analysis(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            analysis_methods=args.methods,
            config_path=args.config
        )
        
        logging.info(f"LLM analysis completed successfully. Results saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"LLM analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()