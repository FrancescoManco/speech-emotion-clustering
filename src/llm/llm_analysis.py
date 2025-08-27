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
import re

# Qwen2.5-Omni imports (with fallback for environments without it)
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    QWEN_AVAILABLE = True
except ImportError:
    logging.warning("Qwen2.5-Omni dependencies not available. Pure audio analysis will be limited.")
    QWEN_AVAILABLE = False


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


def extract_speaker_from_path(path: str) -> str:
    """
    Extract speaker identifier (e.g., 'SPEAKER_00') from file path.
    Assumes format like '.../XXX_SPEAKER_YY_ZZZ-WWW.wav'.
    """
    filename = path.split('/')[-1]
    match = re.search(r'(SPEAKER_\d{2})', filename)
    if match:
        return match.group(1)
    return "SPEAKER_UNKNOWN"


def select_representative_segments(cluster_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Select representative segments from a cluster based on distance and duration.
    Uses the same algorithm as in the notebook.
    
    Args:
        cluster_df: DataFrame with segments from a single cluster
        config: Configuration dictionary
        
    Returns:
        DataFrame with top representative segments
    """
    pure_audio_config = config.get('pure_audio', {})
    n_segments = pure_audio_config.get('segments_per_cluster', 5)
    weight_distance = pure_audio_config.get('weight_distance', 0.6)
    weight_duration = pure_audio_config.get('weight_duration', 0.4)
    
    c_df = cluster_df.copy()
    
    # Check for required columns
    if 'distance_from_centroid' not in c_df.columns:
        logging.warning("Missing 'distance_from_centroid' column, using random selection")
        return c_df.sample(min(n_segments, len(c_df)))
    
    # Normalize both metrics
    distance_min = c_df['distance_from_centroid'].min()
    distance_max = c_df['distance_from_centroid'].max()
    duration_min = c_df['duration'].min() 
    duration_max = c_df['duration'].max()
    
    if distance_max != distance_min:
        c_df['norm_distance'] = (c_df['distance_from_centroid'] - distance_min) / (distance_max - distance_min)
    else:
        c_df['norm_distance'] = 0.0
        
    if duration_max != duration_min:
        c_df['norm_duration'] = (c_df['duration'] - duration_min) / (duration_max - duration_min)
    else:
        c_df['norm_duration'] = 0.0

    # Calculate composite score (higher is better)
    # Invert distance so lower distance = higher score
    c_df['score'] = -(weight_distance * c_df['norm_distance'] - weight_duration * c_df['norm_duration'])

    # Select top segments
    top_segments = c_df.nlargest(n_segments, 'score')
    
    logging.debug(f"Selected {len(top_segments)} representative segments from cluster")
    return top_segments


def extract_assistant_response(text: str) -> str:
    """
    Extract the assistant's response from the generated text.
    """
    parts = text.split("assistant", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return text.strip()


@torch.no_grad()
def analyze_cluster_qwen(cluster_id: int, audio_paths: List[str], model, processor) -> str:
    """
    Analyze a cluster using Qwen2.5-Omni multimodal model.
    
    Args:
        cluster_id: ID of the cluster
        audio_paths: List of audio file paths to analyze
        model: Qwen2.5-Omni model
        processor: Qwen2.5-Omni processor
        
    Returns:
        Analysis response from the model
    """
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }
        ]
    }

    user_prompt_text = """
    Listen to these audio clips. Respond ONLY in this exact format:

    "Identify the shared emotional tone that led to the grouping and respond in exactly this format:

    1. Main Emotion: [emotion name] - [low/medium/high]
    2. Keywords: [word1, word2, word3]
    3. Emotional Label: [maximum 2 words]
    4. Explanation: [ maximum 25 words explaining the emotion]
    5. Main Topic: [maximum 5 words]

    Use only English. No extra comments.
    IMPORTANT: Start directly with "1. Main Emotion:"
    IMPORTANT: Do not insert any other requests or comments.
    """ 

    # Build user content with available audio files (max 3 due to limitations)
    user_content = []
    valid_audio_paths = [path for path in audio_paths[:3] if path and Path(path).exists()]
    
    if not valid_audio_paths:
        raise ValueError("No valid audio paths found")
    
    for path in valid_audio_paths:
        user_content.append({"type": "audio", "audio": path})
    
    user_content.append({"type": "text", "text": user_prompt_text})

    user_prompt = {
        "role": "user",
        "content": user_content
    }

    conversation = [system_prompt, user_prompt]
    
    # Multimodal preprocessing
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate text response
    text_ids = model.generate(
        **inputs, 
        use_audio_in_video=False, 
        return_audio=False,
        do_sample=False,
    ) 
    output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output[0] if output else "No response generated"


def analyze_pure_audio(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyze segments using pure audio with Qwen2.5-Omni multimodal model.
    
    Args:
        df: Input dataframe with representative segments 
        config: Configuration dictionary
        
    Returns:
        DataFrame with analysis results
    """
    logging.info("Starting pure audio analysis with Qwen2.5-Omni...")
    
    # Debug: print config
    logging.info(f"Config loaded: {config.get('models', {}).get('multimodal', {})}")
    
    if not QWEN_AVAILABLE:
        logging.error("Qwen2.5-Omni dependencies not available")
        return pd.DataFrame([{
            'cluster': -1,
            'num_segments': 0,
            'analysis_method': 'pure_audio',
            'llm_response': "Error: Qwen2.5-Omni dependencies not available",
            'model_used': 'N/A',
            'audio_paths': []
        }])
    
    try:
        # Initialize Qwen2.5-Omni model and processor
        model_config = config['models']['multimodal']
        model_name = model_config['name']
        
        # Get HuggingFace token from .env file or environment
        import os
        from pathlib import Path
        
        # Try to load from .env file first
        env_file = Path('.env')
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv()
                logging.info("Loaded environment from .env file")
            except ImportError:
                logging.warning("python-dotenv not installed, skipping .env file loading")
        
        hf_token = os.getenv('HF_TOKEN')
        
        logging.info(f"Loading Qwen2.5-Omni processor: {model_name}")
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name, token=hf_token)
        
        logging.info(f"Loading Qwen2.5-Omni model: {model_name}")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=model_config.get('torch_dtype', "auto"),
            device_map=model_config.get('device_map', "auto"),
            token=hf_token
        )
        model.disable_talker()
        
        logging.info(f"Successfully loaded Qwen2.5-Omni model: {model_name}")
        
    except Exception as e:
        logging.error(f"Failed to load Qwen2.5-Omni model: {e}")
        logging.error(f"Model config: {config['models']['multimodal']}")
        return pd.DataFrame([{
            'cluster': -1,
            'num_segments': 0,
            'analysis_method': 'pure_audio',
            'llm_response': f"Error loading model: {str(e)}",
            'model_used': model_config['name'],
            'audio_paths': []
        }])
    
    results = []
    
    # Group by cluster for analysis
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        
        logging.info(f"Analyzing cluster {cluster_id} with {len(cluster_df)} segments")
        
        try:
            # Select representative segments using the notebook algorithm
            representative_segments = select_representative_segments(cluster_df, config)
            
            # Extract audio paths - check multiple possible column names
            audio_paths = []
            for _, row in representative_segments.iterrows():
                audio_path = None
                # Try different possible column names for audio paths
                for col in ['segment_file', 'path', 'audio_path', 'file_path']:
                    if col in row and row[col] and str(row[col]) != 'nan':
                        audio_path = row[col]
                        break
                
                if audio_path and Path(audio_path).exists():
                    audio_paths.append(audio_path)
            
            if not audio_paths:
                logging.warning(f"No valid audio paths found for cluster {cluster_id}")
                result = {
                    'cluster': cluster_id,
                    'num_segments': len(cluster_df),
                    'analysis_method': 'pure_audio',
                    'llm_response': "No valid audio paths found",
                    'model_used': model_config['name'],
                    'audio_paths': []
                }
                results.append(result)
                continue
            
            # Analyze cluster with Qwen2.5-Omni
            analysis_response = analyze_cluster_qwen(cluster_id, audio_paths, model, processor)
            
            # Extract the clean response
            clean_response = extract_assistant_response(analysis_response)
            
            # Create result for this cluster
            result = {
                'cluster': cluster_id,
                'num_segments': len(cluster_df),
                'representative_segments': len(representative_segments),
                'audio_files_analyzed': len(audio_paths),
                'analysis_method': 'pure_audio',
                'llm_response': clean_response,
                'model_used': model_config['name'],
                'audio_paths': audio_paths[:3]  # Store first 3 paths for reference
            }
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error analyzing cluster {cluster_id}: {e}")
            result = {
                'cluster': cluster_id,
                'num_segments': len(cluster_df),
                'analysis_method': 'pure_audio',
                'llm_response': f"Analysis error: {str(e)}",
                'model_used': model_config['name'],
                'audio_paths': []
            }
            results.append(result)
    
    logging.info(f"Completed pure audio analysis for {len(results)} clusters")
    return pd.DataFrame(results)


def llm_analysis(
    input_csv: str, 
    output_csv: str, 
    analysis_methods: List[str] = None,
    config_path: str = None
) -> List[str]:
    """
    Main LLM analysis function for pipeline integration.
    
    Args:
        input_csv: Path to input CSV with clustered segments
        output_csv: Base path for output CSV files (method name will be appended)
        analysis_methods: List of analysis methods to use
        config_path: Path to configuration file
        
    Returns:
        List of paths to created CSV files
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
    
    created_files = []
    
    # Prepare base output path
    output_path = Path(output_csv)
    base_name = output_path.stem
    output_dir = output_path.parent
    output_ext = output_path.suffix
    
    # Run each analysis method and save separately
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
        
        # Create method-specific output file
        method_output_file = output_dir / f"{base_name}_{method}{output_ext}"
        
        # Save results for this method
        results.to_csv(method_output_file, index=False)
        logging.info(f"Saved {len(results)} {method} analysis results to {method_output_file}")
        
        created_files.append(str(method_output_file))
    
    return created_files


def main():
    """CLI interface for LLM analysis."""
    parser = argparse.ArgumentParser(description="LLM Analysis for Speech Emotion Clustering")
    parser.add_argument("--input_csv", help="Input CSV file with clustered segments")
    parser.add_argument("--output_csv", help="Base output CSV file path (method names will be appended)")
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
        output_files = llm_analysis(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            analysis_methods=args.methods,
            config_path=args.config
        )
        
        logging.info(f"LLM analysis completed successfully. Created {len(output_files)} files:")
        for file_path in output_files:
            logging.info(f"  - {file_path}")
        
    except Exception as e:
        logging.error(f"LLM analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()