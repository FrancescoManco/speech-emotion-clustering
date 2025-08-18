#!/usr/bin/env python3
"""
LLM-based Cluster Analysis Module

Provides comprehensive cluster analysis including weight optimization, 
LLM emotion analysis, NLP evaluation metrics, and audio file export.
"""

import argparse
import os
import sys
import re
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

# LLM and NLP evaluation
try:
    from ollama import chat
except ImportError:
    print("Warning: ollama not installed. LLM analysis will not be available.")
    chat = None

try:
    from evaluate import load
except ImportError:
    print("Warning: evaluate not installed. NLP metrics will not be available.")
    load = None

import warnings
warnings.filterwarnings("ignore")


@dataclass
class LLMAnalysisConfig:
    """Configuration for LLM-based cluster analysis."""
    
    # Data configuration
    data_file: str
    output_dir: str
    
    # Weight optimization
    n_representative_points: int = 10
    weight_steps: int = 11
    weight_distance: float = 0.6
    weight_duration: float = 0.4
    
    # LLM configuration
    llm_model: str = "gemma3:12b-it-q4_K_M"
    analysis_type: str = "short"  # "short" or "detailed"
    
    # NLP evaluation
    enable_nlp_evaluation: bool = True
    evaluation_mode: str = "individual"  # "individual" or "concatenated" or "both"
    
    # Audio export
    enable_audio_export: bool = True
    create_zip_archive: bool = True
    
    # Output options
    save_results: bool = True
    create_google_forms_data: bool = True
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> 'LLMAnalysisConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                yaml_config = yaml.safe_load(file)
            
            # Extract llm_analysis-specific configuration
            analysis_config = yaml_config.get('llm_analysis', {})
            
            config = cls(
                data_file=analysis_config['data_file'],
                output_dir=analysis_config['output_dir'],
                n_representative_points=analysis_config.get('n_representative_points', 10),
                weight_steps=analysis_config.get('weight_steps', 11),
                weight_distance=analysis_config.get('weight_distance', 0.6),
                weight_duration=analysis_config.get('weight_duration', 0.4),
                llm_model=analysis_config.get('llm_model', 'gemma3:12b-it-q4_K_M'),
                analysis_type=analysis_config.get('analysis_type', 'short'),
                enable_nlp_evaluation=analysis_config.get('enable_nlp_evaluation', True),
                evaluation_mode=analysis_config.get('evaluation_mode', 'individual'),
                enable_audio_export=analysis_config.get('enable_audio_export', True),
                create_zip_archive=analysis_config.get('create_zip_archive', True),
                save_results=analysis_config.get('save_results', True),
                create_google_forms_data=analysis_config.get('create_google_forms_data', True)
            )
            
            print(f"LLM analysis configuration loaded from: {yaml_path}")
            return config
            
        except KeyError as e:
            raise ValueError(f"Missing required key in YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {e}")


class ClusterWeightOptimizer:
    """Optimizes weights for selecting representative cluster elements."""
    
    def __init__(self, config: LLMAnalysisConfig):
        self.config = config
    
    def extract_speaker_from_path(self, path: str) -> str:
        """Extract speaker identifier from file path."""
        if pd.isna(path):
            return "SPEAKER_UNKNOWN"
        
        filename = path.split('/')[-1]
        match = re.search(r'(SPEAKER_\d{2})', filename)
        return match.group(1) if match else "SPEAKER_UNKNOWN"
    
    def analyze_cluster_weights(self, cluster_id: int, df_cluster: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze different weight configurations for a cluster."""
        
        # Calculate cluster statistics
        cluster_stats = {
            'avg_duration': df_cluster['duration'].mean(),
            'avg_text_length': df_cluster['text'].str.len().mean(),
            'avg_distance': df_cluster['distance_from_centroid'].mean()
        }
        
        print(f"📊 CLUSTER {cluster_id} STATISTICS")
        print(f"   Duration: {cluster_stats['avg_duration']:.2f}s")
        print(f"   Text length: {cluster_stats['avg_text_length']:.2f} chars")
        print(f"   Distance: {cluster_stats['avg_distance']:.4f}")
        
        # Prepare DataFrame for normalization
        c_df = df_cluster.copy()
        c_df['text_length'] = c_df['text'].str.len()
        
        # Normalize metrics
        c_df['norm_distance'] = (c_df['distance_from_centroid'] - c_df['distance_from_centroid'].min()) / (c_df['distance_from_centroid'].max() - c_df['distance_from_centroid'].min())
        c_df['norm_duration'] = (c_df['duration'] - c_df['duration'].min()) / (c_df['duration'].max() - c_df['duration'].min())
        
        # Generate weight combinations
        weight_range = np.linspace(0.0, 1.0, self.config.weight_steps)
        weight_combinations = []
        
        for w_dist in weight_range:
            for w_dur in weight_range:
                if abs(w_dist + w_dur - 1.0) < 0.001:
                    weight_combinations.append((w_dist, w_dur))
        
        # Test each combination
        results = []
        
        for w_distance, w_duration in weight_combinations:
            # Calculate composite score
            c_df['score'] = -(w_distance * c_df['norm_distance'] - w_duration * c_df['norm_duration'])
            
            # Select best points
            best_points = c_df.nlargest(self.config.n_representative_points, 'score')
            
            # Calculate metrics on selected points
            selected_stats = {
                'selected_avg_duration': best_points['duration'].mean(),
                'selected_avg_text_length': best_points['text_length'].mean(),
                'selected_avg_distance': best_points['distance_from_centroid'].mean()
            }
            
            # Calculate quality score
            max_duration = df_cluster['duration'].max()
            max_text_length = df_cluster['text'].str.len().max()
            min_distance = df_cluster['distance_from_centroid'].min()
            max_distance = df_cluster['distance_from_centroid'].max()
            
            duration_score = selected_stats['selected_avg_duration'] / max_duration if max_duration > 0 else 0
            text_length_score = selected_stats['selected_avg_text_length'] / max_text_length if max_text_length > 0 else 0
            
            if max_distance > min_distance:
                distance_score = 1 - ((selected_stats['selected_avg_distance'] - min_distance) / (max_distance - min_distance))
            else:
                distance_score = 1
            
            quality_score = duration_score + text_length_score + distance_score
            
            results.append({
                'weight_distance': w_distance,
                'weight_duration': w_duration,
                'selected_avg_duration': selected_stats['selected_avg_duration'],
                'selected_avg_text_length': selected_stats['selected_avg_text_length'],
                'selected_avg_distance': selected_stats['selected_avg_distance'],
                'duration_score': duration_score,
                'text_length_score': text_length_score,
                'distance_score': distance_score,
                'quality_score': quality_score
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('quality_score', ascending=False)
        
        return results_df, cluster_stats
    
    def get_representative_elements(self, df_cluster: pd.DataFrame, 
                                  weight_distance: Optional[float] = None,
                                  weight_duration: Optional[float] = None) -> pd.DataFrame:
        """Get representative elements using specified or optimized weights."""
        
        if weight_distance is None:
            weight_distance = self.config.weight_distance
        if weight_duration is None:
            weight_duration = self.config.weight_duration
        
        c_df = df_cluster.copy()
        
        # Normalize metrics
        if len(c_df) > 1:
            c_df['norm_distance'] = (c_df['distance_from_centroid'] - c_df['distance_from_centroid'].min()) / (c_df['distance_from_centroid'].max() - c_df['distance_from_centroid'].min())
            c_df['norm_duration'] = (c_df['duration'] - c_df['duration'].min()) / (c_df['duration'].max() - c_df['duration'].min())
        else:
            c_df['norm_distance'] = 0
            c_df['norm_duration'] = 0
        
        # Calculate composite score
        c_df['score'] = -(weight_distance * c_df['norm_distance'] - weight_duration * c_df['norm_duration'])
        
        # Select best points
        n_points = min(self.config.n_representative_points, len(c_df))
        nearest_points = c_df.nlargest(n_points, 'score')
        
        return nearest_points
    
    def prepare_texts_with_metadata(self, representative_points: pd.DataFrame) -> List[Dict]:
        """Prepare texts with metadata for LLM analysis."""
        
        texts_with_metadata = []
        for _, row in representative_points.iterrows():
            speaker = self.extract_speaker_from_path(row.get('path', ''))
            texts_with_metadata.append({
                'text': row['text'],
                'speaker': speaker,
                'distance': row['distance_from_centroid'],
                'file_path': row.get('path', None),
                'impression': row.get('impression', None),
                'duration': row.get('duration', 0)
            })
        
        return texts_with_metadata


class LLMAnalyzer:
    """Handles LLM-based emotion analysis of clusters using Ollama."""
    
    def __init__(self, config: LLMAnalysisConfig):
        self.config = config
        
        if chat is None:
            raise ImportError("Ollama package not installed. Cannot perform LLM analysis.")
    
    def analyze_cluster_detailed(self, cluster_id: int, texts_with_metadata: List[Dict]) -> str:
        """Perform detailed cluster analysis."""
        
        system_prompt = \"\"\"
        Sei un esperto in analisi emotiva specializzato nell'interpretazione di cluster derivati da embeddings audio. 
        Il modello è stato addestrato specificamente sulle 7 emozioni fondamentali: gioia, tristezza, rabbia, paura, disgusto, sorpresa e neutro.

        Ti verranno forniti frammenti di trascrizioni audio estratti da diverse parti di una conversazione. 
        IMPORTANT: questi NON sono scambi consecutivi, ma campioni isolati selezionati dal modello di clustering perché emotivamente simili.Ciascun frammento è pronunciato da uno speaker identificato .

        I frammenti nel cluster che analizzerai sono stati raggruppati automaticamente perché presentano caratteristiche emotive simili, pur provenendo da momenti diversi della conversazione.

        Per questo cluster, dovrai:

        1. *Identificazione emotiva*: Determina l'emozione predominante che meglio descrive il cluster, indicando anche il grado di intensità (lieve, moderata, intensa) e l'eventuale presenza di emozioni secondarie.
         Verifica se ci sono pattern emotivi comuni tra gli speaker.

        2. *Profilo tonale*: Analizza il registro linguistico e i pattern comunicativi. Identifica 3-5 parole chiave che catturano l'essenza emotiva del cluster. Considera come questa emozione si manifesta indipendentemente dal contesto sequenziale.

        3. *Etichettatura sintetica*: Crea un titolo conciso di 3-4 parole che sintetizzi efficacemente il nucleo emotivo del cluster.

        Rispondi in italiano con uno stile analitico ma accessibile, strutturando la risposta in tre paragrafi numerati corrispondenti ai punti sopra indicati.
        \"\"\"

        # Format texts with speakers
        formatted_texts = ""
        for i, item in enumerate(texts_with_metadata):
            formatted_texts += f"Campione #{i+1} | Speaker: {item['speaker']} | Distanza dal centroide: {item['distance']:.4f}\\n\\\"Trascrizione: {item['text']}\\\"\\n\\n"

        user_prompt = f\"\"\"Analizza il seguente cluster emotivo:

        === CLUSTER {cluster_id} ===

        I seguenti frammenti rappresentano gli elementi più vicini al centroide del cluster e quindi i più rappresentativi della sua configurazione emotiva.
        NOTA BENE: questi NON sono scambi conversazionali consecutivi, ma frammenti isolati provenienti da momenti diversi della conversazione, raggruppati perché emotivamente simili:

        {formatted_texts}

        Concentrandoti esclusivamente su questo cluster, identifica l'emozione predominante che accomuna questi frammenti, traccia il profilo tonale ed elabora un'etichetta sintetica per questa categoria emotiva.
        \"\"\"

        # Get LLM response
        response = chat(model=self.config.llm_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ], stream=False)
        
        return response['message']['content']
    
    def analyze_cluster_short(self, cluster_id: int, texts_with_metadata: List[Dict]) -> str:
        """Perform short cluster analysis with structured output."""
        
        system_prompt = \"\"\"
Sei un esperto in analisi emotiva che classifica cluster di frammenti audio-linguistici.

CONTESTO: I frammenti che analizzerai sono stati raggruppati automaticamente in un cluster perché condividono caratteristiche emotive simili. Identifica il tono emotivo condiviso che ha portato al raggruppamento.

FORMATO OUTPUT OBBLIGATORIO - Rispondi SOLO con questa struttura, SENZA introduzioni o frasi aggiuntive:

1. *Emozione principale*: Una frase (max 10 parole) con l'emozione predominante e intensità.
2. *Parole chiave*: 3 parole che catturano l'essenza emotiva.
3. *Etichetta Emotiva*: Un titolo di 1-2 parole che rappresenti l'emozione prevalente (es. gioia, paura, rabbia).
4. Spiegazione: Una breve motivazione(max 50 parole) che descriva in modo semplice il tono e l'impressione generale trasmessa dai frammenti, spiegando come i frammenti e le loro caratteristiche linguistiche presenti nelle frasi supportano l'identificazione dell'emozione.
5. *Argomento principale*: Una frase (max 10 parole) che riassuma il tema centrale del cluster.
NON aggiungere frasi introduttive come "Ecco l'analisi" o spiegazioni aggiuntive. Inizia DIRETTAMENTE con "1. *Emozione del cluster*:"

Analizza il cluster come un'entità emotiva unica, non i singoli frammenti separatamente.
\"\"\"

        # Format texts
        formatted_texts = ""
        for item in texts_with_metadata:
            formatted_texts += f" {item['speaker']}: {item['text']}\\\"\\n"

        user_prompt = f\"\"\"
CLUSTER {cluster_id} - ANALISI EMOTIVA

FRAMMENTI RAPPRESENTATIVI DEL CLUSTER:
{formatted_texts}

Questi frammenti sono stati raggruppati automaticamente perché condividono un profilo emotivo simile.
Identifica il tono emotivo condiviso che ha portato al raggruppamento.
Focus: Cerca l'emozione più SPECIFICA e DISTINTIVA possibile per questo cluster.
IMPORTANTE: Rispondi DIRETTAMENTE con l'elenco numerato, senza introduzioni.
\"\"\"

        # Get LLM response
        response = chat(model=self.config.llm_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ], stream=False)
        
        return response['message']['content']
    
    def analyze_cluster(self, cluster_id: int, texts_with_metadata: List[Dict]) -> str:
        """Analyze cluster using configured analysis type."""
        
        if self.config.analysis_type == "detailed":
            return self.analyze_cluster_detailed(cluster_id, texts_with_metadata)
        else:
            return self.analyze_cluster_short(cluster_id, texts_with_metadata)


class NLPEvaluator:
    """Evaluates LLM analyses using NLP metrics (BLEU, ROUGE, BERTScore)."""
    
    def __init__(self, config: LLMAnalysisConfig):
        self.config = config
        
        if load is None:
            raise ImportError("evaluate package not installed. Cannot perform NLP evaluation.")
        
        # Initialize metrics
        self.bleu_metric = load('bleu')
        self.rouge_metric = load('rouge')
        self.bert_metric = load('bertscore')
    
    def evaluate_individual_texts(self, cluster_analyses: pd.DataFrame, 
                                cluster_dfs: Dict, optimizer: ClusterWeightOptimizer) -> pd.DataFrame:
        """Evaluate LLM analyses against individual texts."""
        
        output = []
        
        print("📊 EVALUATING INDIVIDUAL TEXTS")
        print("=" * 50)
        
        for _, analysis_row in cluster_analyses.iterrows():
            cluster_id = analysis_row['clusterid']
            llm_analysis = analysis_row['analisi']
            
            if cluster_id not in cluster_dfs:
                continue
            
            df_cluster = cluster_dfs[cluster_id]
            print(f"   Cluster {cluster_id}: {len(df_cluster)} points")
            
            # Get representative elements
            representative_points = optimizer.get_representative_elements(df_cluster)
            
            # Evaluate each text
            for idx, row in representative_points.iterrows():
                text = row['text']
                
                # Calculate metrics
                bleu_res = self.bleu_metric.compute(
                    predictions=[llm_analysis],
                    references=[[text]],
                    smooth=True
                )
                
                rouge_res = self.rouge_metric.compute(
                    predictions=[llm_analysis],
                    references=[text],
                    use_stemmer=True
                )
                
                bert_res = self.bert_metric.compute(
                    predictions=[llm_analysis],
                    references=[text],
                    lang='it'
                )
                
                output.append({
                    'cluster_id': cluster_id,
                    'text_index': idx,
                    'text': text,
                    'llm_analysis': llm_analysis,
                    'bleu_score': bleu_res['bleu'],
                    'rouge1_f1': rouge_res['rouge1'],
                    'rougeL_f1': rouge_res['rougeL'],
                    'bert_f1': bert_res['f1'][0],
                    'bert_precision': bert_res['precision'][0],
                    'bert_recall': bert_res['recall'][0]
                })
        
        return pd.DataFrame(output)
    
    def evaluate_concatenated_texts(self, cluster_analyses: pd.DataFrame,
                                  cluster_dfs: Dict, optimizer: ClusterWeightOptimizer) -> pd.DataFrame:
        """Evaluate LLM analyses against concatenated cluster texts."""
        
        output = []
        
        print("📊 EVALUATING CONCATENATED TEXTS")
        print("=" * 50)
        
        for _, analysis_row in cluster_analyses.iterrows():
            cluster_id = analysis_row['clusterid']
            llm_analysis = analysis_row['analisi']
            
            if cluster_id not in cluster_dfs:
                continue
            
            df_cluster = cluster_dfs[cluster_id]
            print(f"   Cluster {cluster_id}: {len(df_cluster)} points")
            
            # Get representative elements and concatenate texts
            representative_points = optimizer.get_representative_elements(df_cluster)
            concatenated_text = " ".join(representative_points['text'].tolist())
            
            # Calculate metrics
            bleu_res = self.bleu_metric.compute(
                predictions=[llm_analysis],
                references=[[concatenated_text]],
                smooth=True
            )
            
            rouge_res = self.rouge_metric.compute(
                predictions=[llm_analysis],
                references=[concatenated_text],
                use_stemmer=True
            )
            
            bert_res = self.bert_metric.compute(
                predictions=[llm_analysis],
                references=[concatenated_text],
                lang='it'
            )
            
            output.append({
                'cluster_id': cluster_id,
                'concatenated_text_length': len(concatenated_text),
                'llm_analysis': llm_analysis,
                'bleu_score': bleu_res['bleu'],
                'rouge1_f1': rouge_res['rouge1'],
                'rougeL_f1': rouge_res['rougeL'],
                'bert_f1': bert_res['f1'][0],
                'bert_precision': bert_res['precision'][0],
                'bert_recall': bert_res['recall'][0]
            })
        
        return pd.DataFrame(output)


class AudioExporter:
    """Handles audio file export and organization."""
    
    def __init__(self, config: LLMAnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.audio_dir = self.output_dir / "exported_audio"
    
    def extract_speaker_from_path(self, path: str) -> str:
        """Extract speaker identifier from file path."""
        if pd.isna(path):
            return "SPEAKER_UNKNOWN"
        
        filename = path.split('/')[-1]
        match = re.search(r'(SPEAKER_\d{2})', filename)
        return match.group(1) if match else "SPEAKER_UNKNOWN"
    
    def export_cluster_audio_files(self, cluster_dfs: Dict, optimizer: ClusterWeightOptimizer) -> pd.DataFrame:
        """Export audio files for all clusters."""
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        all_files_data = []
        
        print("🎵 EXPORTING AUDIO FILES")
        print("=" * 50)
        
        for cluster_id, df_cluster in tqdm(cluster_dfs.items(), desc="Processing clusters"):
            print(f"\\n📁 Processing CLUSTER {cluster_id}")
            
            # Create cluster directory
            cluster_dir = self.audio_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            # Get representative elements
            representative_points = optimizer.get_representative_elements(df_cluster)
            
            # Copy audio files
            for i, (idx, row) in enumerate(representative_points.iterrows()):
                source_path = row.get('path', '')
                
                if pd.isna(source_path) or not os.path.exists(source_path):
                    print(f"   ⚠️ File not found: {source_path}")
                    continue
                
                # Create standardized filename
                speaker = self.extract_speaker_from_path(source_path)
                file_extension = os.path.splitext(source_path)[1]
                new_filename = f"cluster_{cluster_id}_audio_{i+1:02d}_{speaker}{file_extension}"
                dest_path = cluster_dir / new_filename
                
                try:
                    # Copy file
                    shutil.copy2(source_path, dest_path)
                    
                    # Save metadata
                    all_files_data.append({
                        'cluster_id': cluster_id,
                        'audio_index': i + 1,
                        'speaker': speaker,
                        'text': row['text'],
                        'duration': row['duration'],
                        'distance_from_centroid': row['distance_from_centroid'],
                        'impression': row.get('impression', ''),
                        'original_path': source_path,
                        'local_path': str(dest_path),
                        'filename': new_filename,
                        'file_size_mb': os.path.getsize(dest_path) / (1024*1024)
                    })
                    
                    print(f"   ✅ Copied: {new_filename}")
                    
                except Exception as e:
                    print(f"   ❌ Error copying {source_path}: {e}")
        
        # Create metadata DataFrame
        files_df = pd.DataFrame(all_files_data)
        
        # Save metadata
        metadata_path = self.audio_dir / "audio_files_metadata.csv"
        files_df.to_csv(metadata_path, index=False, encoding='utf-8')
        
        print(f"\\n📊 EXPORT SUMMARY:")
        print(f"   Total files: {len(files_df)}")
        print(f"   Total size: {files_df['file_size_mb'].sum():.2f} MB")
        print(f"   Metadata: {metadata_path}")
        
        return files_df
    
    def create_google_forms_data(self, audio_files_df: pd.DataFrame, 
                               cluster_analyses: pd.DataFrame) -> pd.DataFrame:
        """Create Google Forms compatible data file."""
        
        forms_data = []
        
        print("📋 CREATING GOOGLE FORMS DATA")
        print("=" * 40)
        
        for _, row in audio_files_df.iterrows():
            cluster_id = row['cluster_id']
            
            # Find LLM analysis for this cluster
            analysis_row = cluster_analyses[cluster_analyses['clusterid'] == cluster_id]
            cluster_analysis = analysis_row['analisi'].iloc[0] if not analysis_row.empty else "Analysis not available"
            
            forms_data.append({
                'cluster_id': cluster_id,
                'audio_index': row['audio_index'],
                'speaker': row['speaker'],
                'text': row['text'],
                'duration_seconds': row['duration'],
                'cluster_analysis': cluster_analysis,
                'audio_filename': row['filename'],
                'local_audio_path': row['local_path'],
                'impression_original': row['impression'],
                'distance_from_centroid': row['distance_from_centroid'],
                # Empty fields for form responses
                'human_emotion_rating': '',
                'emotion_intensity': '',
                'audio_quality_rating': '',
                'text_clarity_rating': '',
                'additional_notes': ''
            })
        
        forms_df = pd.DataFrame(forms_data)
        forms_df = forms_df.sort_values(['cluster_id', 'audio_index'])
        
        # Save file
        forms_csv_path = self.audio_dir / "google_forms_data.csv"
        forms_df.to_csv(forms_csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Google Forms data: {forms_csv_path}")
        print(f"   Records: {len(forms_df)}")
        
        return forms_df
    
    def create_audio_archive(self) -> Path:
        """Create ZIP archive with all audio files."""
        
        zip_path = self.audio_dir / "cluster_audio_files.zip"
        
        print("📦 CREATING ZIP ARCHIVE")
        print("=" * 30)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add audio files
            for cluster_dir in self.audio_dir.iterdir():
                if cluster_dir.is_dir() and cluster_dir.name.startswith('cluster_'):
                    for audio_file in cluster_dir.iterdir():
                        if audio_file.suffix in ['.wav', '.mp3', '.flac']:
                            arcname = f"{cluster_dir.name}/{audio_file.name}"
                            zipf.write(audio_file, arcname)
                            print(f"   ✅ Added: {arcname}")
            
            # Add metadata files
            metadata_files = ['audio_files_metadata.csv', 'google_forms_data.csv']
            for filename in metadata_files:
                file_path = self.audio_dir / filename
                if file_path.exists():
                    zipf.write(file_path, filename)
                    print(f"   ✅ Added: {filename}")
        
        zip_size_mb = os.path.getsize(zip_path) / (1024*1024)
        print(f"\\n📦 Archive created: {zip_path}")
        print(f"   Size: {zip_size_mb:.2f} MB")
        
        return zip_path


class LLMClusterAnalyzer:
    """Main class orchestrating the complete LLM cluster analysis workflow."""
    
    def __init__(self, config: LLMAnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.optimizer = ClusterWeightOptimizer(config)
        self.llm_analyzer = LLMAnalyzer(config)
        
        if config.enable_nlp_evaluation:
            self.nlp_evaluator = NLPEvaluator(config)
        
        if config.enable_audio_export:
            self.audio_exporter = AudioExporter(config)
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load data and prepare cluster dictionaries."""
        
        print("📂 LOADING DATA")
        print("=" * 30)
        
        # Load main dataset
        df = pd.read_csv(self.config.data_file)
        print(f"Loaded {len(df)} records from {self.config.data_file}")
        
        # Create cluster dictionaries
        clusters = df['cluster'].unique()
        cluster_dfs = {c: df[df['cluster'] == c] for c in clusters}
        
        print(f"Found {len(clusters)} clusters: {sorted(clusters)}")
        
        return df, cluster_dfs
    
    def run_llm_analysis(self, cluster_dfs: Dict) -> pd.DataFrame:
        """Run LLM analysis on all clusters."""
        
        print("\\n🤖 RUNNING LLM ANALYSIS")
        print("=" * 40)
        
        results = []
        
        for cluster_id, df_cluster in tqdm(cluster_dfs.items(), desc="Analyzing clusters"):
            print(f"\\nAnalyzing cluster {cluster_id} with {len(df_cluster)} points")
            
            try:
                # Get representative elements
                representative_points = self.optimizer.get_representative_elements(df_cluster)
                
                # Prepare metadata
                texts_with_metadata = self.optimizer.prepare_texts_with_metadata(representative_points)
                
                # Run LLM analysis
                analysis = self.llm_analyzer.analyze_cluster(cluster_id, texts_with_metadata)
                
                results.append({
                    'clusterid': cluster_id,
                    'analisi': analysis.replace('\\n', ' ').strip()
                })
                
                print(f"✅ Completed cluster {cluster_id}")
                
            except Exception as e:
                print(f"❌ Error in cluster {cluster_id}: {e}")
                results.append({
                    'clusterid': cluster_id,
                    'analisi': f"ERROR: {e}"
                })
        
        # Create results DataFrame
        df_results = pd.DataFrame(results).sort_values(by='clusterid')
        
        # Save results
        if self.config.save_results:
            results_path = self.output_dir / "cluster_analyses.csv"
            df_results.to_csv(results_path, index=False, encoding='utf-8')
            print(f"\\n💾 LLM analyses saved: {results_path}")
        
        return df_results
    
    def run_nlp_evaluation(self, cluster_analyses: pd.DataFrame, cluster_dfs: Dict) -> Dict[str, pd.DataFrame]:
        """Run NLP evaluation metrics."""
        
        if not self.config.enable_nlp_evaluation:
            return {}
        
        print("\\n📊 RUNNING NLP EVALUATION")
        print("=" * 40)
        
        evaluation_results = {}
        
        if self.config.evaluation_mode in ["individual", "both"]:
            individual_results = self.nlp_evaluator.evaluate_individual_texts(
                cluster_analyses, cluster_dfs, self.optimizer
            )
            evaluation_results['individual'] = individual_results
            
            if self.config.save_results:
                path = self.output_dir / "nlp_evaluation_individual.csv"
                individual_results.to_csv(path, index=False, encoding='utf-8')
                print(f"💾 Individual evaluation saved: {path}")
        
        if self.config.evaluation_mode in ["concatenated", "both"]:
            concatenated_results = self.nlp_evaluator.evaluate_concatenated_texts(
                cluster_analyses, cluster_dfs, self.optimizer
            )
            evaluation_results['concatenated'] = concatenated_results
            
            if self.config.save_results:
                path = self.output_dir / "nlp_evaluation_concatenated.csv"
                concatenated_results.to_csv(path, index=False, encoding='utf-8')
                print(f"💾 Concatenated evaluation saved: {path}")
        
        return evaluation_results
    
    def run_audio_export(self, cluster_analyses: pd.DataFrame, cluster_dfs: Dict) -> Optional[Dict]:
        """Run audio file export."""
        
        if not self.config.enable_audio_export:
            return None
        
        print("\\n🎵 RUNNING AUDIO EXPORT")
        print("=" * 40)
        
        # Export audio files
        audio_files_df = self.audio_exporter.export_cluster_audio_files(cluster_dfs, self.optimizer)
        
        export_results = {'audio_files': audio_files_df}
        
        # Create Google Forms data
        if self.config.create_google_forms_data:
            forms_df = self.audio_exporter.create_google_forms_data(audio_files_df, cluster_analyses)
            export_results['google_forms'] = forms_df
        
        # Create ZIP archive
        if self.config.create_zip_archive:
            zip_path = self.audio_exporter.create_audio_archive()
            export_results['zip_path'] = zip_path
        
        return export_results
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis workflow."""
        
        print("🚀 STARTING COMPLETE LLM CLUSTER ANALYSIS")
        print("=" * 60)
        print(f"Configuration: {self.config.data_file}")
        print(f"Output directory: {self.config.output_dir}")
        print(f"LLM model: {self.config.llm_model}")
        print(f"Analysis type: {self.config.analysis_type}")
        print("=" * 60)
        
        results = {}
        
        try:
            # 1. Load data
            df, cluster_dfs = self.load_and_prepare_data()
            results['data'] = {'dataframe': df, 'cluster_dfs': cluster_dfs}
            
            # 2. Run LLM analysis
            cluster_analyses = self.run_llm_analysis(cluster_dfs)
            results['llm_analyses'] = cluster_analyses
            
            # 3. Run NLP evaluation
            evaluation_results = self.run_nlp_evaluation(cluster_analyses, cluster_dfs)
            results['nlp_evaluation'] = evaluation_results
            
            # 4. Run audio export
            export_results = self.run_audio_export(cluster_analyses, cluster_dfs)
            results['audio_export'] = export_results
            
            print("\\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved in: {self.config.output_dir}")
            
            return results
            
        except Exception as e:
            print(f"\\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main CLI function for LLM cluster analysis."""
    
    parser = argparse.ArgumentParser(
        description="LLM-based Cluster Analysis - Emotion analysis, NLP evaluation, and audio export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\"\"\"
Examples:
  # Basic analysis with YAML config
  python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml

  # Override data file and output directory
  python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --data-file data.csv --output-dir results/

  # Run only LLM analysis without NLP evaluation or audio export
  python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --no-nlp-eval --no-audio-export

  # Use detailed analysis mode
  python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --analysis-type detailed
        \"\"\"
    )
    
    # Required arguments
    parser.add_argument("--config", "-c", default="llm_analysis_config.yaml",
                       help="YAML configuration file path (default: llm_analysis_config.yaml)")
    
    # Override options
    parser.add_argument("--data-file", 
                       help="Override data file path from config")
    parser.add_argument("--output-dir", "-o",
                       help="Override output directory from config")
    parser.add_argument("--llm-model",
                       help="Override LLM model from config")
    parser.add_argument("--analysis-type", choices=["short", "detailed"],
                       help="Override analysis type from config")
    parser.add_argument("--evaluation-mode", choices=["individual", "concatenated", "both"],
                       help="Override NLP evaluation mode from config")
    
    # Feature toggles
    parser.add_argument("--no-nlp-eval", action="store_true",
                       help="Disable NLP evaluation")
    parser.add_argument("--no-audio-export", action="store_true",
                       help="Disable audio file export")
    parser.add_argument("--no-save", action="store_true",
                       help="Disable result saving")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = LLMAnalysisConfig.load_from_yaml(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading configuration: {e}")
        print(f"💡 Make sure {args.config} exists and contains required configuration parameters.")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.data_file:
        config.data_file = args.data_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.analysis_type:
        config.analysis_type = args.analysis_type
    if args.evaluation_mode:
        config.evaluation_mode = args.evaluation_mode
    if args.no_nlp_eval:
        config.enable_nlp_evaluation = False
    if args.no_audio_export:
        config.enable_audio_export = False
    if args.no_save:
        config.save_results = False
    
    # Validate required paths
    if not Path(config.data_file).exists():
        print(f"❌ Error: Data file not found: {config.data_file}")
        sys.exit(1)
    
    # Run analysis
    analyzer = LLMClusterAnalyzer(config)
    results = analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()