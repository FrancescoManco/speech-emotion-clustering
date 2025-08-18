#!/usr/bin/env python3
"""
Dataset Loader Module

Handles loading and preprocessing of Italian emotion recognition datasets.
Supports EMOVO and emozionalmente datasets with unified interface.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

import pandas as pd
import yaml


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    dataset_choice: str  # "emovo", "emozionalmente", "both"
    emotion_mapping: Dict[str, str]
    output_file: Optional[str] = None  # Optional output CSV file
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> 'DatasetConfig':
        """Load dataset configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                yaml_config = yaml.safe_load(file)
            
            # Extract values from YAML sections
            data_config = yaml_config.get('data', {})
            emotion_mapping = yaml_config.get('emotion_mapping', {})
            
            config = cls(
                dataset_choice=data_config['dataset_choice'],
                emotion_mapping=emotion_mapping,
                output_file=None  # Will be set by CLI if needed
            )
            
            print(f"Dataset configuration loaded from: {yaml_path}")
            return config
            
        except KeyError as e:
            raise ValueError(f"Missing required key in YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {e}")


class DatasetLoader:
    """Manages loading and preprocessing of emotion recognition datasets."""
    
    def __init__(self, config: DatasetConfig, data_dir: str):
        """Initialize dataset loader.
        
        Args:
            config: Dataset configuration
            data_dir: Root directory containing dataset subdirectories
        """
        self.config = config
        self.data_dir = Path(data_dir)
        
        # EMOVO emotion identifiers (filename prefix -> emotion)
        self.emovo_emotions = {
            'neu': 'neutral', 'gio': 'happy', 'tri': 'sad', 'rab': 'angry',
            'pau': 'fearful', 'dis': 'disgust', 'sor': 'surprised'
        }
        
    def load_emovo_dataset(self) -> pd.DataFrame:
        """Load and process EMOVO dataset.
        
        Returns:
            DataFrame with columns: ['path', 'emotion', 'source']
        """
        emovo_path = self.data_dir / "EMOVO"
        
        if not emovo_path.exists():
            print(f"EMOVO directory not found: {emovo_path}")
            return pd.DataFrame()
            
        print("Loading EMOVO dataset...")
        
        data = []
        samples_counter = {emotion: 0 for emotion in self.emovo_emotions.values()}
        
        for root, dirs, files in os.walk(emovo_path):
            for file in files:
                if file.endswith(".wav"):
                    # Parse filename: emotion-speaker-sentence.wav
                    f = file.split('.')[0].split('-')
                    if len(f) >= 3 and f[0] in self.emovo_emotions:
                        emotion = self.emovo_emotions[f[0]]
                        samples_counter[emotion] += 1
                        
                        data.append({
                            "path": os.path.join(root, file),
                            "emotion": emotion,
                            "source": "emovo"
                        })
        
        df = pd.DataFrame(data)
        print(f"EMOVO loaded: {len(df)} samples")
        print(f"Distribution: {samples_counter}")
        return df
    
    def load_emozionalmente_dataset(self) -> pd.DataFrame:
        """Load and process emozionalmente dataset.
        
        Returns:
            DataFrame with columns: ['path', 'emotion', 'source']
        """
        emozionalmente_path = self.data_dir / "emozionalmente_dataset"
        metadata_path = emozionalmente_path / "metadata" / "samples.csv"
        audio_path = emozionalmente_path / "audio"
        
        if not metadata_path.exists():
            print(f"Emozionalmente metadata not found: {metadata_path}")
            return pd.DataFrame()
            
        print("Loading emozionalmente dataset...")
        
        df = pd.read_csv(metadata_path)
        df = df[["file_name", "emotion_expressed"]].copy()
        df.columns = ["file_name", "emotion"]
        df["path"] = df["file_name"].apply(lambda x: str(audio_path / x))
        df["source"] = "emozionalmente"
        
        # Verify file existence
        df = df[df["path"].apply(os.path.exists)].copy()
        
        print(f"Emozionalmente loaded: {len(df)} samples")
        print(f"Distribution: {df['emotion'].value_counts().to_dict()}")
        return df
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """Create unified dataset based on configuration.
        
        Returns:
            Combined and processed DataFrame with emotion mapping applied
        """
        datasets = []
        
        # Load requested datasets
        if self.config.dataset_choice in ['emovo', 'both']:
            emovo_df = self.load_emovo_dataset()
            if not emovo_df.empty:
                datasets.append(emovo_df)
        
        if self.config.dataset_choice in ['emozionalmente', 'both']:
            emo_df = self.load_emozionalmente_dataset()
            if not emo_df.empty:
                datasets.append(emo_df)
        
        if not datasets:
            raise ValueError(f"No valid dataset found for choice: {self.config.dataset_choice}")
        
        # Combine datasets
        final_df = pd.concat(datasets, ignore_index=True)
        
        # Apply emotion mapping
        final_df["emotion"] = final_df["emotion"].replace(self.config.emotion_mapping)
        
        # Display final statistics
        print(f"\n" + "="*50)
        print(f"UNIFIED DATASET CREATED")
        print(f"="*50)
        print(f"Total samples: {len(final_df)}")
        print(f"Classes: {sorted(final_df['emotion'].unique())}")
        print(f"Distribution:")
        for emotion, count in final_df['emotion'].value_counts().items():
            print(f"   {emotion}: {count}")
        print(f"="*50)
        
        return final_df

    def save_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """Save dataset to CSV file.
        
        Args:
            df: Dataset DataFrame to save
            output_path: Output CSV file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset saved to: {output_path}")


def main():
    """Main CLI function for dataset creation."""
    parser = argparse.ArgumentParser(
        description="Dataset Loader CLI - Italian Emotion Recognition Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create unified dataset using YAML config
  python -m src.dataset.loader --data-dir data/ --output unified_dataset.csv

  # Use custom YAML configuration  
  python -m src.dataset.loader --data-dir data/ --config my_config.yaml --output dataset.csv

  # Load specific dataset only
  python -m src.dataset.loader --data-dir data/ --dataset emovo --output emovo_only.csv

  # Just show dataset statistics (no output file)
  python -m src.dataset.loader --data-dir data/ --dataset both
        """
    )
    
    # Required arguments
    parser.add_argument("--data-dir", required=True,
                       help="Directory containing datasets (should have EMOVO/ and/or emozionalmente_dataset/ subdirs)")
    
    # Configuration file
    parser.add_argument("--config", "-c", default="train_config.yaml",
                       help="YAML configuration file path (default: train_config.yaml)")
    
    # Output options
    parser.add_argument("--output", "-o",
                       help="Output CSV file path (optional - if not provided, just shows statistics)")
    
    # Dataset override options
    parser.add_argument("--dataset", choices=["emovo", "emozionalmente", "both"],
                       help="Override dataset choice from config")
    
    args = parser.parse_args()
    
    # Validate inputs
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Load configuration from YAML file
    try:
        config = DatasetConfig.load_from_yaml(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading configuration: {e}")
        print(f"💡 Make sure {args.config} exists and contains required configuration parameters.")
        sys.exit(1)
    
    # Override dataset choice if provided via CLI
    if args.dataset:
        config.dataset_choice = args.dataset
    
    # Set output file if provided
    if args.output:
        config.output_file = args.output
    
    # Check dataset directories exist
    if config.dataset_choice in ["emovo", "both"] and not (data_dir / "EMOVO").exists():
        print(f"❌ Error: EMOVO directory not found: {data_dir / 'EMOVO'}")
        sys.exit(1)
        
    if config.dataset_choice in ["emozionalmente", "both"] and not (data_dir / "emozionalmente_dataset").exists():
        print(f"❌ Error: emozionalmente_dataset directory not found: {data_dir / 'emozionalmente_dataset'}")
        sys.exit(1)
    
    print("=" * 60)
    print("DATASET LOADER")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset choice: {config.dataset_choice}")
    print(f"Output file: {config.output_file or 'None (statistics only)'}")
    print("=" * 60)
    
    # Create dataset
    try:
        loader = DatasetLoader(config, args.data_dir)
        dataset = loader.create_unified_dataset()
        
        # Save to file if requested
        if config.output_file:
            loader.save_dataset(dataset, config.output_file)
        
        print(f"\n✅ Dataset processing completed successfully!")
        if config.output_file:
            print(f"📁 Dataset saved to: {config.output_file}")
        else:
            print(f"📊 Use --output to save dataset to CSV file")
        
    except Exception as e:
        print(f"\n❌ Dataset processing failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()