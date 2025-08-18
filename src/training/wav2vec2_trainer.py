"""
Usage:
    python -m src/training/wav2vec2_trainer.py --data-dir data/ --output-dir output/
"""

import argparse
import os
import sys
import random
import warnings
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import librosa
import torchaudio
import soundfile as sf
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Audio augmentation
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, PolarityInversion

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "disabled"


@dataclass
class TrainingConfig:
    """Configuration for Wav2Vec2 fine-tuning training loaded from YAML."""
    
    # Data configuration 
    dataset_choice: str
    use_augmentation: bool
    augmentation_type: str
    augmentation_mode: str
    force_recreate_augmented: bool
    
    # Model configuration
    model_name: str
    
    # Audio processing
    sampling_rate: int
    max_length: int
    
    # Training parameters
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_ratio: float
    weight_decay: float
    gradient_accumulation_steps: int
    lr_scheduler_type: str
    
    # Early stopping
    use_early_stopping: bool
    early_stopping_patience: int
    metric_for_best_model: str
    
    # Data splitting
    test_size: float
    val_size: float
    
    # Reproducibility
    seed: int
    
    # Emotion mapping
    emotion_mapping: Dict[str, str]
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                yaml_config = yaml.safe_load(file)
            
            # Extract values from YAML sections
            data_config = yaml_config.get('data', {})
            model_config = yaml_config.get('model', {})
            audio_config = yaml_config.get('audio', {})
            training_config = yaml_config.get('training', {})
            early_stopping_config = yaml_config.get('early_stopping', {})
            data_split_config = yaml_config.get('data_split', {})
            repro_config = yaml_config.get('reproducibility', {})
            emotion_mapping = yaml_config.get('emotion_mapping', {})
            
            # Create configuration with YAML values
            config = cls(
                # Data configuration
                dataset_choice=data_config['dataset_choice'],
                use_augmentation=data_config['use_augmentation'],
                augmentation_type=data_config['augmentation_type'],
                augmentation_mode=data_config['augmentation_mode'],
                force_recreate_augmented=data_config['force_recreate_augmented'],
                
                # Model configuration
                model_name=model_config['model_name'],
                
                # Audio processing
                sampling_rate=audio_config['sampling_rate'],
                max_length=audio_config['max_length'],
                
                # Training parameters
                batch_size=training_config['batch_size'],
                learning_rate=training_config['learning_rate'],
                num_epochs=training_config['num_epochs'],
                warmup_ratio=training_config['warmup_ratio'],
                weight_decay=training_config['weight_decay'],
                gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
                lr_scheduler_type=training_config['lr_scheduler_type'],
                
                # Early stopping
                use_early_stopping=early_stopping_config['use_early_stopping'],
                early_stopping_patience=early_stopping_config['early_stopping_patience'],
                metric_for_best_model=early_stopping_config['metric_for_best_model'],
                
                # Data splitting
                test_size=data_split_config['test_size'],
                val_size=data_split_config['val_size'],
                
                # Reproducibility
                seed=repro_config['seed'],
                
                # Emotion mapping
                emotion_mapping=emotion_mapping
            )
            
            print(f"Configuration loaded from: {yaml_path}")
            return config
            
        except KeyError as e:
            raise ValueError(f"Missing required key in YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {e}")


def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DatasetManager:
    """Manages loading and preprocessing of datasets."""
    
    def __init__(self, config: TrainingConfig, data_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        
        # EMOVO emotion identifiers
        self.emovo_emotions = {
            'neu': 'neutral', 'gio': 'happy', 'tri': 'sad', 'rab': 'angry',
            'pau': 'fearful', 'dis': 'disgust', 'sor': 'surprised'
        }
        
    def load_emovo_dataset(self) -> pd.DataFrame:
        """Load and process EMOVO dataset."""
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
        """Load and process emozionalmente dataset."""
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
        """Create unified dataset based on configuration."""
        datasets = []
        
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
        
        print(f"\nFINAL DATASET:")
        print(f"Total samples: {len(final_df)}")
        print(f"Classes: {sorted(final_df['emotion'].unique())}")
        print(f"Distribution:")
        for emotion, count in final_df['emotion'].value_counts().items():
            print(f"   {emotion}: {count}")
        
        return final_df


class AugmentationManager:
    """Manages augmentation pipelines."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_augmentation_pipeline(self):
        """Create augmentation pipeline based on configuration."""
        if not self.config.use_augmentation:
            return None
        
        aug_type = self.config.augmentation_type
        
        if aug_type == 'Gaussian':
            return Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.7)])
        
        elif aug_type == 'Gain':
            return Compose([Gain(min_gain_db=-6, max_gain_db=6, p=0.7)])
        
        elif aug_type == 'Pitch':
            return Compose([PitchShift(min_semitones=-4, max_semitones=4, p=0.8)])
        
        elif aug_type == 'Time Shift':
            return Compose([Shift(min_shift=-0.1, max_shift=0.1, p=0.6)])
        
        elif aug_type == 'Time Stretch':
            return Compose([TimeStretch(min_rate=0.85, max_rate=1.15, p=0.8)])
        
        elif aug_type == 'Polarity Inversion':
            return Compose([PolarityInversion(p=0.3)])
        
        elif aug_type == 'All':
            return Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.6),
                PitchShift(min_semitones=-5, max_semitones=5, p=0.6),
                Shift(min_shift=-0.15, max_shift=0.15, p=0.4),
                Gain(min_gain_db=-9, max_gain_db=9, p=0.5),
                PolarityInversion(p=0.2),
            ])
        
        else:
            raise ValueError(f"Invalid augmentation type: {aug_type}")


class AugmentedDatasetCreator:
    """Creates and saves augmented datasets for reproducibility."""
    
    def __init__(self, config: TrainingConfig, experiment_name: str, output_dir: str):
        self.config = config
        self.experiment_name = experiment_name
        self.aug_dir = Path(output_dir) / f"augmented_{experiment_name}"
        
    def create_augmented_dataset(self, train_df: pd.DataFrame, augmentation_pipeline) -> pd.DataFrame:
        """Create physically augmented files and return new dataframe."""
        
        if not self.config.use_augmentation or augmentation_pipeline is None:
            print("No augmentation requested, using original dataset")
            return train_df.copy()
        
        print(f"Creating augmented dataset in: {self.aug_dir}")
        self.aug_dir.mkdir(parents=True, exist_ok=True)
        
        augmented_entries = []
        failed_augmentations = []
        
        # Add original files
        for _, row in train_df.iterrows():
            augmented_entries.append({
                'path': row['path'],
                'emotion': row['emotion'], 
                'emotion_id': row['emotion_id'],
                'source': row['source'],
                'augmentation_type': 'original'
            })
        
        # Create augmented files
        print("Generating augmented files...")
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating augmented files"):
            original_path = row['path']
            
            try:
                # Load original audio
                speech, sr = librosa.load(original_path, sr=self.config.sampling_rate)
                
                # Apply augmentation
                augmented_speech = augmentation_pipeline(samples=speech, sample_rate=sr)
                
                # Create augmented filename
                original_filename = os.path.basename(original_path)
                name, ext = os.path.splitext(original_filename)
                aug_filename = f"{name}_aug_{self.config.augmentation_type.lower().replace(' ', '_')}{ext}"
                aug_path = self.aug_dir / aug_filename
                
                # Save augmented file
                torchaudio.save(str(aug_path), torch.tensor([augmented_speech]), self.config.sampling_rate)
                
                # Add augmented entry
                augmented_entries.append({
                    'path': str(aug_path),
                    'emotion': row['emotion'],
                    'emotion_id': row['emotion_id'], 
                    'source': row['source'],
                    'augmentation_type': self.config.augmentation_type,
                    'original_path': original_path
                })
                
            except Exception as e:
                print(f"Augmentation error for {original_path}: {e}")
                failed_augmentations.append({
                    'original_path': original_path,
                    'error': str(e)
                })
        
        # Create final DataFrame
        augmented_df = pd.DataFrame(augmented_entries)
        
        # Save augmented dataset CSV
        csv_path = self.aug_dir / f"train_augmented_{self.experiment_name}.csv"
        augmented_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save error log if any
        if failed_augmentations:
            error_log_path = self.aug_dir / f"augmentation_errors_{self.experiment_name}.json"
            with open(error_log_path, 'w') as f:
                json.dump(failed_augmentations, f, indent=2)
            print(f"WARNING: {len(failed_augmentations)} files not processed. Log: {error_log_path}")
        
        # Save metadata
        metadata = {
            'experiment_name': self.experiment_name,
            'augmentation_type': self.config.augmentation_type,
            'original_train_size': len(train_df),
            'augmented_train_size': len(augmented_df),
            'augmented_files_created': len([e for e in augmented_entries if e['augmentation_type'] != 'original']),
            'failed_augmentations': len(failed_augmentations),
            'timestamp': pd.Timestamp.now().isoformat(),
            'seed': self.config.seed
        }
        
        metadata_path = self.aug_dir / f"augmentation_metadata_{self.experiment_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Augmented dataset created:")
        print(f"  - Original files: {len(train_df)}")
        print(f"  - Augmented files: {len(augmented_df) - len(train_df)}")
        print(f"  - Total: {len(augmented_df)}")
        print(f"  - CSV saved: {csv_path}")
        print(f"  - Metadata: {metadata_path}")
        
        return augmented_df
    
    def load_existing_augmented_dataset(self) -> pd.DataFrame:
        """Load existing augmented dataset."""
        csv_path = self.aug_dir / f"train_augmented_{self.experiment_name}.csv"
        
        if csv_path.exists():
            print(f"Loading existing augmented dataset: {csv_path}")
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Verify files still exist
            existing_files = df['path'].apply(os.path.exists)
            if not existing_files.all():
                missing_count = (~existing_files).sum()
                print(f"WARNING: {missing_count} augmented files missing. Recreation needed.")
                raise FileNotFoundError(f"{missing_count} augmented files missing")
            
            return df
        else:
            raise FileNotFoundError(f"Augmented dataset not found: {csv_path}")


class SpeechEmotionDataset(Dataset):
    """Dataset class with optional augmentation."""
    
    def __init__(self, dataframe: pd.DataFrame, processor, config: TrainingConfig, 
                 is_train: bool = False, augmentation_pipeline=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.config = config
        self.is_train = is_train
        self.augmentation_pipeline = augmentation_pipeline
        
        print(f"Dataset created: {len(self.dataframe)} samples")
        if is_train and augmentation_pipeline:
            print(f"   Augmentation: Active")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        audio_file = self.dataframe.iloc[idx]['path']
        label = self.dataframe.iloc[idx]['emotion_id']
        
        try:
            # Load audio
            speech, sr = librosa.load(audio_file, sr=self.config.sampling_rate)
            
            # Apply augmentation if needed (on-the-fly mode)
            if self.is_train and self.augmentation_pipeline is not None:
                try:
                    speech = self.augmentation_pipeline(samples=speech, sample_rate=sr)
                except Exception as e:
                    print(f"Augmentation error on {audio_file}: {e}")
            
            # Process with processor
            inputs = self.processor(
                speech,
                sampling_rate=self.config.sampling_rate,
                return_tensors="pt",
                padding=True,
                max_length=self.config.max_length,
                truncation=True
            )
            
            return {
                "input_values": inputs.input_values.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0) if inputs.attention_mask is not None else None,
                "labels": torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return empty tensor as fallback
            return {
                "input_values": torch.zeros(self.config.max_length),
                "attention_mask": torch.zeros(self.config.max_length),
                "labels": torch.tensor(0, dtype=torch.long)
            }


def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    metrics = {'accuracy': acc}
    
    for avg in ['macro', 'weighted']:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=avg, zero_division=0
        )
        metrics[f'precision_{avg}'] = precision
        metrics[f'recall_{avg}'] = recall
        metrics[f'f1_{avg}'] = f1
    
    return metrics


def train_wav2vec2_model(
    config: TrainingConfig,
    data_dir: str,
    output_dir: str,
    experiment_name: str = "wav2vec2_training"
):
    """Main training function."""
    
    # Set seed
    set_seed(config.seed)
    print(f"Seed set to: {config.seed}")
    
    # Load dataset
    dataset_manager = DatasetManager(config, data_dir)
    df = dataset_manager.create_unified_dataset()
    
    # Save final dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    final_dataset_path = output_path / "final_dataset.csv"
    df.to_csv(final_dataset_path, index=False, encoding="utf-8")
    print(f"Dataset saved: {final_dataset_path}")
    
    # Create label mapping
    unique_emotions = sorted(df['emotion'].unique())
    label_mapping = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    inverse_label_mapping = {idx: emotion for emotion, idx in label_mapping.items()}
    
    print(f"Label mapping: {label_mapping}")
    
    # Apply numeric mapping
    df['emotion_id'] = df['emotion'].map(label_mapping)
    
    # Split dataset (BEFORE augmentation to avoid data leakage!)
    train_val_df, test_df = train_test_split(
        df, test_size=config.test_size, random_state=config.seed, 
        stratify=df['emotion_id']
    )
    
    # Split train/validation
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.val_size, random_state=config.seed,
        stratify=train_val_df['emotion_id']
    )
    
    print(f"DATASET SPLITTING:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Save splits
    train_df.to_csv(output_path / "train_dataset.csv", index=False)
    val_df.to_csv(output_path / "val_dataset.csv", index=False)
    test_df.to_csv(output_path / "test_dataset.csv", index=False)
    
    # Setup augmentation
    aug_manager = AugmentationManager(config)
    augmentation_pipeline = aug_manager.get_augmentation_pipeline()
    
    print(f"Augmentation: {'Configured' if augmentation_pipeline else 'Disabled'}")
    if augmentation_pipeline:
        print(f"   Type: {config.augmentation_type}")
        print(f"   Mode: {config.augmentation_mode}")
    
    # Load processor and model
    print(f"Loading model: {config.model_name}")
    processor = Wav2Vec2Processor.from_pretrained(config.model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=len(label_mapping)
    )
    
    # Freeze feature extractor
    model.freeze_feature_extractor()
    print(f"Model loaded with {len(label_mapping)} classes")
    
    # Handle augmentation based on mode
    if config.use_augmentation and config.augmentation_mode == 'pre_generated':
        print("PRE-GENERATED mode: Creating/loading physical augmented files")
        
        aug_creator = AugmentedDatasetCreator(config, experiment_name, str(output_path))
        
        if config.force_recreate_augmented:
            print("Forcing recreation of augmented dataset...")
            final_train_df = aug_creator.create_augmented_dataset(train_df, augmentation_pipeline)
        else:
            try:
                final_train_df = aug_creator.load_existing_augmented_dataset()
                print("Existing augmented dataset loaded!")
            except FileNotFoundError:
                print("Augmented dataset not found, creating...")
                final_train_df = aug_creator.create_augmented_dataset(train_df, augmentation_pipeline)
        
        # Create dataset WITHOUT on-the-fly augmentation
        train_dataset = SpeechEmotionDataset(
            dataframe=final_train_df,
            processor=processor,
            config=config,
            is_train=False,  # No on-the-fly augmentation
            augmentation_pipeline=None
        )
        
    elif config.use_augmentation and config.augmentation_mode == 'on_the_fly':
        print("ON-THE-FLY mode: Augmentation during training")
        
        final_train_df = train_df.copy()
        final_train_df['augmentation_type'] = 'on_the_fly'
        
        # Create dataset WITH on-the-fly augmentation
        train_dataset = SpeechEmotionDataset(
            dataframe=final_train_df,
            processor=processor,
            config=config,
            is_train=True,  # Enable on-the-fly augmentation
            augmentation_pipeline=augmentation_pipeline
        )
        
    else:
        print("No augmentation applied")
        
        final_train_df = train_df.copy()
        final_train_df['augmentation_type'] = 'original'
        
        train_dataset = SpeechEmotionDataset(
            dataframe=final_train_df,
            processor=processor,
            config=config,
            is_train=False,
            augmentation_pipeline=None
        )
    
    # Create validation and test datasets (always without augmentation)
    val_dataset = SpeechEmotionDataset(
        dataframe=val_df,
        processor=processor,
        config=config,
        is_train=False,
        augmentation_pipeline=None
    )
    
    test_dataset = SpeechEmotionDataset(
        dataframe=test_df,
        processor=processor,
        config=config,
        is_train=False,
        augmentation_pipeline=None
    )
    
    print(f"\nFinal datasets created:")
    print(f"Train: {len(final_train_df)} samples")
    if config.use_augmentation and config.augmentation_mode == 'pre_generated':
        try:
            original_count = len(final_train_df[final_train_df['augmentation_type'] == 'original'])
            augmented_count = len(final_train_df[final_train_df['augmentation_type'] != 'original'])
            print(f"   Original: {original_count}")
            print(f"   Augmented: {augmented_count}")
        except:
            pass
    elif config.use_augmentation and config.augmentation_mode == 'on_the_fly':
        print(f"   Augmentation: On-the-fly during training")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=processor, padding="longest")
    
    # Training arguments
    training_output_dir = output_path / experiment_name
    training_args = TrainingArguments(
        output_dir=str(training_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        save_total_limit=2,
        report_to="none",
        logging_dir=str(training_output_dir / "logs"),
        logging_steps=50,
        seed=config.seed,
    )
    
    # Setup callbacks
    callbacks = []
    if config.use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
    
    print(f"\nTRAINING CONFIGURATION:")
    print(f"   Experiment: {experiment_name}")
    print(f"   Output directory: {training_output_dir}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Number of epochs: {config.num_epochs}")
    print(f"   Warmup ratio: {config.warmup_ratio}")
    print(f"   Weight decay: {config.weight_decay}")
    print(f"   Metric for best model: {config.metric_for_best_model}")
    print(f"   Early stopping: {'Active' if config.use_early_stopping else 'Disabled'}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    print("\n" + "="*50)
    print("STARTING TRAINING...")
    print("="*50)
    
    # Start training
    training_result = trainer.train()
    
    print("="*50)
    print("TRAINING COMPLETED!")
    print(f"Final training loss: {training_result.training_loss:.4f}")
    
    # Validation evaluation
    print("\nVALIDATION EVALUATION")
    val_results = trainer.evaluate()
    
    print("\nValidation results:")
    for key, value in val_results.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            if isinstance(value, float):
                print(f"   {metric_name}: {value:.4f}")
    
    # Test evaluation
    print("\nTEST EVALUATION")
    test_results = trainer.predict(test_dataset)
    
    print("\nTest results:")
    for key, value in test_results.metrics.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '')
            if isinstance(value, float):
                print(f"   {metric_name}: {value:.4f}")
    
    # Save final model
    final_model_dir = output_path / f"final_{experiment_name}"
    final_model_dir.mkdir(exist_ok=True)
    
    model.save_pretrained(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    
    # Save configuration and mappings
    config_dict = {
        'experiment_name': experiment_name,
        'dataset_choice': config.dataset_choice,
        'use_augmentation': config.use_augmentation,
        'augmentation_type': config.augmentation_type if config.use_augmentation else None,
        'augmentation_mode': config.augmentation_mode if config.use_augmentation else None,
        'model_name': config.model_name,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'seed': config.seed,
        'label_mapping': label_mapping,
        'inverse_label_mapping': inverse_label_mapping,
        'dataset_sizes': {
            'train': len(final_train_df),
            'val': len(val_df),
            'test': len(test_df)
        },
        'final_metrics': {k: float(v) for k, v in test_results.metrics.items() if isinstance(v, (int, float))}
    }
    
    with open(final_model_dir / 'experiment_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\nModel saved to: {final_model_dir}")
    print(f"Configuration saved to: experiment_config.json")
    
    # Print experiment summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"Experiment name: {experiment_name}")
    print(f"Dataset used: {config.dataset_choice.upper()}")
    if config.use_augmentation:
        print(f"Data augmentation: {config.augmentation_type} ({config.augmentation_mode})")
    else:
        print(f"Data augmentation: Disabled")
    
    print(f"Model: {config.model_name.split('/')[-1]}")
    print(f"Training parameters:")
    print(f"   • Epochs: {config.num_epochs}")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Early stopping: {'Active' if config.use_early_stopping else 'Disabled'}")
    print(f"   • Seed: {config.seed}")
    
    print(f"\nFINAL RESULTS:")
    best_accuracy = max([v for k, v in test_results.metrics.items() if 'accuracy' in k])
    best_f1 = max([v for k, v in test_results.metrics.items() if 'f1' in k and 'macro' in k])
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   F1-score (macro): {best_f1:.4f}")
    
    print(f"\nFiles saved:")
    print(f"   Model: {final_model_dir}")
    
    print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return {
        'model': model,
        'processor': processor,
        'config': config_dict,
        'test_results': test_results.metrics,
        'model_path': str(final_model_dir)
    }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Wav2Vec2 Fine-tuning Trainer - Italian Speech Emotion Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default YAML config (both_full_aug experiment)
  python -m src.training.wav2vec2_trainer --data-dir data/ --output-dir output/

  # Use custom YAML configuration file
  python -m src.training.wav2vec2_trainer --data-dir data/ --output-dir output/ --config my_config.yaml

  # Use hardcoded defaults (ignore YAML)
  python -m src.training.wav2vec2_trainer --data-dir data/ --output-dir output/ --no-config

  # Override YAML config with CLI arguments
  python -m src.training.wav2vec2_trainer --data-dir data/ --output-dir output/ --epochs 15 --batch-size 16

  # Train with EMOVO only, override YAML augmentation setting
  python -m src.training.wav2vec2_trainer --data-dir data/ --output-dir output/ --dataset emovo --no-augmentation
        """
    )
    
    # Required arguments
    parser.add_argument("--data-dir", required=True,
                       help="Directory containing datasets (should have EMOVO/ and/or emozionalmente_dataset/ subdirs)")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for results and models")
    
    # Configuration file
    parser.add_argument("--config", "-c", default="train_config.yaml",
                       help="YAML configuration file path (default: train_config.yaml)")
    
    # Dataset options  
    parser.add_argument("--dataset", choices=["emovo", "emozionalmente", "both"], default="both",
                       help="Dataset choice (default: both)")
    
    # Augmentation options (based on both_full_aug defaults)
    parser.add_argument("--no-augmentation", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--aug-type", default="All",
                       choices=["All", "Gaussian", "Pitch", "Gain", "Time Shift", "Time Stretch", "Polarity Inversion"],
                       help="Augmentation type (default: All)")
    parser.add_argument("--aug-mode", choices=["pre_generated", "on_the_fly"], default="pre_generated",
                       help="Augmentation mode (default: pre_generated)")
    parser.add_argument("--force-recreate-aug", action="store_true",
                       help="Force recreation of augmented files")
    
    # Model options
    parser.add_argument("--model", default="jonatasgrosman/wav2vec2-large-xlsr-53-italian",
                       help="Pretrained model name (default: jonatasgrosman/wav2vec2-large-xlsr-53-italian)")
    
    # Training options (both_full_aug defaults)
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--lr", "--learning-rate", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                       help="Warmup ratio (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                       help="Weight decay (default: 0.001)")
    
    # Early stopping
    parser.add_argument("--early-stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience (default: 3)")
    parser.add_argument("--metric", default="f1_macro",
                       choices=["accuracy", "f1_macro", "f1_weighted"],
                       help="Metric for best model selection (default: f1_macro)")
    
    # Data splitting
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1,
                       help="Validation set size from remaining data (default: 0.1)")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--experiment-name", default="wav2vec2_training",
                       help="Experiment name (default: wav2vec2_training)")
    
    args = parser.parse_args()
    
    # Validate inputs
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check dataset directories exist
    if args.dataset in ["emovo", "both"] and not (data_dir / "EMOVO").exists():
        print(f"Error: EMOVO directory not found: {data_dir / 'EMOVO'}")
        sys.exit(1)
        
    if args.dataset in ["emozionalmente", "both"] and not (data_dir / "emozionalmente_dataset").exists():
        print(f"Error: emozionalmente_dataset directory not found: {data_dir / 'emozionalmente_dataset'}")
        sys.exit(1)
    
    # Load configuration from YAML file
    try:
        config = TrainingConfig.load_from_yaml(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading configuration: {e}")
        print(f"💡 Make sure {args.config} exists and contains all required configuration parameters.")
        sys.exit(1)
    
    # Override YAML configuration with CLI arguments (if provided)
    # CLI arguments are applied directly if they differ from argparse defaults
    
    # Note: We use 'dataset' instead of 'dataset_choice' for CLI consistency
    if hasattr(args, 'dataset') and args.dataset:
        config.dataset_choice = args.dataset
        
    # Augmentation overrides
    if hasattr(args, 'no_augmentation') and args.no_augmentation:
        config.use_augmentation = False
    if hasattr(args, 'aug_type') and args.aug_type:
        config.augmentation_type = args.aug_type
    if hasattr(args, 'aug_mode') and args.aug_mode:
        config.augmentation_mode = args.aug_mode
    if hasattr(args, 'force_recreate_aug') and args.force_recreate_aug:
        config.force_recreate_augmented = True
        
    # Model and training overrides  
    if hasattr(args, 'model') and args.model:
        config.model_name = args.model
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'lr') and args.lr:
        config.learning_rate = args.lr
    if hasattr(args, 'epochs') and args.epochs:
        config.num_epochs = args.epochs
    if hasattr(args, 'warmup_ratio') and args.warmup_ratio:
        config.warmup_ratio = args.warmup_ratio
    if hasattr(args, 'weight_decay') and args.weight_decay:
        config.weight_decay = args.weight_decay
        
    # Early stopping overrides
    if hasattr(args, 'early_stopping') and args.early_stopping:
        config.use_early_stopping = True
    if hasattr(args, 'patience') and args.patience:
        config.early_stopping_patience = args.patience
    if hasattr(args, 'metric') and args.metric:
        config.metric_for_best_model = args.metric
        
    # Data splitting overrides
    if hasattr(args, 'test_size') and args.test_size:
        config.test_size = args.test_size
    if hasattr(args, 'val_size') and args.val_size:
        config.val_size = args.val_size
        
    # Other overrides
    if hasattr(args, 'seed') and args.seed:
        config.seed = args.seed
    
    print("=" * 60)
    print("WAV2VEC2 FINE-TUNING TRAINER")
    print("=" * 60)
    config_source = "YAML config" if not args.no_config else "hardcoded defaults"
    print(f"Configuration: {config_source} {'(' + args.config + ')' if not args.no_config else ''}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset: {config.dataset_choice}")
    print(f"Augmentation: {'Enabled (' + config.augmentation_type + ', ' + config.augmentation_mode + ')' if config.use_augmentation else 'Disabled'}")
    print(f"Model: {config.model_name.split('/')[-1]}")
    print(f"Epochs: {config.num_epochs}, Batch size: {config.batch_size}, LR: {config.learning_rate}")
    print("=" * 60)
    
    # Run training
    try:
        results = train_wav2vec2_model(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {results['model_path']}")
        print(f"🎯 Test Accuracy: {results['test_results']['test_accuracy']:.4f}")
        print(f"📊 Test F1-Macro: {results['test_results']['test_f1_macro']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()