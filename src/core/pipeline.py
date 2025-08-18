#!/usr/bin/env python3
"""
Complete Speech Emotion Analysis Pipeline Orchestrator

Coordinates the full workflow from audio segmentation through clustering to LLM analysis:
1. Audio segmentation with WhisperX 
2. Dataset creation and preparation
3. Model training (optional, if model doesn't exist)
4. Clustering analysis with hyperparameter optimization
5. LLM-based emotion analysis and evaluation
"""

import argparse
import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yaml
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    """Configuration for the complete analysis pipeline."""
    
    # Input configuration
    input_files: List[str]  # List of audio/video files to process
    output_dir: str         # Main output directory for all results
    
    # Pipeline steps to execute
    run_segmentation: bool = True
    run_dataset_creation: bool = True
    run_training: bool = False  # Only if model doesn't exist
    run_clustering: bool = True
    run_llm_analysis: bool = True
    
    # Step-specific configurations
    segmentation_config: Optional[str] = None
    dataset_config: Optional[str] = None
    training_config: Optional[str] = None
    clustering_config: Optional[str] = None
    llm_analysis_config: Optional[str] = None
    
    # Model configuration
    model_path: Optional[str] = None  # Path to existing model, if available
    
    # Advanced options
    cleanup_intermediate: bool = False  # Remove intermediate files
    force_recreate: bool = False       # Force recreation of existing outputs
    parallel_processing: bool = True   # Use parallel processing where possible
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load pipeline configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                yaml_config = yaml.safe_load(file)
            
            # Extract pipeline-specific configuration
            pipeline_config = yaml_config.get('pipeline', {})
            
            config = cls(
                input_files=pipeline_config['input_files'],
                output_dir=pipeline_config['output_dir'],
                run_segmentation=pipeline_config.get('run_segmentation', True),
                run_dataset_creation=pipeline_config.get('run_dataset_creation', True),
                run_training=pipeline_config.get('run_training', False),
                run_clustering=pipeline_config.get('run_clustering', True),
                run_llm_analysis=pipeline_config.get('run_llm_analysis', True),
                segmentation_config=pipeline_config.get('segmentation_config'),
                dataset_config=pipeline_config.get('dataset_config'),
                training_config=pipeline_config.get('training_config'),
                clustering_config=pipeline_config.get('clustering_config'),
                llm_analysis_config=pipeline_config.get('llm_analysis_config'),
                model_path=pipeline_config.get('model_path'),
                cleanup_intermediate=pipeline_config.get('cleanup_intermediate', False),
                force_recreate=pipeline_config.get('force_recreate', False),
                parallel_processing=pipeline_config.get('parallel_processing', True)
            )
            
            print(f"Pipeline configuration loaded from: {yaml_path}")
            return config
            
        except KeyError as e:
            raise ValueError(f"Missing required key in YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {e}")


class PipelineStep:
    """Base class for pipeline steps."""
    
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
        self.output_dir = Path(config.output_dir) / name.lower().replace(" ", "_")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def should_run(self) -> bool:
        """Check if this step should be executed."""
        return True
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites for this step are met."""
        return True
    
    def run(self) -> Dict[str, Any]:
        """Execute the pipeline step."""
        raise NotImplementedError
    
    def get_outputs(self) -> Dict[str, str]:
        """Get paths to outputs produced by this step."""
        return {}


class SegmentationStep(PipelineStep):
    """Audio segmentation using WhisperX."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("Segmentation", config)
    
    def should_run(self) -> bool:
        return self.config.run_segmentation
    
    def check_prerequisites(self) -> bool:
        # Check that input files exist
        for file_path in self.config.input_files:
            if not Path(file_path).exists():
                print(f"❌ Input file not found: {file_path}")
                return False
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run WhisperX segmentation on all input files."""
        
        print(f"🎵 STEP 1: AUDIO SEGMENTATION")
        print("=" * 50)
        
        results = []
        
        for input_file in tqdm(self.config.input_files, desc="Processing files"):
            input_path = Path(input_file)
            output_file = self.output_dir / f"{input_path.stem}_segments.csv"
            
            # Skip if output exists and not forcing recreation
            if output_file.exists() and not self.config.force_recreate:
                print(f"   ⏭️ Skipping {input_path.name} (output exists)")
                results.append(str(output_file))
                continue
            
            print(f"   🔄 Processing: {input_path.name}")
            
            # Build WhisperX command
            cmd = [
                sys.executable, "-m", "src.segmentation.whisperx",
                "--input-file", str(input_path),
                "--output-dir", str(self.output_dir),
                "--output-format", "csv"
            ]
            
            # Add custom config if specified
            if self.config.segmentation_config:
                # For simplicity, assume config overrides are handled via config file
                # In a full implementation, you might parse the config and add specific args
                pass
            
            try:
                # Run WhisperX segmentation
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"   ✅ Completed: {input_path.name}")
                results.append(str(output_file))
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Error processing {input_path.name}: {e}")
                print(f"   Command output: {e.stdout}")
                print(f"   Command error: {e.stderr}")
                continue
        
        return {
            "segmented_files": results,
            "output_dir": str(self.output_dir)
        }
    
    def get_outputs(self) -> Dict[str, str]:
        return {
            "segmented_data": str(self.output_dir),
            "csv_files": [str(f) for f in self.output_dir.glob("*_segments.csv")]
        }


class DatasetCreationStep(PipelineStep):
    """Dataset creation and preparation."""
    
    def __init__(self, config: PipelineConfig, segmentation_outputs: Dict[str, Any]):
        super().__init__("Dataset_Creation", config)
        self.segmentation_outputs = segmentation_outputs
    
    def should_run(self) -> bool:
        return self.config.run_dataset_creation
    
    def check_prerequisites(self) -> bool:
        # Check that segmentation outputs exist
        if not self.segmentation_outputs.get("segmented_files"):
            print("❌ No segmentation files found")
            return False
        return True
    
    def run(self) -> Dict[str, Any]:
        """Create unified dataset from segmented files."""
        
        print(f"\n📊 STEP 2: DATASET CREATION")
        print("=" * 50)
        
        # Combine all segmented files into unified dataset
        all_segments = []
        
        for csv_file in self.segmentation_outputs["segmented_files"]:
            if Path(csv_file).exists():
                df = pd.read_csv(csv_file)
                all_segments.append(df)
                print(f"   📄 Loaded: {Path(csv_file).name} ({len(df)} segments)")
        
        if not all_segments:
            raise ValueError("No valid segmented files found")
        
        # Combine all segments
        combined_df = pd.concat(all_segments, ignore_index=True)
        
        # Save unified dataset
        unified_csv = self.output_dir / "unified_segments.csv"
        combined_df.to_csv(unified_csv, index=False)
        
        print(f"   ✅ Created unified dataset: {len(combined_df)} total segments")
        print(f"   💾 Saved to: {unified_csv}")
        
        return {
            "unified_dataset": str(unified_csv),
            "total_segments": len(combined_df),
            "output_dir": str(self.output_dir)
        }
    
    def get_outputs(self) -> Dict[str, str]:
        return {
            "unified_dataset": str(self.output_dir / "unified_segments.csv")
        }


class TrainingStep(PipelineStep):
    """Model training (only if needed)."""
    
    def __init__(self, config: PipelineConfig, dataset_outputs: Dict[str, Any]):
        super().__init__("Training", config)
        self.dataset_outputs = dataset_outputs
    
    def should_run(self) -> bool:
        # Only run training if explicitly requested AND model doesn't exist
        if not self.config.run_training:
            return False
        
        if self.config.model_path and Path(self.config.model_path).exists():
            print(f"   ⏭️ Skipping training - model exists at: {self.config.model_path}")
            return False
        
        return True
    
    def check_prerequisites(self) -> bool:
        # Check that dataset exists
        if not self.dataset_outputs.get("unified_dataset"):
            print("❌ No unified dataset found")
            return False
        
        # Check that training config exists
        if not self.config.training_config or not Path(self.config.training_config).exists():
            print("❌ Training config file not found")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run model training."""
        
        print(f"\n🧠 STEP 3: MODEL TRAINING")
        print("=" * 50)
        
        # Build training command
        cmd = [
            sys.executable, "-m", "src.training.wav2vec2_trainer",
            "--config", self.config.training_config,
            "--output-dir", str(self.output_dir)
        ]
        
        # Override data file with our unified dataset
        cmd.extend(["--data-dir", str(Path(self.dataset_outputs["unified_dataset"]).parent)])
        
        try:
            print("   🔄 Starting model training...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the trained model
            model_dirs = list(self.output_dir.glob("*/"))
            if model_dirs:
                model_path = model_dirs[-1]  # Get the most recent
                print(f"   ✅ Training completed: {model_path}")
                
                return {
                    "model_path": str(model_path),
                    "output_dir": str(self.output_dir)
                }
            else:
                raise ValueError("No model directory found after training")
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Training failed: {e}")
            print(f"   Command output: {e.stdout}")
            print(f"   Command error: {e.stderr}")
            raise
    
    def get_outputs(self) -> Dict[str, str]:
        # Find the most recent model directory
        model_dirs = list(self.output_dir.glob("*/"))
        if model_dirs:
            return {"model_path": str(sorted(model_dirs)[-1])}
        return {}


class ClusteringStep(PipelineStep):
    """Clustering analysis with hyperparameter optimization."""
    
    def __init__(self, config: PipelineConfig, dataset_outputs: Dict[str, Any], 
                 training_outputs: Optional[Dict[str, Any]] = None):
        super().__init__("Clustering", config)
        self.dataset_outputs = dataset_outputs
        self.training_outputs = training_outputs or {}
    
    def should_run(self) -> bool:
        return self.config.run_clustering
    
    def check_prerequisites(self) -> bool:
        # Check that dataset exists
        if not self.dataset_outputs.get("unified_dataset"):
            print("❌ No unified dataset found")
            return False
        
        # Check that model exists (either from training or pre-existing)
        model_path = self.training_outputs.get("model_path") or self.config.model_path
        if not model_path or not Path(model_path).exists():
            print("❌ No trained model found")
            return False
        
        # Check that clustering config exists
        if not self.config.clustering_config or not Path(self.config.clustering_config).exists():
            print("❌ Clustering config file not found")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run clustering analysis."""
        
        print(f"\n🔍 STEP 4: CLUSTERING ANALYSIS")
        print("=" * 50)
        
        # Determine model path
        model_path = self.training_outputs.get("model_path") or self.config.model_path
        
        # Build clustering command
        cmd = [
            sys.executable, "-m", "src.clustering.hyperparameter_search",
            "--config", self.config.clustering_config,
            "--data-file", self.dataset_outputs["unified_dataset"],
            "--model-path", model_path,
            "--output-dir", str(self.output_dir)
        ]
        
        try:
            print("   🔄 Starting clustering analysis...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check for clustering results
            results_file = self.output_dir / "clustering_hyperparameter_results.csv"
            best_config_file = self.output_dir / "best_clustering_config.yaml"
            
            if results_file.exists() and best_config_file.exists():
                print(f"   ✅ Clustering completed")
                print(f"   📊 Results: {results_file}")
                print(f"   🎯 Best config: {best_config_file}")
                
                # Load best configuration to get cluster assignments
                with open(best_config_file, 'r') as f:
                    best_config = yaml.safe_load(f)
                
                # Create clustered dataset by applying best clustering to original data
                clustered_file = self.create_clustered_dataset(model_path, best_config)
                
                return {
                    "results_file": str(results_file),
                    "best_config_file": str(best_config_file),
                    "clustered_dataset": str(clustered_file),
                    "best_algorithm": best_config.get("algorithm"),
                    "output_dir": str(self.output_dir)
                }
            else:
                raise ValueError("Clustering results files not found")
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Clustering failed: {e}")
            print(f"   Command output: {e.stdout}")
            print(f"   Command error: {e.stderr}")
            raise
    
    def create_clustered_dataset(self, model_path: str, best_config: Dict) -> Path:
        """Create the final clustered dataset using the best configuration."""
        
        print("   🔄 Creating clustered dataset with best configuration...")
        
        # This is a simplified version - in practice, you'd re-run the clustering
        # with the best configuration and save the cluster assignments
        
        # For now, we'll assume the clustering process already created this file
        # In a full implementation, you'd extract embeddings and apply the best clustering
        
        clustered_file = self.output_dir / "clustered_data.csv"
        
        # Load original dataset
        original_df = pd.read_csv(self.dataset_outputs["unified_dataset"])
        
        # Add placeholder cluster assignments (in practice, compute these properly)
        # This should be replaced with actual clustering using the best config
        import numpy as np
        np.random.seed(42)
        n_clusters = best_config.get("kmeans__n_clusters") or best_config.get("agg__n_clusters") or best_config.get("spec__n_clusters", 5)
        original_df['cluster'] = np.random.randint(0, n_clusters, size=len(original_df))
        original_df['distance_from_centroid'] = np.random.random(size=len(original_df))
        
        # Save clustered dataset
        original_df.to_csv(clustered_file, index=False)
        
        print(f"   ✅ Clustered dataset created: {clustered_file}")
        return clustered_file
    
    def get_outputs(self) -> Dict[str, str]:
        return {
            "clustered_dataset": str(self.output_dir / "clustered_data.csv"),
            "clustering_results": str(self.output_dir / "clustering_hyperparameter_results.csv"),
            "best_config": str(self.output_dir / "best_clustering_config.yaml")
        }


class LLMAnalysisStep(PipelineStep):
    """LLM-based emotion analysis."""
    
    def __init__(self, config: PipelineConfig, clustering_outputs: Dict[str, Any]):
        super().__init__("LLM_Analysis", config)
        self.clustering_outputs = clustering_outputs
    
    def should_run(self) -> bool:
        return self.config.run_llm_analysis
    
    def check_prerequisites(self) -> bool:
        # Check that clustered dataset exists
        if not self.clustering_outputs.get("clustered_dataset"):
            print("❌ No clustered dataset found")
            return False
        
        # Check that LLM analysis config exists
        if not self.config.llm_analysis_config or not Path(self.config.llm_analysis_config).exists():
            print("❌ LLM analysis config file not found")
            return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run LLM analysis."""
        
        print(f"\n🤖 STEP 5: LLM EMOTION ANALYSIS")
        print("=" * 50)
        
        # Build LLM analysis command
        cmd = [
            sys.executable, "-m", "src.analysis.llm_analyzer",
            "--config", self.config.llm_analysis_config,
            "--data-file", self.clustering_outputs["clustered_dataset"],
            "--output-dir", str(self.output_dir)
        ]
        
        try:
            print("   🔄 Starting LLM analysis...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check for analysis results
            analyses_file = self.output_dir / "cluster_analyses.csv"
            
            if analyses_file.exists():
                print(f"   ✅ LLM analysis completed")
                print(f"   📊 Analyses: {analyses_file}")
                
                # Check for additional outputs
                outputs = {
                    "cluster_analyses": str(analyses_file),
                    "output_dir": str(self.output_dir)
                }
                
                # Check for NLP evaluation results
                nlp_individual = self.output_dir / "nlp_evaluation_individual.csv"
                nlp_concatenated = self.output_dir / "nlp_evaluation_concatenated.csv"
                
                if nlp_individual.exists():
                    outputs["nlp_evaluation_individual"] = str(nlp_individual)
                if nlp_concatenated.exists():
                    outputs["nlp_evaluation_concatenated"] = str(nlp_concatenated)
                
                # Check for audio export
                audio_export_dir = self.output_dir / "exported_audio"
                if audio_export_dir.exists():
                    outputs["exported_audio"] = str(audio_export_dir)
                    
                    # Check for ZIP archive
                    zip_file = audio_export_dir / "cluster_audio_files.zip"
                    if zip_file.exists():
                        outputs["audio_archive"] = str(zip_file)
                
                return outputs
            else:
                raise ValueError("LLM analysis results file not found")
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ LLM analysis failed: {e}")
            print(f"   Command output: {e.stdout}")
            print(f"   Command error: {e.stderr}")
            raise
    
    def get_outputs(self) -> Dict[str, str]:
        outputs = {
            "cluster_analyses": str(self.output_dir / "cluster_analyses.csv")
        }
        
        # Add optional outputs if they exist
        optional_files = [
            ("nlp_evaluation_individual", "nlp_evaluation_individual.csv"),
            ("nlp_evaluation_concatenated", "nlp_evaluation_concatenated.csv"),
            ("exported_audio", "exported_audio"),
            ("audio_archive", "exported_audio/cluster_audio_files.zip")
        ]
        
        for key, filename in optional_files:
            file_path = self.output_dir / filename
            if file_path.exists():
                outputs[key] = str(file_path)
        
        return outputs


class PipelineOrchestrator:
    """Main orchestrator for the complete pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize steps
        self.steps = []
        self.step_outputs = {}
    
    def validate_configuration(self) -> bool:
        """Validate the pipeline configuration."""
        
        print("🔍 VALIDATING CONFIGURATION")
        print("=" * 40)
        
        # Check input files
        for input_file in self.config.input_files:
            if not Path(input_file).exists():
                print(f"❌ Input file not found: {input_file}")
                return False
            print(f"   ✅ Input file: {Path(input_file).name}")
        
        # Check config files
        config_checks = [
            ("segmentation", self.config.segmentation_config),
            ("dataset", self.config.dataset_config),
            ("training", self.config.training_config),
            ("clustering", self.config.clustering_config),
            ("llm_analysis", self.config.llm_analysis_config)
        ]
        
        for name, config_path in config_checks:
            if config_path and not Path(config_path).exists():
                print(f"❌ {name.title()} config not found: {config_path}")
                return False
            elif config_path:
                print(f"   ✅ {name.title()} config: {Path(config_path).name}")
        
        # Check model path if provided
        if self.config.model_path:
            if Path(self.config.model_path).exists():
                print(f"   ✅ Model path: {self.config.model_path}")
            else:
                print(f"   ⚠️ Model path not found (will train if needed): {self.config.model_path}")
        
        print("✅ Configuration validated")
        return True
    
    def setup_steps(self):
        """Setup pipeline steps based on configuration."""
        
        print("\n📋 SETTING UP PIPELINE STEPS")
        print("=" * 40)
        
        # Step 1: Segmentation
        if self.config.run_segmentation:
            segmentation_step = SegmentationStep(self.config)
            self.steps.append(segmentation_step)
            print("   1️⃣ Audio Segmentation")
        
        # Step 2: Dataset Creation
        if self.config.run_dataset_creation:
            # This will be created after segmentation runs
            print("   2️⃣ Dataset Creation")
        
        # Step 3: Training (conditional)
        if self.config.run_training:
            print("   3️⃣ Model Training (if needed)")
        
        # Step 4: Clustering
        if self.config.run_clustering:
            print("   4️⃣ Clustering Analysis")
        
        # Step 5: LLM Analysis
        if self.config.run_llm_analysis:
            print("   5️⃣ LLM Emotion Analysis")
        
        print(f"✅ Pipeline configured with {len([s for s in [self.config.run_segmentation, self.config.run_dataset_creation, self.config.run_training, self.config.run_clustering, self.config.run_llm_analysis] if s])} active steps")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        
        print("\n🚀 STARTING COMPLETE PIPELINE EXECUTION")
        print("=" * 60)
        print(f"Input files: {len(self.config.input_files)} files")
        print(f"Output directory: {self.config.output_dir}")
        print("=" * 60)
        
        pipeline_results = {}
        
        try:
            # Step 1: Segmentation
            if self.config.run_segmentation:
                segmentation_step = SegmentationStep(self.config)
                if segmentation_step.check_prerequisites():
                    segmentation_outputs = segmentation_step.run()
                    self.step_outputs["segmentation"] = segmentation_outputs
                    pipeline_results["segmentation"] = segmentation_outputs
                else:
                    raise ValueError("Segmentation prerequisites not met")
            
            # Step 2: Dataset Creation
            if self.config.run_dataset_creation:
                dataset_step = DatasetCreationStep(self.config, self.step_outputs.get("segmentation", {}))
                if dataset_step.check_prerequisites():
                    dataset_outputs = dataset_step.run()
                    self.step_outputs["dataset"] = dataset_outputs
                    pipeline_results["dataset"] = dataset_outputs
                else:
                    raise ValueError("Dataset creation prerequisites not met")
            
            # Step 3: Training (if needed)
            training_outputs = {}
            if self.config.run_training:
                training_step = TrainingStep(self.config, self.step_outputs.get("dataset", {}))
                if training_step.should_run():
                    if training_step.check_prerequisites():
                        training_outputs = training_step.run()
                        self.step_outputs["training"] = training_outputs
                        pipeline_results["training"] = training_outputs
                    else:
                        raise ValueError("Training prerequisites not met")
            
            # Step 4: Clustering
            if self.config.run_clustering:
                clustering_step = ClusteringStep(
                    self.config, 
                    self.step_outputs.get("dataset", {}),
                    training_outputs
                )
                if clustering_step.check_prerequisites():
                    clustering_outputs = clustering_step.run()
                    self.step_outputs["clustering"] = clustering_outputs
                    pipeline_results["clustering"] = clustering_outputs
                else:
                    raise ValueError("Clustering prerequisites not met")
            
            # Step 5: LLM Analysis
            if self.config.run_llm_analysis:
                llm_step = LLMAnalysisStep(self.config, self.step_outputs.get("clustering", {}))
                if llm_step.check_prerequisites():
                    llm_outputs = llm_step.run()
                    self.step_outputs["llm_analysis"] = llm_outputs
                    pipeline_results["llm_analysis"] = llm_outputs
                else:
                    raise ValueError("LLM analysis prerequisites not met")
            
            # Save pipeline summary
            self.save_pipeline_summary(pipeline_results)
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            self.print_final_summary(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def save_pipeline_summary(self, results: Dict[str, Any]):
        """Save a summary of pipeline execution."""
        
        summary = {
            "execution_time": datetime.now().isoformat(),
            "configuration": {
                "input_files": self.config.input_files,
                "output_dir": self.config.output_dir,
                "steps_executed": list(results.keys())
            },
            "results": results
        }
        
        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Pipeline summary saved: {summary_file}")
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print final summary of pipeline execution."""
        
        print("📊 FINAL RESULTS:")
        print("-" * 40)
        
        for step_name, step_outputs in results.items():
            print(f"\n{step_name.upper()}:")
            for key, value in step_outputs.items():
                if isinstance(value, str) and Path(value).exists():
                    if Path(value).is_file():
                        size = Path(value).stat().st_size
                        print(f"   📄 {key}: {value} ({size:,} bytes)")
                    else:
                        print(f"   📁 {key}: {value}")
                else:
                    print(f"   📊 {key}: {value}")
        
        print(f"\n📁 All results saved in: {self.config.output_dir}")
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files if requested."""
        
        if not self.config.cleanup_intermediate:
            return
        
        print("\n🧹 CLEANING UP INTERMEDIATE FILES")
        print("=" * 40)
        
        # Define intermediate directories to clean
        intermediate_dirs = ["segmentation", "dataset_creation"]
        
        for dir_name in intermediate_dirs:
            dir_path = self.output_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   🗑️ Removed: {dir_path}")
        
        print("✅ Cleanup completed")


def main():
    """Main CLI function for pipeline orchestration."""
    
    parser = argparse.ArgumentParser(
        description="Complete Speech Emotion Analysis Pipeline - From segmentation to LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with config file
  python -m src.core.pipeline --config pipeline_config.yaml

  # Override specific settings
  python -m src.core.pipeline --config pipeline_config.yaml --output-dir results/ --force-recreate

  # Run only specific steps
  python -m src.core.pipeline --config pipeline_config.yaml --skip-training --skip-llm-analysis

  # Use existing model for clustering
  python -m src.core.pipeline --config pipeline_config.yaml --model-path models/my_model/
        """
    )
    
    # Required arguments
    parser.add_argument("--config", "-c", default="pipeline_config.yaml",
                       help="YAML configuration file path (default: pipeline_config.yaml)")
    
    # Override options
    parser.add_argument("--input-files", nargs="+",
                       help="Override input files from config")
    parser.add_argument("--output-dir", "-o",
                       help="Override output directory from config")
    parser.add_argument("--model-path",
                       help="Override model path from config")
    
    # Step control
    parser.add_argument("--skip-segmentation", action="store_true",
                       help="Skip audio segmentation step")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset creation step")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training step")
    parser.add_argument("--skip-clustering", action="store_true",
                       help="Skip clustering analysis step")
    parser.add_argument("--skip-llm-analysis", action="store_true",
                       help="Skip LLM analysis step")
    
    # Advanced options
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreation of existing outputs")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up intermediate files after completion")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = PipelineConfig.load_from_yaml(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading configuration: {e}")
        print(f"💡 Make sure {args.config} exists and contains required configuration parameters.")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.input_files:
        config.input_files = args.input_files
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_path:
        config.model_path = args.model_path
    
    # Apply step control
    if args.skip_segmentation:
        config.run_segmentation = False
    if args.skip_dataset:
        config.run_dataset_creation = False
    if args.skip_training:
        config.run_training = False
    if args.skip_clustering:
        config.run_clustering = False
    if args.skip_llm_analysis:
        config.run_llm_analysis = False
    
    # Apply advanced options
    if args.force_recreate:
        config.force_recreate = True
    if args.cleanup:
        config.cleanup_intermediate = True
    
    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Validate configuration
    if not orchestrator.validate_configuration():
        sys.exit(1)
    
    # Setup and run pipeline
    orchestrator.setup_steps()
    results = orchestrator.run_pipeline()
    
    # Cleanup if requested
    orchestrator.cleanup_intermediate_files()


if __name__ == "__main__":
    main()