# 🚀 Complete Speech Emotion Analysis Refactoring Documentation

## 📋 Overview

This document provides a comprehensive guide to the complete refactoring of the Speech Emotion Analysis project from Jupyter notebooks to a modular, production-ready Python CLI system. The refactoring transforms notebook-based workflows into reusable, configurable modules with proper dependency management.

## 🏗️ Project Architecture

The refactored system follows a modular architecture with the following structure:

```
speech-emotion-clustering/
├── src/
│   ├── core/                    # Pipeline orchestration
│   ├── segmentation/            # Audio segmentation with WhisperX
│   ├── dataset/                 # Dataset creation and management
│   ├── training/                # Model training with Wav2Vec2
│   ├── clustering/              # Clustering analysis
│   └── analysis/                # LLM analysis and evaluation
├── *.yaml                       # Configuration files
├── pyproject.toml              # Project dependencies
└── requirements.txt            # Alternative dependency format
```

---

## 📁 Module-by-Module Documentation

### 🎵 1. Segmentation Module (`src/segmentation/`)

**Purpose:** Audio transcription, alignment, diarization, and segmentation using WhisperX.

#### Files Created:

##### `src/segmentation/__init__.py`
- **What it does:** Module initialization and documentation
- **Usage:** Automatic import when using the segmentation module

##### `src/segmentation/whisperx.py`
- **Source:** Converted from `notebooks/Segmentation/whisperX.ipynb`
- **What it does:** 
  - Audio transcription using WhisperX models
  - Speaker diarization with pyannote.audio
  - Word-level alignment for precise timing
  - Audio segmentation into individual clips
  - CSV export with metadata (speaker, timing, text)

- **Key Classes:**
  - `WhisperXConfig`: Configuration management
  - `AudioProcessor`: Main processing logic
  - `SegmentationPipeline`: Complete workflow orchestration

- **Usage Examples:**
```bash
# Basic transcription
python -m src.segmentation.whisperx --input-file audio.wav --output-dir results/

# With custom model and language
python -m src.segmentation.whisperx --input-file video.mp4 --model large-v2 --language it --output-dir results/

# Enable diarization
python -m src.segmentation.whisperx --input-file meeting.wav --enable-diarization --output-dir results/

# Segment audio files
python -m src.segmentation.whisperx --input-file podcast.mp3 --segment-audio --output-dir results/
```

- **Outputs:**
  - `transcription.csv`: Transcribed segments with timing
  - `segments/`: Individual audio clips (if enabled)
  - `metadata.json`: Processing metadata

---

### 📊 2. Dataset Module (`src/dataset/`)

**Purpose:** Dataset creation, loading, and preprocessing for training and analysis.

#### Files Created:

##### `src/dataset/__init__.py`
- **What it does:** Module initialization and documentation
- **Usage:** Automatic import when using the dataset module

##### `src/dataset/loader.py`
- **Source:** Extracted from training logic in wav2vec2 notebooks
- **What it does:**
  - Load and combine EMOVO and emozionalmente datasets
  - Apply emotion mapping and filtering
  - Data augmentation with audiomentations
  - Train/validation/test splitting
  - Export unified datasets

- **Key Classes:**
  - `DatasetConfig`: Configuration for dataset parameters
  - `DatasetLoader`: Main dataset loading and processing

- **Usage Examples:**
```bash
# Create unified dataset
python -m src.dataset.loader --output-dir datasets/ --dataset both_full_aug

# Custom dataset configuration
python -m src.dataset.loader --dataset emovo --use-augmentation --output-dir my_dataset/

# Override specific settings
python -m src.dataset.loader --dataset both_full_aug --test-size 0.3 --output-dir datasets/
```

- **Configuration:** Works with any YAML config file or standalone
- **Outputs:**
  - `train_dataset.csv`: Training data
  - `val_dataset.csv`: Validation data  
  - `test_dataset.csv`: Test data
  - `dataset_info.json`: Dataset statistics

---

### 🧠 3. Training Module (`src/training/`)

**Purpose:** Wav2Vec2 model fine-tuning for emotion recognition.

#### Files Created:

##### `src/training/__init__.py`
- **What it does:** Module initialization and documentation
- **Usage:** Automatic import when using the training module

##### `src/training/wav2vec2_trainer.py`
- **Source:** Converted from `notebooks/FineTuning/10epochs-noearlystop.ipynb`
- **What it does:**
  - Fine-tune pre-trained Wav2Vec2 models for emotion classification
  - Support for data augmentation (pre-generated and on-the-fly)
  - Comprehensive training configuration via YAML
  - Evaluation metrics and model checkpointing
  - Integration with HuggingFace Transformers

- **Key Classes:**
  - `TrainingConfig`: YAML-based configuration management
  - `SpeechDataset`: Custom dataset for audio emotion classification
  - `AudioAugmentation`: Data augmentation pipeline

- **Usage Examples:**
```bash
# Basic training with config file
python -m src.training.wav2vec2_trainer --config train_config.yaml

# Override specific parameters
python -m src.training.wav2vec2_trainer --config train_config.yaml --num-epochs 15 --learning-rate 1e-4

# Custom data and output directories
python -m src.training.wav2vec2_trainer --config train_config.yaml --data-dir data/ --output-dir models/my_model/
```

##### `train_config.yaml`
- **What it does:** Complete training configuration with experiment defaults
- **Key sections:**
  - Data configuration (dataset choice, augmentation)
  - Model parameters (Wav2Vec2 variant)
  - Training hyperparameters (learning rate, epochs, batch size)
  - Audio processing settings
  - Emotion mapping

- **Usage:** Referenced by training module, customizable for different experiments

---

### 🔍 4. Clustering Module (`src/clustering/`)

**Purpose:** Unified clustering analysis with hyperparameter optimization across multiple algorithms.

#### Files Created:

##### `src/clustering/__init__.py`
- **What it does:** Module initialization and documentation
- **Usage:** Automatic import when using the clustering module

##### `src/clustering/hyperparameter_search.py`
- **Source:** Unified from 3 notebooks: `K_means.ipynb`, `agglomerative.ipynb`, `spectral.ipynb`
- **What it does:**
  - Extract embeddings using fine-tuned Wav2Vec2 models
  - Hyperparameter optimization for K-means, Agglomerative, and Spectral clustering
  - UMAP dimensionality reduction with parameter tuning
  - Two-step algorithm selection process:
    1. Best K per algorithm (mean silhouette score)
    2. Best absolute configurations per optimal K
    3. Final best algorithm selection
  - Comprehensive visualization and result reporting

- **Key Classes:**
  - `ClusteringConfig`: YAML-based configuration
  - `EmbeddingExtractor`: Wav2Vec2 embedding extraction
  - `ClusteringAlgorithms`: Algorithm definitions and parameter grids
  - `HyperparameterSearcher`: Main optimization logic

- **Usage Examples:**
```bash
# Basic clustering analysis
python -m src.clustering.hyperparameter_search --config clustering_config.yaml

# Override data and model paths
python -m src.clustering.hyperparameter_search --config clustering_config.yaml --data-file data.csv --model-path models/my_model/

# Custom cluster range
python -m src.clustering.hyperparameter_search --config clustering_config.yaml --k-range 3 4 5 6 7 8

# Disable visualizations
python -m src.clustering.hyperparameter_search --config clustering_config.yaml --no-plots
```

##### `clustering_config.yaml`
- **What it does:** Configuration for clustering analysis parameters
- **Key sections:**
  - Data and model paths
  - Embedding extraction parameters
  - Cluster range and evaluation metrics
  - Visualization and output options

- **Outputs:**
  - `clustering_hyperparameter_results.csv`: All tested configurations
  - `best_clustering_config.yaml`: Optimal configuration found
  - `clustering_summary.txt`: Human-readable summary
  - Visualization plots (mean silhouette by K, top configurations)

---

### 🤖 5. Analysis Module (`src/analysis/`)

**Purpose:** LLM-based emotion analysis, NLP evaluation, and audio export.

#### Files Created:

##### `src/analysis/__init__.py`
- **What it does:** Module initialization and documentation
- **Usage:** Automatic import when using the analysis module

##### `src/analysis/llm_analyzer.py`
- **Source:** Converted from `notebooks/Inference/LLM/analyze_texts.ipynb`
- **What it does:**
  - Weight optimization for selecting representative cluster elements
  - LLM emotion analysis using Ollama (gemma3:12b-it-q4_K_M)
  - NLP evaluation metrics (BLEU, ROUGE, BERTScore)
  - Audio file export and organization
  - Google Forms data generation for human evaluation

- **Key Classes:**
  - `LLMAnalysisConfig`: YAML-based configuration
  - `ClusterWeightOptimizer`: Representative element selection with quality scoring
  - `LLMAnalyzer`: Ollama integration with structured prompts
  - `NLPEvaluator`: Evaluation metrics calculation
  - `AudioExporter`: File organization and export
  - `LLMClusterAnalyzer`: Main orchestrator

- **Usage Examples:**
```bash
# Basic LLM analysis
python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml

# Override analysis settings
python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --analysis-type detailed --llm-model gemma2:27b

# Disable specific components
python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --no-nlp-eval --no-audio-export

# Custom evaluation mode
python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --evaluation-mode concatenated
```

##### `llm_analysis_config.yaml`
- **What it does:** Configuration for LLM analysis parameters
- **Key sections:**
  - Data input and output paths
  - Weight optimization settings
  - LLM model configuration
  - NLP evaluation parameters
  - Audio export options

- **Outputs:**
  - `cluster_analyses.csv`: LLM emotion analyses
  - `nlp_evaluation_*.csv`: NLP evaluation metrics
  - `exported_audio/`: Organized audio files by cluster
  - `google_forms_data.csv`: Human evaluation ready data
  - `cluster_audio_files.zip`: Complete audio archive

---

### 🎯 6. Core Module (`src/core/`)

**Purpose:** Pipeline orchestration for complete end-to-end workflow.

#### Files Created:

##### `src/core/__init__.py`
- **What it does:** Module initialization and documentation
- **Usage:** Automatic import when using the core module

##### `src/core/pipeline.py`
- **What it does:**
  - Orchestrates complete workflow from audio files to LLM analysis
  - Manages dependencies between pipeline steps
  - Validates configurations and prerequisites
  - Provides comprehensive error handling and logging
  - Generates pipeline execution summary

- **Pipeline Steps:**
  1. **Segmentation**: WhisperX audio processing
  2. **Dataset Creation**: Unified dataset from segments
  3. **Training**: Model fine-tuning (optional)
  4. **Clustering**: Hyperparameter optimization
  5. **LLM Analysis**: Emotion analysis and evaluation

- **Key Classes:**
  - `PipelineConfig`: Complete pipeline configuration
  - `PipelineStep`: Base class for pipeline steps
  - `SegmentationStep`, `DatasetCreationStep`, `TrainingStep`, `ClusteringStep`, `LLMAnalysisStep`: Individual step implementations
  - `PipelineOrchestrator`: Main orchestration logic

- **Usage Examples:**
```bash
# Complete pipeline execution
python -m src.core.pipeline --config pipeline_config.yaml

# Skip specific steps
python -m src.core.pipeline --config pipeline_config.yaml --skip-training --skip-llm-analysis

# Override settings
python -m src.core.pipeline --config pipeline_config.yaml --output-dir results/ --force-recreate

# Custom input files
python -m src.core.pipeline --config pipeline_config.yaml --input-files audio1.wav video1.mp4

# Use existing model
python -m src.core.pipeline --config pipeline_config.yaml --model-path models/my_model/
```

##### `pipeline_config.yaml`
- **What it does:** Master configuration for complete pipeline
- **Key sections:**
  - Input files and output directory
  - Step execution control
  - References to step-specific configs
  - Advanced options (force recreate, cleanup, parallel processing)

---

## 🔧 Configuration System

### YAML-First Approach
All modules use YAML configuration files as the primary method for parameter specification:

- **`train_config.yaml`**: Training parameters with experiment defaults
- **`clustering_config.yaml`**: Clustering algorithm and evaluation settings  
- **`llm_analysis_config.yaml`**: LLM analysis and evaluation configuration
- **`pipeline_config.yaml`**: Master pipeline orchestration config

### CLI Override Pattern
Every module supports CLI overrides for key parameters:
```bash
# YAML + CLI override pattern used throughout
python -m src.module.name --config config.yaml --param value --other-param value
```

### Configuration Inheritance
The pipeline config references step-specific configs, allowing modular configuration management while maintaining centralized control.

---

## 🚀 Usage Workflows

### 1. Individual Module Usage

Each module can be used independently for specific tasks:

```bash
# Only segmentation
python -m src.segmentation.whisperx --input-file audio.wav --output-dir segments/

# Only clustering analysis  
python -m src.clustering.hyperparameter_search --config clustering_config.yaml

# Only LLM analysis
python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml
```

### 2. Partial Pipeline Usage

Skip steps as needed for partial workflows:

```bash
# Segmentation + Dataset creation only
python -m src.core.pipeline --config pipeline_config.yaml --skip-training --skip-clustering --skip-llm-analysis

# Clustering + LLM analysis only (with existing data)
python -m src.core.pipeline --config pipeline_config.yaml --skip-segmentation --skip-dataset --skip-training
```

### 3. Complete Pipeline Usage

Full end-to-end processing:

```bash
# Standard complete pipeline
python -m src.core.pipeline --config pipeline_config.yaml

# With custom settings
python -m src.core.pipeline --config pipeline_config.yaml --output-dir my_results/ --force-recreate
```

---

## 📊 Output Structure

### Individual Module Outputs

Each module produces organized outputs in its designated directory:

```
results/
├── segmentation/
│   ├── audio1_segments.csv
│   ├── audio2_segments.csv
│   └── segments/
├── clustering/
│   ├── clustering_hyperparameter_results.csv
│   ├── best_clustering_config.yaml
│   └── visualizations/
└── llm_analysis/
    ├── cluster_analyses.csv
    ├── nlp_evaluation_individual.csv
    ├── nlp_evaluation_concatenated.csv
    └── exported_audio/
```

### Complete Pipeline Output

The pipeline orchestrator creates a comprehensive results structure:

```
results/complete_pipeline/
├── segmentation/                    # WhisperX outputs
├── dataset_creation/                # Unified datasets
├── training/                        # Trained models (if enabled)
├── clustering/                      # Clustering analysis
├── llm_analysis/                    # LLM analyses and exports
├── pipeline_summary.json           # Execution summary
└── [intermediate files]             # Cleaned up if requested
```

---

## 🔗 Dependencies and Requirements

### Core Dependencies
- **Python 3.11+**: Required for all modules
- **PyTorch**: Deep learning framework
- **HuggingFace Transformers**: Pre-trained models
- **WhisperX**: Audio transcription and diarization
- **scikit-learn**: Clustering algorithms
- **pandas**: Data manipulation
- **PyYAML**: Configuration management

### Specialized Dependencies
- **ollama**: LLM analysis (requires local Ollama installation)
- **evaluate**: NLP evaluation metrics
- **audiomentations**: Audio data augmentation
- **umap-learn**: Dimensionality reduction
- **sentence-transformers**: Additional embeddings

### Installation
```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

---

## 🎯 Key Benefits of Refactoring

### 1. **Modularity**
- Each component is independently usable
- Clear separation of concerns
- Reusable across different projects

### 2. **Configurability** 
- YAML-based configuration system
- CLI overrides for flexibility
- Environment-specific settings

### 3. **Reproducibility**
- Version-controlled configurations
- Consistent parameter management
- Detailed execution logging

### 4. **Scalability**
- Parallel processing support
- Batch file processing
- Pipeline orchestration

### 5. **Maintainability**
- Clean code architecture
- Comprehensive error handling
- Modular testing capabilities

### 6. **Production-Ready**
- CLI interfaces for automation
- Proper logging and monitoring
- Integration-friendly APIs

---

## 🔄 Migration from Notebooks

### Original Notebook Structure
```
notebooks/
├── Segmentation/whisperX.ipynb
├── FineTuning/10epochs-noearlystop.ipynb
├── Inference/clustering/[K_means.ipynb, agglomerative.ipynb, spectral.ipynb]
└── Inference/LLM/analyze_texts.ipynb
```

### Refactored Module Structure
```
src/
├── segmentation/whisperx.py      # ← whisperX.ipynb
├── training/wav2vec2_trainer.py  # ← 10epochs-noearlystop.ipynb
├── clustering/hyperparameter_search.py  # ← 3 clustering notebooks unified
├── analysis/llm_analyzer.py      # ← analyze_texts.ipynb
└── core/pipeline.py              # ← New: complete orchestration
```

### Preserved Functionality
- **100% logic preservation**: All notebook functionality maintained
- **Enhanced reliability**: Better error handling and validation
- **Improved usability**: CLI interfaces and configuration management
- **Added features**: Pipeline orchestration and batch processing

---

## 🛠️ Development and Testing

### Running Individual Tests
```bash
# Test segmentation with sample audio
python -m src.segmentation.whisperx --input-file test_audio.wav --output-dir test_results/

# Test clustering with sample data
python -m src.clustering.hyperparameter_search --config clustering_config.yaml --data-file sample_data.csv

# Test LLM analysis
python -m src.analysis.llm_analyzer --config llm_analysis_config.yaml --data-file clustered_sample.csv
```

### Debugging Pipeline Issues
```bash
# Run pipeline with verbose output and force recreation
python -m src.core.pipeline --config pipeline_config.yaml --force-recreate

# Test individual steps
python -m src.core.pipeline --config pipeline_config.yaml --skip-clustering --skip-llm-analysis
```

### Configuration Validation
Each module validates its configuration on startup and provides clear error messages for missing or invalid parameters.

---

## 📚 Additional Resources

### Configuration Templates
All YAML configuration files include:
- Comprehensive parameter documentation
- Usage examples
- Default value explanations
- Output descriptions

### Error Handling
- Detailed error messages with suggested fixes
- Graceful failure with partial results preservation
- Validation checks before expensive operations

### Logging
- Progress tracking for long-running operations
- Detailed execution summaries
- Performance metrics and timing information

---

## 🎉 Conclusion

This refactoring transforms a collection of Jupyter notebooks into a **production-ready, modular system** suitable for:

- **Research workflows**: Flexible experimentation with individual components
- **Production deployments**: Automated pipelines for speech emotion analysis  
- **Batch processing**: Large-scale audio analysis with parallel processing
- **Integration projects**: APIs and modules for larger systems

The system maintains **100% compatibility** with the original notebook functionality while adding **significant improvements** in usability, reliability, and scalability.

Each module can be used independently or as part of the complete pipeline, providing maximum flexibility for different use cases and deployment scenarios.