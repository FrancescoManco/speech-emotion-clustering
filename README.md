# Speech Emotion Clustering Pipeline

Complete end-to-end pipeline for Italian speech emotion analysis with clustering and LLM interpretation.

## 🚀 Quick Start

```bash
# Complete pipeline (6 steps)
python src/core/main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/wav2vec2/model

# With topic modeling filter
python src/core/main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/wav2vec2/model --enable-topic-modeling

# With custom LLM methods
python src/core/main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/wav2vec2/model --llm-methods text_only text_audio
```

## 🏗️ Pipeline Architecture

The system consists of 6 modular steps that can run together or independently:

### 1. **Audio Segmentation** (`src/segmentation/segmentation.py`)
- **WhisperX integration**: Transcription, alignment, and speaker diarization
- **Audio splitting**: Individual segment files with timing information
- **Output**: Segmented CSV with speaker labels and transcriptions

### 2. **Feature Extraction** (`src/SpeechCueLLM/extract_audio_feature.py`)
- **Praat-based analysis**: Duration, intensity, pitch, articulation rate
- **Comprehensive features**: 15+ acoustic characteristics per segment
- **Output**: Numerical features CSV

### 3. **Post-processing** (`src/SpeechCueLLM/postprocess_audio_feature.py`)
- **Categorization**: Converts numerical features to interpretable categories (low/medium/high)
- **Impression generation**: Textual descriptions of vocal characteristics
- **Output**: Final enriched CSV with categories and impressions

### 4. **Topic Modeling** (`src/topic_modeling/topic_modeling.py`) - Optional
- **BERTopic analysis**: Semantic topic extraction from transcriptions
- **LLM labeling**: 5 different representation strategies with Ollama
- **Filtering**: Selects relevant segments for clustering
- **Output**: Topic analysis + filtered CSV for clustering

### 5. **Clustering** (`src/clustering/clustering.py`)
- **Wav2Vec2 embeddings**: Deep audio representations
- **Multiple algorithms**: K-means, Agglomerative, Spectral clustering
- **Hyperparameter optimization**: Automatic parameter tuning
- **Input**: Either all segments or topic-filtered segments
- **Output**: Clusters with assignments and centroids

### 6. **LLM Analysis** (`src/llm/llm_analysis.py`) - Mandatory
- **Cluster interpretation**: Emotion analysis per cluster using Ollama
- **3 analysis modes**: text_only, text_audio, pure_audio
- **Representative selection**: Smart segment selection from clusters
- **Output**: LLM emotion analysis for each cluster

## 📁 Module Details

### Core Pipeline (`src/core/main.py`)
Main orchestrator with progress tracking and error handling. Coordinates all modules with proper data flow and optional step execution.

### Segmentation Module (`src/segmentation/`)
- **Primary**: `segmentation.py` - WhisperX wrapper for transcription and diarization
- **Models**: Supports all WhisperX models (large-v3-turbo default)
- **Languages**: Configurable language support (Italian default)
- **Output**: `segmented_{audio_name}.csv`

### Feature Extraction (`src/SpeechCueLLM/`)
- **extract_audio_feature.py**: Praat-based acoustic feature extraction
- **postprocess_audio_feature.py**: Feature categorization and impression generation  
- **syllable_nuclei.py**: Articulation rate calculation utility
- **Output**: Enriched CSV with numerical and categorical features

### Topic Modeling (`src/topic_modeling/`)
- **BERTopic workflow**: Embedding → UMAP → Clustering → LLM labeling
- **5 representation models**: Base, DeepSeek, Chat, Claude, Summarization
- **Visualization**: 2D/3D plots, topic overview, document clustering
- **Configuration**: `configs/topic_modeling_config.yaml`

### Clustering Analysis (`src/clustering/`)
- **Wav2Vec2 embeddings**: Uses fine-tuned models for audio representation
- **Algorithm support**: K-means, Agglomerative, Spectral clustering
- **Optimization**: Hyperparameter search with silhouette scoring
- **Configuration**: `configs/clustering_config.yaml`

### LLM Analysis (`src/llm/`)
- **Ollama integration**: Local LLM inference with Gemma3
- **Cluster-level analysis**: Analyzes clusters as emotional entities
- **3 prompting strategies**: Text-only, text+audio features, pure audio
- **Configuration**: `configs/llm_config.yaml`

### Dataset & Training (`src/dataset/`, `src/training/`)
- **Dataset loader**: EMOVO and emozionalmente dataset integration
- **Wav2Vec2 trainer**: Fine-tuning for emotion recognition
- **Configuration**: `configs/train_config.yaml`

## 🎯 Key Features

### **Modular Design**
- Each module works standalone or integrated
- CLI interfaces for individual testing
- Clean separation of concerns

### **Configuration-Driven**
- YAML configuration files for all parameters
- CLI overrides for quick adjustments
- Environment-specific settings

### **Flexible Workflow**
- Optional topic modeling for segment filtering
- Multiple LLM analysis strategies
- Configurable clustering algorithms

### **Production-Ready**
- Error handling and validation
- Progress tracking with tqdm
- Logging and debugging support

## ⚙️ Configuration Files

All modules use YAML configuration files in `configs/`:

- **`pipeline_config.yaml`**: Master pipeline settings
- **`clustering_config.yaml`**: Clustering parameters and models
- **`llm_config.yaml`**: LLM prompts and analysis settings
- **`topic_modeling_config.yaml`**: BERTopic workflow configuration
- **`train_config.yaml`**: Model training parameters

## 🔧 Requirements

### Core Dependencies
- Python 3.11+
- PyTorch + CUDA support
- WhisperX (for segmentation)
- Transformers (Wav2Vec2 models)
- BERTopic (topic modeling)
- Ollama (LLM analysis)

### Audio Processing
- librosa, soundfile, torchaudio
- Praat (via parselmouth)
- audiomentations (augmentation)

### ML & Analysis
- scikit-learn, pandas, numpy
- umap-learn, hdbscan
- sentence-transformers

### Installation
```bash
# Using pip
pip install -r requirements.txt

# Or with uv (recommended)
uv sync
```

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required model
ollama pull gemma3:12b-it-q4_K_M
```

## 🚀 Usage Examples

### Complete Pipeline
```bash
# Basic analysis (all 6 steps)
python src/core/main.py \
  --input-audio audio.wav \
  --output-csv results.csv \
  --model-path /path/to/wav2vec2/model

# With topic modeling filter
python src/core/main.py \
  --input-audio audio.wav \
  --output-csv results.csv \
  --model-path /path/to/wav2vec2/model \
  --enable-topic-modeling \
  --topic-representation claude

# Multiple LLM analysis methods
python src/core/main.py \
  --input-audio audio.wav \
  --output-csv results.csv \
  --model-path /path/to/wav2vec2/model \
  --llm-methods text_only text_audio

# Complete with custom configs
python src/core/main.py \
  --input-audio audio.wav \
  --output-csv results.csv \
  --model-path /path/to/wav2vec2/model \
  --enable-topic-modeling \
  --clustering-config custom_clustering.yaml \
  --llm-config custom_llm.yaml \
  --topic-config custom_topic.yaml
```

### Standalone Module Usage
```bash
# Only segmentation
python src/segmentation/segmentation.py audio.wav output_dir/

# Only clustering
python src/clustering/clustering.py input.csv output.csv --model-path /path/to/model

# Only LLM analysis  
python src/llm/llm_analysis.py input.csv output.csv --methods text_only

# Only topic modeling
python src/topic_modeling/topic_modeling.py input.csv output_dir/ --representation chat
```

## 📊 Output Structure

```
results/
├── audio_results.csv                    # Main results with features & clustering
├── audio_results_llm_analysis.csv       # LLM emotion analysis per cluster
├── audio_results_topic_modeling/        # Topic modeling analysis (if enabled)
│   ├── topic_info.csv                  # Topic details & labels
│   ├── document_info.csv               # Document-topic assignments
│   ├── filtered_for_clustering.csv     # Segments selected for clustering
│   └── visualizations/                 # 2D/3D plots & topic visualizations
└── temp/                               # Temporary files (cleaned up)
```

## 🔄 Pipeline Flow

### Standard Flow (without topic modeling):
1. **Audio** → Segmentation → Features → Post-processing → **All segments**
2. **All segments** → Clustering → LLM Analysis → **Results**

### With Topic Modeling Filter:
1. **Audio** → Segmentation → Features → Post-processing → **All segments**
2. **All segments** → Topic Modeling → **Filtered segments**
3. **Filtered segments** → Clustering → LLM Analysis → **Results**

## 🐛 Troubleshooting

### Common Issues

**cuDNN library error (Google Colab):**
```bash
!sudo apt-get install libcudnn8 libcudnn8-dev
```

**Ollama connection issues:**
- Ensure Ollama is running: `ollama serve`
- Check model is available: `ollama list`
- Verify correct base URL in config

**Memory issues with large audio files:**
- Use smaller WhisperX model (base, small)
- Reduce max_segments_total in LLM config
- Enable chunking in segmentation

**Topic modeling optimization:**
- Set `enable_optimization: false` in config to use manual parameters
- Reduce k_range for faster processing
- Disable visualizations for headless environments

## 📈 Performance Notes

- **GPU recommended** for WhisperX and Wav2Vec2 operations
- **Large audio files**: Consider chunking for memory efficiency  
- **Topic modeling**: Most computationally intensive step
- **LLM analysis**: Speed depends on Ollama model size

## 🔮 Future Enhancements

- [ ] Real-time processing for streaming audio
- [ ] Multi-language support expansion
- [ ] Additional LLM provider integration (OpenAI, Anthropic)
- [ ] Web interface for pipeline execution
- [ ] Batch processing for multiple files
- [ ] Export to various formats (JSON, Parquet)

## 📄 License

[License information]

## 🤝 Contributing

[Contributing guidelines]