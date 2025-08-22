import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import core functions from modules
from segmentation.segmentation import process_audio_file
from SpeechCueLLM.extract_audio_feature import process_segmented_csv
from SpeechCueLLM.postprocess_audio_feature import process_audio_features
from clustering.clustering import clustering_analysis
from llm.llm_analysis import llm_analysis
from topic_modeling.topic_modeling import topic_modeling_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Speech emotion analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
    # Basic analysis:
    python main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/wav2vec2/model
    
    # With custom parameters:
    python main.py --input-audio audio.mp3 --output-csv output.csv --num-classes 3 --speakers 2 --model-path /path/to/model
    
    # With custom LLM methods:
    python main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/model --llm-methods text_only text_audio
    
    # With topic modeling:
    python main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/model --enable-topic-modeling --topic-representation claude
    
    # With all features:
    python main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/model --llm-methods text_only text_audio --enable-topic-modeling --clustering-config custom_config.yaml
        """
    )
    
    # Required arguments
    parser.add_argument('--input-audio', required=True, 
                       help='Input audio file (wav, mp3, flac, m4a)')
    parser.add_argument('--output-csv', required=True, 
                       help='Output CSV file with final results')
    
    # Optional arguments
    parser.add_argument('--num-classes', type=int, default=5, choices=[3,4,5,6], 
                       help='Number of classes for feature categorization (default: 5)')
    parser.add_argument('--speakers', type=int,
                       help='Number of speakers if known (optional)')
    parser.add_argument('--model', default="large-v3-turbo",
                       help='WhisperX model to use (default: large-v3-turbo)')
    parser.add_argument('--language', default="it",
                       help='Audio language code (default: it)')
    parser.add_argument('--model-path', required=True,
                       help='Path to Wav2Vec2 model for clustering')
    parser.add_argument('--clustering-config', 
                       help='Path to clustering YAML config file (optional)')
    parser.add_argument('--llm-methods', nargs='+', 
                       choices=['text_only', 'text_audio', 'pure_audio'],
                       default=['text_only'],
                       help='LLM analysis methods to use (default: text_only)')
    parser.add_argument('--llm-config',
                       help='Path to LLM config YAML file (optional)')
    parser.add_argument('--enable-topic-modeling', action='store_true',
                       help='Enable topic modeling analysis (optional)')
    parser.add_argument('--topic-representation', 
                       choices=['base', 'deep_seek', 'chat', 'claude', 'summarization'],
                       default='chat',
                       help='Type of LLM representation for topic modeling (default: chat)')
    parser.add_argument('--topic-config',
                       help='Path to topic modeling config YAML file (optional)')
    
    args = parser.parse_args()
    
    # Verify input file exists
    audio_path = Path(args.input_audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.input_audio}")
        sys.exit(1)
    
    # Create temporary directory for intermediate files
    output_path = Path(args.output_csv)
    temp_dir = output_path.parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate intermediate file names
    audio_name = audio_path.stem
    segmented_csv = temp_dir / f"{audio_name}_segmented.csv"
    features_csv = temp_dir / f"{audio_name}_features.csv"
    
    # Verify model path for clustering
    if not Path(args.model_path).exists():
        print(f"Error: Wav2Vec2 model path not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Initialize progress bar - 6 steps total (topic modeling optional but still counted)
        with tqdm(total=6, desc="Pipeline Progress", unit="step") as pbar:
            
            # STEP 1: Audio Segmentation with WhisperX
            pbar.set_description("Audio segmentation...")
            segments_df = process_audio_file(
                audio_file=str(audio_path),
                output_dir=str(temp_dir),
                num_speakers=args.speakers,
                model_name=args.model,
                language=args.language
            )
            
            # Find the actual segmented CSV file created by process_audio_file
            actual_segmented_csv = temp_dir / f"segmented_{audio_name}.csv"
            if not actual_segmented_csv.exists():
                print(f"Error: Segmentation file not created: {actual_segmented_csv}")
                sys.exit(1)
            pbar.update(1)
            
            # STEP 2: Audio Feature Extraction
            pbar.set_description("Feature extraction...")
            features_df = process_segmented_csv(
                csv_file=str(actual_segmented_csv),
                output_file=str(features_csv)
            )
            pbar.update(1)
            
            # STEP 3: Post-processing and Description Generation
            pbar.set_description("Post-processing...")
            processed_df = process_audio_features(
                input_csv=str(features_csv),
                output_csv=str(args.output_csv),
                num_classes=args.num_classes
            )
            pbar.update(1)
            
            # STEP 4: Topic Modeling Analysis (optional - determines clustering input)
        clustering_input_csv = str(args.output_csv)  # Default input for clustering
        
        if args.enable_topic_modeling:
            pbar.set_description("Topic modeling...")
            # Generate topic modeling output directory
            output_path = Path(args.output_csv)
            topic_output_dir = output_path.parent / f"{output_path.stem}_topic_modeling"
            
            # Run topic modeling and get filtered CSV for clustering
            filtered_csv_path = topic_modeling_analysis(
                input_csv=str(args.output_csv),
                output_dir=str(topic_output_dir),
                config_path=args.topic_config,
                representation_type=args.topic_representation,
                return_filtered_csv=True  # Get filtered CSV for clustering
            )
            clustering_input_csv = filtered_csv_path  # Use filtered data for clustering
            pbar.update(1)
        else:
            # Skip topic modeling step in progress bar
            pbar.set_description("Skipping topic modeling...")
            pbar.update(1)
        
        # STEP 5: Clustering Analysis (mandatory - on filtered or original data)
        pbar.set_description("Clustering analysis...")
        # Use either original CSV or topic-filtered CSV
        final_df = clustering_analysis(
            input_csv=clustering_input_csv,
            output_csv=str(args.output_csv),  # Still save to original output path
            model_path=args.model_path,
            config_path=args.clustering_config
        )
        pbar.update(1)
        
        # STEP 6: LLM Analysis (mandatory)
        pbar.set_description("LLM analysis...")
        # Generate LLM output CSV name
        output_path = Path(args.output_csv)
        llm_output_csv = output_path.parent / f"{output_path.stem}_llm_analysis.csv"
        
        llm_analysis(
            input_csv=str(args.output_csv),
            output_csv=str(llm_output_csv),
            analysis_methods=args.llm_methods,
            config_path=args.llm_config
        )
        pbar.update(1)
        
        # Clean up temporary files
        try:
            actual_segmented_csv.unlink()
            features_csv.unlink()
            temp_dir.rmdir()
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
        
        print("Pipeline completed successfully!")
        print(f"Final result saved to: {args.output_csv}")
        print(f"LLM analysis results saved to: {llm_output_csv}")
        
        if args.enable_topic_modeling:
            print(f"Topic modeling results saved to: {topic_output_dir}")
        
        # Display final summary
        summary_text = "Results include audio features, categories, textual descriptions, and LLM emotion analysis."
        if args.enable_topic_modeling:
            summary_text += " Topic modeling was used to filter segments before clustering analysis."
        else:
            summary_text += " Clustering was performed on all segments."
        print(summary_text)
        
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()