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
    
    # With topic modeling filtering:
    python main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/model --enable-topic-filtering --topic-representation chat
    
    # With all features:
    python main.py --input-audio audio.wav --output-csv results.csv --model-path /path/to/model --llm-methods text_only text_audio --enable-topic-filtering --clustering-config custom_config.yaml
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
    
    # Topic modeling arguments
    parser.add_argument('--enable-topic-filtering', action='store_true',
                       help='Enable topic modeling to filter segments before clustering (optional)')
    parser.add_argument('--topic-representation', 
                       choices=['base', 'chat', 'summarization'],
                       default='chat',
                       help='Type of LLM representation for topic modeling (default: chat)')
    parser.add_argument('--topic-config',
                       help='Path to topic modeling config YAML file (optional)')
    parser.add_argument('--topic-search-keywords', nargs='+',
                       help='Keywords to search for similar topics (optional, requires --enable-topic-filtering)')
    
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
        # Calculate total steps based on options
        total_steps = 5  # Base steps: segmentation, feature extraction, post-processing, clustering, LLM
        if args.enable_topic_filtering:
            total_steps += 1
        
        # Initialize progress bar
        with tqdm(total=total_steps, desc="Pipeline Progress", unit="step") as pbar:
            
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
                # Try alternative naming pattern
                actual_segmented_csv = temp_dir / f"{audio_name}_segmented.csv"
            
            if not actual_segmented_csv.exists():
                print(f"Error: Segmentation file not created")
                sys.exit(1)
            pbar.update(1)
            
            # STEP 2: Optional Topic Modeling Analysis (on segmented data)
            clustering_input_csv = str(actual_segmented_csv)  # Default input for next steps
            
            if args.enable_topic_filtering:
                pbar.set_description("Topic modeling and filtering...")
                
                # Generate topic modeling output directory
                topic_output_dir = output_path.parent / f"{output_path.stem}_topic_modeling"
                
                # Prepare config if keywords are provided via CLI
                topic_config = None
                if args.topic_search_keywords:
                    # Create a temporary config with search keywords
                    import yaml
                    topic_config_dict = {
                        'topic_search': {
                            'enable_search': True,
                            'search_keywords': args.topic_search_keywords,
                            'top_n': 5
                        }
                    }
                    # If user provided a config file, load and update it
                    if args.topic_config:
                        with open(args.topic_config, 'r') as f:
                            user_config = yaml.safe_load(f)
                            user_config.update(topic_config_dict)
                            topic_config_dict = user_config
                    
                    # Save temporary config
                    temp_config_path = temp_dir / "topic_config_temp.yaml"
                    with open(temp_config_path, 'w') as f:
                        yaml.dump(topic_config_dict, f)
                    topic_config = str(temp_config_path)
                else:
                    topic_config = args.topic_config
                
                # Run topic modeling and get filtered CSV
                filtered_csv_path = topic_modeling_analysis(
                    input_csv=clustering_input_csv,
                    output_dir=str(topic_output_dir),
                    config_path=topic_config,
                    representation_type=args.topic_representation,
                    return_filtered_csv=True  # This returns the filtered CSV path
                )
                
                if filtered_csv_path and Path(filtered_csv_path).exists():
                    clustering_input_csv = filtered_csv_path
                    print(f"Topic filtering applied. Using filtered data for next steps.")
                else:
                    print("Warning: Topic filtering failed, using original segmented data")
                
                pbar.update(1)
            
            # STEP 3: Audio Feature Extraction (on filtered or original data)
            pbar.set_description("Feature extraction...")
            features_df = process_segmented_csv(
                csv_file=clustering_input_csv,
                output_file=str(features_csv)
            )
            pbar.update(1)
            
            # STEP 4: Post-processing and Description Generation
            pbar.set_description("Post-processing...")
            processed_csv = temp_dir / f"{audio_name}_processed.csv"
            processed_df = process_audio_features(
                input_csv=str(features_csv),
                output_csv=str(processed_csv),
                num_classes=args.num_classes
            )
            pbar.update(1)
            
            # STEP 5: Clustering Analysis
            pbar.set_description("Clustering analysis...")
            clustered_csv = temp_dir / f"{audio_name}_clustered.csv"
            final_df = clustering_analysis(
                input_csv=str(processed_csv),
                output_csv=str(clustered_csv),
                model_path=args.model_path,
                config_path=args.clustering_config
            )
            pbar.update(1)
            
            # STEP 6: LLM Analysis
            pbar.set_description("LLM analysis...")
            llm_analysis(
                input_csv=str(clustered_csv),
                output_csv=str(args.output_csv),  # Final output
                analysis_methods=args.llm_methods,
                config_path=args.llm_config
            )
            pbar.update(1)
        
        # Generate summary report
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final result saved to: {args.output_csv}")
        
        # Count segments at different stages
        import pandas as pd
        original_segments = len(pd.read_csv(actual_segmented_csv))
        final_segments = len(pd.read_csv(args.output_csv))
        
        print(f"\nSegment Statistics:")
        print(f"  Original segments: {original_segments}")
        
        if args.enable_topic_filtering:
            filtered_segments = len(pd.read_csv(clustering_input_csv))
            print(f"  After topic filtering: {filtered_segments}")
            print(f"  Segments removed by filtering: {original_segments - filtered_segments}")
            print(f"  Topic modeling results: {topic_output_dir}")
        
        print(f"  Final segments: {final_segments}")
        
        # Display methods used
        print(f"\nMethods Used:")
        print(f"  Segmentation model: {args.model}")
        print(f"  Language: {args.language}")
        print(f"  Feature classes: {args.num_classes}")
        print(f"  LLM methods: {', '.join(args.llm_methods)}")
        
        if args.enable_topic_filtering:
            print(f"  Topic filtering: ENABLED")
            print(f"  Topic representation: {args.topic_representation}")
            if args.topic_search_keywords:
                print(f"  Search keywords: {', '.join(args.topic_search_keywords)}")
        else:
            print(f"  Topic filtering: DISABLED")
        
        # Clean up temporary files (optional)
        cleanup = input("\nClean up temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print("Temporary files removed.")
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")
        else:
            print(f"Temporary files preserved in: {temp_dir}")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()