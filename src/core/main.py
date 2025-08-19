#!/usr/bin/env python3
"""
Speech emotion analysis pipeline
Executes in sequence: Segmentation -> Feature Extraction -> Post-processing
"""
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


def main():
    parser = argparse.ArgumentParser(
        description="Speech emotion analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
    # Basic analysis:
    python main.py --input-audio audio.wav --output-csv results.csv
    
    # With specific parameters:
    python main.py --input-audio audio.mp3 --output-csv output.csv --num-classes 3 --speakers 2
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
    
    try:
        # Initialize progress bar for all pipeline steps
        with tqdm(total=3, desc="Pipeline Progress", unit="step") as pbar:
            
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
        
        # TODO: Add clustering analysis step here
        # This step will group similar segments based on audio features
        # and generate cluster labels for each segment
        
        # TODO: Add LLM analysis step here
        # This step will use language models to analyze transcriptions
        # and generate emotion predictions based on text content
        
        # Clean up temporary files
        try:
            actual_segmented_csv.unlink()
            features_csv.unlink()
            temp_dir.rmdir()
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
        
        print("Pipeline completed successfully!")
        print(f"Final result saved to: {args.output_csv}")
        
    except KeyboardInterrupt:
        print("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()