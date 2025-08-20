import argparse
import gc
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import whisperx
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class WhisperXConfig:
    """Configuration for WhisperX processing."""
    model_name: str = "large-v3-turbo"
    language: str = "it"
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "int8"
    
    # Segmentation parameters
    merge_duration_threshold: float = 1.5  # seconds
    split_duration_threshold: float = 10.0  # seconds
    min_words_per_segment: int = 2
    
    # Audio segmentation
    export_margin: float = 0.1  # seconds
    export_format: str = "wav"


class WhisperXProcessor:
    """Segmentation pipeline."""
    
    def __init__(self, config: WhisperXConfig, hf_token: str):
        self.config = config
        self.hf_token = hf_token
        self.model = None
        self.diarize_model = None
        
    def load_model(self) -> None:
        """Load WhisperX model."""
        print(f'Loading WhisperX model {self.config.model_name}...')
        self.model = whisperx.load_model(
            self.config.model_name, 
            self.config.device, 
            compute_type=self.config.compute_type,
            language=self.config.language
        )
        print(f'WhisperX model ready!')
        
    def load_audio(self, audio_file: str) -> Any:
        """Load audio file."""
        print(f'Loading audio file: {audio_file}')
        audio = whisperx.load_audio(audio_file)
        print(f'Audio loaded successfully!')
        return audio
        
    def transcribe(self, audio: Any) -> Dict:
        """Transcribe audio."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print(f'Transcribing audio...')
        transcription = self.model.transcribe(
            audio, 
            batch_size=self.config.batch_size,
            language=self.config.language
        )
        print(f'Audio transcription completed!')
        return transcription
        
    def align(self, transcription: Dict, audio: Any) -> Dict:
        """Align transcription with audio."""
        print(f'Aligning audio transcription...')
        model_a, metadata = whisperx.load_align_model(
            language_code=transcription["language"], 
            device=self.config.device
        )
        alignment = whisperx.align(
            transcription["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.config.device, 
            return_char_alignments=False
        )
        print(f'Audio alignment completed!')
        return alignment
        
    def diarize(self, audio: Any, alignment: Dict, audio_name: str, num_speakers: Optional[int] = None) -> Dict:
        """Perform speaker diarization."""
        print(f'Diarizing audio {audio_name}...')
        
        
        if self.diarize_model is None:
            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.hf_token, 
                device=self.config.device
            )
        
        # Set speaker parameters
        diarize_kwargs = {}
        if num_speakers:
            diarize_kwargs['min_speakers'] = num_speakers
            diarize_kwargs['max_speakers'] = num_speakers
        
        diarize_segments = self.diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, alignment)
        
        print(f'Diarization completed!')
        return result
        
    def process_file(self, audio_file: str, num_speakers: Optional[int] = None) -> pd.DataFrame:
        """Complete processing pipeline for a single audio file."""
        audio_name = Path(audio_file).stem
        
        try:
            # Load and process
            audio = self.load_audio(audio_file)
            transcription = self.transcribe(audio)
            alignment = self.align(transcription, audio)
            result = self.diarize(audio, alignment, audio_name, num_speakers)
            
            # Process segments to DataFrame
            print("Processing segments to DataFrame...")
            df = self._process_segments_to_dataframe(result["segments"])
            print("DataFrame created successfully!")
            
            # Clean up memory
            del audio, transcription, alignment, result
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return df
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            raise
    
    def _find_split_index(self, words: List[Dict], target_split_time: float, min_words_per_segment: int = 1) -> int:
        if not words or len(words) < min_words_per_segment * 2:
            return -1
        
        valid_indices_range = range(min_words_per_segment - 1, len(words) - min_words_per_segment)
        if not valid_indices_range:
            return -1
        
        min_diff = float('inf')
        split_idx = -1
        
        for i in valid_indices_range:
            end_time = words[i].get('end', 0)
            diff = abs(end_time - target_split_time)
            if diff < min_diff:
                min_diff = diff
                split_idx = i
        
        return split_idx
        
    def _create_text_from_words(self, word_list: List[Dict]) -> str:
        return ' '.join([w.get('word', '') for w in word_list if w.get('word')]).strip()
        
    def _validate_word(self, word: Dict) -> Tuple[bool, Optional[Dict]]:
        word_copy = word.copy()
        
        required_fields = ['word', 'start', 'end']
        for field in required_fields:
            if field not in word_copy or word_copy[field] is None:
                return False, None
        
        if word_copy['end'] < word_copy['start']:
            word_copy['end'] = word_copy['start']
        
        if 'speaker' not in word_copy or word_copy['speaker'] is None:
            word_copy['speaker'] = 'unknown'
        
        return True, word_copy
        
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []
        
        result = [segments[0]]
        threshold = self.config.merge_duration_threshold
        
        for current in segments[1:]:
            previous = result[-1]
            
            if (previous['duration'] < threshold and 
                previous['speaker'] == current['speaker']):
                
                merged = {
                    'start': previous['start'],
                    'end': current['end'],
                    'speaker': previous['speaker'],
                    'words': previous['words'] + current['words'],
                    'duration': current['end'] - previous['start']
                }
                result[-1] = merged
            else:
                result.append(current)
        
        return result
        
    def _split_long_segment(self, segment: Dict) -> List[Dict]:
        threshold = self.config.split_duration_threshold
        min_words = self.config.min_words_per_segment
        
        if segment['duration'] <= threshold or len(segment['words']) < min_words * 2:
            segment['text'] = self._create_text_from_words(segment['words'])
            return [segment]
        
        target_split = segment['start'] + (segment['duration'] / 2.0)
        split_idx = self._find_split_index(segment['words'], target_split, min_words)
        
        if split_idx == -1:
            segment['text'] = self._create_text_from_words(segment['words'])
            return [segment]
        
        words1 = segment['words'][:split_idx + 1]
        words2 = segment['words'][split_idx + 1:]
        
        segment1 = {
            'start': words1[0]['start'],
            'end': words1[-1]['end'],
            'speaker': segment['speaker'],
            'words': words1,
            'duration': words1[-1]['end'] - words1[0]['start'],
            'text': self._create_text_from_words(words1)
        }
        
        segment2 = {
            'start': words2[0]['start'],
            'end': words2[-1]['end'],
            'speaker': segment['speaker'],
            'words': words2,
            'duration': words2[-1]['end'] - words2[0]['start'],
            'text': self._create_text_from_words(words2)
        }
        
        result = []
        for seg in [segment1, segment2]:
            if seg['duration'] > threshold and len(seg['words']) >= min_words * 2:
                result.extend(self._split_long_segment(seg))
            else:
                result.append(seg)
        
        return result
        
    def _process_segments_to_dataframe(self, segments: List[Dict]) -> pd.DataFrame:
        if not segments:
            return pd.DataFrame(columns=['start', 'end', 'speaker', 'text', 'duration'])
        
        initial_processed_data = []
        
        for segment in segments:
            if not segment.get('words'):
                continue
            
            current_speaker = None
            current_words = []
            start_time = None
            end_time = None
            
            for word in segment.get('words', []):
                is_valid, validated_word = self._validate_word(word)
                if not is_valid:
                    continue
                
                word_speaker = validated_word['speaker']
                word_start = validated_word['start']
                word_end = validated_word['end']
                
                if current_speaker is None:
                    current_speaker = word_speaker
                    current_words = [validated_word]
                    start_time = word_start
                    end_time = word_end
                elif word_speaker != current_speaker:
                    if current_words:
                        duration = end_time - start_time
                        initial_processed_data.append({
                            'start': start_time,
                            'end': end_time,
                            'speaker': current_speaker,
                            'words': current_words.copy(),
                            'duration': duration
                        })
                    
                    current_speaker = word_speaker
                    current_words = [validated_word]
                    start_time = word_start
                    end_time = word_end
                else:
                    current_words.append(validated_word)
                    end_time = word_end
            
            if current_speaker is not None and current_words:
                duration = end_time - start_time
                initial_processed_data.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': current_speaker,
                    'words': current_words.copy(),
                    'duration': duration
                })
        
        merged_data = self._merge_consecutive_segments(initial_processed_data)
        
        final_data = []
        for segment in merged_data:
            split_segments = self._split_long_segment(segment)
            final_data.extend(split_segments)
        
        if not final_data:
            return pd.DataFrame(columns=['start', 'end', 'speaker', 'text', 'duration'])
        
        df = pd.DataFrame(final_data)
        if 'words' in df.columns:
            df = df.drop(columns=['words'])
        
        columns_order = ['start', 'end', 'speaker', 'text', 'duration']
        df = df[columns_order] if set(columns_order).issubset(df.columns) else df
        
        numeric_columns = ['start', 'end', 'duration']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(3)
        
        return df


class AudioSegmenter:
    """Creates individual audio files from segments."""
    
    def __init__(self, config: WhisperXConfig):
        self.config = config
        
    def segment_audio(self, audio_path: str, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        audio = AudioSegment.from_file(audio_path)
        
        df = df.copy()
        df['start_ms'] = (df['start'] * 1000).astype(int)
        df['end_ms'] = (df['end'] * 1000).astype(int)
        
        processed_segments = []
        margin_ms = int(self.config.export_margin * 1000)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating audio segments"):
            try:
                start = max(0, row['start_ms'] - margin_ms)
                end = min(len(audio), row['end_ms'] + margin_ms)
                
                segment = audio[start:end]
                
                speaker = row['speaker'].replace(" ", "_")
                filename = f"{idx:03d}_{speaker}_{row['start']:.2f}-{row['end']:.2f}.{self.config.export_format}"
                filepath = Path(output_dir) / filename
                
                segment.export(str(filepath), format=self.config.export_format)
                
                processed_segments.append({
                    'original_file': audio_path,
                    'segment_file': str(filepath),
                    'start': row['start'],
                    'end': row['end'],
                    'speaker': row['speaker'],
                    'duration': row['duration'],
                    'text': row['text'],
                    'success': True
                })
                
            except Exception as e:
                processed_segments.append({
                    'original_file': audio_path,
                    'segment_file': None,
                    'start': row['start'],
                    'end': row['end'],
                    'speaker': row['speaker'],
                    'duration': row['duration'],
                    'text': row['text'],
                    'success': False,
                    'error': str(e)
                })
        
        return pd.DataFrame(processed_segments)


def process_audio_file(audio_file, output_dir, num_speakers=None, model_name="large-v3-turbo", language="it", hf_token=None):
    """ Process a single audio file for segmentation, transcription, alignment, and diarization.
    Args:
        audio_file (str): Audio file path
        output_dir (str): Audio file directory
        num_speakers (int, optional): Number of speakers (if known)
        model_name (str): WhisperX model name
        language (str): language code (default: "it")
        hf_token (str): HuggingFace token for diarization (if not in .env file)
    Returns:
        pd.DataFrame: DataFrame with segmented audio information
    """
    # Get HuggingFace token from env if not provided
    if not hf_token:
        hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN in .env file or provide via hf_token parameter")
    
    # Verify audio file exists
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create configuration
    config = WhisperXConfig(
        model_name=model_name,
        language=language,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = WhisperXProcessor(config, hf_token)
    processor.load_model()
    segmenter = AudioSegmenter(config)
    
    print(f"Processing: {Path(audio_file).name}")
    
    try:
        # Step 1: Process transcription (includes transcribe, align, diarize, segment processing)
        print("Transcribing & processing segments...")
        transcript_df = processor.process_file(str(audio_file), num_speakers)
        audio_name = Path(audio_file).stem
        
        # Step 2: Create audio segments  
        print("Creating audio clips...")
        segments_dir = output_path / f"segmented_{audio_name}"
        segments_report_df = segmenter.segment_audio(str(audio_file), transcript_df, str(segments_dir))
        
        # Step 3: Save results
        print("Saving results...")
        segments_report_path = output_path / f"segmented_{audio_name}.csv"
        segments_report_df.to_csv(segments_report_path, index=False)
        
        successful_segments = segments_report_df['success'].sum()
        print(f"Audio segments created: {successful_segments}/{len(segments_report_df)}")
        print(f"Results saved: {segments_report_path}")
        
        return segments_report_df
        
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        raise


def main():
    """
    Audio Segmentation - Transcription, alignment, diarization and segmentation.

    Usage:
        # Single file segmentation:
        python segmentation.py --audio audio_file.wav --output results/

        # Multiple files segmentation:
        python segmentation.py --input-dir audio_files/ --output results/

        # Example with specific parameters:
        python segmentation.py --audio audio_file.mp3 --output results/ --speakers 3 --language it
    """
    parser = argparse.ArgumentParser(
        description="Segmentation Process - Audio transcription, alignment, diarization and segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", "-a", 
                             help="Path to single audio file (e.g. song.mp3)")
    input_group.add_argument("--input-dir", "-i", 
                             help="Directory containing audio files to process")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for results")
    parser.add_argument("--token", "-t", 
                       help="HuggingFace token for diarization (if not in .env file)")
    parser.add_argument("--model", default="large-v3-turbo",
                       help="WhisperX model name (default: large-v3-turbo)")
    parser.add_argument("--language", "-l", default="it",
                       help="Audio language code (default: it)")
    parser.add_argument("--device", 
                       choices=["cuda", "cpu", "auto"],
                       default="auto",
                       help="Processing device (default: auto)")
    parser.add_argument("--speakers", "-s", type=int,
                       help="Number of speakers (if known)")
    args = parser.parse_args()
    
    # Get HuggingFace token from env or args
    hf_token = args.token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("Error: HuggingFace token required. Provide via --token argument or set HF_TOKEN in .env file")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create configuration
    config = WhisperXConfig(
        model_name=args.model,
        language=args.language,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        )
    
    # Collect audio files to process
    audio_files = []
    
    if args.audio:
        if not Path(args.audio).exists():
            print(f"Error: Audio file not found: {args.audio}")
            sys.exit(1)
        audio_files = [args.audio]
    
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)
            
        # Audio extensions to search for
        extensions = [".wav", ".mp3", ".m4a", ".flac", ".mp4", ".avi"]
        for ext in extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            print(f"No audio files found in {args.input_dir} with extensions {extensions}")
            sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s) to process")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = WhisperXProcessor(config, hf_token)
    processor.load_model()
    

    segmenter = AudioSegmenter(config)
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {Path(audio_file).name}")
        
        with tqdm(total=3, desc="Progress", leave=False) as pbar:
            try:
                # Step 1: Process transcription (includes transcribe, align, diarize, segment processing)
                pbar.set_description("Transcribing & processing segments")
                transcript_df = processor.process_file(str(audio_file), args.speakers)
                audio_name = Path(audio_file).stem
                pbar.update(1)
                
                # Step 2: Create audio segments
                pbar.set_description("Creating audio clips")
                segments_dir = output_path / f"segmented_{audio_name}"
                segments_report_df = segmenter.segment_audio(str(audio_file), transcript_df, str(segments_dir))
                pbar.update(1)
                
                # Step 3: Save results
                pbar.set_description("Saving results")
                segments_report_path = output_path / f"segmented_{audio_name}.csv"
                segments_report_df.to_csv(segments_report_path, index=False)
                pbar.update(1)
                
                successful_segments = segments_report_df['success'].sum()
                print(f"Audio segments created: {successful_segments}/{len(segments_report_df)}")
                print(f"Results saved: {segments_report_path}")
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
    
    print("\n Processing completed!")


if __name__ == "__main__":
    main()