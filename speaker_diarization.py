# instantiate the pipeline
from pyannote.audio import Pipeline
import whisper
import os
import json
import torch
import sys
import numpy as np
from pydub import AudioSegment
import tempfile
import subprocess
import pandas as pd
from typing import List, Tuple, Dict, Set
import uuid
from datetime import datetime

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration parameters
# Short segment thresholds (in seconds)
SHORT_SEGMENT_THRESHOLD = 0.5  # General short segment threshold
FULL_OVERLAP_SHORT_THRESHOLD = 1.0  # Threshold for full overlaps
PARTIAL_OVERLAP_SHORT_THRESHOLD = 2.0  # Threshold for partial overlaps
SIGNIFICANT_OVERLAP_THRESHOLD = 0.6  # Threshold for significant overlap duration

# Enable/disable overlap resolution strategies
ENABLE_SHORT_SEGMENT_FILTERING = True  # Enable short segment filtering
ENABLE_FULL_OVERLAP_RESOLUTION = True  # Enable full overlap resolution
ENABLE_PARTIAL_OVERLAP_HANDLING = True  # Enable partial overlap handling
ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT = True  # Enable intelligent segment assignment

# Enable/disable detailed transcript generation
ENABLE_DETAILED_TRANSCRIPT = True  # Enable detailed transcript with word-level timestamps

# Helper function to run FFmpeg commands
def run_ffmpeg(input_file, output_file, options=None):
    """Run FFmpeg command to convert/repair audio file"""
    cmd = ["ffmpeg", "-y", "-i", input_file]
    
    if options:
        cmd.extend(options)
        
    cmd.append(output_file)
    
    print(f"Running FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False
    return True

def seconds_to_ticks(seconds):
    """Convert seconds to ticks (1 tick = 100 nanoseconds)"""
    return int(seconds * 10_000_000)

def create_lexical_text(display_text):
    """Convert display text to lexical format (lowercase, no punctuation)"""
    import re
    # Remove punctuation and convert to lowercase
    lexical = re.sub(r'[^\w\s]', '', display_text.lower())
    # Remove extra whitespace
    lexical = ' '.join(lexical.split())
    return lexical

def generate_detailed_transcript_segment(segment_data, speaker_assignments, segment_id=None):
    """
    Generate a detailed transcript segment in the target JSON format
    
    Args:
        segment_data: Whisper segment with word-level timestamps
        speaker_assignments: Dict mapping segment indices to speaker IDs
        segment_id: Optional segment ID, will generate UUID if not provided
    
    Returns:
        Dict in the target JSON format
    """
    if segment_id is None:
        segment_id = str(uuid.uuid4()).replace('-', '')
    
    # Get the full text and timing
    display_text = segment_data.get('text', '').strip()
    segment_start = segment_data.get('start', 0)
    segment_end = segment_data.get('end', 0)
    segment_duration = segment_end - segment_start
    
    # Get speaker ID from assignments
    speaker_id = speaker_assignments.get(segment_data.get('segment_index', 0), 'SPEAKER_UNKNOWN')
    
    # Create lexical version
    lexical_text = create_lexical_text(display_text)
    
    # Process words if available
    words_data = []
    if 'words' in segment_data and segment_data['words']:
        for word_info in segment_data['words']:
            word_start = word_info.get('start', 0)
            word_end = word_info.get('end', 0)
            word_duration = word_end - word_start
            word_text = word_info.get('word', '').strip()
            
            words_data.append({
                "Duration": seconds_to_ticks(word_duration),
                "Offset": seconds_to_ticks(word_start),
                "Word": word_text
            })
    
    # Create the segment structure
    segment_result = {
        "Id": segment_id,
        "DisplayText": display_text,
        "Duration": seconds_to_ticks(segment_duration),
        "Offset": seconds_to_ticks(segment_start),
        "SpeakerId": speaker_id,
        "RecognitionStatus": "Success",
        "NBest": [
            {
                "Display": display_text,
                "Lexical": lexical_text,
                "Words": words_data
            }
        ]
    }
    
    return segment_result

try:
    # Try to read auth token from config file
    pyannote_auth_token = None
    config_file = "auth_config.txt"
    
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_line = f.read().strip()
            # Parse the config line to extract the token
            if "=" in config_line:
                key, value = config_line.split("=", 1)
                if key == "pyannote_auth_token":
                    pyannote_auth_token = value
    
    if not pyannote_auth_token:
        print(f"Error: pyannote_auth_token not found in {config_file}")
        print(f"Please create a file named {config_file} with your Hugging Face token in the format:")
        print("pyannote_auth_token=your_token_here")
        sys.exit(1)
    
    print("Loading pipeline...")
    # load the pipeline with your token
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=pyannote_auth_token)

    input_audio_path = "inputs/harvard.wav"
    
    # Convert audio file to WAV format using FFmpeg
    print("Converting audio file to WAV format with FFmpeg...")
    os.makedirs("temp", exist_ok=True)
    wav_path = os.path.join("temp", "audio_converted.wav")
    
    # Convert to WAV with optimal settings for Whisper
    if run_ffmpeg(input_audio_path, wav_path, ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]):
        audio_path = wav_path
        print(f"Audio converted to WAV: {wav_path}")
    else:
        print("FFmpeg conversion failed. Trying to use the original file.")
        audio_path = input_audio_path

    print("Running speaker diarization...")
    diarization = pipeline(audio_path)

    # Save diarization RTTM output
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/diarization.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
        
    # Add timestamp output in a more readable format with durations
    print("Generating speaker timestamps...")
    timestamp_lines = []
    
    # Store speaker turns for later use and convert to DataFrame for easier processing
    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time
        timestamp_lines.append(f"Speaker {speaker}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
        diarization_segments.append({
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "duration": duration
        })
    
    with open("outputs/speaker_timestamps.txt", "w") as f:
        for line in timestamp_lines:
            f.write(line + "\n")

    # Convert to DataFrame for easier processing
    diar_df = pd.DataFrame(diarization_segments)
    # Sort by start time
    diar_df = diar_df.sort_values(by=['start']).reset_index(drop=True)
    
    print("Loading Whisper model...")
    # Use the smallest model to reduce memory usage
    model = whisper.load_model("small", device=device)
    
    # Load audio 
    print("Processing audio...")
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000  # in seconds
    
    # Process in chunks of 10 seconds (smaller chunks)
    CHUNK_SIZE = 10 * 1000  # 10 seconds in milliseconds
    all_segments = []
    all_segments_with_words = []  # Store segments with word-level data for detailed transcript
    
    print(f"Total audio duration: {audio_duration:.2f} seconds")
    print(f"Processing in chunks of {CHUNK_SIZE/1000:.0f} seconds")
    
    chunk_count = int(np.ceil(len(audio) / CHUNK_SIZE))
    
    for i in range(chunk_count):
        print(f"Processing chunk {i+1}/{chunk_count}...")
        start_ms = i * CHUNK_SIZE
        end_ms = min((i + 1) * CHUNK_SIZE, len(audio))
        
        chunk = audio[start_ms:end_ms]
        
        # Save chunk to temporary file (still using WAV for chunks as it's more reliable)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            chunk.export(temp_path, format="wav")
        
        try:
            # Transcribe the chunk - enable word timestamps for detailed output
            result = model.transcribe(
                temp_path,
                word_timestamps=True,  # Enable word timestamps for detailed transcript generation
                verbose=False,
                fp16=False
            )
            
            # Adjust timestamps to account for chunk position
            for idx, segment in enumerate(result["segments"]):
                segment["start"] += start_ms / 1000
                segment["end"] += start_ms / 1000
                segment["segment_index"] = len(all_segments) + idx  # Add global segment index
                
                # Adjust word timestamps if they exist
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        word["start"] += start_ms / 1000
                        word["end"] += start_ms / 1000
                
            all_segments.extend(result["segments"])
            
            # Store segments with word data for detailed transcript generation
            if ENABLE_DETAILED_TRANSCRIPT:
                all_segments_with_words.extend(result["segments"])
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        # Force garbage collection to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine all results
    combined_result = {"segments": all_segments}
    
    # Save Whisper result as JSON
    with open("outputs/whisper_result.json", "w") as f:
        json.dump(combined_result, f, indent=2)

    # Sort segments by start time for correct processing
    all_segments.sort(key=lambda x: x["start"])
    
    # Process overlaps using the GitHub recommendation approach
    print("Processing speaker overlaps...")
    
    # Convert diarization DataFrame to detect overlaps
    # Add a column to track if we should keep the segment
    diar_df['keep'] = True
    diar_df['reason'] = None

    # Check for overlaps and mark segments based on GitHub recommendation
    for i in range(len(diar_df)):
        # Skip segments already marked for unassignment
        if not diar_df.at[i, 'keep']:
            continue
        
        # First filter: short segments under general threshold
        if ENABLE_SHORT_SEGMENT_FILTERING and diar_df.at[i, 'duration'] < SHORT_SEGMENT_THRESHOLD:
            diar_df.at[i, 'keep'] = False
            diar_df.at[i, 'reason'] = "short_segment"
            continue
        
        # Look for overlaps with other segments
        current_start = diar_df.at[i, 'start']
        current_end = diar_df.at[i, 'end']
        
        # Check for overlaps with subsequent segments
        for j in range(i+1, len(diar_df)):
            # Skip segments already marked for unassignment
            if not diar_df.at[j, 'keep']:
                continue
                
            next_start = diar_df.at[j, 'start']
            next_end = diar_df.at[j, 'end']
            
            # No overlap if the next segment starts after current ends
            if next_start >= current_end:
                continue
                
            # Check for full overlap (one segment completely within another)
            is_i_inside_j = (current_start >= next_start and current_end <= next_end)
            is_j_inside_i = (next_start >= current_start and next_end <= current_end)
            
            if ENABLE_FULL_OVERLAP_RESOLUTION and (is_i_inside_j or is_j_inside_i):
                # Full overlap case
                
                # Get the shorter segment
                shorter_idx = i if diar_df.at[i, 'duration'] < diar_df.at[j, 'duration'] else j
                longer_idx = j if shorter_idx == i else i
                
                # If the shorter segment is less than threshold, mark for unassignment
                if diar_df.at[shorter_idx, 'duration'] < FULL_OVERLAP_SHORT_THRESHOLD:
                    diar_df.at[shorter_idx, 'keep'] = False
                    diar_df.at[shorter_idx, 'reason'] = "short_full_overlap"
                else:
                    # Longer full overlaps: divide the longer segment
                    # Just mark the overlap part in the longer segment
                    # This is a simplification - in a full implementation, we would actually
                    # split the longer segment into two parts
                    
                    # Mark for logical processing of the overlap later
                    if longer_idx == i:
                        diar_df.at[i, 'end'] = diar_df.at[j, 'start']
                        diar_df.at[i, 'reason'] = "adjusted_for_full_overlap"
                    else:
                        diar_df.at[j, 'start'] = diar_df.at[i, 'end']
                        diar_df.at[j, 'reason'] = "adjusted_for_full_overlap"
            elif ENABLE_PARTIAL_OVERLAP_HANDLING and not (is_i_inside_j or is_j_inside_i):
                # Partial overlap case
                overlap_start = max(current_start, next_start)
                overlap_end = min(current_end, next_end)
                overlap_duration = overlap_end - overlap_start
                
                if (diar_df.at[i, 'duration'] < PARTIAL_OVERLAP_SHORT_THRESHOLD and 
                    overlap_duration > SIGNIFICANT_OVERLAP_THRESHOLD):
                    # Short segment with significant overlap
                    diar_df.at[i, 'keep'] = False
                    diar_df.at[i, 'reason'] = "short_significant_partial_overlap"
                elif (diar_df.at[j, 'duration'] < PARTIAL_OVERLAP_SHORT_THRESHOLD and 
                      overlap_duration > SIGNIFICANT_OVERLAP_THRESHOLD):
                    # Next segment is short with significant overlap
                    diar_df.at[j, 'keep'] = False
                    diar_df.at[j, 'reason'] = "short_significant_partial_overlap"
                else:
                    # Longer partial overlap - adjust end of first segment
                    if i < j:  # Ensure the first segment is modified
                        diar_df.at[i, 'end'] = next_start
                        diar_df.at[i, 'reason'] = "adjusted_for_partial_overlap"
    
    # Dictionary to store which transcript segments are assigned
    used_segments = set()
    
    # Dictionary to store speaker assignments for detailed transcript
    segment_speaker_assignments = {}
    
    # Process transcription segments for output
    output_lines = []
    unassigned_segments = []
    
    # Apply the diarization to the transcription segments
    # Only use segments marked 'keep'
    keep_df = diar_df[diar_df['keep']].copy()
    keep_df = keep_df.sort_values(by=['start']).reset_index(drop=True)
    
    for _, row in keep_df.iterrows():
        speaker = row['speaker']
        start_time = row['start']
        end_time = row['end']
        
        # Find matching transcript segments
        segment_texts = []
        for i, seg in enumerate(all_segments):
            if i in used_segments:
                continue  # Skip already used segments
                
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Calculate overlap
            overlap_start = max(start_time, seg_start)
            overlap_end = min(end_time, seg_end)
            
            if overlap_end > overlap_start:
                # There is some overlap
                overlap_duration = overlap_end - overlap_start
                segment_duration = seg_end - seg_start
                
                # Use intelligent segment assignment if enabled, otherwise use any overlap
                if ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT:
                    # Use it if at least 30% of the segment overlaps with this turn
                    if overlap_duration > 0.3 * segment_duration:
                            segment_texts.append(seg["text"].strip())
                            used_segments.add(i)
                            # Store speaker assignment for detailed transcript
                            segment_speaker_assignments[seg.get("segment_index", i)] = speaker
                else:
                    # Use any overlap, no matter how small
                    segment_texts.append(seg["text"].strip())
                    used_segments.add(i)
                    # Store speaker assignment for detailed transcript
                    segment_speaker_assignments[seg.get("segment_index", i)] = speaker
        
        combined_text = " ".join(segment_texts)
        if combined_text.strip():
            note = f" [{row['reason']}]" if row['reason'] else ""
            output_lines.append(f"Speaker {speaker} | {start_time:.2f} - {end_time:.2f} | {combined_text}{note}")
    
    # Record unassigned diarization segments with reasons
    for _, row in diar_df[~diar_df['keep']].iterrows():
        speaker = row['speaker']
        start_time = row['start']
        end_time = row['end']
        reason = row['reason'] if row['reason'] else "unknown"
        unassigned_segments.append(f"Unassigned Speaker {speaker} | {start_time:.2f} - {end_time:.2f} | [{reason}]")
    
    # Include remaining transcript segments that weren't assigned
    for i, seg in enumerate(all_segments):
        if i not in used_segments and seg["text"].strip():
            unassigned_segments.append(f"Unassigned Transcript | {seg['start']:.2f} - {seg['end']:.2f} | {seg['text'].strip()} [no_matching_speaker]")
    
    # Sort output lines by start time
    output_lines.sort(key=lambda x: float(x.split(" | ")[1].split(" - ")[0]))
    
    # Save final transcript
    print("Saving final transcript...")
    with open("outputs/speaker_transcript.txt", "w") as f:
        f.write("# Overlap resolution settings:\n")
        f.write(f"# - Short Segment Filtering: {'Enabled' if ENABLE_SHORT_SEGMENT_FILTERING else 'Disabled'}\n")
        f.write(f"# - Full Overlap Resolution: {'Enabled' if ENABLE_FULL_OVERLAP_RESOLUTION else 'Disabled'}\n")
        f.write(f"# - Partial Overlap Handling: {'Enabled' if ENABLE_PARTIAL_OVERLAP_HANDLING else 'Disabled'}\n")
        f.write(f"# - Intelligent Segment Assignment: {'Enabled' if ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT else 'Disabled'}\n")
        f.write(f"# - Detailed Transcript Generation: {'Enabled' if ENABLE_DETAILED_TRANSCRIPT else 'Disabled'}\n\n")
        
        for line in output_lines:
            f.write(line + "\n")
        
        if unassigned_segments:
            f.write("\n# Unassigned speech segments:\n")
            for line in unassigned_segments:
                f.write(line + "\n")
    
    # Generate detailed transcript if enabled
    if ENABLE_DETAILED_TRANSCRIPT:
        print("Generating detailed transcript with word-level timestamps...")
        
        detailed_transcript_results = []
        
        for segment in all_segments_with_words:
            if segment.get("text", "").strip():  # Only process non-empty segments
                detailed_segment = generate_detailed_transcript_segment(
                    segment, 
                    segment_speaker_assignments
                )
                detailed_transcript_results.append(detailed_segment)
        
        # Create the final detailed transcript structure
        detailed_transcript = {
            "Result": detailed_transcript_results
        }
        
        # Save detailed transcript
        with open("outputs/detailed_transcript.json", "w") as f:
            json.dump(detailed_transcript, f, indent=2)
        
        print(f"Detailed transcript saved with {len(detailed_transcript_results)} segments")
            
    print("Processing completed successfully!")
    
    # Clean up temporary files
    if os.path.exists(wav_path):
        os.remove(wav_path)

except Exception as e:
    print(f"An error occurred: {str(e)}", file=sys.stderr)
    sys.exit(1)
