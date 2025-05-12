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

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration parameters
# Short segment thresholds (in seconds)
SHORT_SEGMENT_THRESHOLD = 0.5  # General short segment threshold
FULL_OVERLAP_SHORT_THRESHOLD = 1.0  # Threshold for full overlaps
PARTIAL_OVERLAP_SHORT_THRESHOLD = 2.0  # Threshold for partial overlaps
SIGNIFICANT_OVERLAP_THRESHOLD = 0.6  # Threshold for significant overlap duration

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

    input_audio_path = "inputs/meeting_trimmed.mp3"
    
    # First, try to repair the MP3 file using FFmpeg
    print("Repairing audio file with FFmpeg...")
    os.makedirs("temp", exist_ok=True)
    repaired_path = os.path.join("temp", "audio_repaired.mp3")
    
    # Try to repair by re-encoding the MP3 (keeping MP3 format)
    if run_ffmpeg(input_audio_path, repaired_path, ["-c:a", "libmp3lame", "-q:a", "2"]):
        audio_path = repaired_path
        print(f"Audio repaired and re-encoded to: {repaired_path}")
    else:
        print("FFmpeg repair failed. Trying to use the original file.")
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
    model = whisper.load_model("tiny", device=device)
    
    # Load audio 
    print("Processing audio...")
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000  # in seconds
    
    # Process in chunks of 30 seconds (much smaller)
    CHUNK_SIZE = 30 * 1000  # 30 seconds in milliseconds
    all_segments = []
    
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
            # Transcribe the chunk - disable word timestamps to reduce complexity
            result = model.transcribe(
                temp_path,
                word_timestamps=False,  # Disable word timestamps to reduce complexity
                verbose=False,
                fp16=False
            )
            
            # Adjust timestamps to account for chunk position
            for segment in result["segments"]:
                segment["start"] += start_ms / 1000
                segment["end"] += start_ms / 1000
                
            all_segments.extend(result["segments"])
            
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
        if diar_df.at[i, 'duration'] < SHORT_SEGMENT_THRESHOLD:
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
            
            if is_i_inside_j or is_j_inside_i:
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
            else:
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
                
                # Use it if at least 30% of the segment overlaps with this turn
                if overlap_duration > 0.3 * segment_duration:
                    segment_texts.append(seg["text"].strip())
                    used_segments.add(i)
        
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
        for line in output_lines:
            f.write(line + "\n")
        
        if unassigned_segments:
            f.write("\n# Unassigned speech segments:\n")
            for line in unassigned_segments:
                f.write(line + "\n")
            
    print("Processing completed successfully!")
    
    # Clean up temporary files
    if os.path.exists(repaired_path):
        os.remove(repaired_path)

except Exception as e:
    print(f"An error occurred: {str(e)}", file=sys.stderr)
    sys.exit(1)
