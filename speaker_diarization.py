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

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration parameters
# Adjust these values to control how strictly segments are assigned
OVERLAP_THRESHOLD = 0.3  # Minimum overlap required (30% instead of 50% to assign more segments)
FORCE_ASSIGN_ALL = True  # If True, all unassigned segments will be assigned to the closest speaker

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
    
    # Store speaker turns for later use
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time
        timestamp_lines.append(f"Speaker {speaker}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
        speaker_turns.append((start_time, end_time, speaker))
    
    with open("outputs/speaker_timestamps.txt", "w") as f:
        for line in timestamp_lines:
            f.write(line + "\n")

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

    # Create a more accurate speech segments map with word boundaries
    print("Creating improved alignment between diarization and transcription...")
    
    # Sort segments by start time for correct processing
    all_segments.sort(key=lambda x: x["start"])
    
    # Dictionary to track which text has been assigned to avoid repetition
    used_segments = set()
    
    # Improved alignment algorithm to prevent repetition
    output_lines = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        
        # Filter segments that significantly overlap with this turn
        # A segment is considered to belong to a turn if at least the overlap threshold is met
        turn_segments = []
        for i, seg in enumerate(all_segments):
            if i in used_segments:
                continue  # Skip already used segments
                
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Calculate overlap
            overlap_start = max(start_time, seg_start)
            overlap_end = min(end_time, seg_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                segment_duration = seg_end - seg_start
                
                # If enough of segment overlaps with this turn, use it
                if overlap_duration > OVERLAP_THRESHOLD * segment_duration:
                    turn_segments.append(seg)
                    used_segments.add(i)  # Mark this segment as used
        
        # Sort segments by start time within this turn
        turn_segments.sort(key=lambda x: x["start"])
        
        # Extract text from segments
        segment_texts = [seg["text"].strip() for seg in turn_segments]
        combined_text = " ".join(segment_texts)
        
        # Only add non-empty lines
        if combined_text.strip():
            output_lines.append(f"Speaker {speaker} | {start_time:.2f} - {end_time:.2f} | {combined_text}")

    # For any remaining segments that weren't assigned
    remaining_segments = []
    for i, seg in enumerate(all_segments):
        if i not in used_segments and seg["text"].strip():
            remaining_segments.append((i, seg))
    
    # Handle unassigned segments
    if remaining_segments and FORCE_ASSIGN_ALL:
        print(f"Assigning {len(remaining_segments)} remaining segments to nearest speakers...")
        for i, seg in remaining_segments:
            seg_mid = (seg["start"] + seg["end"]) / 2  # Middle point of segment
            
            # Find the closest speaker turn
            min_distance = float('inf')
            closest_speaker = None
            
            for start, end, speaker in speaker_turns:
                # Check if segment is inside a turn
                if start <= seg_mid <= end:
                    closest_speaker = speaker
                    break
                    
                # Calculate distance to turn boundaries
                dist_to_start = abs(seg_mid - start)
                dist_to_end = abs(seg_mid - end)
                min_dist_to_turn = min(dist_to_start, dist_to_end)
                
                if min_dist_to_turn < min_distance:
                    min_distance = min_dist_to_turn
                    closest_speaker = speaker
            
            if closest_speaker:
                output_lines.append(f"Speaker {closest_speaker} | {seg['start']:.2f} - {seg['end']:.2f} | {seg['text'].strip()} [reassigned]")
                used_segments.add(i)
    
    # Any segments still unassigned
    unassigned_texts = []
    for i, seg in enumerate(all_segments):
        if i not in used_segments and seg["text"].strip():
            unassigned_texts.append(f"Unassigned | {seg['start']:.2f} - {seg['end']:.2f} | {seg['text'].strip()}")
    
    # Sort output lines by start time for chronological order
    output_lines.sort(key=lambda x: float(x.split(" | ")[1].split(" - ")[0]))
    
    print("Saving final transcript...")
    with open("outputs/speaker_transcript.txt", "w") as f:
        for line in output_lines:
            f.write(line + "\n")
        
        # Add any still unassigned segments at the end
        if unassigned_texts:
            f.write("\n# Unassigned speech segments:\n")
            for line in unassigned_texts:
                f.write(line + "\n")
            
    print("Processing completed successfully!")
    
    # Clean up temporary files
    if os.path.exists(repaired_path):
        os.remove(repaired_path)

except Exception as e:
    print(f"An error occurred: {str(e)}", file=sys.stderr)
    sys.exit(1)
