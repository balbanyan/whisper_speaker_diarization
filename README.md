# Speaker Diarization with Pyannote and Whisper

This project combines Pyannote Audio for speaker diarization and OpenAI's Whisper for transcription to create accurate speaker-labeled transcripts from audio recordings.

## Features

- Speaker diarization using Pyannote Audio
- Transcription using OpenAI's Whisper
- Intelligent overlap handling for improved speaker attribution
- Advanced overlap resolution strategies based on segment duration
- Chunk-based processing to avoid memory issues
- Automatic audio repair for corrupted MP3 files

## Advanced Overlap Handling

The speaker diarization implements sophisticated overlap resolution strategies:

- **Short Segment Filtering**: Automatically filters out segments shorter than a configurable threshold
- **Full Overlap Resolution**: When one segment is fully contained within another, the shorter segment is handled based on duration
- **Partial Overlap Handling**: Adjusts segment boundaries to eliminate overlaps while preserving speech content
- **Intelligent Segment Assignment**: Uses overlap percentage to assign transcription to the most likely speaker

### Overlap Resolution Logic

The script implements a multi-step approach to handle the complex problem of overlapping speech:

1. **Initial Filtering**:
   - Very short segments (< 0.5s by default) are filtered out, as they often represent noise or artifacts
   - Each remaining segment is evaluated for potential overlaps with other segments

2. **Full Overlap Handling**:
   - When segment A is completely contained within segment B:
     - If the shorter segment is below threshold (< 1.0s), it's removed
     - For longer overlaps, the algorithm adjusts the longer segment's boundaries to avoid the overlap
     - This handles cases where one speaker briefly interjects during another's speech

3. **Partial Overlap Processing**:
   - When two segments partially overlap:
     - If a short segment (< 2.0s) has significant overlap (> 60%), it's removed
     - Otherwise, the boundary is adjusted to split the speech at the overlap point
     - Earlier segment's end time is adjusted to the start of the later segment

4. **Tracking and Annotation**:
   - Each segment maintains a "reason" field that tracks what adjustments were made
   - Filtered segments are moved to an "Unassigned" section with their removal reason
   - Final transcript includes annotations showing which segments were adjusted

5. **Transcript Assignment**:
   - After overlap resolution, transcribed text is assigned to speaker segments
   - Assignment requires at least 30% overlap between a transcript segment and speaker turn
   - Each transcript segment is assigned to only one speaker to prevent duplication

This approach significantly improves transcript readability by eliminating the confusion caused by overlapping speech segments, while preserving all meaningful content.

## Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install pyannote.audio openai-whisper torch pydub numpy pandas
   ```

3. **Hugging Face Authentication**:
   - Get access to the pyannote/speaker-diarization model: [Hugging Face Model](https://huggingface.co/pyannote/speaker-diarization)
   - Create an auth_config.txt file with your token:
     ```
     pyannote_auth_token=your_token_here
     ```

4. **Prepare audio**:
   - Place your audio file in the `inputs/` directory
   - Default filename is "meeting_trimmed.mp3" (modify in script if needed)

## Configuration

The script includes several configurable parameters:

```python
# Short segment thresholds (in seconds)
SHORT_SEGMENT_THRESHOLD = 0.5        # General short segment threshold
FULL_OVERLAP_SHORT_THRESHOLD = 1.0   # Threshold for full overlaps
PARTIAL_OVERLAP_SHORT_THRESHOLD = 2.0 # Threshold for partial overlaps
SIGNIFICANT_OVERLAP_THRESHOLD = 0.6   # Threshold for significant overlap duration
```

Adjust these values to fine-tune the overlap handling for your specific audio content:

- **Lower SHORT_SEGMENT_THRESHOLD**: To keep more very brief utterances (e.g., "yes", "uh-huh")
- **Higher FULL_OVERLAP_SHORT_THRESHOLD**: To be more aggressive in removing contained segments
- **Lower SIGNIFICANT_OVERLAP_THRESHOLD**: To require less overlap before considering it significant

## Usage

Run the script:
```bash
python speaker_diarization.py
```

## Output

The script produces several output files in the `outputs/` directory:

- `diarization.rttm`: Raw diarization output in RTTM format
- `speaker_timestamps.txt`: Speaker timestamps with durations in a readable format
- `whisper_result.json`: Complete transcript with timestamps
- `speaker_transcript.txt`: Final transcript with speaker labels, including annotations for adjusted segments

### Example Output

```
Speaker SPEAKER_00 | 10.25 - 15.64 | I think we should proceed with the new project plan.
Speaker SPEAKER_01 | 15.64 - 22.31 | I agree, but we need to address the budget concerns first. [adjusted_for_partial_overlap]
Speaker SPEAKER_02 | 22.31 - 25.18 | Has finance approved the initial allocation?

# Unassigned speech segments:
Unassigned Speaker SPEAKER_00 | 21.89 - 22.15 | [short_significant_partial_overlap]
```

## Troubleshooting

If you encounter memory issues:
- Try reducing the chunk size in the script
- Use a smaller Whisper model (tiny, base, etc.)
- Ensure FFmpeg is installed for audio processing

## Requirements

- Python 3.7+
- FFmpeg (required by pydub for audio processing)
- CUDA-compatible GPU recommended but not required 