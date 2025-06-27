# Speaker Diarization Codebase Context

## Project Overview

This project combines **Pyannote Audio** for speaker diarization with **OpenAI's Whisper** for transcription to create accurate speaker-labeled transcripts from audio recordings. The system addresses the complex problem of overlapping speech through sophisticated overlap resolution strategies.

## Core Goals

1. **Accurate Speaker Attribution**: Correctly identify who said what in multi-speaker audio
2. **Overlap Resolution**: Handle cases where speakers talk over each other
3. **Flexible Processing**: Allow toggling of different processing strategies for experimentation
4. **Memory Efficiency**: Process long audio files without running out of memory
5. **Transparency**: Provide clear annotations showing what adjustments were made

## Architecture Overview

### Two-Stage Processing
1. **Speaker Diarization** (Pyannote): Processes entire audio file to identify speaker segments
2. **Transcription** (Whisper): Processes audio in 30-second chunks for memory efficiency
3. **Alignment**: Combines diarization and transcription results using overlap analysis

### Key Design Decisions

- **Pyannote processes full audio**: Needs complete context for speaker pattern recognition
- **Whisper uses chunking**: 30-second chunks to manage memory usage
- **Segment-based alignment**: Uses 30% overlap threshold for transcript assignment
- **Preserves all content**: Unassigned segments are tracked, not lost

## Overlap Resolution Strategies (All Toggleable)

### 1. Short Segment Filtering
- **Purpose**: Remove noise and artifacts
- **Threshold**: 0.5 seconds (configurable)
- **Toggle**: `ENABLE_SHORT_SEGMENT_FILTERING`
- **Reason Code**: `short_segment`

### 2. Full Overlap Resolution
- **Purpose**: Handle when one speaker is completely contained within another's segment
- **Logic**:
  - If shorter segment < 1.0s: Remove it (`short_full_overlap`)
  - If longer: Adjust boundaries (`adjusted_for_full_overlap`)
- **Toggle**: `ENABLE_FULL_OVERLAP_RESOLUTION`

### 3. Partial Overlap Handling
- **Purpose**: Resolve partial overlaps between speakers
- **Logic**:
  - Short segment (< 2.0s) with significant overlap (> 60%): Remove (`short_significant_partial_overlap`)
  - Otherwise: Adjust earlier segment's end time (`adjusted_for_partial_overlap`)
- **Toggle**: `ENABLE_PARTIAL_OVERLAP_HANDLING`

### 4. Intelligent Segment Assignment
- **Purpose**: Assign transcript text to most likely speaker
- **Logic**:
  - If enabled: Requires 30% overlap between transcript and speaker segment
  - If disabled: Uses any overlap, no matter how small
- **Toggle**: `ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT`
- **Unassigned reason**: `no_matching_speaker`

## Configuration Parameters

```python
# Thresholds (in seconds)
SHORT_SEGMENT_THRESHOLD = 0.5           # General short segment threshold
FULL_OVERLAP_SHORT_THRESHOLD = 1.0      # Threshold for full overlaps
PARTIAL_OVERLAP_SHORT_THRESHOLD = 2.0   # Threshold for partial overlaps
SIGNIFICANT_OVERLAP_THRESHOLD = 0.6     # Threshold for significant overlap duration

# Strategy toggles
ENABLE_SHORT_SEGMENT_FILTERING = True
ENABLE_FULL_OVERLAP_RESOLUTION = True
ENABLE_PARTIAL_OVERLAP_HANDLING = True
ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT = True

# Processing settings
CHUNK_SIZE = 30 * 1000  # 30 seconds in milliseconds for Whisper
```

## File Structure

```
speaker-diarization/
├── speaker_diarization.py      # Main processing script
├── README.md                   # User documentation
├── LLM_CODEBASE_CONTEXT.md     # This context file for LLM reference
├── auth_config.txt             # Hugging Face token (gitignored)
├── inputs/                     # Audio files for processing
├── outputs/                    # Generated results
│   ├── diarization.rttm        # Raw diarization output
│   ├── speaker_timestamps.txt  # Readable timestamps
│   ├── whisper_result.json     # Complete transcription
│   └── speaker_transcript.txt  # Final aligned transcript
└── temp/                       # Temporary files during processing
```

## Processing Flow

### 1. Audio Preparation
- Loads audio file (default: `inputs/meeting_trimmed.mp3`)
- Attempts FFmpeg repair if needed
- Converts to appropriate format for processing

### 2. Speaker Diarization (Pyannote)
- Processes entire audio file at once
- Generates speaker segments with timestamps
- No chunking - needs full context for speaker identification
- Output: DataFrame with speaker, start, end, duration

### 3. Transcription (Whisper)
- Processes in 30-second chunks to manage memory
- Uses "tiny" model by default for speed
- Word timestamps disabled by default (but can be enabled)
- Adjusts timestamps to account for chunk position
- Output: List of segments with text and timestamps

### 4. Overlap Resolution
- Applies enabled strategies in order:
  1. Short segment filtering
  2. Full overlap resolution
  3. Partial overlap handling
- Tracks reasons for all modifications
- Maintains `keep` flag and `reason` field for each segment

### 5. Transcript Assignment
- Calculates overlap between transcript and speaker segments
- Uses 30% threshold (if intelligent assignment enabled)
- Assigns each transcript segment to only one speaker
- Tracks used segments to prevent duplication

### 6. Output Generation
- Creates final transcript with speaker labels
- Includes reason annotations for adjusted segments
- Lists unassigned segments with explanations
- Shows which strategies were enabled

## Key Data Structures

### Diarization DataFrame
```python
{
    'speaker': 'SPEAKER_00',
    'start': 10.5,
    'end': 15.2,
    'duration': 4.7,
    'keep': True,
    'reason': 'adjusted_for_partial_overlap'
}
```

### Whisper Segments
```python
{
    'start': 10.2,
    'end': 15.8,
    'text': 'This is what was said.'
}
```

## Output Format

### Main Transcript
```
Speaker SPEAKER_00 | 10.25 - 15.64 | I think we should proceed. [adjusted_for_partial_overlap]
Speaker SPEAKER_01 | 15.64 - 22.31 | I agree with that approach.
```

### Unassigned Segments
```
# Unassigned speech segments:
Unassigned Speaker SPEAKER_02 | 21.89 - 22.15 | [short_significant_partial_overlap]
Unassigned Transcript | 52.32 - 54.32 | Oh, yeah. [no_matching_speaker]
```

## Memory Management

- **Chunking**: Whisper processes 30-second chunks to avoid memory issues
- **Cleanup**: Temporary files deleted after each chunk
- **CUDA Management**: `torch.cuda.empty_cache()` called after each chunk
- **Garbage Collection**: Forced after processing chunks

## Authentication

- Uses Hugging Face token for Pyannote model access
- Token stored in `auth_config.txt` (format: `pyannote_auth_token=your_token`)
- File is gitignored for security

## Known Limitations

1. **Speaker Limit**: No hard limit, but accuracy decreases with more speakers
2. **Single Speaker Once**: If someone speaks only once, higher chance of misclassification
3. **Audio Quality**: Performance depends on clear audio with minimal background noise
4. **Real-time**: Not optimized for real-time processing

## Experimental Features

- **Word-level timestamps**: Can be enabled in Whisper but currently disabled for performance
- **Different Whisper models**: Can use larger models (base, small, medium, large) for better accuracy
- **Chunk size adjustment**: Can modify 30-second chunks for different memory/accuracy tradeoffs

## Testing Strategy

The toggleable overlap strategies allow systematic testing:
1. Run with all strategies enabled (baseline)
2. Disable one strategy at a time to measure impact
3. Compare results to identify optimal combination for specific audio types
4. Use output annotations to understand what changes were made

## Change Tracking

This section tracks modifications made to the codebase and the reasoning behind them:

### May 13, 2025
- **Added toggleable overlap resolution strategies**: Implemented configuration flags for all 4 overlap strategies to allow experimental testing
- **Enhanced documentation**: Updated README.md with toggleable strategy information
- **Fixed indentation error**: Resolved Python syntax error in speaker_diarization.py around line 317-319
- **Created context documentation**: Established this LLM_CODEBASE_CONTEXT.md file for tracking project state and changes

### Future Change Template
```
### [Date]
- **[Change Type]**: [Description of what was changed]
- **Reason**: [Why the change was made]
- **Impact**: [How it affects the system]
- **Files Modified**: [List of files changed]
```

## Recent Changes

- **May 13, 2025**: Added toggleable overlap resolution strategies
- **Previous**: Implemented advanced overlap handling with improved documentation
- All strategies can now be individually enabled/disabled via configuration flags
- Output includes strategy status information

## Future Considerations

1. **Speaker Verification**: Add confidence scoring for speaker assignments
2. **Real-time Processing**: Optimize for streaming audio
3. **Advanced Overlap Handling**: More sophisticated boundary adjustment algorithms 
4. **Make code more modular**: Optimize codebase to be more flexible and readable
5. **Add word-level timestamps**: Add word-level timestamps with whipser