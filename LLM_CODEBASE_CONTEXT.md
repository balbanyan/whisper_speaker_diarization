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

# Output format toggles
ENABLE_DETAILED_TRANSCRIPT = True       # Enable detailed transcript with word-level timestamps

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
├── test_speaker_diarization.py # Unit tests for core functionality
├── test_current_system.py      # Integration tests for system verification
├── run_tests.py                # Test runner script
├── TESTING.md                  # Testing documentation and guidelines
├── inputs/                     # Audio files for processing
├── outputs/                    # Generated results
│   ├── diarization.rttm        # Raw diarization output
│   ├── speaker_timestamps.txt  # Readable timestamps
│   ├── whisper_result.json     # Complete transcription
│   ├── speaker_transcript.txt  # Final aligned transcript
│   └── detailed_transcript.json # Detailed transcript with word-level timestamps
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
- Word timestamps enabled for detailed transcript generation
- Adjusts timestamps to account for chunk position
- Output: List of segments with text and word-level timestamps

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
- Generates detailed transcript with word-level timestamps (if enabled)

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
    'text': 'This is what was said.',
    'words': [
        {'start': 10.2, 'end': 10.8, 'word': 'This'},
        {'start': 10.9, 'end': 11.2, 'word': 'is'},
        {'start': 11.3, 'end': 11.6, 'word': 'what'},
        {'start': 11.7, 'end': 12.0, 'word': 'was'},
        {'start': 12.1, 'end': 12.5, 'word': 'said.'}
    ]
}
```

### Detailed Transcript Segments
```python
{
    "Id": "ea4798edfdae4c698195f1ffe83ba928",
    "DisplayText": "This is what was said.",
    "Duration": 58000000,  # Duration in ticks (100ns units)
    "Offset": 102000000,   # Start time in ticks
    "SpeakerId": "SPEAKER_00",
    "RecognitionStatus": "Success",
    "NBest": [
        {
            "Display": "This is what was said.",
            "Lexical": "this is what was said",
            "Words": [
                {
                    "Duration": 6000000,
                    "Offset": 102000000,
                    "Word": "This"
                },
                # ... more words
            ]
        }
    ]
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

### Detailed Transcript Output (JSON)
```json
{
    "Result": [
        {
            "Id": "ea4798edfdae4c698195f1ffe83ba928",
            "DisplayText": "Hello, how are you today?",
            "Duration": 25000000,
            "Offset": 5000000,
            "SpeakerId": "SPEAKER_00",
            "RecognitionStatus": "Success",
            "NBest": [
                {
                    "Display": "Hello, how are you today?",
                    "Lexical": "hello how are you today",
                    "Words": [
                        {
                            "Duration": 5000000,
                            "Offset": 5000000,
                            "Word": "Hello"
                        }
                    ]
                }
            ]
        }
    ]
}
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

### Unit Tests (`test_speaker_diarization.py`)
Comprehensive test suite covering:
- **Overlap Resolution**: Short segment filtering, full/partial overlap detection
- **Speaker Assignment**: Overlap calculation, 30% threshold logic
- **Configuration**: Parameter validation, strategy toggles
- **Audio Processing**: Chunking logic, timestamp adjustment
- **Output Generation**: Line parsing, timestamp sorting
- **Error Handling**: Empty data, invalid inputs
- **Memory Management**: CUDA availability, chunk size validation
- **Integration**: File creation, JSON structure validation

### Integration Tests (`test_current_system.py`)
System-level verification:
- Audio file creation and processing infrastructure
- Configuration parameter accessibility
- Expected output file structure
- Test environment setup and cleanup

### Experimental Testing
The toggleable overlap strategies allow systematic testing:
1. Run with all strategies enabled (baseline)
2. Disable one strategy at a time to measure impact
3. Compare results to identify optimal combination for specific audio types
4. Use output annotations to understand what changes were made

### Running Tests
```bash
# Run all tests (recommended)
python run_tests.py

# Or run individual test suites:
# Run unit tests
python test_speaker_diarization.py

# Run integration tests  
python test_current_system.py

# All test suites should show 100% success rate before making changes
```

## Change Tracking

This section tracks modifications made to the codebase and the reasoning behind them:

### May 13, 2025 (Latest)
- **Implemented detailed transcript generation**: Added word-level timestamp support with JSON output format
- **Added new configuration flag**: `ENABLE_DETAILED_TRANSCRIPT` to toggle detailed transcript generation
- **Enhanced Whisper processing**: Enabled word-level timestamps for detailed output generation
- **Created helper functions**: `seconds_to_ticks()`, `create_lexical_text()`, `generate_detailed_transcript_segment()`
- **Added new output file**: `outputs/detailed_transcript.json` with structured JSON format
- **Updated test suite**: Added 6 new tests for detailed transcript functionality (25 total tests)
- **Enhanced documentation**: Updated context file with detailed transcript structure and examples

### May 13, 2025 (Earlier)
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

## Implemented Features

1. **Detailed Transcript Output**: ✅ **COMPLETED**
   - **DisplayText**: Direct Whisper output with punctuation/capitalization
   - **Lexical**: Lowercase, no punctuation version (implemented)
   - **Word-level timestamps**: Full word timing in ticks format
   - **Speaker integration**: Uses existing overlap resolution logic
   - **JSON structure**: Compatible format with structured output
   - **Toggleable**: Can be enabled/disabled via `ENABLE_DETAILED_TRANSCRIPT`

## Planned Features (To Be Added Later)

1. **Enhanced Text Normalization**:
   - **ITN Support**: Inverse Text Normalization for English/Arabic (convert spoken numbers to written form)
   - **Advanced formatting**: "twenty two" → "22", "percent" → "%"
   - **Arabic numerals**: Proper handling of Arabic numerals and spoken forms
   - **Currency, dates, abbreviations**: For both English and Arabic

2. **Enhanced Mixed Language Processing**: 
   - Currently Whisper processes segments by dominant language
   - Future: Word-level language detection and appropriate processing
   - Challenge: Arabic words may be transliterated as English phonetics and vice versa

3. **Confidence Scores**: If/when available from Whisper or alternative libraries
   - Would require external libraries like `whisper-timestamped`
   - Currently excluded to maintain simplicity

4. **NBest Alternatives**: Multiple transcription hypotheses (when Whisper supports it)