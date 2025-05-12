# Speaker Diarization with Pyannote and Whisper

This project combines Pyannote Audio for speaker diarization and OpenAI's Whisper for transcription to create accurate speaker-labeled transcripts from audio recordings.

## Features

- Speaker diarization using Pyannote Audio
- Transcription using OpenAI's Whisper
- Alignment of diarization with transcription
- Chunk-based processing to avoid memory issues

## Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install pyannote.audio openai-whisper torch pydub numpy
   ```

3. **Hugging Face Authentication**:
   - Get access to the pyannote/speaker-diarization model: [Hugging Face Model](https://huggingface.co/pyannote/speaker-diarization)
   - Create an auth_config.txt file with your token:
     ```
     pyannote_auth_token=your_token_here
     ```

4. **Prepare audio**:
   - Place your audio file in the `inputs/` directory
   - Default filename is `debate.mp3` (modify in script if needed)

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
- `debate_speaker_transcript.txt`: Final transcript with speaker labels

## Troubleshooting

If you encounter memory issues:
- Try reducing the chunk size in the script
- Use a smaller Whisper model (tiny, base, etc.)
- Ensure FFmpeg is installed for audio processing

## Requirements

- Python 3.7+
- FFmpeg (required by pydub for audio processing)
- CUDA-compatible GPU recommended but not required 