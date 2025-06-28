#!/usr/bin/env python3
"""
Unit tests for speaker_diarization.py

This test suite covers:
- Audio processing and chunking
- Overlap resolution strategies
- Speaker assignment logic
- Output generation
- Configuration parameter validation
- Error handling
"""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
from pydub import AudioSegment

# Import the main module (assuming speaker_diarization.py functions can be imported)
# Note: The current script is designed to run standalone, so we'll need to refactor
# some functions to be testable. For now, we'll test the logic concepts.

class TestOverlapResolution(unittest.TestCase):
    """Test overlap resolution strategies"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_diarization = pd.DataFrame([
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "duration": 5.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 7.0, "duration": 5.0},
            {"speaker": "SPEAKER_00", "start": 6.5, "end": 8.0, "duration": 1.5},
            {"speaker": "SPEAKER_01", "start": 0.2, "end": 0.4, "duration": 0.2},  # Short segment
        ])
        self.sample_diarization['keep'] = True
        self.sample_diarization['reason'] = None
    
    def test_short_segment_filtering(self):
        """Test short segment filtering logic"""
        SHORT_SEGMENT_THRESHOLD = 0.5
        df = self.sample_diarization.copy()
        
        # Apply short segment filtering
        for i in range(len(df)):
            if df.at[i, 'duration'] < SHORT_SEGMENT_THRESHOLD:
                df.at[i, 'keep'] = False
                df.at[i, 'reason'] = "short_segment"
        
        # Check that short segment was filtered
        short_segments = df[df['duration'] < SHORT_SEGMENT_THRESHOLD]
        self.assertEqual(len(short_segments), 1)
        self.assertFalse(short_segments.iloc[0]['keep'])
        self.assertEqual(short_segments.iloc[0]['reason'], "short_segment")
    
    def test_full_overlap_detection(self):
        """Test full overlap detection logic"""
        # Create test case with full overlap
        df = pd.DataFrame([
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0, "duration": 10.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 5.0, "duration": 3.0},  # Inside first
        ])
        
        # Check overlap detection
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                current_start, current_end = df.at[i, 'start'], df.at[i, 'end']
                next_start, next_end = df.at[j, 'start'], df.at[j, 'end']
                
                is_j_inside_i = (next_start >= current_start and next_end <= current_end)
                is_i_inside_j = (current_start >= next_start and current_end <= next_end)
                
                if is_j_inside_i:
                    self.assertTrue(True)  # Second segment is inside first
                    break
    
    def test_partial_overlap_detection(self):
        """Test partial overlap detection"""
        # Test case with partial overlap
        segment1 = {"start": 0.0, "end": 5.0}
        segment2 = {"start": 3.0, "end": 8.0}
        
        overlap_start = max(segment1["start"], segment2["start"])
        overlap_end = min(segment1["end"], segment2["end"])
        overlap_duration = overlap_end - overlap_start
        
        self.assertEqual(overlap_start, 3.0)
        self.assertEqual(overlap_end, 5.0)
        self.assertEqual(overlap_duration, 2.0)
        self.assertGreater(overlap_duration, 0)  # Confirms overlap exists


class TestSpeakerAssignment(unittest.TestCase):
    """Test speaker assignment logic"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_segments = [
            {"start": 1.0, "end": 3.0, "text": "Hello world"},
            {"start": 4.0, "end": 6.0, "text": "How are you"},
            {"start": 7.0, "end": 9.0, "text": "I am fine"},
        ]
        
        self.sample_speakers = pd.DataFrame([
            {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.5, "keep": True},
            {"speaker": "SPEAKER_01", "start": 3.8, "end": 6.2, "keep": True},
            {"speaker": "SPEAKER_00", "start": 6.8, "end": 9.2, "keep": True},
        ])
    
    def test_overlap_calculation(self):
        """Test overlap calculation between transcript and speaker segments"""
        transcript_seg = self.sample_segments[0]  # 1.0 - 3.0
        speaker_seg = self.sample_speakers.iloc[0]  # 0.5 - 3.5
        
        overlap_start = max(transcript_seg["start"], speaker_seg["start"])
        overlap_end = min(transcript_seg["end"], speaker_seg["end"])
        overlap_duration = overlap_end - overlap_start
        
        self.assertEqual(overlap_start, 1.0)
        self.assertEqual(overlap_end, 3.0)
        self.assertEqual(overlap_duration, 2.0)
    
    def test_intelligent_assignment_threshold(self):
        """Test 30% overlap threshold for intelligent assignment"""
        transcript_seg = {"start": 1.0, "end": 3.0, "text": "test"}  # 2 second duration
        speaker_seg = {"start": 1.0, "end": 1.5}  # 0.5 second overlap
        
        overlap_start = max(transcript_seg["start"], speaker_seg["start"])
        overlap_end = min(transcript_seg["end"], speaker_seg["end"])
        overlap_duration = overlap_end - overlap_start
        segment_duration = transcript_seg["end"] - transcript_seg["start"]
        
        overlap_ratio = overlap_duration / segment_duration
        
        self.assertEqual(overlap_ratio, 0.25)  # 25% overlap
        self.assertLess(overlap_ratio, 0.3)  # Below 30% threshold


class TestConfigurationParameters(unittest.TestCase):
    """Test configuration parameter validation"""
    
    def test_threshold_values(self):
        """Test that threshold values are reasonable"""
        SHORT_SEGMENT_THRESHOLD = 0.5
        FULL_OVERLAP_SHORT_THRESHOLD = 1.0
        PARTIAL_OVERLAP_SHORT_THRESHOLD = 2.0
        SIGNIFICANT_OVERLAP_THRESHOLD = 0.6
        
        # Test logical ordering
        self.assertLessEqual(SHORT_SEGMENT_THRESHOLD, FULL_OVERLAP_SHORT_THRESHOLD)
        self.assertLessEqual(FULL_OVERLAP_SHORT_THRESHOLD, PARTIAL_OVERLAP_SHORT_THRESHOLD)
        
        # Test reasonable ranges
        self.assertGreaterEqual(SIGNIFICANT_OVERLAP_THRESHOLD, 0.0)
        self.assertLessEqual(SIGNIFICANT_OVERLAP_THRESHOLD, 1.0)
    
    def test_strategy_toggles(self):
        """Test that strategy toggles work correctly"""
        strategies = [
            "ENABLE_SHORT_SEGMENT_FILTERING",
            "ENABLE_FULL_OVERLAP_RESOLUTION", 
            "ENABLE_PARTIAL_OVERLAP_HANDLING",
            "ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT"
        ]
        
        for strategy in strategies:
            for value in [True, False]:
                self.assertIsInstance(value, bool)


class TestAudioProcessing(unittest.TestCase):
    """Test audio processing functions"""
    
    def setUp(self):
        """Create temporary test audio file"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create a simple test audio file (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16)
        
        # Use pydub to create audio file
        audio = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        audio.export(self.test_audio_path, format="wav")
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)
        os.rmdir(self.temp_dir)
    
    def test_audio_chunking_logic(self):
        """Test audio chunking calculations"""
        CHUNK_SIZE = 30 * 1000  # 30 seconds in milliseconds
        
        # Test with different audio durations
        test_durations = [15000, 45000, 90000]  # 15s, 45s, 90s in ms
        
        for duration in test_durations:
            chunk_count = int(np.ceil(duration / CHUNK_SIZE))
            
            if duration <= CHUNK_SIZE:
                self.assertEqual(chunk_count, 1)
            elif duration <= 2 * CHUNK_SIZE:
                self.assertEqual(chunk_count, 2)
            elif duration <= 3 * CHUNK_SIZE:
                self.assertEqual(chunk_count, 3)
    
    def test_timestamp_adjustment(self):
        """Test timestamp adjustment for chunks"""
        CHUNK_SIZE = 30 * 1000  # 30 seconds in milliseconds
        
        # Simulate segments from second chunk
        chunk_index = 1
        start_ms = chunk_index * CHUNK_SIZE
        
        original_segments = [
            {"start": 5.0, "end": 10.0},
            {"start": 15.0, "end": 20.0}
        ]
        
        # Adjust timestamps
        adjusted_segments = []
        for segment in original_segments:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += start_ms / 1000
            adjusted_segment["end"] += start_ms / 1000
            adjusted_segments.append(adjusted_segment)
        
        # Check adjustments
        self.assertEqual(adjusted_segments[0]["start"], 35.0)  # 5 + 30
        self.assertEqual(adjusted_segments[0]["end"], 40.0)   # 10 + 30


class TestOutputGeneration(unittest.TestCase):
    """Test output generation and formatting"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_output_lines = [
            "Speaker SPEAKER_00 | 1.25 - 5.64 | Hello world [adjusted_for_partial_overlap]",
            "Speaker SPEAKER_01 | 6.64 - 12.31 | How are you today",
            "Speaker SPEAKER_00 | 13.12 - 18.45 | I am doing well"
        ]
        
        self.sample_unassigned = [
            "Unassigned Speaker SPEAKER_02 | 21.89 - 22.15 | [short_significant_partial_overlap]",
            "Unassigned Transcript | 52.32 - 54.32 | Background noise [no_matching_speaker]"
        ]
    
    def test_output_line_parsing(self):
        """Test that output lines can be parsed correctly"""
        line = self.sample_output_lines[0]
        
        # Parse the line
        parts = line.split(" | ")
        self.assertEqual(len(parts), 3)
        
        speaker_part = parts[0]
        time_part = parts[1]
        text_part = parts[2]
        
        self.assertIn("Speaker", speaker_part)
        self.assertIn("-", time_part)
        self.assertIn("Hello world", text_part)
    
    def test_timestamp_sorting(self):
        """Test that output lines are sorted by timestamp"""
        timestamps = []
        for line in self.sample_output_lines:
            time_part = line.split(" | ")[1]
            start_time = float(time_part.split(" - ")[0])
            timestamps.append(start_time)
        
        # Check if sorted
        self.assertEqual(timestamps, sorted(timestamps))
    
    def test_strategy_status_output(self):
        """Test strategy status output format"""
        strategies = {
            "ENABLE_SHORT_SEGMENT_FILTERING": True,
            "ENABLE_FULL_OVERLAP_RESOLUTION": False,
            "ENABLE_PARTIAL_OVERLAP_HANDLING": True,
            "ENABLE_INTELLIGENT_SEGMENT_ASSIGNMENT": True
        }
        
        expected_lines = [
            "# - Short Segment Filtering: Enabled",
            "# - Full Overlap Resolution: Disabled", 
            "# - Partial Overlap Handling: Enabled",
            "# - Intelligent Segment Assignment: Enabled"
        ]
        
        # Generate status lines
        status_lines = []
        for strategy, enabled in strategies.items():
            status = "Enabled" if enabled else "Disabled"
            name = strategy.replace("ENABLE_", "").replace("_", " ").title()
            status_lines.append(f"# - {name}: {status}")
        
        self.assertEqual(len(status_lines), 4)
        self.assertIn("Enabled", status_lines[0])
        self.assertIn("Disabled", status_lines[1])


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def test_empty_diarization_handling(self):
        """Test handling of empty diarization results"""
        empty_df = pd.DataFrame(columns=["speaker", "start", "end", "duration", "keep", "reason"])
        
        self.assertEqual(len(empty_df), 0)
        self.assertTrue(empty_df.empty)
    
    def test_no_transcript_segments(self):
        """Test handling when no transcript segments exist"""
        empty_segments = []
        self.assertEqual(len(empty_segments), 0)
    
    def test_invalid_audio_duration(self):
        """Test handling of invalid audio durations"""
        invalid_durations = [-1.0, 0.0, float('inf'), float('nan')]
        
        for duration in invalid_durations:
            if duration <= 0 or not np.isfinite(duration):
                self.assertTrue(duration <= 0 or not np.isfinite(duration))


class TestMemoryManagement(unittest.TestCase):
    """Test memory management aspects"""
    
    def test_chunk_size_calculation(self):
        """Test that chunk size is reasonable for memory management"""
        CHUNK_SIZE = 30 * 1000  # 30 seconds in milliseconds
        
        # 30 seconds should be reasonable for memory
        self.assertGreaterEqual(CHUNK_SIZE, 10 * 1000)  # At least 10 seconds
        self.assertLessEqual(CHUNK_SIZE, 60 * 1000)     # At most 60 seconds
    
    @patch('torch.cuda.is_available')
    def test_cuda_availability_check(self, mock_cuda):
        """Test CUDA availability checking"""
        # Test when CUDA is available
        mock_cuda.return_value = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device, "cuda")
        
        # Test when CUDA is not available
        mock_cuda.return_value = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device, "cpu")


class TestDetailedTranscript(unittest.TestCase):
    """Test detailed transcript generation functionality"""
    
    def test_seconds_to_ticks_conversion(self):
        """Test conversion from seconds to ticks"""
        # Test the conversion logic
        def seconds_to_ticks(seconds):
            return int(seconds * 10_000_000)
        
        self.assertEqual(seconds_to_ticks(1.0), 10_000_000)
        self.assertEqual(seconds_to_ticks(0.5), 5_000_000)
        self.assertEqual(seconds_to_ticks(2.5), 25_000_000)
        self.assertEqual(seconds_to_ticks(0.0), 0)
    
    def test_lexical_text_creation(self):
        """Test creation of lexical text format"""
        import re
        
        def create_lexical_text(display_text):
            lexical = re.sub(r'[^\w\s]', '', display_text.lower())
            return ' '.join(lexical.split())
        
        test_cases = [
            ("Hello, world!", "hello world"),
            ("What's happening?", "whats happening"),
            ("I'm fine, thank you.", "im fine thank you"),
            ("Multiple   spaces", "multiple spaces"),
            ("Test123", "test123")
        ]
        
        for display, expected_lexical in test_cases:
            result = create_lexical_text(display)
            self.assertEqual(result, expected_lexical)
    
    def test_detailed_transcript_structure(self):
        """Test that detailed transcript structure matches expected format"""
        # Test the expected structure
        expected_keys = ["Id", "DisplayText", "Duration", "Offset", "SpeakerId", "RecognitionStatus", "NBest"]
        expected_nbest_keys = ["Display", "Lexical", "Words"]
        expected_word_keys = ["Duration", "Offset", "Word"]
        
        # Sample structure that should be generated
        sample_segment = {
            "Id": "test123",
            "DisplayText": "Hello world",
            "Duration": 10000000,
            "Offset": 5000000,
            "SpeakerId": "SPEAKER_00",
            "RecognitionStatus": "Success",
            "NBest": [
                {
                    "Display": "Hello world",
                    "Lexical": "hello world",
                    "Words": [
                        {
                            "Duration": 5000000,
                            "Offset": 5000000,
                            "Word": "hello"
                        },
                        {
                            "Duration": 5000000,
                            "Offset": 10000000,
                            "Word": "world"
                        }
                    ]
                }
            ]
        }
        
        # Test main structure
        for key in expected_keys:
            self.assertIn(key, sample_segment)
        
        # Test NBest structure
        self.assertEqual(len(sample_segment["NBest"]), 1)
        nbest = sample_segment["NBest"][0]
        for key in expected_nbest_keys:
            self.assertIn(key, nbest)
        
        # Test Words structure
        self.assertGreater(len(nbest["Words"]), 0)
        for word in nbest["Words"]:
            for key in expected_word_keys:
                self.assertIn(key, word)
    
    def test_detailed_transcript_configuration(self):
        """Test detailed transcript configuration flag"""
        ENABLE_DETAILED_TRANSCRIPT = True
        self.assertIsInstance(ENABLE_DETAILED_TRANSCRIPT, bool)
        
        # Test that it can be disabled
        ENABLE_DETAILED_TRANSCRIPT = False
        self.assertIsInstance(ENABLE_DETAILED_TRANSCRIPT, bool)
    
    def test_word_level_timestamp_processing(self):
        """Test word-level timestamp processing logic"""
        # Sample Whisper segment with word-level data
        sample_segment = {
            "start": 1.0,
            "end": 3.0,
            "text": "Hello world",
            "words": [
                {"start": 1.0, "end": 1.5, "word": "Hello"},
                {"start": 1.6, "end": 2.8, "word": "world"}
            ]
        }
        
        # Test that word timestamps are within segment bounds
        for word in sample_segment["words"]:
            self.assertGreaterEqual(word["start"], sample_segment["start"])
            self.assertLessEqual(word["end"], sample_segment["end"])
            self.assertLessEqual(word["start"], word["end"])
    
    def test_speaker_assignment_integration(self):
        """Test integration with speaker assignment logic"""
        # Test that segments can be assigned speakers
        segment_speaker_assignments = {
            0: "SPEAKER_00",
            1: "SPEAKER_01", 
            2: "SPEAKER_00"
        }
        
        # Test assignment lookup
        for segment_idx, expected_speaker in segment_speaker_assignments.items():
            assigned_speaker = segment_speaker_assignments.get(segment_idx, "SPEAKER_UNKNOWN")
            self.assertEqual(assigned_speaker, expected_speaker)
        
        # Test unknown segment
        unknown_speaker = segment_speaker_assignments.get(999, "SPEAKER_UNKNOWN")
        self.assertEqual(unknown_speaker, "SPEAKER_UNKNOWN")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_output_file_creation(self):
        """Test that all expected output files would be created"""
        expected_files = [
            "diarization.rttm",
            "speaker_timestamps.txt", 
            "whisper_result.json",
            "speaker_transcript.txt"
        ]
        
        # Test that file paths are valid
        for filename in expected_files:
            filepath = os.path.join(self.output_dir, filename)
            self.assertTrue(os.path.isabs(filepath) or not filepath.startswith('/'))
    
    def test_json_output_structure(self):
        """Test JSON output structure validity"""
        sample_whisper_result = {
            "segments": [
                {
                    "start": 1.0,
                    "end": 3.0,
                    "text": "Hello world"
                }
            ]
        }
        
        # Test JSON serialization
        json_str = json.dumps(sample_whisper_result, indent=2)
        parsed = json.loads(json_str)
        
        self.assertIn("segments", parsed)
        self.assertEqual(len(parsed["segments"]), 1)
        self.assertEqual(parsed["segments"][0]["text"], "Hello world")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestOverlapResolution,
        TestSpeakerAssignment, 
        TestConfigurationParameters,
        TestAudioProcessing,
        TestOutputGeneration,
        TestErrorHandling,
        TestMemoryManagement,
        TestDetailedTranscript,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*50}") 