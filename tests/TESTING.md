# Testing Documentation

This document describes the test suite for the Speaker Diarization system.

## Overview

The test suite ensures that the speaker diarization system works correctly and helps prevent regressions when adding new features. It consists of unit tests, integration tests, and a test runner.

## Test Files

### `test_speaker_diarization.py`
**Unit Tests** - Tests individual components and logic:

- **TestOverlapResolution**: Tests overlap detection and resolution strategies
- **TestSpeakerAssignment**: Tests speaker-to-transcript assignment logic  
- **TestConfigurationParameters**: Tests configuration validation
- **TestAudioProcessing**: Tests audio chunking and timestamp adjustment
- **TestOutputGeneration**: Tests output formatting and parsing
- **TestErrorHandling**: Tests error scenarios and edge cases
- **TestMemoryManagement**: Tests memory-related functionality
- **TestDetailedTranscript**: Tests detailed transcript generation with word-level timestamps
- **TestIntegration**: Tests data structure validation

### `test_current_system.py`
**Integration Tests** - Tests system-level functionality:

- Audio file creation and processing infrastructure
- Configuration parameter accessibility
- Expected output file structure
- Test environment setup and cleanup

### `run_tests.py`
**Test Runner** - Runs all tests with summary reporting:

- Executes all test suites in sequence
- Provides real-time output
- Reports overall success/failure status
- Returns appropriate exit codes for CI/CD

## Running Tests

### Quick Test (Recommended)
```bash
python run_tests.py
```

### Individual Test Suites
```bash
# Unit tests only
python test_speaker_diarization.py

# Integration tests only  
python test_current_system.py
```

## Expected Output

All tests should pass with 100% success rate:

```
🎉 ALL TESTS PASSED!
✅ System is ready for development!
Success rate: 100.0%
```

**Current Test Count**: 25 tests total
- 19 original tests (overlap resolution, speaker assignment, etc.)
- 6 new tests for detailed transcript functionality

## Test Requirements

The tests require the following Python packages:
- `unittest` (built-in)
- `pandas` 
- `numpy`
- `torch`
- `pydub`
- `tempfile` (built-in)
- `json` (built-in)
- `os` (built-in)

## When to Run Tests

### Before Making Changes
Always run tests before starting development:
```bash
python run_tests.py
```

### After Making Changes
Run tests after any code modifications to ensure nothing broke:
```bash
python run_tests.py
```

### Before Committing
Always ensure tests pass before committing code to version control.

## Test Coverage

The current test suite covers:

- ✅ Overlap resolution algorithms
- ✅ Speaker assignment logic
- ✅ Configuration parameter validation
- ✅ Audio processing workflows
- ✅ Output generation and formatting
- ✅ Error handling scenarios
- ✅ Memory management aspects
- ✅ Detailed transcript generation (word-level timestamps)
- ✅ JSON structure validation
- ✅ Tick conversion and lexical text processing
- ✅ System integration points

## Adding New Tests

When adding new features, also add corresponding tests:

1. **Unit Tests**: Add to `test_speaker_diarization.py`
   - Create new test class for new functionality
   - Follow existing naming conventions
   - Test both success and failure cases

2. **Integration Tests**: Add to `test_current_system.py`
   - Test end-to-end functionality
   - Verify file creation and structure
   - Test with realistic data

3. **Update Test Runner**: If needed, update `run_tests.py`
   - Add new test commands
   - Update documentation

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all required packages are installed in your virtual environment.

**File Not Found**: Run tests from the project root directory where `speaker_diarization.py` is located.

**Permission Errors**: Ensure write permissions for temporary file creation.

### Getting Help

If tests fail unexpectedly:

1. Run individual test suites to isolate the issue
2. Check that all dependencies are installed
3. Verify you're in the correct directory
4. Check for any recent changes that might have broken functionality

## Recent Test Additions

**Detailed Transcript Tests** (Added May 13, 2025):
- ✅ Seconds to ticks conversion validation
- ✅ Lexical text creation (punctuation removal, lowercase)
- ✅ JSON structure validation for detailed transcript format
- ✅ Word-level timestamp processing logic
- ✅ Speaker assignment integration with detailed transcripts
- ✅ Configuration flag testing

## Future Test Enhancements

Planned test improvements:
- [ ] Performance benchmarking tests
- [ ] Memory usage validation
- [ ] Real audio file testing (with sample files)
- [ ] Mixed language processing tests
- [ ] ITN (Inverse Text Normalization) validation tests
- [ ] End-to-end detailed transcript accuracy tests 