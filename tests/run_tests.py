#!/usr/bin/env python3
"""
Test Runner for Speaker Diarization System

This script runs all tests to ensure the system is working correctly
before making any changes.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 SPEAKER DIARIZATION SYSTEM - TEST RUNNER")
    print("=" * 60)
    print("Running all tests to verify system integrity...")
    
    # Check if we're in the right directory
    if not os.path.exists("speaker_diarization.py"):
        print("❌ Error: speaker_diarization.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # List of tests to run
    tests = [
        ("python test_speaker_diarization.py", "Unit Tests"),
        ("python test_current_system.py", "Integration Tests")
    ]
    
    # Track results
    passed_tests = 0
    total_tests = len(tests)
    
    # Run each test
    for command, description in tests:
        if run_command(command, description):
            passed_tests += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready for development!")
        success_rate = 100.0
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please fix failing tests before making changes!")
        success_rate = (passed_tests / total_tests) * 100
    
    print(f"Success rate: {success_rate:.1f}%")
    print('='*60)
    
    # Exit with appropriate code
    sys.exit(0 if passed_tests == total_tests else 1)

if __name__ == "__main__":
    main() 