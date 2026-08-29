from evaluation_suite import NeuroscribeEvaluator
import os
from pathlib import Path
from paths import recordings_dir as _recordings_dir

# Initialize evaluator
evaluator = NeuroscribeEvaluator()

# Find the most recent processed meeting
recordings_dir = _recordings_dir()
# Get most recent files
files = {
    'audio': list(recordings_dir.glob("*_final.mp4")) or list(recordings_dir.glob("*.mp4")),
    'transcript': list(recordings_dir.glob("*_complete_analysis.txt")) or list(recordings_dir.glob("*.txt")),
    'diarized': list(recordings_dir.glob("*_diarized.txt")),
    'prosody': [Path("Prosody_annotations.txt")] if Path("Prosody_annotations.txt").exists() else [],
    'intents': list(recordings_dir.glob("*_intents.txt")),
    'summary': []
}

# Get most recent of each type
test_files = {}

if files['audio']:
    test_files['audio_file'] = str(max(files['audio'], key=lambda x: x.stat().st_mtime))
    
if files['transcript']:
    test_files['transcript_file'] = str(max(files['transcript'], key=lambda x: x.stat().st_mtime))
    
if files['diarized']:
    test_files['diarized_file'] = str(max(files['diarized'], key=lambda x: x.stat().st_mtime))
    
if files['prosody']:
    test_files['prosody_file'] = str(files['prosody'][0])
    
if files['intents']:
    test_files['intent_file'] = str(max(files['intents'], key=lambda x: x.stat().st_mtime))

# Find summary in complete_analysis file
if 'transcript_file' in test_files:
    test_files['summary_file'] = test_files['transcript_file']  # Summary is in the same file

print("📂 Found test files:")
for key, path in test_files.items():
    print(f"  ✓ {key}: {path}")

# Run evaluation
print("\n🚀 Starting comprehensive evaluation...\n")
results = evaluator.run_comprehensive_evaluation(**test_files)

print("\n✅ Evaluation complete! Check 'evaluation_results' folder for detailed report.")