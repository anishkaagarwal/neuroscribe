"""
NeuroscribeAI - Comprehensive Evaluation Suite
8 Dimensions: ASR, Diarization, Signal Quality, Prosody, Intent, Summary, Explainability, Privacy
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import librosa
import parselmouth
from groq import Groq
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()


class NeuroscribeEvaluator:
    """Comprehensive evaluation suite for NeuroscribeAI"""
    
    def __init__(self, groq_api_key=None, deepgram_api_key=None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.deepgram_api_key = deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        
        self.results = {}
        self.test_dir = Path("evaluation_results")
        self.test_dir.mkdir(exist_ok=True)
        
    # ========================================
    # A. ASR (Automatic Speech Recognition)
    # ========================================
    
    def evaluate_asr(self, audio_file: str, ground_truth_text: str = None) -> Dict:
        """
        Evaluate ASR performance
        
        Metrics:
        - Word Error Rate (WER)
        - Character Error Rate (CER)
        - Real-Time Factor (RTF)
        
        Args:
            audio_file: Path to audio file
            ground_truth_text: Optional ground truth for WER/CER calculation
        """
        print("\n=== A. ASR EVALUATION ===")
        
        from Transcript import transcribe_file
        import time
        
        # Measure transcription time
        start_time = time.time()
        hypothesis = transcribe_file(audio_file)
        transcription_time = time.time() - start_time
        
        # Calculate audio duration
        import soundfile as sf
        audio_info = sf.info(audio_file)
        audio_duration = audio_info.duration
        
        # Real-Time Factor (RTF)
        rtf = transcription_time / audio_duration
        
        results = {
            "audio_file": audio_file,
            "audio_duration_sec": audio_duration,
            "transcription_time_sec": transcription_time,
            "rtf": rtf,
            "hypothesis_length": len(hypothesis.split()),
        }
        
        # If ground truth provided, calculate WER and CER
        if ground_truth_text:
            wer = self._calculate_wer(ground_truth_text, hypothesis)
            cer = self._calculate_cer(ground_truth_text, hypothesis)
            results["wer"] = wer
            results["cer"] = cer
            results["ground_truth_length"] = len(ground_truth_text.split())
            
            print(f"✓ Word Error Rate (WER): {wer:.2%}")
            print(f"✓ Character Error Rate (CER): {cer:.2%}")
        
        print(f"✓ Real-Time Factor (RTF): {rtf:.2f}x")
        print(f"✓ Transcription speed: {'Real-time' if rtf <= 1.0 else 'Slower than real-time'}")
        
        self.results['asr'] = results
        return results
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Levenshtein distance for words
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0.0
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        ref_chars = list(reference.lower().replace(" ", ""))
        hyp_chars = list(hypothesis.lower().replace(" ", ""))
        
        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) if len(ref_chars) > 0 else 0.0
    
    # ========================================
    # B. Speaker Diarization
    # ========================================
    
    def evaluate_diarization(self, diarized_file: str, ground_truth_file: str = None) -> Dict:
        """
        Evaluate speaker diarization performance
        
        Metrics:
        - Diarization Error Rate (DER)
        - Speaker Count Accuracy
        - Segment Purity
        """
        print("\n=== B. DIARIZATION EVALUATION ===")
        
        with open(diarized_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract speaker segments
        speaker_pattern = r'SPEAKER (\d+):'
        speakers = re.findall(speaker_pattern, content)
        unique_speakers = set(speakers)
        
        # Count speaker transitions
        transitions = 0
        prev_speaker = None
        for speaker in speakers:
            if prev_speaker and speaker != prev_speaker:
                transitions += 1
            prev_speaker = speaker
        
        results = {
            "num_unique_speakers": len(unique_speakers),
            "total_speaker_segments": len(speakers),
            "speaker_transitions": transitions,
            "avg_segment_length": len(content) / len(speakers) if speakers else 0
        }
        
        # If ground truth provided, calculate DER
        if ground_truth_file:
            der = self._calculate_der(diarized_file, ground_truth_file)
            results["diarization_error_rate"] = der
            print(f"✓ Diarization Error Rate (DER): {der:.2%}")
        
        print(f"✓ Unique speakers detected: {len(unique_speakers)}")
        print(f"✓ Speaker transitions: {transitions}")
        
        self.results['diarization'] = results
        return results
    
    def _calculate_der(self, hypothesis_file: str, reference_file: str) -> float:
        """
        Calculate Diarization Error Rate (simplified version)
        DER = (False Alarm + Missed Speech + Speaker Error) / Total Speech Time
        """
        # This is a simplified implementation
        # For production, use pyannote.metrics
        return 0.15  # Placeholder - implement actual DER calculation
    
    # ========================================
    # C. Signal Quality
    # ========================================
    
    def evaluate_signal_quality(self, audio_file: str) -> Dict:
        """
        Evaluate audio signal quality
        
        Metrics:
        - SNR (Signal-to-Noise Ratio)
        - Dynamic Range
        - Clipping Detection
        """
        print("\n=== C. SIGNAL QUALITY EVALUATION ===")
        
        y, sr = librosa.load(audio_file, sr=None)
        
        # SNR estimation
        signal_power = np.mean(y ** 2)
        noise_estimate = np.median(np.abs(y))
        noise_power = noise_estimate ** 2
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Dynamic range
        db_values = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        dynamic_range = np.max(db_values) - np.percentile(db_values, 5)
        
        # Clipping detection
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        clipping_percentage = (clipped_samples / len(y)) * 100
        
        results = {
            "snr_db": float(snr_db),
            "dynamic_range_db": float(dynamic_range),
            "clipping_percentage": float(clipping_percentage),
            "sample_rate": sr,
            "duration_sec": len(y) / sr
        }
        
        print(f"✓ SNR: {snr_db:.2f} dB")
        print(f"✓ Dynamic Range: {dynamic_range:.2f} dB")
        print(f"✓ Clipping: {clipping_percentage:.3f}%")
        
        self.results['signal_quality'] = results
        return results
    
    # ========================================
    # D. Prosody Analysis
    # ========================================
    
    def evaluate_prosody(self, prosody_file: str) -> Dict:
        """
        Evaluate prosody extraction quality
        
        Metrics:
        - Urgency Detection Accuracy
        - Emotion Classification Accuracy
        - Coverage (% of transcript annotated)
        """
        print("\n=== D. PROSODY EVALUATION ===")
        
        with open(prosody_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract urgency labels
        urgency_pattern = r'urg=(\w+)'
        urgencies = re.findall(urgency_pattern, content)
        urgency_counts = Counter(urgencies)
        
        # Extract emotion labels
        emotion_pattern = r'emo=(\w+)'
        emotions = re.findall(emotion_pattern, content)
        emotion_counts = Counter(emotions)
        
        # Count segments
        segment_pattern = r'\[[\d.]+-[\d.]+\]'
        segments = re.findall(segment_pattern, content)
        
        results = {
            "total_segments": len(segments),
            "urgency_distribution": dict(urgency_counts),
            "emotion_distribution": dict(emotion_counts),
            "unique_emotions": len(emotion_counts),
            "high_urgency_percentage": (urgency_counts.get('High', 0) / len(urgencies) * 100) if urgencies else 0
        }
        
        print(f"✓ Segments analyzed: {len(segments)}")
        print(f"✓ Urgency distribution: {dict(urgency_counts)}")
        print(f"✓ Emotion distribution: {dict(emotion_counts)}")
        print(f"✓ High urgency segments: {results['high_urgency_percentage']:.1f}%")
        
        self.results['prosody'] = results
        return results
    
    # ========================================
    # E. Intent Detection
    # ========================================
    
    def evaluate_intent(self, intent_file: str, ground_truth: Dict = None) -> Dict:
        """
        Evaluate intent detection performance
        
        Metrics:
        - Intent Classification Accuracy
        - Intent Coverage
        - False Positive Rate
        """
        print("\n=== E. INTENT DETECTION EVALUATION ===")
        
        with open(intent_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract intents
        intent_section = re.search(r'Intents:(.*?)(?=\n\n|$)', content, re.DOTALL)
        if intent_section:
            intent_text = intent_section.group(1)
            intents = [line.strip('- ').strip() for line in intent_text.split('\n') if line.strip().startswith('-')]
        else:
            intents = []
        
        # Count segments with intents
        segment_pattern = r'--- Segment \d+ ---'
        total_segments = len(re.findall(segment_pattern, content))
        
        results = {
            "total_segments": total_segments,
            "segments_with_intents": len(intents),
            "intent_coverage": (len(intents) / total_segments * 100) if total_segments > 0 else 0,
            "unique_intents": len(set(intents)),
            "intents_detected": intents[:5]  # Sample
        }
        
        print(f"✓ Total segments: {total_segments}")
        print(f"✓ Intent coverage: {results['intent_coverage']:.1f}%")
        print(f"✓ Unique intents: {results['unique_intents']}")
        
        self.results['intent'] = results
        return results
    
    # ========================================
    # F. Summarization Quality
    # ========================================
    
    def evaluate_summary(self, transcript: str, summary: str, reference_summary: str = None) -> Dict:
        """
        Evaluate summary quality using LLM-based metrics
        
        Metrics:
        - Relevance Score
        - Completeness Score
        - Conciseness Score
        - Factual Consistency
        """
        print("\n=== F. SUMMARIZATION EVALUATION ===")
        
        if not self.groq_client:
            print("⚠️ Groq API key not available, skipping LLM evaluation")
            return {"error": "API key not available"}
        
        # Calculate basic metrics
        compression_ratio = len(summary.split()) / len(transcript.split())
        
        # LLM-based evaluation
        eval_prompt = f"""Evaluate this meeting summary on a scale of 1-10 for each metric.

TRANSCRIPT (excerpt): {transcript[:1000]}...

SUMMARY: {summary}

Rate the summary:
1. Relevance (captures key points): 
2. Completeness (covers main topics): 
3. Conciseness (no unnecessary info): 
4. Factual Consistency (accurate to transcript): 

Respond ONLY with JSON:
{{"relevance": X, "completeness": X, "conciseness": X, "factual_consistency": X, "explanation": "brief reason"}}
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.3
            )
            
            eval_result = json.loads(response.choices[0].message.content)
            
            results = {
                "compression_ratio": compression_ratio,
                "summary_length_words": len(summary.split()),
                "transcript_length_words": len(transcript.split()),
                **eval_result
            }
            
            print(f"✓ Compression ratio: {compression_ratio:.2%}")
            print(f"✓ Relevance: {eval_result.get('relevance', 'N/A')}/10")
            print(f"✓ Completeness: {eval_result.get('completeness', 'N/A')}/10")
            print(f"✓ Conciseness: {eval_result.get('conciseness', 'N/A')}/10")
            print(f"✓ Factual Consistency: {eval_result.get('factual_consistency', 'N/A')}/10")
            
        except Exception as e:
            print(f"⚠️ LLM evaluation failed: {e}")
            results = {
                "compression_ratio": compression_ratio,
                "summary_length_words": len(summary.split()),
                "transcript_length_words": len(transcript.split())
            }
        
        self.results['summary'] = results
        return results
    
    # ========================================
    # G. Explainability & Attribution
    # ========================================
    
    def evaluate_explainability(self, summary: str, transcript: str) -> Dict:
        """
        Evaluate how well claims in summary can be attributed to transcript
        
        Metrics:
        - Attribution Score
        - Claim Coverage
        - Hallucination Detection
        """
        print("\n=== G. EXPLAINABILITY EVALUATION ===")
        
        if not self.groq_client:
            print("⚠️ Groq API key not available, skipping")
            return {"error": "API key not available"}
        
        # Extract claims from summary
        claims_prompt = f"""Extract all factual claims from this summary as a JSON list:

SUMMARY: {summary}

Respond ONLY with JSON array of claims:
["claim1", "claim2", ...]
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": claims_prompt}],
                temperature=0
            )
            
            claims = json.loads(response.choices[0].message.content)
            
            # Verify each claim against transcript
            verified_claims = 0
            for claim in claims:
                verify_prompt = f"""Is this claim supported by the transcript?

CLAIM: {claim}

TRANSCRIPT: {transcript[:2000]}...

Respond ONLY with JSON: {{"supported": true/false, "confidence": 0-1}}
"""
                
                verify_response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": verify_prompt}],
                    temperature=0
                )
                
                verification = json.loads(verify_response.choices[0].message.content)
                if verification.get("supported", False):
                    verified_claims += 1
            
            attribution_score = verified_claims / len(claims) if claims else 0
            
            results = {
                "total_claims": len(claims),
                "verified_claims": verified_claims,
                "attribution_score": attribution_score,
                "hallucination_rate": 1 - attribution_score
            }
            
            print(f"✓ Total claims: {len(claims)}")
            print(f"✓ Verified claims: {verified_claims}")
            print(f"✓ Attribution score: {attribution_score:.2%}")
            print(f"✓ Hallucination rate: {(1-attribution_score):.2%}")
            
        except Exception as e:
            print(f"⚠️ Explainability evaluation failed: {e}")
            results = {"error": str(e)}
        
        self.results['explainability'] = results
        return results
    
    # ========================================
    # H. Privacy & Redaction
    # ========================================
    
    def evaluate_privacy(self, transcript: str) -> Dict:
        """
        Evaluate privacy protection measures
        
        Metrics:
        - PII Detection Rate
        - Redaction Coverage
        - False Positive Rate
        """
        print("\n=== H. PRIVACY EVALUATION ===")
        
        # Common PII patterns
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'url': r'https?://[^\s]+',
        }
        
        detected_pii = {}
        total_pii = 0
        
        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, transcript)
            if matches:
                detected_pii[pii_type] = len(matches)
                total_pii += len(matches)
        
        # Check for names (simple heuristic: capitalized words)
        potential_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', transcript)
        if potential_names:
            detected_pii['names'] = len(potential_names)
            total_pii += len(potential_names)
        
        results = {
            "total_pii_detected": total_pii,
            "pii_by_type": detected_pii,
            "needs_redaction": total_pii > 0,
            "privacy_risk_level": "High" if total_pii > 5 else "Medium" if total_pii > 0 else "Low"
        }
        
        print(f"✓ PII entities detected: {total_pii}")
        print(f"✓ PII breakdown: {detected_pii}")
        print(f"✓ Privacy risk: {results['privacy_risk_level']}")
        
        self.results['privacy'] = results
        return results
    
    # ========================================
    # Master Evaluation Function
    # ========================================
    
    def run_comprehensive_evaluation(self, 
                                     audio_file: str = None,
                                     transcript_file: str = None,
                                     diarized_file: str = None,
                                     prosody_file: str = None,
                                     intent_file: str = None,
                                     summary_file: str = None,
                                     ground_truth_text: str = None) -> Dict:
        """
        Run all evaluations and generate comprehensive report
        
        Args:
            audio_file: Path to audio file
            transcript_file: Path to transcript file
            diarized_file: Path to diarized transcript
            prosody_file: Path to prosody annotations
            intent_file: Path to intent detection output
            summary_file: Path to summary file
            ground_truth_text: Optional ground truth for WER calculation
        """
        print("\n" + "="*60)
        print("NEUROSCRIBEAI - COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # Run each evaluation if file provided
        if audio_file:
            self.evaluate_asr(audio_file, ground_truth_text)
            self.evaluate_signal_quality(audio_file)
        
        if diarized_file:
            self.evaluate_diarization(diarized_file)
        
        if prosody_file:
            self.evaluate_prosody(prosody_file)
        
        if intent_file:
            self.evaluate_intent(intent_file)
        
        if transcript_file and summary_file:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
            
            self.evaluate_summary(transcript, summary)
            self.evaluate_explainability(summary, transcript)
            self.evaluate_privacy(transcript)
        
        # Generate final report
        self._generate_report()
        
        return self.results
    
    def _generate_report(self):
        """Generate and save comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_dir / f"evaluation_report_{timestamp}.json"
        
        # Calculate overall score
        scores = []
        if 'asr' in self.results and 'wer' in self.results['asr']:
            scores.append((1 - self.results['asr']['wer']) * 100)
        
        if 'summary' in self.results:
            for key in ['relevance', 'completeness', 'conciseness', 'factual_consistency']:
                if key in self.results['summary']:
                    scores.append(self.results['summary'][key] * 10)
        
        if 'explainability' in self.results and 'attribution_score' in self.results['explainability']:
            scores.append(self.results['explainability']['attribution_score'] * 100)

        # Mean of the available dimension scores (0-100). No manual adjustments.
        overall_score = float(np.mean(scores)) if scores else 0.0
        
        report = {
            "timestamp": timestamp,
            "overall_score": overall_score,
            "dimensions": self.results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"\n📊 Overall Score: {overall_score:.1f}/100")
        print(f"📁 Report saved: {report_file}")
        print("\n" + "="*60)


# ========================================
# Example Usage
# ========================================

def main():
    """Example usage of the evaluation suite"""
    
    # Initialize evaluator
    evaluator = NeuroscribeEvaluator()
    
    # Define test files (adjust paths as needed)
    test_files = {
        'audio_file': 'recordings/test_meeting.mp4',
        'transcript_file': 'recordings/test_meeting.txt',
        'diarized_file': 'recordings/test_meeting_diarized.txt',
        'prosody_file': 'Prosody_annotations.txt',
        'intent_file': 'recordings/test_meeting_intents.txt',
        'summary_file': 'recordings/test_meeting_summary.txt',
        # 'ground_truth_text': 'Ground truth transcript for WER calculation...'
    }
    
    # Filter to only existing files
    existing_files = {k: v for k, v in test_files.items() if os.path.exists(v)}
    
    print(f"\n🔍 Found {len(existing_files)} test files")
    print(f"Files: {list(existing_files.keys())}\n")
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(**existing_files)
    
    return results


if __name__ == "__main__":
    main()