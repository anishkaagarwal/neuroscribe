"""
Signal Quality & Noise Evaluation (Section C)
Metrics: PESQ, STOI, SNR Gain

Run: python signal_quality_eval.py
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew


class SignalQualityEvaluator:
    """Evaluate audio signal quality with advanced metrics"""
    
    def __init__(self, audio_file: str):
        self.audio_file = Path(audio_file)
        self.results = {}
        
        # Load audio
        print(f"📂 Loading audio: {self.audio_file.name}")
        self.audio, self.sr = librosa.load(str(audio_file), sr=None, mono=True)
        self.duration = len(self.audio) / self.sr
        print(f"✓ Duration: {self.duration:.2f}s, Sample rate: {self.sr} Hz\n")
    
    # =========================================
    # 1. PESQ (Perceptual Evaluation of Speech Quality)
    # =========================================
    
    def calculate_pesq_approximation(self):
        """
        PESQ approximation using spectral quality metrics
        Real PESQ requires reference audio - this gives similar insight
        
        Returns: Quality score 1-5 (higher is better)
        """
        print("🔊 Calculating PESQ Approximation...")
        
        # Calculate spectral quality indicators
        
        # 1. Spectral flatness (how noise-like vs tone-like)
        spec_flatness = librosa.feature.spectral_flatness(y=self.audio)[0]
        avg_flatness = np.mean(spec_flatness)
        
        # 2. Spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=self.audio, sr=self.sr)[0]
        avg_rolloff = np.mean(rolloff)
        
        # 3. Zero crossing rate (indicates noisiness)
        zcr = librosa.feature.zero_crossing_rate(self.audio)[0]
        avg_zcr = np.mean(zcr)
        
        # 4. Spectral centroid (brightness of sound)
        centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
        avg_centroid = np.mean(centroid)
        
        # 5. RMS energy stability (consistent vs fluctuating)
        rms = librosa.feature.rms(y=self.audio)[0]
        rms_std = np.std(rms)
        rms_mean = np.mean(rms)
        rms_stability = 1 - (rms_std / (rms_mean + 1e-6))
        
        # Combine metrics into quality score (1-5 scale)
        # Lower flatness = more tonal = better (speech is tonal)
        flatness_score = (1 - avg_flatness) * 2  # 0-2 points
        
        # Moderate ZCR is good (too high = noisy, too low = problematic)
        zcr_score = 1 - abs(avg_zcr - 0.1) * 10  # 0-1 points
        zcr_score = max(0, min(1, zcr_score))
        
        # RMS stability score (0-1.5 points)
        stability_score = rms_stability * 1.5
        
        # Rolloff should be reasonable (not too low)
        rolloff_normalized = avg_rolloff / (self.sr / 2)
        rolloff_score = rolloff_normalized * 0.5  # 0-0.5 points
        
        # Total PESQ approximation (1-5 scale)
        pesq_approx = 1 + flatness_score + zcr_score + stability_score + rolloff_score
        pesq_approx = max(1.0, min(5.0, pesq_approx))
        
        self.results['pesq_approximation'] = {
            'overall_score': float(pesq_approx),
            'scale': '1-5 (higher is better)',
            'interpretation': self._interpret_pesq(pesq_approx),
            'components': {
                'spectral_flatness': float(avg_flatness),
                'spectral_rolloff_hz': float(avg_rolloff),
                'zero_crossing_rate': float(avg_zcr),
                'spectral_centroid_hz': float(avg_centroid),
                'rms_stability': float(rms_stability)
            }
        }
        
        print(f"✓ PESQ Approximation: {pesq_approx:.2f}/5.0 ({self._interpret_pesq(pesq_approx)})")
        print(f"  - Spectral flatness: {avg_flatness:.4f} (lower=better for speech)")
        print(f"  - RMS stability: {rms_stability:.4f} (higher=better)")
        
        return pesq_approx
    
    def _interpret_pesq(self, score):
        """Interpret PESQ score"""
        if score >= 4.0:
            return "Excellent"
        elif score >= 3.5:
            return "Good"
        elif score >= 3.0:
            return "Fair"
        elif score >= 2.5:
            return "Poor"
        else:
            return "Bad"
    
    # =========================================
    # 2. STOI (Short-Time Objective Intelligibility)
    # =========================================
    
    def calculate_stoi_approximation(self):
        """
        STOI approximation using spectral coherence and modulation metrics
        Real STOI requires clean reference - this approximates intelligibility
        
        Returns: Intelligibility score 0-1 (higher is better)
        """
        print("\n🎯 Calculating STOI (Intelligibility)...")
        
        # Calculate features related to speech intelligibility
        
        # 1. Spectral contrast (clarity of speech formants)
        contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr)
        avg_contrast = np.mean(contrast)
        
        # 2. Modulation spectrum (temporal envelope modulation)
        # Speech has characteristic modulation in 2-8 Hz range
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)    # 10ms hop
        
        rms = librosa.feature.rms(y=self.audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # FFT of RMS envelope to get modulation spectrum
        mod_spectrum = np.abs(np.fft.rfft(rms))
        mod_freqs = np.fft.rfftfreq(len(rms), d=hop_length/self.sr)
        
        # Energy in speech modulation range (2-8 Hz)
        speech_mod_idx = np.where((mod_freqs >= 2) & (mod_freqs <= 8))[0]
        if len(speech_mod_idx) > 0:
            speech_mod_energy = np.sum(mod_spectrum[speech_mod_idx])
            total_mod_energy = np.sum(mod_spectrum)
            mod_ratio = speech_mod_energy / (total_mod_energy + 1e-6)
        else:
            mod_ratio = 0
        
        # 3. High-frequency energy ratio (presence of consonants)
        stft = librosa.stft(self.audio)
        power_spectrum = np.abs(stft) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # High freq (2-8 kHz important for consonants)
        high_freq_idx = np.where((freqs >= 2000) & (freqs <= 8000))[0]
        low_freq_idx = np.where(freqs < 2000)[0]
        
        high_energy = np.sum(power_spectrum[high_freq_idx, :])
        low_energy = np.sum(power_spectrum[low_freq_idx, :])
        hf_ratio = high_energy / (low_energy + high_energy + 1e-6)
        
        # 4. Cepstral analysis (vocal tract resonances)
        mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_dynamic_range = np.mean(mfcc_std)
        
        # Combine into STOI approximation (0-1 scale)
        
        # Spectral contrast score (0-0.3)
        contrast_norm = min(avg_contrast / 30, 1.0)  # Normalize
        contrast_score = contrast_norm * 0.3
        
        # Modulation score (0-0.3)
        mod_score = mod_ratio * 0.3
        
        # High-frequency score (0-0.2)
        hf_score = hf_ratio * 0.2
        
        # MFCC dynamic range score (0-0.2)
        mfcc_score = min(mfcc_dynamic_range / 10, 1.0) * 0.2
        
        stoi_approx = contrast_score + mod_score + hf_score + mfcc_score
        stoi_approx = max(0.0, min(1.0, stoi_approx))
        
        self.results['stoi'] = {
            'overall_score': float(stoi_approx),
            'scale': '0-1 (higher is better)',
            'interpretation': self._interpret_stoi(stoi_approx),
            'components': {
                'spectral_contrast': float(avg_contrast),
                'speech_modulation_ratio': float(mod_ratio),
                'high_freq_ratio': float(hf_ratio),
                'mfcc_dynamic_range': float(mfcc_dynamic_range)
            }
        }
        
        print(f"✓ STOI: {stoi_approx:.3f}/1.0 ({self._interpret_stoi(stoi_approx)})")
        print(f"  - Spectral contrast: {avg_contrast:.2f} dB")
        print(f"  - Speech modulation ratio: {mod_ratio:.3f}")
        print(f"  - High-freq energy ratio: {hf_ratio:.3f}")
        
        return stoi_approx
    
    def _interpret_stoi(self, score):
        """Interpret STOI score"""
        if score >= 0.85:
            return "Excellent intelligibility"
        elif score >= 0.70:
            return "Good intelligibility"
        elif score >= 0.55:
            return "Fair intelligibility"
        elif score >= 0.40:
            return "Poor intelligibility"
        else:
            return "Very poor intelligibility"
    
    # =========================================
    # 3. SNR Gain (Signal-to-Noise Ratio)
    # =========================================
    
    def calculate_snr_gain(self):
        """
        Calculate SNR and noise characteristics
        
        Returns: SNR in dB and noise floor metrics
        """
        print("\n📊 Calculating SNR Gain...")
        
        # Method 1: Simple energy-based SNR
        rms = librosa.feature.rms(y=self.audio)[0]
        
        # Estimate noise from quietest 20% of frames
        noise_threshold = np.percentile(rms, 20)
        noise_frames = rms[rms <= noise_threshold]
        
        if len(noise_frames) > 0:
            noise_power = np.mean(noise_frames ** 2)
        else:
            noise_power = np.min(rms) ** 2
        
        signal_power = np.mean(rms ** 2)
        
        snr_simple = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Method 2: Spectral SNR
        stft = librosa.stft(self.audio)
        magnitude = np.abs(stft)
        
        # Estimate noise spectrum from quietest frames
        frame_energy = np.sum(magnitude, axis=0)
        quiet_frame_idx = np.argsort(frame_energy)[:int(len(frame_energy) * 0.1)]
        noise_spectrum = np.median(magnitude[:, quiet_frame_idx], axis=1)
        
        # Average signal spectrum
        signal_spectrum = np.mean(magnitude, axis=1)
        
        # Spectral SNR
        spectral_snr = 10 * np.log10(
            np.sum(signal_spectrum) / (np.sum(noise_spectrum) + 1e-10)
        )
        
        # Noise floor characteristics
        noise_floor_db = 20 * np.log10(np.mean(noise_frames) + 1e-10)
        
        # Dynamic range
        max_amplitude = np.max(np.abs(self.audio))
        noise_amplitude = np.mean(noise_frames)
        dynamic_range = 20 * np.log10(max_amplitude / (noise_amplitude + 1e-10))
        
        # SNR variability (how consistent is SNR across time)
        windowed_snr = []
        window_size = int(1.0 * self.sr)  # 1-second windows
        hop = window_size // 2
        
        for i in range(0, len(self.audio) - window_size, hop):
            window = self.audio[i:i+window_size]
            win_signal_power = np.mean(window ** 2)
            win_noise_power = np.percentile(window ** 2, 10)
            win_snr = 10 * np.log10(win_signal_power / (win_noise_power + 1e-10))
            windowed_snr.append(win_snr)
        
        snr_std = np.std(windowed_snr)
        snr_mean = np.mean(windowed_snr)
        
        self.results['snr_gain'] = {
            'snr_simple_db': float(snr_simple),
            'snr_spectral_db': float(spectral_snr),
            'snr_mean_db': float(snr_mean),
            'snr_std_db': float(snr_std),
            'noise_floor_db': float(noise_floor_db),
            'dynamic_range_db': float(dynamic_range),
            'interpretation': self._interpret_snr(snr_mean)
        }
        
        print(f"✓ SNR (Simple): {snr_simple:.2f} dB")
        print(f"✓ SNR (Spectral): {spectral_snr:.2f} dB")
        print(f"✓ SNR (Mean): {snr_mean:.2f} dB ({self._interpret_snr(snr_mean)})")
        print(f"  - SNR variability: {snr_std:.2f} dB")
        print(f"  - Noise floor: {noise_floor_db:.2f} dB")
        print(f"  - Dynamic range: {dynamic_range:.2f} dB")
        
        return snr_mean
    
    def _interpret_snr(self, snr):
        """Interpret SNR value"""
        if snr >= 30:
            return "Excellent (very clean)"
        elif snr >= 20:
            return "Good (clean)"
        elif snr >= 15:
            return "Fair (acceptable)"
        elif snr >= 10:
            return "Poor (noisy)"
        else:
            return "Very poor (very noisy)"
    
    # =========================================
    # Additional Metrics
    # =========================================
    
    def calculate_additional_metrics(self):
        """Calculate supplementary quality metrics"""
        print("\n📈 Calculating Additional Metrics...")
        
        # 1. Clipping detection
        clipping_threshold = 0.99
        clipped = np.sum(np.abs(self.audio) > clipping_threshold)
        clipping_pct = (clipped / len(self.audio)) * 100
        
        # 2. Peak-to-Average Ratio (PAR)
        peak = np.max(np.abs(self.audio))
        average = np.mean(np.abs(self.audio))
        par_db = 20 * np.log10(peak / (average + 1e-10))
        
        # 3. Crest Factor
        rms_val = np.sqrt(np.mean(self.audio ** 2))
        crest_factor = peak / (rms_val + 1e-10)
        crest_factor_db = 20 * np.log10(crest_factor)
        
        # 4. Harmonic-to-Noise Ratio (HNR) approximation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio, fmin=75, fmax=500, sr=self.sr
        )
        
        # Count voiced frames
        voiced_frames = np.sum(~np.isnan(f0))
        total_frames = len(f0)
        voicing_ratio = voiced_frames / total_frames if total_frames > 0 else 0
        
        # 5. Spectral entropy (measure of noisiness)
        stft = librosa.stft(self.audio)
        mag = np.abs(stft)
        
        # Normalize to probability distribution
        mag_norm = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(mag_norm * np.log2(mag_norm + 1e-10), axis=0)
        avg_entropy = np.mean(entropy)
        
        self.results['additional_metrics'] = {
            'clipping_percentage': float(clipping_pct),
            'peak_to_average_ratio_db': float(par_db),
            'crest_factor_db': float(crest_factor_db),
            'voicing_ratio': float(voicing_ratio),
            'spectral_entropy': float(avg_entropy)
        }
        
        print(f"✓ Clipping: {clipping_pct:.3f}%")
        print(f"✓ Peak-to-Average Ratio: {par_db:.2f} dB")
        print(f"✓ Crest Factor: {crest_factor_db:.2f} dB")
        print(f"✓ Voicing Ratio: {voicing_ratio:.2%}")
        print(f"✓ Spectral Entropy: {avg_entropy:.2f}")
    
    # =========================================
    # Visualization
    # =========================================
    
    def plot_quality_analysis(self, save_path=None):
        """Generate visualization of signal quality metrics"""
        print("\n📊 Generating visualization...")
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Signal Quality Analysis: {self.audio_file.name}', fontsize=14, fontweight='bold')
        
        # 1. Waveform
        time = np.linspace(0, self.duration, len(self.audio))
        axes[0, 0].plot(time, self.audio, linewidth=0.5)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
        
        # 3. RMS Energy
        rms = librosa.feature.rms(y=self.audio)[0]
        frames = range(len(rms))
        t_rms = librosa.frames_to_time(frames, sr=self.sr)
        axes[1, 0].plot(t_rms, rms)
        axes[1, 0].set_title('RMS Energy')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
        t_cent = librosa.frames_to_time(range(len(centroid)), sr=self.sr)
        axes[1, 1].plot(t_cent, centroid)
        axes[1, 1].set_title('Spectral Centroid (Brightness)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Quality Metrics Summary
        axes[2, 0].axis('off')
        metrics_text = (
            f"PESQ Approximation: {self.results['pesq_approximation']['overall_score']:.2f}/5.0\n"
            f"STOI: {self.results['stoi']['overall_score']:.3f}/1.0\n"
            f"SNR: {self.results['snr_gain']['snr_mean_db']:.2f} dB\n"
            f"Dynamic Range: {self.results['snr_gain']['dynamic_range_db']:.2f} dB\n"
            f"Clipping: {self.results['additional_metrics']['clipping_percentage']:.3f}%\n"
            f"Voicing Ratio: {self.results['additional_metrics']['voicing_ratio']:.2%}"
        )
        axes[2, 0].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 0].set_title('Quality Metrics Summary')
        
        # 6. Power Spectrum
        freqs = librosa.fft_frequencies(sr=self.sr)
        stft = librosa.stft(self.audio)
        power = np.mean(np.abs(stft) ** 2, axis=1)
        power_db = 10 * np.log10(power + 1e-10)
        
        axes[2, 1].plot(freqs, power_db)
        axes[2, 1].set_title('Average Power Spectrum')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Power (dB)')
        axes[2, 1].set_xlim([0, 8000])  # Focus on speech range
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to: {save_path}")
        else:
            plt.savefig('signal_quality_analysis.png', dpi=150, bbox_inches='tight')
            print("✓ Saved plot to: signal_quality_analysis.png")
        
        plt.close()
    
    # =========================================
    # Run All Evaluations
    # =========================================
    
    def run_complete_evaluation(self):
        """Run all signal quality evaluations"""
        print("\n" + "="*60)
        print("SIGNAL QUALITY EVALUATION (Section C)")
        print("="*60)
        
        # Run all evaluations
        self.calculate_pesq_approximation()
        self.calculate_stoi_approximation()
        self.calculate_snr_gain()
        self.calculate_additional_metrics()
        
        # Generate visualization
        self.plot_quality_analysis()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"signal_quality_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'audio_file': str(self.audio_file),
                'duration_sec': self.duration,
                'sample_rate': self.sr,
                'timestamp': timestamp,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Visualization saved to: signal_quality_analysis.png")
        print("="*60 + "\n")
        
        return self.results


# =========================================
# Main Execution
# =========================================

def main():
    """Run signal quality evaluation on audio file"""
    import sys
    
    # Get audio file from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Find most recent audio file in recordings
        recordings_dir = Path("recordings")
        audio_files = (
            list(recordings_dir.glob("*_final.mp4")) +
            list(recordings_dir.glob("*.mp4")) +
            list(recordings_dir.glob("*.wav")) +
            list(recordings_dir.glob("*.m4a"))
        )
        
        if audio_files:
            audio_file = max(audio_files, key=lambda x: x.stat().st_mtime)
            print(f"📂 Using most recent file: {audio_file}")
        else:
            print("❌ No audio files found!")
            print("Usage: python signal_quality_eval.py <audio_file>")
            return
    
    # Run evaluation
    evaluator = SignalQualityEvaluator(audio_file)
    results = evaluator.run_complete_evaluation()
    
    return results


if __name__ == "__main__":
    main()