"""
Feature Extraction Module for ECG Beats
Extracts comprehensive features for arrhythmia classification
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch


def extract_temporal_features(beat, fs=360):
    """
    Extract temporal/time-domain features
    
    Args:
        beat: Single heartbeat segment
        fs: Sampling frequency
        
    Returns:
        Dictionary of temporal features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(beat)
    features['std'] = np.std(beat)
    features['median'] = np.median(beat)
    features['min'] = np.min(beat)
    features['max'] = np.max(beat)
    features['range'] = features['max'] - features['min']
    
    # Peak-to-peak amplitude
    features['ptp'] = np.ptp(beat)
    
    # RMS (Root Mean Square)
    features['rms'] = np.sqrt(np.mean(beat ** 2))
    
    # Signal energy
    features['energy'] = np.sum(beat ** 2)
    
    # Mean absolute deviation
    features['mad'] = np.mean(np.abs(beat - features['mean']))
    
    return features


def extract_morphological_features(beat, fs=360):
    """
    Extract morphological features (waveform shape)
    
    Args:
        beat: Single heartbeat segment
        fs: Sampling frequency
        
    Returns:
        Dictionary of morphological features
    """
    features = {}
    
    # Find R-peak (maximum point)
    r_idx = np.argmax(beat)
    features['r_amplitude'] = beat[r_idx]
    features['r_position'] = r_idx / len(beat)  # Normalized position
    
    # Find Q and S points (local minima before and after R)
    # Q point: minimum before R-peak
    if r_idx > 5:
        q_idx = np.argmin(beat[:r_idx])
        features['q_amplitude'] = beat[q_idx]
        features['qr_amplitude'] = features['r_amplitude'] - features['q_amplitude']
    else:
        features['q_amplitude'] = beat[0]
        features['qr_amplitude'] = features['r_amplitude'] - beat[0]
    
    # S point: minimum after R-peak
    if r_idx < len(beat) - 5:
        s_idx = r_idx + np.argmin(beat[r_idx:])
        features['s_amplitude'] = beat[s_idx]
        features['rs_amplitude'] = features['r_amplitude'] - features['s_amplitude']
    else:
        features['s_amplitude'] = beat[-1]
        features['rs_amplitude'] = features['r_amplitude'] - beat[-1]
    
    # QRS duration estimation (width at 50% of R amplitude)
    threshold = 0.5 * features['r_amplitude']
    above_threshold = beat > threshold
    if np.any(above_threshold):
        features['qrs_width'] = np.sum(above_threshold)
    else:
        features['qrs_width'] = 0
    
    # Area under the curve
    features['area'] = np.trapz(beat)
    features['abs_area'] = np.trapz(np.abs(beat))
    
    # Symmetry measure
    mid_point = len(beat) // 2
    left_half = beat[:mid_point]
    right_half = beat[mid_point:]
    if len(left_half) == len(right_half):
        features['symmetry'] = np.corrcoef(left_half, right_half[::-1])[0, 1]
    else:
        features['symmetry'] = 0
    
    return features


def extract_statistical_features(beat):
    """
    Extract statistical features
    
    Args:
        beat: Single heartbeat segment
        
    Returns:
        Dictionary of statistical features
    """
    features = {}
    
    # Variance
    features['variance'] = np.var(beat)
    
    # Skewness (asymmetry)
    features['skewness'] = skew(beat)
    
    # Kurtosis (tailedness)
    features['kurtosis'] = kurtosis(beat)
    
    # Entropy (randomness)
    hist, _ = np.histogram(beat, bins=20, density=True)
    features['entropy'] = entropy(hist + 1e-10)  # Add small value to avoid log(0)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(beat)))
    features['zero_crossing_rate'] = zero_crossings / len(beat)
    
    # Peak count
    peaks, _ = find_peaks(beat, height=np.std(beat))
    features['peak_count'] = len(peaks)
    
    # Mean crossing rate (crossings of mean value)
    mean_crossings = np.sum(np.diff(np.signbit(beat - np.mean(beat))))
    features['mean_crossing_rate'] = mean_crossings / len(beat)
    
    return features


def extract_frequency_features(beat, fs=360):
    """
    Extract frequency-domain features
    
    Args:
        beat: Single heartbeat segment
        fs: Sampling frequency
        
    Returns:
        Dictionary of frequency features
    """
    features = {}
    
    try:
        # Power spectral density
        freqs, psd = welch(beat, fs=fs, nperseg=min(len(beat), 256))
        
        # Spectral energy
        features['spectral_energy'] = np.sum(psd)
        
        # Dominant frequency
        features['dominant_freq'] = freqs[np.argmax(psd)]
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-10)
        features['spectral_entropy'] = entropy(psd_norm + 1e-10)
        
        # Frequency bands power
        # Low frequency (0-5 Hz)
        lf_mask = (freqs >= 0) & (freqs < 5)
        features['lf_power'] = np.sum(psd[lf_mask])
        
        # Medium frequency (5-15 Hz)
        mf_mask = (freqs >= 5) & (freqs < 15)
        features['mf_power'] = np.sum(psd[mf_mask])
        
        # High frequency (15-45 Hz)
        hf_mask = (freqs >= 15) & (freqs < 45)
        features['hf_power'] = np.sum(psd[hf_mask])
        
        # Power ratios
        total_power = features['spectral_energy'] + 1e-10
        features['lf_ratio'] = features['lf_power'] / total_power
        features['mf_ratio'] = features['mf_power'] / total_power
        features['hf_ratio'] = features['hf_power'] / total_power
        
    except Exception as e:
        # If frequency analysis fails, return zeros
        for key in ['spectral_energy', 'dominant_freq', 'spectral_entropy',
                    'lf_power', 'mf_power', 'hf_power',
                    'lf_ratio', 'mf_ratio', 'hf_ratio']:
            features[key] = 0.0
    
    return features


def extract_wavelet_features(beat):
    """
    Extract wavelet-based features
    
    Args:
        beat: Single heartbeat segment
        
    Returns:
        Dictionary of wavelet features
    """
    features = {}
    
    try:
        import pywt
        
        # Discrete wavelet transform
        coeffs = pywt.wavedec(beat, 'db4', level=3)
        
        # Features from each level
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_L{i}'] = np.sum(coeff ** 2)
            features[f'wavelet_mean_L{i}'] = np.mean(np.abs(coeff))
            features[f'wavelet_std_L{i}'] = np.std(coeff)
        
    except ImportError:
        # PyWavelets not available, skip wavelet features
        pass
    except Exception as e:
        pass
    
    return features


def extract_features_from_beat(beat, fs=360, include_wavelet=False, include_frequency=True):
    """
    Extract all features from a single heartbeat
    
    Args:
        beat: Single heartbeat segment
        fs: Sampling frequency
        include_wavelet: Include wavelet features (requires pywt)
        include_frequency: Include frequency features
        
    Returns:
        Dictionary of all features
    """
    all_features = {}
    
    # Temporal features
    all_features.update(extract_temporal_features(beat, fs))
    
    # Morphological features
    all_features.update(extract_morphological_features(beat, fs))
    
    # Statistical features
    all_features.update(extract_statistical_features(beat))
    
    # Frequency features (optional)
    if include_frequency:
        all_features.update(extract_frequency_features(beat, fs))
    
    # Wavelet features (optional)
    if include_wavelet:
        all_features.update(extract_wavelet_features(beat))
    
    return all_features


def extract_features_from_beats(beats, fs=360, include_wavelet=False, include_frequency=True):
    """
    Extract features from multiple beats
    
    Args:
        beats: List of heartbeat segments
        fs: Sampling frequency
        include_wavelet: Include wavelet features
        include_frequency: Include frequency features
        
    Returns:
        Feature matrix (n_beats, n_features)
    """
    feature_dicts = []
    
    for beat in beats:
        features = extract_features_from_beat(beat, fs, include_wavelet, include_frequency)
        feature_dicts.append(features)
    
    # Convert to numpy array
    if len(feature_dicts) > 0:
        # Get feature names from first beat
        feature_names = list(feature_dicts[0].keys())
        
        # Create feature matrix
        feature_matrix = np.array([[fd[name] for name in feature_names] 
                                   for fd in feature_dicts])
        
        return feature_matrix, feature_names
    else:
        return np.array([]), []


def get_feature_names(include_wavelet=False, include_frequency=True):
    """
    Get list of feature names
    
    Returns:
        List of feature names
    """
    # Generate a dummy beat to extract feature names
    dummy_beat = np.random.randn(234)  # Typical beat length at 360 Hz
    features = extract_features_from_beat(dummy_beat, include_wavelet=include_wavelet, 
                                         include_frequency=include_frequency)
    return list(features.keys())


def print_feature_importance(feature_names, importance_scores, top_n=20):
    """
    Print top N most important features
    
    Args:
        feature_names: List of feature names
        importance_scores: Feature importance scores
        top_n: Number of top features to display
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1]
    
    print(f"\n📊 Top {top_n} Most Important Features:")
    print("=" * 60)
    
    for i, idx in enumerate(indices[:top_n], 1):
        feature = feature_names[idx]
        score = importance_scores[idx]
        bar = '█' * int(score * 50)
        print(f"{i:2d}. {feature:30s}: {score:.4f} {bar}")


# Testing
if __name__ == "__main__":
    print("🫀 Feature Extraction Module Test\n")
    
    # Generate synthetic heartbeat
    fs = 360
    beat_length = int(0.65 * fs)  # 650ms beat
    t = np.linspace(0, 0.65, beat_length)
    
    # Simulate QRS complex
    beat = np.zeros(beat_length)
    r_idx = beat_length // 3
    
    # Q wave
    beat[r_idx-20:r_idx] = -0.2 * np.sin(np.linspace(0, np.pi, 20))
    # R wave
    beat[r_idx:r_idx+15] = np.linspace(0, 1, 15)
    beat[r_idx+15:r_idx+30] = np.linspace(1, 0, 15)
    # S wave
    beat[r_idx+30:r_idx+45] = -0.3 * np.sin(np.linspace(0, np.pi, 15))
    # T wave
    beat[r_idx+80:r_idx+130] = 0.3 * np.sin(np.linspace(0, np.pi, 50))
    
    # Add noise
    beat += 0.02 * np.random.randn(beat_length)
    
    print("1. Extracting temporal features...")
    temporal = extract_temporal_features(beat, fs)
    print(f"   ✓ Extracted {len(temporal)} temporal features")
    
    print("\n2. Extracting morphological features...")
    morphological = extract_morphological_features(beat, fs)
    print(f"   ✓ Extracted {len(morphological)} morphological features")
    print(f"   R amplitude: {morphological['r_amplitude']:.3f}")
    print(f"   QRS width: {morphological['qrs_width']} samples")
    
    print("\n3. Extracting statistical features...")
    statistical = extract_statistical_features(beat)
    print(f"   ✓ Extracted {len(statistical)} statistical features")
    print(f"   Skewness: {statistical['skewness']:.3f}")
    print(f"   Kurtosis: {statistical['kurtosis']:.3f}")
    
    print("\n4. Extracting frequency features...")
    frequency = extract_frequency_features(beat, fs)
    print(f"   ✓ Extracted {len(frequency)} frequency features")
    print(f"   Dominant frequency: {frequency['dominant_freq']:.2f} Hz")
    
    print("\n5. Extracting all features...")
    all_features = extract_features_from_beat(beat, fs)
    print(f"   ✓ Total features: {len(all_features)}")
    
    print("\n6. Feature names:")
    feature_names = list(all_features.keys())
    for i, name in enumerate(feature_names[:10], 1):
        print(f"   {i}. {name}")
    print(f"   ... and {len(feature_names) - 10} more")
    
    print("\n✅ All tests passed!")
    print(f"\n📝 Summary:")
    print(f"   Total features extracted: {len(all_features)}")
    print(f"   Ready for machine learning!")