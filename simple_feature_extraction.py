"""
Simplified Feature Extraction for Real-Time Prediction
Extracts essential features from ECG beats for classification
"""

import numpy as np
from scipy.signal import find_peaks


def extract_beat_features(beat, fs=360):
    """
    Extract features from a single heartbeat segment
    
    Args:
        beat: ECG beat segment (numpy array)
        fs: Sampling frequency (default 360 Hz)
        
    Returns:
        dict: Dictionary of features
    """
    features = {}
    
    # Ensure beat is numpy array
    beat = np.array(beat)
    
    # 1. TEMPORAL FEATURES (10 features)
    features['mean'] = np.mean(beat)
    features['std'] = np.std(beat)
    features['median'] = np.median(beat)
    features['min'] = np.min(beat)
    features['max'] = np.max(beat)
    features['range'] = features['max'] - features['min']
    features['ptp'] = np.ptp(beat)
    features['rms'] = np.sqrt(np.mean(beat ** 2))
    features['energy'] = np.sum(beat ** 2)
    features['mad'] = np.mean(np.abs(beat - features['mean']))
    
    # 2. MORPHOLOGICAL FEATURES (10 features)
    try:
        # Find R-peak (maximum point)
        r_idx = np.argmax(beat)
        features['r_amplitude'] = beat[r_idx]
        features['r_position'] = r_idx / len(beat)
        
        # Q point (minimum before R)
        if r_idx > 5:
            q_idx = np.argmin(beat[:r_idx])
            features['q_amplitude'] = beat[q_idx]
            features['qr_amplitude'] = features['r_amplitude'] - features['q_amplitude']
        else:
            features['q_amplitude'] = beat[0]
            features['qr_amplitude'] = features['r_amplitude'] - beat[0]
        
        # S point (minimum after R)
        if r_idx < len(beat) - 5:
            s_idx = r_idx + np.argmin(beat[r_idx:])
            features['s_amplitude'] = beat[s_idx]
            features['rs_amplitude'] = features['r_amplitude'] - features['s_amplitude']
        else:
            features['s_amplitude'] = beat[-1]
            features['rs_amplitude'] = features['r_amplitude'] - beat[-1]
        
        # QRS width estimation
        threshold = 0.5 * features['r_amplitude']
        above_threshold = beat > threshold
        features['qrs_width'] = np.sum(above_threshold) if np.any(above_threshold) else 0
        
        # Area measures
        features['area'] = np.trapz(beat)
        features['abs_area'] = np.trapz(np.abs(beat))
        
        # Symmetry
        mid_point = len(beat) // 2
        left_half = beat[:mid_point]
        right_half = beat[mid_point:mid_point + len(left_half)]
        if len(left_half) == len(right_half) and len(left_half) > 0:
            features['symmetry'] = np.corrcoef(left_half, right_half[::-1])[0, 1]
        else:
            features['symmetry'] = 0
            
    except Exception as e:
        # If morphological extraction fails, use defaults
        features['r_amplitude'] = features['max']
        features['r_position'] = 0.5
        features['q_amplitude'] = features['min']
        features['qr_amplitude'] = features['range']
        features['s_amplitude'] = features['min']
        features['rs_amplitude'] = features['range']
        features['qrs_width'] = len(beat) * 0.3
        features['area'] = np.sum(beat)
        features['abs_area'] = np.sum(np.abs(beat))
        features['symmetry'] = 0
    
    # 3. STATISTICAL FEATURES (8 features)
    features['variance'] = np.var(beat)
    
    from scipy.stats import skew, kurtosis, entropy
    features['skewness'] = skew(beat)
    features['kurtosis'] = kurtosis(beat)
    
    # Entropy
    hist, _ = np.histogram(beat, bins=20, density=True)
    features['entropy'] = entropy(hist + 1e-10)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(beat)))
    features['zero_crossing_rate'] = zero_crossings / len(beat)
    
    # Peak count
    try:
        peaks, _ = find_peaks(beat, height=np.std(beat))
        features['peak_count'] = len(peaks)
    except:
        features['peak_count'] = 1
    
    # Mean crossing rate
    mean_crossings = np.sum(np.diff(np.signbit(beat - np.mean(beat))))
    features['mean_crossing_rate'] = mean_crossings / len(beat)
    
    # 4. FREQUENCY FEATURES (4 simplified features)
    try:
        from scipy.signal import welch
        freqs, psd = welch(beat, fs=fs, nperseg=min(len(beat), 256))
        
        features['spectral_energy'] = np.sum(psd)
        features['dominant_freq'] = freqs[np.argmax(psd)]
        
        # Low frequency power (0-5 Hz)
        lf_mask = (freqs >= 0) & (freqs < 5)
        features['lf_power'] = np.sum(psd[lf_mask])
        
        # High frequency power (15-45 Hz)
        hf_mask = (freqs >= 15) & (freqs < 45)
        features['hf_power'] = np.sum(psd[hf_mask])
        
    except Exception as e:
        features['spectral_energy'] = 0
        features['dominant_freq'] = 0
        features['lf_power'] = 0
        features['hf_power'] = 0
    
    return features


def extract_features_array(beat, fs=360):
    """
    Extract features and return as ordered numpy array
    Compatible with trained model input
    
    Args:
        beat: ECG beat segment
        fs: Sampling frequency
        
    Returns:
        numpy array of features (32 features)
    """
    features_dict = extract_beat_features(beat, fs)
    
    # Order matters! Must match training order
    feature_order = [
        # Temporal (10)
        'mean', 'std', 'median', 'min', 'max', 'range', 'ptp', 'rms', 'energy', 'mad',
        # Morphological (10)
        'r_amplitude', 'r_position', 'q_amplitude', 'qr_amplitude', 's_amplitude',
        'rs_amplitude', 'qrs_width', 'area', 'abs_area', 'symmetry',
        # Statistical (8)
        'variance', 'skewness', 'kurtosis', 'entropy', 
        'zero_crossing_rate', 'peak_count', 'mean_crossing_rate',
        # Frequency (4)
        'spectral_energy', 'dominant_freq', 'lf_power', 'hf_power'
    ]
    
    # Extract in correct order (32 features total)
    features_array = np.array([features_dict.get(f, 0) for f in feature_order[:32]])
    
    return features_array


def segment_and_extract_features(signal, r_peaks, fs=360):
    """
    Segment beats around R-peaks and extract features from each
    
    Args:
        signal: Full ECG signal
        r_peaks: R-peak locations (indices)
        fs: Sampling frequency
        
    Returns:
        features_matrix: (n_beats, n_features) array
    """
    # Beat window: 250ms before, 400ms after R-peak
    before_samples = int(0.25 * fs)  # 90 samples at 360 Hz
    after_samples = int(0.4 * fs)    # 144 samples at 360 Hz
    
    features_list = []
    
    for r_idx in r_peaks:
        # Extract beat segment
        start = max(0, r_idx - before_samples)
        end = min(len(signal), r_idx + after_samples)
        
        beat = signal[start:end]
        
        # Skip if beat is too short
        if len(beat) < 50:
            continue
        
        # Normalize beat
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-10)
        
        # Extract features
        features = extract_features_array(beat, fs)
        features_list.append(features)
    
    if len(features_list) == 0:
        return np.array([])
    
    return np.array(features_list)


# Testing
if __name__ == "__main__":
    print("🧪 Testing Simple Feature Extraction\n")
    
    # Generate test beat
    fs = 360
    beat_length = int(0.65 * fs)
    t = np.linspace(0, 0.65, beat_length)
    
    # Simulate QRS complex
    beat = np.zeros(beat_length)
    r_idx = beat_length // 3
    beat[r_idx:r_idx+15] = np.linspace(0, 1, 15)
    beat[r_idx+15:r_idx+30] = np.linspace(1, 0, 15)
    beat += 0.02 * np.random.randn(beat_length)
    
    # Extract features
    features = extract_beat_features(beat, fs)
    print(f"✅ Extracted {len(features)} features")
    
    # Extract as array
    features_array = extract_features_array(beat, fs)
    print(f"✅ Feature array shape: {features_array.shape}")
    print(f"   First 5 features: {features_array[:5]}")
    
    print("\n✅ Feature extraction ready!")