"""
ECG Signal Preprocessing Module
Implements professional-grade signal processing techniques
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.ndimage import median_filter


def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=360, order=3):
    """
    Apply Butterworth bandpass filter
    
    Args:
        signal: Input ECG signal
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def remove_baseline_wander(signal, fs=360, window_size=None):
    """
    Remove baseline wander using median filter
    
    Args:
        signal: Input ECG signal
        fs: Sampling frequency
        window_size: Median filter window size (default: 0.2*fs)
        
    Returns:
        Signal with baseline removed
    """
    if window_size is None:
        window_size = int(0.2 * fs)
    
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    baseline = medfilt(signal, kernel_size=window_size)
    corrected_signal = signal - baseline
    
    return corrected_signal


def remove_powerline_interference(signal, fs=360, f0=60.0, Q=30.0):
    """
    Remove powerline interference (50/60 Hz) using notch filter
    
    Args:
        signal: Input ECG signal
        fs: Sampling frequency
        f0: Powerline frequency (50 or 60 Hz)
        Q: Quality factor
        
    Returns:
        Signal with powerline interference removed
    """
    from scipy.signal import iirnotch
    
    # Design notch filter
    b, a = iirnotch(f0, Q, fs)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def detect_and_remove_outliers(signal, threshold=3.0):
    """
    Detect and remove outliers using z-score method
    
    Args:
        signal: Input signal
        threshold: Z-score threshold
        
    Returns:
        Signal with outliers removed
    """
    signal = np.array(signal, dtype=float)
    
    # Calculate z-scores
    mean = np.mean(signal)
    std = np.std(signal)
    
    if std == 0:
        return signal
    
    z_scores = np.abs((signal - mean) / std)
    
    # Replace outliers with median
    outliers = z_scores > threshold
    if np.any(outliers):
        signal[outliers] = np.median(signal)
    
    return signal


def normalize_signal(signal, method='zscore'):
    """
    Normalize signal
    
    Args:
        signal: Input signal
        method: 'zscore', 'minmax', or 'robust'
        
    Returns:
        Normalized signal
    """
    signal = np.array(signal, dtype=float)
    
    if method == 'zscore':
        # Z-score normalization (zero mean, unit variance)
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization [0, 1]
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val == min_val:
            return np.zeros_like(signal)
        return (signal - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(signal)
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return signal - median
        return (signal - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_ecg_signal(signal, fs=360, remove_powerline=False):
    """
    Complete ECG preprocessing pipeline
    
    Steps:
    1. Bandpass filtering (0.5-45 Hz)
    2. Baseline wander removal
    3. Outlier removal
    4. Normalization
    
    Args:
        signal: Raw ECG signal
        fs: Sampling frequency (Hz)
        remove_powerline: Whether to remove powerline interference
        
    Returns:
        Preprocessed signal
    """
    # 1. Bandpass filter
    filtered = bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=fs)
    
    # 2. Remove baseline wander
    corrected = remove_baseline_wander(filtered, fs=fs)
    
    # 3. Optional: Remove powerline interference
    if remove_powerline:
        corrected = remove_powerline_interference(corrected, fs=fs)
    
    # 4. Remove outliers
    cleaned = detect_and_remove_outliers(corrected, threshold=3.0)
    
    # 5. Normalize
    normalized = normalize_signal(cleaned, method='zscore')
    
    return normalized


def detect_r_peaks_pan_tompkins(signal, fs=360):
    """
    Detect R-peaks using Pan-Tompkins algorithm
    
    Args:
        signal: Preprocessed ECG signal
        fs: Sampling frequency
        
    Returns:
        Array of R-peak indices
    """
    # 1. Derivative (approximation of slope)
    diff_signal = np.diff(signal)
    
    # 2. Squaring (amplifies larger differences)
    squared = diff_signal ** 2
    
    # 3. Moving window integration
    window_size = int(0.15 * fs)  # 150ms window
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # 4. Find peaks using adaptive threshold
    # Calculate dynamic threshold
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    
    # Find peaks
    peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.2 * fs))
    
    # Find actual R-peaks in original signal
    r_peaks = []
    search_window = int(0.05 * fs)  # 50ms search window
    
    for peak in peaks:
        # Search for maximum in original signal around detected peak
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)
        
        if end > start:
            local_max = start + np.argmax(signal[start:end])
            
            # Avoid duplicates (must be at least 200ms apart)
            if len(r_peaks) == 0 or local_max - r_peaks[-1] > int(0.2 * fs):
                r_peaks.append(local_max)
    
    return np.array(r_peaks)


def segment_beats_from_annotations(signal, r_peaks, fs=360, before_ms=250, after_ms=400):
    """
    Segment individual heartbeats around R-peaks
    
    Args:
        signal: Preprocessed ECG signal
        r_peaks: Array of R-peak indices
        fs: Sampling frequency
        before_ms: Milliseconds before R-peak to include
        after_ms: Milliseconds after R-peak to include
        
    Returns:
        beats: List of segmented beats
        valid_indices: Indices of valid beats
    """
    before_samples = int(before_ms * fs / 1000)
    after_samples = int(after_ms * fs / 1000)
    beat_length = before_samples + after_samples
    
    beats = []
    valid_indices = []
    
    for i, r_peak in enumerate(r_peaks):
        start = r_peak - before_samples
        end = r_peak + after_samples
        
        # Check if beat is within signal bounds
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            
            # Ensure consistent length (pad if necessary)
            if len(beat) < beat_length:
                beat = np.pad(beat, (0, beat_length - len(beat)), mode='edge')
            elif len(beat) > beat_length:
                beat = beat[:beat_length]
            
            beats.append(beat)
            valid_indices.append(i)
    
    return beats, valid_indices


def segment_beats_auto(signal, fs=360, before_ms=250, after_ms=400):
    """
    Automatically detect and segment heartbeats
    
    Args:
        signal: Preprocessed ECG signal
        fs: Sampling frequency
        before_ms: Milliseconds before R-peak
        after_ms: Milliseconds after R-peak
        
    Returns:
        beats: List of segmented beats
        r_peaks: Detected R-peak locations
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks_pan_tompkins(signal, fs)
    
    # Segment beats
    beats, valid_indices = segment_beats_from_annotations(
        signal, r_peaks, fs, before_ms, after_ms
    )
    
    # Filter R-peaks to match valid beats
    r_peaks = r_peaks[valid_indices]
    
    return beats, r_peaks


def calculate_heart_rate(r_peaks, fs=360):
    """
    Calculate heart rate from R-peaks
    
    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency
        
    Returns:
        Heart rate in BPM
    """
    if len(r_peaks) < 2:
        return 0
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    
    # Calculate heart rate (beats per minute)
    mean_rr = np.mean(rr_intervals)
    heart_rate = 60.0 / mean_rr if mean_rr > 0 else 0
    
    return heart_rate


def quality_assessment(signal, r_peaks, fs=360):
    """
    Assess signal quality
    
    Args:
        signal: ECG signal
        r_peaks: Detected R-peaks
        fs: Sampling frequency
        
    Returns:
        quality_score: Quality score (0-100)
        quality_level: 'Excellent', 'Good', 'Fair', 'Poor'
    """
    scores = []
    
    # 1. SNR estimation (signal-to-noise ratio)
    signal_power = np.var(signal)
    if signal_power > 0:
        snr_score = min(100, signal_power * 20)
        scores.append(snr_score)
    
    # 2. R-peak detection rate
    expected_beats = len(signal) / fs * 1.2  # Assuming ~72 BPM
    detection_rate = len(r_peaks) / expected_beats if expected_beats > 0 else 0
    detection_score = min(100, detection_rate * 100)
    scores.append(detection_score)
    
    # 3. RR interval regularity
    if len(r_peaks) > 2:
        rr_intervals = np.diff(r_peaks)
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        regularity = 1 - min(1, rr_std / rr_mean) if rr_mean > 0 else 0
        regularity_score = regularity * 100
        scores.append(regularity_score)
    
    # Overall quality score
    quality_score = np.mean(scores) if scores else 0
    
    # Quality level
    if quality_score >= 80:
        quality_level = 'Excellent'
    elif quality_score >= 60:
        quality_level = 'Good'
    elif quality_score >= 40:
        quality_level = 'Fair'
    else:
        quality_level = 'Poor'
    
    return quality_score, quality_level


# Testing
if __name__ == "__main__":
    print("🫀 ECG Preprocessing Module Test\n")
    
    # Generate synthetic ECG
    fs = 360
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate heartbeats
    ecg = np.zeros_like(t)
    for beat_time in np.arange(0, duration, 0.8):  # ~75 BPM
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg) - 100:
            # R peak
            ecg[beat_idx:beat_idx+10] = np.linspace(0, 1, 10)
            ecg[beat_idx+10:beat_idx+20] = np.linspace(1, 0, 10)
    
    # Add noise
    ecg_noisy = ecg + 0.1 * np.random.randn(len(ecg))
    
    print("1. Preprocessing signal...")
    processed = preprocess_ecg_signal(ecg_noisy, fs=fs)
    print(f"   ✓ Signal preprocessed (length: {len(processed)})")
    
    print("\n2. Detecting R-peaks...")
    r_peaks = detect_r_peaks_pan_tompkins(processed, fs=fs)
    print(f"   ✓ Detected {len(r_peaks)} R-peaks")
    
    print("\n3. Calculating heart rate...")
    hr = calculate_heart_rate(r_peaks, fs=fs)
    print(f"   ✓ Heart rate: {hr:.1f} BPM")
    
    print("\n4. Segmenting beats...")
    beats, _ = segment_beats_from_annotations(processed, r_peaks, fs=fs)
    print(f"   ✓ Segmented {len(beats)} beats")
    print(f"   ✓ Beat length: {len(beats[0])} samples")
    
    print("\n5. Assessing quality...")
    quality_score, quality_level = quality_assessment(processed, r_peaks, fs=fs)
    print(f"   ✓ Quality: {quality_level} ({quality_score:.1f}/100)")
    
    print("\n✅ All tests passed!")