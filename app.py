"""
Flask Backend API for Arrhythmia Detection System
Complete integration with MIT-BIH database
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_loader import MITBIHDataLoader
from preprocessing import (preprocess_ecg_signal, segment_beats_auto, 
                          detect_r_peaks_pan_tompkins, calculate_heart_rate)
from feature_extraction import extract_features_from_beat
from models.train_models import ArrhythmiaClassifier

app = Flask(__name__)
CORS(app)

# Global variables
classifier = ArrhythmiaClassifier()
data_loader = MITBIHDataLoader()
current_dataset = None
is_trained = False


# ============= UTILITY FUNCTIONS =============

def create_ecg_plot(signal, r_peaks=None, sampling_rate=360, title="ECG Signal"):
    """Create ECG signal plot"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time_axis = np.arange(len(signal)) / sampling_rate
    ax.plot(time_axis, signal, 'b-', linewidth=1, label='ECG Signal')
    
    if r_peaks is not None and len(r_peaks) > 0:
        ax.plot(r_peaks/sampling_rate, signal[r_peaks], 'ro', 
               markersize=8, label=f'R-peaks ({len(r_peaks)})')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def create_results_plot(signal, r_peaks, predictions, sampling_rate=360):
    """Create comprehensive results visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original signal with R-peaks
    time_axis = np.arange(len(signal)) / sampling_rate
    ax1.plot(time_axis, signal, 'b-', linewidth=1, alpha=0.7, label='ECG Signal')
    
    if len(r_peaks) > 0:
        ax1.plot(r_peaks/sampling_rate, signal[r_peaks], 'ro', 
                markersize=8, label='R-peaks', zorder=5)
        
        # Color code beats by prediction
        colors = {'N': 'green', 'S': 'orange', 'V': 'red'}
        labels_added = set()
        
        for i, (r_peak, pred) in enumerate(zip(r_peaks, predictions)):
            if i < len(predictions):
                color = colors.get(pred, 'gray')
                label = f'{pred} ({"Normal" if pred=="N" else "Supraventricular" if pred=="S" else "Ventricular"})'
                
                if pred not in labels_added:
                    ax1.axvspan(r_peak/sampling_rate - 0.2, r_peak/sampling_rate + 0.2,
                              alpha=0.3, color=color, label=label)
                    labels_added.add(pred)
                else:
                    ax1.axvspan(r_peak/sampling_rate - 0.2, r_peak/sampling_rate + 0.2,
                              alpha=0.3, color=color)
    
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('ECG Signal with Arrhythmia Detection', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Class distribution pie chart
    unique, counts = np.unique(predictions, return_counts=True)
    colors_pie = [colors.get(cls, 'gray') for cls in unique]
    labels_pie = [f'{cls} ({"Normal" if cls=="N" else "Supraventricular" if cls=="S" else "Ventricular"})' 
                  for cls in unique]
    
    ax2.pie(counts, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Beat Classification Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def create_comparison_plot(results):
    """Create model comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] * 100 for model in models]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def generate_ventricular_beat(ecg, beat_idx, sampling_rate):
    """
    Generate a Ventricular Ectopic Beat (PVC)
    Characteristics:
    - Wide QRS (>120ms) - CRITICAL FEATURE
    - Bizarre morphology (different shape)
    - Large amplitude
    - No preceding P-wave
    - Inverted T-wave
    """
    # VERY WIDE QRS complex (180ms - clearly abnormal)
    width = 0.18 * sampling_rate  # 65 samples = 180ms (VERY WIDE!)
    qrs_start = max(0, beat_idx - int(width // 2))
    qrs_end = min(len(ecg), beat_idx + int(width // 2))
    
    # Much higher amplitude
    amplitude = 3.0  # Increased from 2.5
    
    # Create bizarre, asymmetric shape with multiple components
    num_samples = qrs_end - qrs_start
    x = np.linspace(-2, 1.5, num_samples)  # Asymmetric range
    
    # Triple-component bizarre morphology
    # Main spike (shifted and asymmetric)
    gaussian1 = amplitude * np.exp(-(x - 0.3)**2 * 6)  # Shifted right
    
    # Secondary notch (characteristic of bundle branch block)
    gaussian2 = amplitude * 0.7 * np.exp(-(x + 0.5)**2 * 10)  # Left notch
    
    # Third component (makes it bizarre)
    gaussian3 = amplitude * 0.4 * np.exp(-(x - 0.8)**2 * 8)  # Right shoulder
    
    # Combine all components
    bizarre_waveform = gaussian1 + gaussian2 + gaussian3
    
    # Add some asymmetry and distortion
    bizarre_waveform += 0.3 * np.sin(np.linspace(0, 4*np.pi, num_samples))
    
    ecg[qrs_start:qrs_end] += bizarre_waveform
    
    # Add prominent inverted T-wave (discordant - very characteristic of PVC)
    t_start = beat_idx + int(0.1 * sampling_rate)  # 100ms after R-peak
    t_end = min(len(ecg), beat_idx + int(0.35 * sampling_rate))  # 350ms window
    if t_end > t_start:
        t_x = np.linspace(-1.5, 1.5, t_end - t_start)
        inverted_t = -0.8 * np.exp(-t_x**2 * 3)  # Deep inverted T-wave
        ecg[t_start:t_end] += inverted_t
    
    # Remove P-wave area (PVCs have no P-wave)
    p_start = max(0, beat_idx - int(0.15 * sampling_rate))
    p_end = beat_idx - int(0.05 * sampling_rate)
    if p_end > p_start and p_end < len(ecg):
        # Flatten the P-wave region
        ecg[p_start:p_end] *= 0.5
    
    return ecg, 'V'


def generate_supraventricular_beat(ecg, beat_idx, sampling_rate, beat_time, beat_interval):
    """
    Generate a Supraventricular Ectopic Beat (PAC)
    Characteristics:
    - Narrow QRS (<100ms) - similar to normal
    - PREMATURE occurrence (key feature!)
    - Abnormal/inverted P-wave
    - Compensatory pause after
    """
    # Make it VERY premature (30% earlier than expected)
    premature_shift = int(-0.3 * beat_interval * sampling_rate)
    beat_idx = beat_idx + premature_shift
    beat_idx = max(0, min(beat_idx, len(ecg) - 100))
    
    # NARROW QRS (80ms - clearly narrow)
    width = 0.08 * sampling_rate  # 29 samples = 80ms (narrow)
    qrs_start = max(0, beat_idx - int(width // 2))
    qrs_end = min(len(ecg), beat_idx + int(width // 2))
    
    # Lower amplitude (characteristic of PAC)
    amplitude = 0.85
    
    # Normal narrow QRS shape
    if qrs_end > qrs_start:
        ecg[qrs_start:qrs_end] += amplitude * np.exp(
            -np.linspace(-1, 1, qrs_end - qrs_start)**2 * 15  # Sharper peak
        )
    
    # ABNORMAL P-WAVE (inverted or different morphology)
    p_start = max(0, beat_idx - int(0.12 * sampling_rate))
    p_end = beat_idx - int(0.03 * sampling_rate)
    if p_end > p_start and p_end < len(ecg):
        p_x = np.linspace(-1, 1, p_end - p_start)
        # Inverted or biphasic P-wave (abnormal)
        abnormal_p = -0.2 * np.exp(-p_x**2 * 5)  # Inverted P-wave
        ecg[p_start:p_end] += abnormal_p
    
    # Slightly different T-wave
    t_start = beat_idx + int(0.08 * sampling_rate)
    t_end = min(len(ecg), beat_idx + int(0.25 * sampling_rate))
    if t_end > t_start:
        t_x = np.linspace(-1, 1, t_end - t_start)
        ecg[t_start:t_end] += 0.35 * np.exp(-t_x**2 * 4)
    
    return ecg, 'S'


def add_p_and_t_waves(ecg, beat_times, sampling_rate):
    """
    Add P-waves and T-waves to make ECG more realistic
    P-wave: Atrial depolarization (before QRS)
    T-wave: Ventricular repolarization (after QRS)
    """
    for beat_time in beat_times:
        beat_idx = int(beat_time * sampling_rate)
        
        if beat_idx >= len(ecg):
            continue
        
        # Add P-wave (60ms before QRS)
        p_start = max(0, beat_idx - int(0.12 * sampling_rate))
        p_end = beat_idx - int(0.06 * sampling_rate)
        if p_end > p_start and p_end < len(ecg):
            p_x = np.linspace(-1, 1, p_end - p_start)
            ecg[p_start:p_end] += 0.2 * np.exp(-p_x**2 * 6)
        
        # Add T-wave (200ms after QRS)
        t_start = beat_idx + int(0.1 * sampling_rate)
        t_end = min(len(ecg), beat_idx + int(0.3 * sampling_rate))
        if t_end > t_start:
            t_x = np.linspace(-1, 1, t_end - t_start)
            ecg[t_start:t_end] += 0.4 * np.exp(-t_x**2 * 3)
    
    return ecg


# ============= API ENDPOINTS =============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Arrhythmia Detection API is running",
        "trained": is_trained,
        "dataset_loaded": current_dataset is not None
    })


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train models with REAL MIT-BIH data"""
    global classifier, current_dataset, is_trained
    
    try:
        data = request.get_json() or {}
        use_real_data = data.get('use_real_data', True)
        num_records = data.get('num_records', 5)
        max_beats_per_record = data.get('max_beats_per_record', 400)
        
        print("\n" + "="*70)
        print("🫀 TRAINING ARRHYTHMIA DETECTION SYSTEM")
        print("="*70)
        
        if use_real_data:
            print(f"\n📥 Loading MIT-BIH Database ({num_records} records)...")
            X, y = data_loader.load_dataset(
                num_records=num_records,
                max_beats_per_record=max_beats_per_record,
                use_cache=True
            )
            current_dataset = {'X': X, 'y': y}
            data_source = f"MIT-BIH Arrhythmia Database ({num_records} records)"
        else:
            print("\n⚠️ Using synthetic data (demo mode)...")
            np.random.seed(42)
            n_samples = 200
            n_features = 40
            
            X = np.vstack([
                np.random.randn(n_samples, n_features) * 0.5,
                np.random.randn(n_samples, n_features) * 0.6 + 1.0,
                np.random.randn(n_samples, n_features) * 0.8 + 2.0
            ])
            y = np.array(['N'] * n_samples + ['S'] * n_samples + ['V'] * n_samples)
            current_dataset = {'X': X, 'y': y}
            data_source = "Synthetic Data (Demo Mode)"
        
        X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
        classifier.train_all_models(X_train, y_train, optimize=False, quick_mode=True)
        results = classifier.evaluate_all_models(X_test, y_test)
        
        # Format results for JSON - FIXED VERSION
        formatted_results = {}
        for name, metrics in results.items():
            formatted_results[name] = {}
            for m, v in metrics.items():
                # Skip non-numeric values (confusion_matrix, predictions, probabilities)
                if m in ['confusion_matrix', 'predictions', 'probabilities']:
                    continue
                # Convert to float if it's a numeric value
                if isinstance(v, (int, float, np.integer, np.floating)):
                    formatted_results[name][m] = float(v)
        
        comparison_plot = create_comparison_plot(formatted_results)
        is_trained = True
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!\n")
        
        return jsonify({
            "status": "success",
            "message": "Models trained successfully",
            "data_source": data_source,
            "total_samples": len(y),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "classes": classifier.label_encoder.classes_.tolist(),
            "best_model": classifier.best_model_name,
            "results": formatted_results,
            "comparison_plot": comparison_plot
        })
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/process-ecg', methods=['POST'])
def process_ecg():
    """Process ECG signal and detect arrhythmias"""
    global classifier, is_trained
    
    try:
        if not is_trained:
            return jsonify({
                "status": "error",
                "message": "Models not trained. Please train models first."
            }), 400
        
        data = request.get_json()
        ecg_signal = np.array(data.get('signal', []))
        sampling_rate = data.get('sampling_rate', 360)
        
        if len(ecg_signal) == 0:
            return jsonify({"status": "error", "message": "No signal provided"}), 400
        
        print(f"\n📊 Processing ECG signal ({len(ecg_signal)} samples)...")
        
        print("   1. Preprocessing...")
        processed_signal = preprocess_ecg_signal(ecg_signal, sampling_rate)
        
        print("   2. Detecting heartbeats...")
        beats, r_peaks = segment_beats_auto(processed_signal, sampling_rate)
        print(f"   ✓ Detected {len(beats)} heartbeats")
        
        print("   3. Extracting features...")
        features_list = []
        for beat in beats:
            beat_features = extract_features_from_beat(beat, sampling_rate)
            features_list.append(list(beat_features.values()))
        
        features = np.array(features_list)
        print(f"   ✓ Extracted features shape: {features.shape}")
        
        print("   4. Classifying arrhythmias...")
        predictions, probabilities = classifier.predict(features)
        
        heart_rate = calculate_heart_rate(r_peaks, sampling_rate)
        pred_counts = dict(zip(*np.unique(predictions, return_counts=True)))
        
        results_plot = create_results_plot(processed_signal, r_peaks, predictions, sampling_rate)
        
        print("✅ Processing complete!\n")
        
        return jsonify({
            "status": "success",
            "heart_rate": float(heart_rate),
            "num_beats": len(beats),
            "r_peaks": r_peaks.tolist(),
            "predictions": predictions.tolist(),
            "prediction_counts": {k: int(v) for k, v in pred_counts.items()},
            "best_model": classifier.best_model_name,
            "visualizations": {
                "results": results_plot
            }
        })
        
    except Exception as e:
        print(f"\n❌ Error processing ECG: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/generate-sample', methods=['GET'])
def generate_sample():
    """Generate synthetic ECG signal for testing with optional arrhythmias"""
    try:
        # Get parameters from query string
        include_arrhythmias = request.args.get('arrhythmias', 'false').lower() == 'true'
        num_ventricular = int(request.args.get('ventricular', 2))
        num_supraventricular = int(request.args.get('supraventricular', 1))
        
        sampling_rate = 360
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)
        ecg = np.zeros_like(t)
        heart_rate = 75
        beat_interval = 60 / heart_rate

        # Generate beat times
        beat_times = list(np.arange(0, duration, beat_interval))
        total_beats = len(beat_times)
        
        # Determine which beats will be abnormal
        ventricular_indices = []
        supraventricular_indices = []
        
        if include_arrhythmias:
            # Randomly select beats to be ventricular (but not first or last)
            if total_beats > 4:
                available_indices = list(range(2, total_beats - 2))
                np.random.shuffle(available_indices)
                
                # Select ventricular beats
                ventricular_indices = available_indices[:min(num_ventricular, len(available_indices))]
                
                # Select supraventricular beats (not overlapping with ventricular)
                remaining_indices = [i for i in available_indices if i not in ventricular_indices]
                supraventricular_indices = remaining_indices[:min(num_supraventricular, len(remaining_indices))]
        
        # Generate beats
        beat_types = []
        for i, beat_time in enumerate(beat_times):
            beat_idx = int(beat_time * sampling_rate)
            
            if beat_idx < len(ecg):
                if i in ventricular_indices:
                    # Generate VENTRICULAR beat
                    ecg, beat_type = generate_ventricular_beat(ecg, beat_idx, sampling_rate)
                    beat_types.append('V')
                    
                elif i in supraventricular_indices:
                    # Generate SUPRAVENTRICULAR beat
                    ecg, beat_type = generate_supraventricular_beat(ecg, beat_idx, sampling_rate, beat_time, beat_interval)
                    beat_types.append('S')
                    # Adjust next beat time (premature = early next beat)
                    if i + 1 < len(beat_times):
                        beat_times[i + 1] = beat_time + beat_interval * 1.1
                    
                else:
                    # Generate NORMAL beat
                    width = 0.05 * sampling_rate  # 50ms
                    qrs_start = max(0, beat_idx - int(width // 2))
                    qrs_end = min(len(ecg), beat_idx + int(width // 2))
                    amplitude = 1.2 if i % 7 != 0 else 1.5
                    
                    ecg[qrs_start:qrs_end] += amplitude * np.exp(
                        -np.linspace(-1, 1, qrs_end - qrs_start)**2 * 12
                    )
                    beat_types.append('N')
        
        # Add realistic noise
        ecg = ecg + 0.05 * np.random.randn(len(ecg))
        
        # Add baseline wander (breathing artifact)
        ecg += 0.1 * np.sin(2 * np.pi * 0.3 * t)
        
        # Add P-waves and T-waves for more realism
        ecg = add_p_and_t_waves(ecg, beat_times, sampling_rate)
        
        response_data = {
            "status": "success",
            "signal": ecg.tolist(),
            "sampling_rate": sampling_rate,
            "duration": duration,
            "include_arrhythmias": include_arrhythmias,
            "beat_types": beat_types,
            "total_beats": len(beat_types),
            "normal_beats": beat_types.count('N'),
            "ventricular_beats": beat_types.count('V'),
            "supraventricular_beats": beat_types.count('S')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n❌ Error generating sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get information about currently loaded dataset"""
    global current_dataset
    
    if current_dataset is None:
        return jsonify({
            "status": "error",
            "message": "No dataset loaded"
        }), 400
    
    X, y = current_dataset['X'], current_dataset['y']
    unique, counts = np.unique(y, return_counts=True)
    
    return jsonify({
        "status": "success",
        "total_samples": len(y),
        "num_features": X.shape[1],
        "classes": unique.tolist(),
        "class_distribution": dict(zip(unique.tolist(), counts.tolist()))
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🫀 ARRHYTHMIA DETECTION SYSTEM - BACKEND API")
    print("="*70)
    print("\n📡 Starting Flask server...")
    print("   URL: http://localhost:5000")
    print("   Health Check: http://localhost:5000/api/health")
    print("\n⚡ Ready to receive requests!\n")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')