"""
Comprehensive Flask Backend API
Shows progression: Basic → Intermediate → Advanced ML models
Perfect for AIML project demonstration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_dataset_loader import MultiDatasetLoader
from comprehensive_training import ComprehensiveArrhythmiaClassifier
from preprocessing import preprocess_ecg_signal, segment_beats_auto, calculate_heart_rate
from feature_extraction import extract_features_from_beat

app = Flask(__name__)
CORS(app)

# Global variables
classifier = ComprehensiveArrhythmiaClassifier()
dataset_loader = MultiDatasetLoader()
current_dataset = None
is_trained = False


def create_comprehensive_comparison_plot(results, classifier):
    """Create detailed comparison plot showing all model categories"""
    
    # Create comparison DataFrame
    comparison_df = classifier.create_comparison_dataframe(results)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. F1-Score comparison by category
    ax1 = fig.add_subplot(gs[0, :])
    
    colors_map = {
        'Basic': '#FF6B6B',
        'Intermediate': '#4ECDC4',
        'Advanced': '#45B7D1'
    }
    
    x_pos = np.arange(len(comparison_df))
    colors = [colors_map.get(cat, '#ccc') for cat in comparison_df['Category']]
    
    bars = ax1.barh(comparison_df['Model'], comparison_df['F1-Score'] * 100, color=colors)
    ax1.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison (Ranked by F1-Score)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 105])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, comparison_df['F1-Score'] * 100)):
        ax1.text(val + 1, i, f'{val:.2f}%', va='center', fontsize=9)
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) 
                      for cat, color in colors_map.items()]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # 2. Accuracy vs Training Time scatter
    ax2 = fig.add_subplot(gs[1, 0])
    
    for category, color in colors_map.items():
        cat_data = comparison_df[comparison_df['Category'] == category]
        ax2.scatter(cat_data['Training Time (s)'], 
                   cat_data['Accuracy'] * 100,
                   c=color, s=100, alpha=0.7, label=category, edgecolors='black')
    
    ax2.set_xlabel('Training Time (seconds)', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy vs Training Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. All metrics comparison for top 5 models
    ax3 = fig.add_subplot(gs[1, 1])
    
    top5 = comparison_df.head(5)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(top5))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = top5[metric] * 100
        ax3.bar(x + i * width, values, width, label=metric)
    
    ax3.set_xlabel('Top 5 Models', fontsize=11)
    ax3.set_ylabel('Score (%)', fontsize=11)
    ax3.set_title('Detailed Metrics for Top 5 Models', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(top5['Model'], rotation=15, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Category-wise average performance
    ax4 = fig.add_subplot(gs[2, 0])
    
    category_avg = comparison_df.groupby('Category')[['Accuracy', 'F1-Score']].mean() * 100
    category_avg.plot(kind='bar', ax=ax4, color=['#667eea', '#764ba2'])
    ax4.set_xlabel('Category', fontsize=11)
    ax4.set_ylabel('Average Score (%)', fontsize=11)
    ax4.set_title('Average Performance by Model Category', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.legend(['Accuracy', 'F1-Score'])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 105])
    
    # 5. Training time by category
    ax5 = fig.add_subplot(gs[2, 1])
    
    category_time = comparison_df.groupby('Category')['Training Time (s)'].mean()
    bars = ax5.bar(category_time.index, category_time.values, 
                   color=[colors_map[cat] for cat in category_time.index])
    ax5.set_xlabel('Category', fontsize=11)
    ax5.set_ylabel('Average Training Time (s)', fontsize=11)
    ax5.set_title('Training Efficiency by Category', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('🫀 Comprehensive Model Comparison Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def create_results_plot(signal, r_peaks, predictions, sampling_rate=360):
    """Create ECG analysis visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # ECG signal with predictions
    time_axis = np.arange(len(signal)) / sampling_rate
    ax1.plot(time_axis, signal, 'b-', linewidth=1, alpha=0.7, label='ECG Signal')
    
    if len(r_peaks) > 0:
        ax1.plot(r_peaks/sampling_rate, signal[r_peaks], 'ro', 
                markersize=8, label='R-peaks', zorder=5)
        
        colors = {'N': 'green', 'S': 'orange', 'V': 'red'}
        labels_added = set()
        
        for i, (r_peak, pred) in enumerate(zip(r_peaks, predictions)):
            if i < len(predictions):
                color = colors.get(pred, 'gray')
                label_text = f'{pred} ({"Normal" if pred=="N" else "Supraventricular" if pred=="S" else "Ventricular"})'
                
                if pred not in labels_added:
                    ax1.axvspan(r_peak/sampling_rate - 0.2, r_peak/sampling_rate + 0.2,
                              alpha=0.3, color=color, label=label_text)
                    labels_added.add(pred)
                else:
                    ax1.axvspan(r_peak/sampling_rate - 0.2, r_peak/sampling_rate + 0.2,
                              alpha=0.3, color=color)
    
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('ECG Signal with Arrhythmia Detection', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Distribution pie chart
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Comprehensive Arrhythmia Detection API",
        "trained": is_trained,
        "dataset_loaded": current_dataset is not None,
        "features": [
            "11 ML models (Basic → Intermediate → Advanced)",
            "Multi-dataset support (Kaggle + MIT-BIH)",
            "SMOTE class balancing",
            "Comprehensive performance analysis"
        ],
        "model_categories": {
            "basic": ["Logistic Regression", "Naive Bayes", "Decision Tree", "Perceptron"],
            "intermediate": ["KNN", "SVM", "Random Forest", "AdaBoost"],
            "advanced": ["Gradient Boosting", "XGBoost", "LightGBM"]
        }
    })


@app.route('/api/train-comprehensive', methods=['POST'])
def train_comprehensive():
    """Train all models with comprehensive analysis"""
    global classifier, current_dataset, is_trained
    
    try:
        data = request.get_json() or {}
        use_kaggle = data.get('use_kaggle', True)
        use_mitbih = data.get('use_mitbih', True)
        balance_data = data.get('balance_data', True)
        kaggle_samples = data.get('kaggle_samples_per_class', 3000)
        mitbih_records = data.get('mitbih_records', 10)
        skip_slow = data.get('skip_slow_models', False)
        
        print("\n" + "="*70)
        print("🫀 COMPREHENSIVE ARRHYTHMIA DETECTION TRAINING")
        print("   Demonstrating: Basic → Intermediate → Advanced ML")
        print("="*70)
        
        # Load datasets
        X, y, info = dataset_loader.load_cached_combined()
        
        if X is None:
            X, y, info = dataset_loader.combine_datasets(
                use_kaggle=use_kaggle,
                use_mitbih=use_mitbih,
                kaggle_samples_per_class=kaggle_samples,
                mitbih_records=mitbih_records,
                mitbih_beats_per_record=500
            )
        
        current_dataset = {'X': X, 'y': y, 'info': info}
        
        # Prepare data
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            X, y,
            balance_train=balance_data
        )
        
        # Train all models
        classifier.train_all_models(X_train, y_train, skip_slow=skip_slow)
        
        # Evaluate
        results = classifier.evaluate_all_models(X_test, y_test)
        
        # Create comparison DataFrame
        comparison_df = classifier.create_comparison_dataframe(results)
        
        # Format results for JSON
        formatted_results = {}
        for name, metrics in results.items():
            formatted_results[name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'training_time': float(metrics['training_time'])
            }
        
        # Create comprehensive comparison plot
        comparison_plot = create_comprehensive_comparison_plot(results, classifier)
        
        # Save models
        classifier.save_models()
        
        is_trained = True
        
        print("\n✅ COMPREHENSIVE TRAINING COMPLETED!\n")
        
        return jsonify({
            "status": "success",
            "message": "All models trained successfully",
            "dataset_info": {
                "total_samples": info['total_samples'],
                "num_features": info['num_features'],
                "sources": info['sources'],
                "class_distribution": info['class_distribution']
            },
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "models_trained": {
                "basic": len(classifier.model_categories['basic']),
                "intermediate": len(classifier.model_categories['intermediate']),
                "advanced": len(classifier.model_categories['advanced']),
                "total": len(classifier.models)
            },
            "best_model": classifier.best_model_name,
            "best_model_category": next(
                (cat for cat, models in classifier.model_categories.items() 
                 if classifier.best_model_name in models), 
                'unknown'
            ),
            "results": formatted_results,
            "comparison_table": comparison_df.to_dict('records'),
            "comparison_plot": comparison_plot
        })
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/process-ecg', methods=['POST'])
def process_ecg():
    """Process ECG signal"""
    global classifier, is_trained
    
    try:
        if not is_trained:
            return jsonify({
                "status": "error",
                "message": "Models not trained. Train models first."
            }), 400
        
        data = request.get_json()
        ecg_signal = np.array(data.get('signal', []))
        sampling_rate = data.get('sampling_rate', 360)
        
        if len(ecg_signal) == 0:
            return jsonify({"status": "error", "message": "No signal provided"}), 400
        
        print(f"\n📊 Processing ECG signal ({len(ecg_signal)} samples)...")
        
        # Process
        processed_signal = preprocess_ecg_signal(ecg_signal, sampling_rate)
        beats, r_peaks = segment_beats_auto(processed_signal, sampling_rate)
        
        # Extract features
        features_list = []
        for beat in beats:
            beat_features = extract_features_from_beat(beat, sampling_rate)
            features_list.append(list(beat_features.values()))
        
        features = np.array(features_list)
        
        # Classify
        predictions, probabilities = classifier.predict(features, use_best=True)
        
        heart_rate = calculate_heart_rate(r_peaks, sampling_rate)
        pred_counts = dict(zip(*np.unique(predictions, return_counts=True)))
        
        # Visualization
        results_plot = create_results_plot(processed_signal, r_peaks, predictions, sampling_rate)
        
        print("✅ Processing complete!\n")
        
        return jsonify({
            "status": "success",
            "heart_rate": float(heart_rate),
            "num_beats": len(beats),
            "predictions": predictions.tolist(),
            "prediction_counts": {k: int(v) for k, v in pred_counts.items()},
            "best_model_used": classifier.best_model_name,
            "visualizations": {
                "results": results_plot
            }
        })
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/generate-sample', methods=['GET'])
def generate_sample():
    """Generate sample ECG (keep simple for now)"""
    try:
        sampling_rate = 360
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)
        ecg = np.zeros_like(t)
        
        # Simple heartbeat generation
        for beat_time in np.arange(0, duration, 0.8):
            beat_idx = int(beat_time * sampling_rate)
            if beat_idx < len(ecg) - 100:
                ecg[beat_idx:beat_idx+10] = np.linspace(0, 1, 10)
                ecg[beat_idx+10:beat_idx+20] = np.linspace(1, 0, 10)
        
        ecg = ecg + 0.05 * np.random.randn(len(ecg))
        
        return jsonify({
            "status": "success",
            "signal": ecg.tolist(),
            "sampling_rate": sampling_rate,
            "duration": duration
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🫀 COMPREHENSIVE ARRHYTHMIA DETECTION SYSTEM")
    print("="*70)
    print("\n📚 AIML Project Features:")
    print("   • 11 ML models across 3 categories")
    print("   • Basic: Logistic Regression, Naive Bayes, Decision Tree, Perceptron")
    print("   • Intermediate: KNN, SVM, Random Forest, AdaBoost")
    print("   • Advanced: Gradient Boosting, XGBoost, LightGBM")
    print("\n📊 Dataset:")
    print("   • Multi-dataset support (Kaggle + MIT-BIH)")
    print("   • SMOTE class balancing")
    print("   • Comprehensive performance analysis")
    print("\n📡 Starting Flask server...")
    print("   URL: http://localhost:5000")
    print("   Health Check: http://localhost:5000/api/health")
    print("\n⚡ Ready!\n")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')