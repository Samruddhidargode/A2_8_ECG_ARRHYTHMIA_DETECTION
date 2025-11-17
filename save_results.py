"""
Save Pre-Trained Model Results
Run this ONCE after training to save all results for instant loading
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import json
import pickle
import os
from pathlib import Path
from datetime import datetime

def save_pretrained_results():
    """Save all training results for instant frontend loading"""
    
    print("\n" + "="*70)
    print("💾 SAVING PRE-TRAINED RESULTS")
    print("="*70)
    
    # Create results directory
    results_dir = Path('pretrained_results')
    results_dir.mkdir(exist_ok=True)
    
    # Load dataset info
    print("\n📊 Loading dataset information...")
    from multi_dataset_loader import MultiDatasetLoader
    loader = MultiDatasetLoader()
    X, y, info = loader.load_cached_combined()
    
    if X is None:
        print("❌ No cached dataset found!")
        print("   Run: python download_all_data.py")
        return False
    
    print(f"✓ Dataset loaded: {len(y):,} samples")
    
    # Load trained models metadata
    print("\n🤖 Loading trained models...")
    import joblib
    
    model_dirs = ['trained_models', 'real_data_models', 'comprehensive_models']
    metadata = None
    
    for model_dir in model_dirs:
        metadata_file = Path(model_dir) / 'metadata.pkl'
        if metadata_file.exists():
            metadata = joblib.load(metadata_file)
            print(f"✓ Loaded metadata from: {model_dir}")
            break
    
    if metadata is None:
        print("❌ No trained models found!")
        print("   Run: python quick_test.py")
        return False
    
    # Your actual test results (from the terminal output)
    results_data = {
        'dataset': {
            'total_samples': 909963,
            'train_samples': 727970,
            'test_samples': 181993,
            'num_features': 32,
            'classes': ['N', 'S', 'V'],
            'class_distribution': {
                'N': 719957,
                'S': 2225,
                'V': 5788
            },
            'class_distribution_percentage': {
                'N': 98.9,
                'S': 0.3,
                'V': 0.8
            }
        },
        'models': {
            'XGBoost': {
                'category': 'Advanced',
                'accuracy': 99.87,
                'precision': 99.86,
                'recall': 99.87,
                'f1_score': 99.86,
                'training_time': 27.49
            },
            'Random Forest': {
                'category': 'Intermediate',
                'accuracy': 99.25,
                'precision': 99.59,
                'recall': 99.25,
                'f1_score': 99.38,
                'training_time': 61.92
            },
            'Decision Tree': {
                'category': 'Basic',
                'accuracy': 97.55,
                'precision': 99.37,
                'recall': 97.55,
                'f1_score': 98.33,
                'training_time': 12.40
            },
            'Logistic Regression': {
                'category': 'Basic',
                'accuracy': 91.74,
                'precision': 99.06,
                'recall': 91.74,
                'f1_score': 94.91,
                'training_time': 286.16
            },
            'Naive Bayes': {
                'category': 'Basic',
                'accuracy': 89.97,
                'precision': 99.04,
                'recall': 89.97,
                'f1_score': 93.95,
                'training_time': 0.95
            }
        },
        'best_model': 'XGBoost',
        'timestamp': datetime.now().isoformat(),
        'notes': 'Results from comprehensive evaluation on 909,963 ECG samples'
    }
    
    # Save as JSON
    print("\n💾 Saving results...")
    
    json_file = results_dir / 'model_results.json'
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved: {json_file}")
    
    # Save as pickle (for Python)
    pkl_file = results_dir / 'model_results.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"✓ Saved: {pkl_file}")
    
    # Create summary report
    print("\n📄 Creating summary report...")
    
    summary = f"""
# Pre-Trained Model Results Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Information

- Total Samples: {results_data['dataset']['total_samples']:,}
- Training Set: {results_data['dataset']['train_samples']:,}
- Testing Set: {results_data['dataset']['test_samples']:,}
- Features: {results_data['dataset']['num_features']}

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (N) | {results_data['dataset']['class_distribution']['N']:,} | {results_data['dataset']['class_distribution_percentage']['N']}% |
| Supraventricular (S) | {results_data['dataset']['class_distribution']['S']:,} | {results_data['dataset']['class_distribution_percentage']['S']}% |
| Ventricular (V) | {results_data['dataset']['class_distribution']['V']:,} | {results_data['dataset']['class_distribution_percentage']['V']}% |

## Model Performance

| Model | Category | Accuracy | F1-Score | Training Time |
|-------|----------|----------|----------|---------------|
"""
    
    # Sort by accuracy
    sorted_models = sorted(results_data['models'].items(), 
                          key=lambda x: x[1]['accuracy'], 
                          reverse=True)
    
    for model, metrics in sorted_models:
        summary += f"| {model} | {metrics['category']} | {metrics['accuracy']:.2f}% | {metrics['f1_score']:.2f}% | {metrics['training_time']:.2f}s |\n"
    
    summary += f"""
## Best Model

**{results_data['best_model']}** achieved the highest performance:
- Accuracy: {results_data['models'][results_data['best_model']]['accuracy']:.2f}%
- F1-Score: {results_data['models'][results_data['best_model']]['f1_score']:.2f}%
- Category: {results_data['models'][results_data['best_model']]['category']}

## Key Findings

1. Advanced models (XGBoost: 99.87%) significantly outperform basic models
2. Naive Bayes (89.97%) shows expected lower performance due to feature correlation
3. Training on 909K samples provides excellent generalization
4. Class imbalance (98.9% Normal) requires careful evaluation

---

*Results ready for instant loading in frontend*
"""
    
    readme_file = results_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(summary)
    print(f"✓ Saved: {readme_file}")
    
    # Create JavaScript export for frontend
    print("\n🌐 Creating frontend data file...")
    
    js_content = f"""// Pre-trained Model Results
// Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// DO NOT EDIT - Run save_results.py to update

const PRETRAINED_RESULTS = {json.dumps(results_data, indent=2)};

// Export for use in frontend
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = PRETRAINED_RESULTS;
}}
"""
    
    js_file = results_dir / 'results.js'
    with open(js_file, 'w') as f:
        f.write(js_content)
    print(f"✓ Saved: {js_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ RESULTS SAVED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📁 Files created:")
    print(f"   • {json_file.name} - JSON format (for API)")
    print(f"   • {pkl_file.name} - Pickle format (for Python)")
    print(f"   • {readme_file.name} - Human-readable summary")
    print(f"   • {js_file.name} - JavaScript data (for frontend)")
    
    print("\n📊 Summary:")
    print(f"   Dataset: {results_data['dataset']['total_samples']:,} samples")
    print(f"   Models: {len(results_data['models'])} trained")
    print(f"   Best: {results_data['best_model']} ({results_data['models'][results_data['best_model']]['accuracy']:.2f}%)")
    
    print("\n🚀 Next Steps:")
    print("   1. Frontend will now load results instantly")
    print("   2. No need to retrain (saves 5-10 minutes)")
    print("   3. Run: python app_comprehensive.py")
    print("   4. Open: frontend/index.html")
    
    print("\n" + "="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = save_pretrained_results()
        if not success:
            print("\n💡 Make sure you have:")
            print("   1. Downloaded dataset: python download_all_data.py")
            print("   2. Trained models: python quick_test.py")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)