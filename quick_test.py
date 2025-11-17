"""
Quick test with automatic data cleaning
Simple and reliable test of all models
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from multi_dataset_loader import MultiDatasetLoader
from comprehensive_training import ComprehensiveArrhythmiaClassifier


def inspect_data_quality(X, y):
    """Inspect and report data quality issues"""
    print("\n" + "="*70)
    print("🔍 DATA QUALITY INSPECTION")
    print("="*70)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Total samples: {len(y)}")
    
    # Check for NaN
    nan_count = np.isnan(X).sum()
    nan_percentage = (nan_count / X.size) * 100
    print(f"\nNaN values: {nan_count} ({nan_percentage:.2f}%)")
    
    if nan_count > 0:
        # NaN per column
        nan_per_col = np.isnan(X).sum(axis=0)
        problematic_cols = np.where(nan_per_col > 0)[0]
        print(f"  Columns with NaN: {len(problematic_cols)}")
        for col in problematic_cols[:5]:  # Show first 5
            print(f"    Column {col}: {nan_per_col[col]} NaN values")
        if len(problematic_cols) > 5:
            print(f"    ... and {len(problematic_cols)-5} more columns")
    
    # Check for infinite
    inf_count = np.isinf(X).sum()
    print(f"\nInfinite values: {inf_count}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = count/len(y)*100
        print(f"  {cls}: {count} ({percentage:.1f}%)")
    
    # Check for extreme values
    print(f"\nValue ranges:")
    print(f"  Min: {np.nanmin(X):.4f}")
    print(f"  Max: {np.nanmax(X):.4f}")
    print(f"  Mean: {np.nanmean(X):.4f}")
    print(f"  Std: {np.nanstd(X):.4f}")
    
    return nan_count, inf_count


def main():
    print("\n" + "="*70)
    print("🫀 QUICK TEST WITH REAL ECG DATA")
    print("="*70)
    
    # Load data
    print("\n📥 Loading dataset...")
    loader = MultiDatasetLoader()
    X, y, info = loader.load_cached_combined()
    
    if X is None:
        print("❌ No cached dataset found")
        print("💡 Run: python multi_dataset_loader.py")
        return
    
    print(f"✅ Loaded {len(y)} samples")
    
    # Inspect data quality
    nan_count, inf_count = inspect_data_quality(X, y)
    
    # Initialize classifier
    print("\n" + "="*70)
    print("🤖 TRAINING MODELS")
    print("="*70)
    
    classifier = ComprehensiveArrhythmiaClassifier()
    
    # Prepare data (with automatic cleaning)
    print("\n🧹 Preparing data (includes automatic cleaning)...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        X, y,
        balance_train=True
    )
    
    # Train selected models (quick test)
    print("\n🏃 Training 6 representative models...")
    print("   (Basic, Intermediate, Advanced)")
    
    try:
        # Basic
        print("\n📚 Basic Models:")
        classifier.train_logistic_regression(X_train, y_train)
        classifier.train_naive_bayes(X_train, y_train)
        
        # Intermediate
        print("\n🎓 Intermediate Models:")
        classifier.train_decision_tree(X_train, y_train)
        classifier.train_random_forest(X_train, y_train)
        
        # Advanced
        print("\n🚀 Advanced Models:")
        classifier.train_xgboost(X_train, y_train)
        classifier.train_lightgbm(X_train, y_train)
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    results = classifier.evaluate_all_models(X_test, y_test)
    
    # Create comparison table
    comparison_df = classifier.create_comparison_dataframe(results)
    
    print("\n📊 MODEL COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Detailed analysis of best and worst
    print("\n" + "="*70)
    print("🔍 DETAILED ANALYSIS")
    print("="*70)
    
    best_model = comparison_df.iloc[0]
    worst_model = comparison_df.iloc[-1]
    
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Category: {best_model['Category']}")
    print(f"   Accuracy: {best_model['Accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_model['F1-Score']*100:.2f}%")
    print(f"   Training Time: {best_model['Training Time (s)']:.2f}s")
    
    print(f"\n📉 WORST MODEL: {worst_model['Model']}")
    print(f"   Category: {worst_model['Category']}")
    print(f"   Accuracy: {worst_model['Accuracy']*100:.2f}%")
    print(f"   F1-Score: {worst_model['F1-Score']*100:.2f}%")
    print(f"   Training Time: {worst_model['Training Time (s)']:.2f}s")
    
    # Per-class performance for best model
    print(f"\n🎯 PER-CLASS PERFORMANCE ({best_model['Model']}):")
    
    from sklearn.metrics import classification_report
    
    best_model_name = classifier.best_model_name
    best_model_obj = classifier.models[best_model_name]
    
    X_test_scaled = classifier.scaler.transform(X_test)
    y_pred = best_model_obj.predict(X_test_scaled)
    
    print(classification_report(
        y_test,
        y_pred,
        target_names=classifier.label_encoder.classes_,
        digits=4
    ))
    
    # Key insights
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS")
    print("="*70)
    
    naive_bayes_row = comparison_df[comparison_df['Model'] == 'Naive Bayes']
    if not naive_bayes_row.empty:
        nb_acc = naive_bayes_row['Accuracy'].values[0] * 100
        best_acc = best_model['Accuracy'] * 100
        diff = best_acc - nb_acc
        
        print(f"\n✓ Naive Bayes accuracy: {nb_acc:.2f}%")
        print(f"✓ Best model ({best_model['Model']}): {best_acc:.2f}%")
        print(f"✓ Improvement: {diff:.2f}%")
        
        if nb_acc < best_acc - 3:
            print(f"\n🎓 WHY Naive Bayes performs worse:")
            print(f"   • Assumes feature independence")
            print(f"   • ECG features are highly correlated")
            print(f"   • Example: RR_interval ↔ heart_rate")
            print(f"   • This violates the independence assumption")
        else:
            print(f"\n✓ All models perform similarly (well-separated classes)")
    
    # Category comparison
    print(f"\n📊 PERFORMANCE BY CATEGORY:")
    category_stats = comparison_df.groupby('Category').agg({
        'Accuracy': 'mean',
        'F1-Score': 'mean',
        'Training Time (s)': 'mean'
    })
    
    for category in ['Basic', 'Intermediate', 'Advanced']:
        if category in category_stats.index:
            stats = category_stats.loc[category]
            print(f"\n{category}:")
            print(f"   Avg Accuracy: {stats['Accuracy']*100:.2f}%")
            print(f"   Avg F1-Score: {stats['F1-Score']*100:.2f}%")
            print(f"   Avg Time: {stats['Training Time (s)']:.2f}s")
    
    # Save models
    print("\n" + "="*70)
    print("💾 SAVING MODELS")
    print("="*70)
    
    classifier.save_models(directory='trained_models')
    print(f"✅ All models saved to: trained_models/")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)
    
    print(f"\n📄 Summary:")
    print(f"   • Dataset: {len(y)} samples, {X.shape[1]} features")
    print(f"   • Models trained: {len(classifier.models)}")
    print(f"   • Best model: {best_model['Model']} ({best_model['Accuracy']*100:.2f}%)")
    print(f"   • Models saved to: trained_models/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Make sure dataset is cached: python multi_dataset_loader.py")
        print("   2. Check dependencies: pip install -r requirements_enhanced.txt")
        print("   3. Verify data integrity")