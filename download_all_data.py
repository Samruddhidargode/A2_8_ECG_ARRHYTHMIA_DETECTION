"""
Download ALL data from Kaggle and MIT-BIH
Maximum dataset for best training results
"""

import sys
sys.path.append('.')

from multi_dataset_loader import MultiDatasetLoader
import os
import shutil

def download_complete_dataset():
    """Download complete dataset with all available data"""
    
    print("\n" + "="*70)
    print("📥 DOWNLOADING COMPLETE DATASET")
    print("   • ALL Kaggle data (no sample limits)")
    print("   • ALL MIT-BIH records (48 records)")
    print("="*70)
    
    # Ask for confirmation
    print("\n⚠️  This will:")
    print("   • Download ~170,000+ samples from Kaggle")
    print("   • Download all 48 MIT-BIH records (~10,000 beats)")
    print("   • Take 10-15 minutes")
    print("   • Require ~2-3 GB disk space")
    
    response = input("\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Cancelled.")
        return
    
    # Delete old cache
    cache_file = 'dataset_cache/combined_dataset.pkl'
    if os.path.exists(cache_file):
        print(f"\n🗑️  Removing old cache...")
        os.remove(cache_file)
        print("   ✓ Old cache deleted")
    
    # Initialize loader
    loader = MultiDatasetLoader()
    
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING KAGGLE DATASET")
    print("="*70)
    
    try:
        # Download Kaggle (no limits = all data)
        print("\n📥 Downloading complete Kaggle dataset...")
        print("   (This may take 5-8 minutes)")
        
        kaggle_path = loader.download_kaggle_dataset()
        
        if kaggle_path:
            print(f"✅ Kaggle dataset downloaded to: {kaggle_path}")
        else:
            print("⚠️  Kaggle download failed, will try MIT-BIH only")
        
    except Exception as e:
        print(f"⚠️  Kaggle error: {e}")
        kaggle_path = None
    
    print("\n" + "="*70)
    print("STEP 2: COMBINING ALL DATA")
    print("="*70)
    
    try:
        # Combine with NO LIMITS
        X, y, info = loader.combine_datasets(
            use_kaggle=True,
            use_mitbih=True,
            kaggle_samples_per_class=None,  # NO LIMIT - use ALL data
            mitbih_records=48,              # ALL 48 records
            mitbih_beats_per_record=None    # NO LIMIT - use ALL beats
        )
        
        print("\n" + "="*70)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*70)
        
        print(f"\n📊 FINAL DATASET:")
        print(f"   Total samples: {len(y):,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Size in memory: ~{X.nbytes / 1024 / 1024:.1f} MB")
        
        print(f"\n📈 CLASS DISTRIBUTION:")
        import numpy as np
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        for cls, count in zip(unique, counts):
            percentage = count/total*100
            bar = '█' * int(percentage / 2)
            print(f"   {cls}: {count:6d} ({percentage:5.1f}%) {bar}")
        
        # Check if balanced enough for SMOTE
        min_samples = min(counts)
        print(f"\n🔍 SMOTE Compatibility:")
        if min_samples >= 6:
            print(f"   ✅ Minimum class has {min_samples} samples (SMOTE will work!)")
        else:
            print(f"   ⚠️  Minimum class has only {min_samples} samples")
            print(f"   💡 SMOTE requires at least 6 samples per class")
        
        print("\n💾 Dataset cached to: dataset_cache/combined_dataset.pkl")
        print("   (Next runs will be instant!)")
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Run comprehensive training:")
        print("   python quick_test.py")
        print("\n2. Or test individual models:")
        print("   python test_real_data.py")
        print("\n3. Or start the web interface:")
        print("   python app_comprehensive.py")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error combining datasets: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_dataset_info():
    """Show info about current cached dataset"""
    print("\n" + "="*70)
    print("📊 CURRENT DATASET INFO")
    print("="*70)
    
    cache_file = 'dataset_cache/combined_dataset.pkl'
    
    if not os.path.exists(cache_file):
        print("\n❌ No cached dataset found")
        print("   Run this script to download data")
        return
    
    # Load and show info
    import pickle
    import numpy as np
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        X, y = data['X'], data['y']
        info = data.get('info', {})
        
        print(f"\n✅ Dataset found:")
        print(f"   Location: {cache_file}")
        print(f"   Total samples: {len(y):,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   File size: {os.path.getsize(cache_file) / 1024 / 1024:.1f} MB")
        
        if 'sources' in info:
            print(f"   Sources: {', '.join(info['sources'])}")
        
        print(f"\n📈 Class Distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count/len(y)*100
            print(f"   {cls}: {count:6d} ({percentage:5.1f}%)")
        
        # Check data quality
        nan_count = np.isnan(X).sum()
        print(f"\n🔍 Data Quality:")
        print(f"   NaN values: {nan_count} ({nan_count/X.size*100:.2f}%)")
        print(f"   Min value: {np.nanmin(X):.4f}")
        print(f"   Max value: {np.nanmax(X):.4f}")
        
    except Exception as e:
        print(f"\n❌ Error reading cache: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download complete ECG dataset')
    parser.add_argument('--info', action='store_true', 
                       help='Show info about current dataset')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print("🫀 Complete Dataset Downloader")
    
    if args.info:
        show_dataset_info()
    else:
        if args.force:
            # Skip confirmation
            cache_file = 'dataset_cache/combined_dataset.pkl'
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            loader = MultiDatasetLoader()
            X, y, info = loader.combine_datasets(
                use_kaggle=True,
                use_mitbih=True,
                kaggle_samples_per_class=None,
                mitbih_records=48,
                mitbih_beats_per_record=None
            )
        else:
            download_complete_dataset()