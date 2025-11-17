"""
Multi-Dataset Loader for Arrhythmia Detection
Combines MIT-BIH, Kaggle, and optionally Chapman datasets
"""

import os
import numpy as np
import pandas as pd
import wfdb
import kagglehub
from pathlib import Path
import pickle
from preprocessing import preprocess_ecg_signal, segment_beats_from_annotations
from feature_extraction import extract_features_from_beat

class MultiDatasetLoader:
    """Unified loader for multiple ECG datasets"""
    
    def __init__(self, cache_dir='dataset_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # AAMI standard classes (common across datasets)
        self.AAMI_CLASSES = {
            'N': ['N', 'L', 'R', 'e', 'j'],      # Normal
            'S': ['A', 'a', 'J', 'S'],            # Supraventricular
            'V': ['V', 'E'],                      # Ventricular
        }
        
    def download_kaggle_dataset(self):
        """Download Kaggle ECG dataset"""
        print("\n" + "="*70)
        print("📥 DOWNLOADING KAGGLE DATASET")
        print("="*70)
        
        cache_file = os.path.join(self.cache_dir, 'kaggle_dataset_path.txt')
        
        # Check if already downloaded
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                path = f.read().strip()
            if os.path.exists(path):
                print(f"✅ Kaggle dataset already downloaded at: {path}")
                return path
        
        try:
            # Download latest version
            path = kagglehub.dataset_download("sadmansakib7/ecg-arrhythmia-classification-dataset")
            print(f"✅ Downloaded to: {path}")
            
            # Cache the path
            with open(cache_file, 'w') as f:
                f.write(path)
                
            return path
            
        except Exception as e:
            print(f"❌ Error downloading Kaggle dataset: {e}")
            print("💡 Please ensure kagglehub is installed and authenticated")
            return None
    
    def load_kaggle_dataset(self, max_samples_per_class=None):
        """
        Load Kaggle pre-extracted features dataset
        
        Args:
            max_samples_per_class: Max samples per class (None = use all)
        
        Returns:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
        """
        print("\n" + "="*70)
        print("📊 LOADING KAGGLE DATASET")
        print("="*70)
        
        dataset_path = self.download_kaggle_dataset()
        if dataset_path is None:
            return None, None, None
        
        # Find CSV files
        from pathlib import Path
        csv_files = list(Path(dataset_path).glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found in dataset!")
            return None, None, None
        
        print(f"\nFound {len(csv_files)} dataset files:")
        for f in csv_files:
            print(f"  • {f.name}")
        
        # Load all CSVs and combine
        all_data = []
        for csv_file in csv_files:
            print(f"\nLoading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                print(f"  Shape: {df.shape}")
                print(f"  Classes: {df['type'].value_counts().to_dict()}")
            except Exception as e:
                print(f"  ⚠️  Error loading {csv_file.name}: {e}")
                continue
        
        if not all_data:
            print("❌ No data loaded successfully!")
            return None, None, None
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Combined dataset shape: {combined_df.shape}")
        
        # Map to AAMI classes
        def map_to_aami(label):
            for aami_class, symbols in self.AAMI_CLASSES.items():
                if label in symbols:
                    return aami_class
            return None
        
        combined_df['aami_class'] = combined_df['type'].apply(map_to_aami)
        combined_df = combined_df[combined_df['aami_class'].notna()]
        
        print(f"\n📊 After AAMI mapping: {len(combined_df)} samples")
        
        # Balance classes if specified
        if max_samples_per_class is not None:
            print(f"\n⚖️  Limiting to {max_samples_per_class} samples per class...")
            balanced_dfs = []
            for cls in ['N', 'S', 'V']:
                cls_df = combined_df[combined_df['aami_class'] == cls]
                if len(cls_df) > max_samples_per_class:
                    cls_df = cls_df.sample(n=max_samples_per_class, random_state=42)
                balanced_dfs.append(cls_df)
            combined_df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"   ✓ Balanced to {len(combined_df)} samples")
        else:
            print(f"\n✅ Using ALL data (no class limits)")
        
        # Extract features and labels
        feature_cols = [col for col in combined_df.columns 
                       if col not in ['record', 'type', 'aami_class']]
        
        X = combined_df[feature_cols].values
        y = combined_df['aami_class'].values
        
        print(f"\n📊 Kaggle Dataset Statistics:")
        print(f"   Total samples: {len(y):,}")
        print(f"   Features: {len(feature_cols)}")
        
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"   {cls}: {count:,} ({count/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
    
    def load_mitbih_dataset(self, num_records=10, max_beats_per_record=500):
        """
        Load MIT-BIH dataset with raw signal processing
        
        Returns:
            X: Feature matrix
            y: Labels
        """
        print("\n" + "="*70)
        print("📊 LOADING MIT-BIH DATASET")
        print("="*70)
        
        from data_loader import MITBIHDataLoader
        
        loader = MITBIHDataLoader(data_dir=os.path.join(self.cache_dir, 'mitbih_data'))
        X, y = loader.load_dataset(
            num_records=num_records,
            max_beats_per_record=max_beats_per_record,
            use_cache=True
        )
        
        return X, y
    
    def harmonize_features(self, X_kaggle, X_mitbih, kaggle_features):
        """
        Harmonize features from different datasets to same dimensions
        
        Strategy:
        1. If same size → concatenate directly
        2. If different → use intersection or pad/truncate
        """
        print("\n" + "="*70)
        print("🔄 HARMONIZING FEATURES")
        print("="*70)
        
        kaggle_dim = X_kaggle.shape[1]
        mitbih_dim = X_mitbih.shape[1]
        
        print(f"Kaggle features: {kaggle_dim}")
        print(f"MIT-BIH features: {mitbih_dim}")
        
        if kaggle_dim == mitbih_dim:
            print("✅ Feature dimensions match! No harmonization needed.")
            return X_kaggle, X_mitbih
        
        # Use minimum dimension (truncate to match)
        min_dim = min(kaggle_dim, mitbih_dim)
        print(f"⚠️  Dimensions differ. Using first {min_dim} features from each.")
        
        X_kaggle_harmonized = X_kaggle[:, :min_dim]
        X_mitbih_harmonized = X_mitbih[:, :min_dim]
        
        return X_kaggle_harmonized, X_mitbih_harmonized
    
    def combine_datasets(self, 
                        use_kaggle=True,
                        use_mitbih=True,
                        kaggle_samples_per_class=5000,
                        mitbih_records=10,
                        mitbih_beats_per_record=500):
        """
        Combine multiple datasets into one unified dataset
        
        Args:
            use_kaggle: Include Kaggle dataset
            use_mitbih: Include MIT-BIH dataset
            kaggle_samples_per_class: Max samples per class from Kaggle
            mitbih_records: Number of MIT-BIH records
            mitbih_beats_per_record: Beats per MIT-BIH record
            
        Returns:
            X_combined: Combined feature matrix
            y_combined: Combined labels
            dataset_info: Dictionary with dataset information
        """
        print("\n" + "="*70)
        print("🔗 COMBINING DATASETS")
        print("="*70)
        
        X_list = []
        y_list = []
        sources = []
        
        # Load Kaggle dataset
        if use_kaggle:
            X_kaggle, y_kaggle, kaggle_features = self.load_kaggle_dataset(
                max_samples_per_class=kaggle_samples_per_class
            )
            if X_kaggle is not None:
                X_list.append(X_kaggle)
                y_list.append(y_kaggle)
                sources.append('kaggle')
                print(f"✅ Kaggle: {len(y_kaggle)} samples")
        
        # Load MIT-BIH dataset
        if use_mitbih:
            X_mitbih, y_mitbih = self.load_mitbih_dataset(
                num_records=mitbih_records,
                max_beats_per_record=mitbih_beats_per_record
            )
            if X_mitbih is not None:
                X_list.append(X_mitbih)
                y_list.append(y_mitbih)
                sources.append('mitbih')
                print(f"✅ MIT-BIH: {len(y_mitbih)} samples")
        
        if len(X_list) == 0:
            raise Exception("No datasets loaded!")
        
        # Harmonize features if multiple datasets
        if len(X_list) > 1:
            X_kaggle_harm, X_mitbih_harm = self.harmonize_features(
                X_list[0], X_list[1], 
                kaggle_features if use_kaggle else None
            )
            X_list = [X_kaggle_harm, X_mitbih_harm]
        
        # Combine
        X_combined = np.vstack(X_list)
        y_combined = np.concatenate(y_list)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]
        
        print("\n" + "="*70)
        print("✅ COMBINED DATASET READY")
        print("="*70)
        print(f"Total samples: {len(y_combined)}")
        print(f"Feature dimensions: {X_combined.shape[1]}")
        print(f"Sources: {', '.join(sources)}")
        
        print("\n📊 Class Distribution:")
        unique, counts = np.unique(y_combined, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count/len(y_combined)*100
            bar = '█' * int(percentage / 2)
            print(f"   {cls}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        dataset_info = {
            'total_samples': len(y_combined),
            'num_features': X_combined.shape[1],
            'sources': sources,
            'class_distribution': dict(zip(unique, counts))
        }
        
        # Cache combined dataset
        cache_file = os.path.join(self.cache_dir, 'combined_dataset.pkl')
        print(f"\n💾 Caching combined dataset to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'X': X_combined,
                'y': y_combined,
                'info': dataset_info
            }, f)
        
        return X_combined, y_combined, dataset_info
    
    def load_cached_combined(self):
        """Load previously cached combined dataset"""
        cache_file = os.path.join(self.cache_dir, 'combined_dataset.pkl')
        
        if os.path.exists(cache_file):
            print(f"📂 Loading cached combined dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded {data['info']['total_samples']} samples")
            return data['X'], data['y'], data['info']
        else:
            print("❌ No cached combined dataset found")
            return None, None, None


# Example usage
if __name__ == "__main__":
    print("🫀 Multi-Dataset Loader for Arrhythmia Detection\n")
    
    loader = MultiDatasetLoader()
    
    # Option 1: Load cached if available
    X, y, info = loader.load_cached_combined()
    
    # Option 2: Combine fresh datasets
    if X is None:
        X, y, info = loader.combine_datasets(
            use_kaggle=True,
            use_mitbih=True,
            kaggle_samples_per_class=3000,  # Balance Kaggle
            mitbih_records=10,              # Use more MIT-BIH records
            mitbih_beats_per_record=500
        )
    
    print("\n✅ Dataset ready for training!")
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {np.unique(y)}")