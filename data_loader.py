"""
MIT-BIH Arrhythmia Database Loader
Loads real ECG data from PhysioNet with all 48 records
"""

import wfdb
import numpy as np
import os
import pickle
from preprocessing import preprocess_ecg_signal, segment_beats_from_annotations
from feature_extraction import extract_features_from_beat

class MITBIHDataLoader:
    """Complete MIT-BIH Arrhythmia Database loader"""
    
    # All 48 MIT-BIH records
    ALL_RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # AAMI standard classification (used in medical practice)
    AAMI_CLASSES = {
        'N': ['N', 'L', 'R', 'e', 'j'],      # Normal beats
        'S': ['A', 'a', 'J', 'S'],            # Supraventricular ectopic beats
        'V': ['V', 'E'],                      # Ventricular ectopic beats
        'F': ['F'],                           # Fusion beats
        'Q': ['/', 'f', 'Q']                 # Unknown/Paced beats
    }
    
    def __init__(self, data_dir='mitbih_data'):
        """Initialize with data directory for caching"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        print(f"📁 Data directory: {data_dir}")
    
    def map_to_aami(self, symbol):
        """Map MIT-BIH beat symbol to AAMI class"""
        for aami_class, symbols in self.AAMI_CLASSES.items():
            if symbol in symbols:
                return aami_class
        return None  # Ignore unknown classes
    
    def download_and_process_record(self, record_name, max_beats=None):
        """
        Download and process a single MIT-BIH record
        
        Args:
            record_name: Record number (e.g., '100', '101')
            max_beats: Maximum number of beats to extract (None for all)
            
        Returns:
            features: numpy array of shape (n_beats, n_features)
            labels: list of AAMI class labels
        """
        try:
            print(f"📥 Downloading record {record_name} from PhysioNet...")
            
            # Download from PhysioNet
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
            
            # Get ECG signal (use MLII lead if available, otherwise first lead)
            signal = record.p_signal[:, 0]
            sampling_rate = record.fs
            
            print(f"   Signal length: {len(signal)} samples ({len(signal)/sampling_rate:.1f} seconds)")
            print(f"   Sampling rate: {sampling_rate} Hz")
            
            # Preprocess signal
            processed_signal = preprocess_ecg_signal(signal, sampling_rate)
            
            # Get R-peak locations and beat labels from annotations
            r_peaks = annotation.sample
            symbols = annotation.symbol
            
            print(f"   Total annotated beats: {len(r_peaks)}")
            
            # Segment beats around R-peaks
            beats, valid_indices = segment_beats_from_annotations(
                processed_signal, r_peaks, sampling_rate
            )
            
            # Map symbols to AAMI classes and filter
            labels = []
            valid_beats = []
            
            for i, beat in zip(valid_indices, beats):
                label = self.map_to_aami(symbols[i])
                # Only keep N, S, V classes (exclude F and Q for better training)
                if label is not None and label in ['N', 'S', 'V']:
                    labels.append(label)
                    valid_beats.append(beat)
            
            # Limit number of beats if specified
            if max_beats and len(valid_beats) > max_beats:
                indices = np.random.choice(len(valid_beats), max_beats, replace=False)
                valid_beats = [valid_beats[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            # Extract features from each beat
            print(f"   Extracting features from {len(valid_beats)} beats...")
            features_list = []
            for beat in valid_beats:
                beat_features = extract_features_from_beat(beat)
                # Convert dict to array of values
                features_list.append(list(beat_features.values()))
            
            features = np.array(features_list)
            
            # Print class distribution
            unique, counts = np.unique(labels, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"   {cls}: {count} beats ({count/len(labels)*100:.1f}%)")
            
            print(f"✅ Record {record_name} processed successfully!")
            print(f"   Features shape: {features.shape}")
            print()
            
            return features, labels
            
        except Exception as e:
            print(f"❌ Error processing record {record_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_dataset(self, num_records=5, records=None, max_beats_per_record=500, use_cache=True):
        """
        Load complete dataset from multiple MIT-BIH records
        
        Args:
            num_records: Number of records to load (default: 5)
            records: Specific record list (overrides num_records)
            max_beats_per_record: Max beats per record for balancing
            use_cache: Use cached data if available
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
        """
        # Determine which records to load
        if records is None:
            records = self.ALL_RECORDS[:num_records]
        
        cache_file = os.path.join(self.data_dir, f'dataset_{len(records)}records.pkl')
        
        # Try loading from cache
        if use_cache and os.path.exists(cache_file):
            print(f"📂 Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            X, y = data['features'], data['labels']
            print(f"✅ Loaded {len(y)} beats from cache\n")
            self._print_dataset_info(X, y)
            return X, y
        
        # Download and process records
        print("=" * 60)
        print(f"🫀 LOADING MIT-BIH ARRHYTHMIA DATABASE")
        print(f"📊 Processing {len(records)} records")
        print("=" * 60 + "\n")
        
        all_features = []
        all_labels = []
        
        for i, record_name in enumerate(records, 1):
            print(f"[{i}/{len(records)}] Processing record {record_name}")
            
            features, labels = self.download_and_process_record(
                record_name, max_beats=max_beats_per_record
            )
            
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_labels.extend(labels)
        
        # Combine all features
        if len(all_features) == 0:
            raise Exception("No records were successfully processed!")
        
        # CRITICAL FIX: Stack features correctly
        # Each element in all_features is (n_beats, n_features)
        # We want final shape: (total_beats, n_features)
        X = np.vstack(all_features)  # This stacks along axis 0
        y = np.array(all_labels)
        
        print("=" * 60)
        print("✅ DATASET LOADED SUCCESSFULLY!")
        print("=" * 60)
        
        # Verify shapes
        print(f"\n🔍 SHAPE VERIFICATION:")
        print(f"   X shape: {X.shape} (should be [n_samples, n_features])")
        print(f"   y shape: {y.shape} (should be [n_samples])")
        print(f"   Match: {X.shape[0] == y.shape[0]}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch! X has {X.shape[0]} samples but y has {y.shape[0]}")
        
        self._print_dataset_info(X, y)
        
        # Save to cache
        print(f"\n💾 Caching dataset to: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({'features': X, 'labels': y, 'records': records}, f)
        print("✅ Dataset cached successfully!\n")
        
        return X, y
    
    def _print_dataset_info(self, X, y):
        """Print dataset statistics"""
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Total beats: {len(y)}")
        print(f"   Feature dimensions: {X.shape[1]}")
        print(f"   Data shape: {X.shape}")
        
        print(f"\n📈 CLASS DISTRIBUTION:")
        unique, counts = np.unique(y, return_counts=True)
        class_names = {
            'N': 'Normal',
            'S': 'Supraventricular',
            'V': 'Ventricular'
        }
        
        total = len(y)
        for cls, count in zip(unique, counts):
            name = class_names.get(cls, cls)
            percentage = count/total*100
            bar = '█' * int(percentage / 2)
            print(f"   {cls} ({name:17s}): {count:5d} beats ({percentage:5.1f}%) {bar}")
    
    def load_balanced_dataset(self, num_records=5, samples_per_class=300):
        """
        Load dataset with balanced classes (equal samples per class)
        
        Args:
            num_records: Number of records to load
            samples_per_class: Samples per class (N, S, V)
            
        Returns:
            X: Balanced feature matrix
            y: Balanced labels array
        """
        # First load full dataset
        X, y = self.load_dataset(num_records, use_cache=True)
        
        print(f"\n⚖️  BALANCING DATASET ({samples_per_class} samples per class)...")
        
        balanced_X = []
        balanced_y = []
        
        for cls in ['N', 'S', 'V']:
            cls_indices = np.where(y == cls)[0]
            
            if len(cls_indices) == 0:
                print(f"   ⚠️  Warning: No samples found for class {cls}")
                continue
            
            if len(cls_indices) >= samples_per_class:
                selected = np.random.choice(cls_indices, samples_per_class, replace=False)
            else:
                print(f"   ⚠️  Warning: Only {len(cls_indices)} samples for class {cls}, using all")
                selected = cls_indices
            
            balanced_X.append(X[selected])
            balanced_y.extend([cls] * len(selected))
        
        X_balanced = np.vstack(balanced_X)
        y_balanced = np.array(balanced_y)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        print(f"✅ Balanced dataset created: {len(y_balanced)} total beats")
        self._print_dataset_info(X_balanced, y_balanced)
        
        return X_balanced, y_balanced
    
    def get_record_info(self, record_name):
        """Get information about a specific record"""
        try:
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
            
            info = {
                'record_name': record_name,
                'duration_sec': len(record.p_signal) / record.fs,
                'sampling_rate': record.fs,
                'num_leads': record.n_sig,
                'lead_names': record.sig_name,
                'total_beats': len(annotation.sample),
                'beat_types': dict(zip(*np.unique(annotation.symbol, return_counts=True)))
            }
            return info
        except Exception as e:
            print(f"Error getting info for record {record_name}: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    print("🫀 MIT-BIH Arrhythmia Database Loader\n")
    
    loader = MITBIHDataLoader()
    
    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET (5 records)")
    print("="*60)
    X, y = loader.load_dataset(num_records=5, max_beats_per_record=400)
    
    print(f"\n✅ Dataset ready for training!")
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Verify shapes match
    assert X.shape[0] == y.shape[0], "Shape mismatch!"
    print(f"   ✓ Shapes match correctly!")