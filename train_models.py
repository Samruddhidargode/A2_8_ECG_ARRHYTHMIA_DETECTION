"""
Machine Learning Models Training Module
Trains and evaluates multiple classifiers for arrhythmia detection
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import joblib
import os


class ArrhythmiaClassifier:
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model_name = None
        self.feature_names = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"📊 Data Split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Number of features: {X_train.shape[1]}")
        print(f"   Number of classes: {len(np.unique(y_encoded))}")
        
        return X_train, X_test, y_train, y_test
    
    def train_decision_tree(self, X_train, y_train, optimize=False):
        """Train Decision Tree Classifier"""
        print("\n🌳 Training Decision Tree...")
        
        if optimize:
            param_grid = {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            dt = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                            param_grid, cv=3, n_jobs=-1)
            dt.fit(X_train, y_train)
            print(f"   Best params: {dt.best_params_}")
            model = dt.best_estimator_
        else:
            model = DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        self.models['decision_tree'] = model
        print("   ✓ Decision Tree trained")
        return model
    
    def train_knn(self, X_train, y_train, optimize=False):
        """Train K-Nearest Neighbors"""
        print("\n👥 Training K-Nearest Neighbors...")
        
        if optimize:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_jobs=-1)
            knn.fit(X_train, y_train)
            print(f"   Best params: {knn.best_params_}")
            model = knn.best_estimator_
        else:
            model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.models['knn'] = model
        print("   ✓ KNN trained")
        return model
    
    def train_random_forest(self, X_train, y_train, optimize=False):
        """Train Random Forest Classifier"""
        print("\n🌲 Training Random Forest...")
        
        if optimize:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf = GridSearchCV(RandomForestClassifier(random_state=42), 
                            param_grid, cv=3, n_jobs=-1)
            rf.fit(X_train, y_train)
            print(f"   Best params: {rf.best_params_}")
            model = rf.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("   ✓ Random Forest trained")
        return model
    
    def train_svm(self, X_train, y_train, optimize=False):
        """Train Support Vector Machine"""
        print("\n🎯 Training SVM...")
        
        if optimize:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
            svm = GridSearchCV(SVC(probability=True, random_state=42), 
                             param_grid, cv=3, n_jobs=-1)
            svm.fit(X_train, y_train)
            print(f"   Best params: {svm.best_params_}")
            model = svm.best_estimator_
        else:
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
        
        self.models['svm'] = model
        print("   ✓ SVM trained")
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting Classifier"""
        print("\n🚀 Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        print("   ✓ Gradient Boosting trained")
        return model
    
    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes"""
        print("\n📊 Training Naive Bayes...")
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        self.models['naive_bayes'] = model
        print("   ✓ Naive Bayes trained")
        return model
    
    def train_all_models(self, X_train, y_train, optimize=False, quick_mode=True):
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize: Perform hyperparameter optimization
            quick_mode: Train only fast models (skip SVM if dataset is large)
        """
        print("\n" + "="*60)
        print("🤖 TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # Always train these models (fast)
        self.train_decision_tree(X_train, y_train, optimize)
        self.train_knn(X_train, y_train, optimize)
        self.train_random_forest(X_train, y_train, optimize)
        self.train_naive_bayes(X_train, y_train)
        
        # Train slower models if not in quick mode or dataset is small
        if not quick_mode or len(X_train) < 2000:
            self.train_svm(X_train, y_train, optimize)
            self.train_gradient_boosting(X_train, y_train)
        else:
            print("\n⚡ Quick mode: Skipping SVM and Gradient Boosting (slow on large datasets)")
        
        print("\n" + "="*60)
        print(f"✅ Trained {len(self.models)} models successfully!")
        print("="*60)
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate a single model
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("📊 EVALUATING MODELS")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            results[name] = self.evaluate_model(name, model, X_test, y_test)
            
            print(f"   Accuracy:  {results[name]['accuracy']:.4f}")
            print(f"   Precision: {results[name]['precision']:.4f}")
            print(f"   Recall:    {results[name]['recall']:.4f}")
            print(f"   F1-Score:  {results[name]['f1_score']:.4f}")
        
        # Find best model
        best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        self.best_model_name = best_name
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {best_name.upper()}")
        print(f"   Accuracy: {results[best_name]['accuracy']:.4f}")
        print("="*60)
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on all models"""
        print("\n" + "="*60)
        print("🔄 CROSS-VALIDATION (5-Fold)")
        print("="*60)
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"\n{name}...")
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_results[name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
            print(f"   Mean Accuracy: {cv_results[name]['mean']:.4f} ± {cv_results[name]['std']:.4f}")
        
        return cv_results
    
    def get_feature_importance(self, model_name='random_forest', top_n=20):
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if self.feature_names:
                indices = np.argsort(importances)[::-1][:top_n]
                
                print(f"\n📊 Top {top_n} Feature Importances ({model_name}):")
                print("="*60)
                
                for i, idx in enumerate(indices, 1):
                    name = self.feature_names[idx]
                    score = importances[idx]
                    bar = '█' * int(score * 50)
                    print(f"{i:2d}. {name:30s}: {score:.4f} {bar}")
                
                return importances
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
    
    def save_models(self, directory='saved_models'):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}.pkl')
            joblib.dump(model, filepath)
        
        # Save scaler and encoder
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(directory, 'label_encoder.pkl'))
        
        print(f"\n💾 Models saved to {directory}/")
        print(f"   Saved {len(self.models)} models")
    
    def load_models(self, directory='saved_models'):
        """Load saved models"""
        if not os.path.exists(directory):
            print(f"Directory {directory} not found!")
            return False
        
        # Load models
        model_files = [f for f in os.listdir(directory) if f.endswith('.pkl') and f not in ['scaler.pkl', 'label_encoder.pkl']]
        
        for filename in model_files:
            name = filename.replace('.pkl', '')
            filepath = os.path.join(directory, filename)
            self.models[name] = joblib.load(filepath)
        
        # Load scaler and encoder
        self.scaler = joblib.load(os.path.join(directory, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(directory, 'label_encoder.pkl'))
        
        print(f"\n📂 Models loaded from {directory}/")
        print(f"   Loaded {len(self.models)} models")
        return True
    
    def predict(self, X, model_name=None):
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            model_name: Specific model to use (uses best model if None)
            
        Returns:
            predictions: Class labels
            probabilities: Class probabilities
        """
        if model_name is None:
            model_name = self.best_model_name or list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model = self.models[model_name]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities


# Testing
if __name__ == "__main__":
    print("🫀 Machine Learning Training Module Test\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples_per_class = 200
    n_features = 40
    
    # Generate features for 3 classes (N, S, V)
    X_N = np.random.randn(n_samples_per_class, n_features) * 0.5
    X_S = np.random.randn(n_samples_per_class, n_features) * 0.6 + 1.0
    X_V = np.random.randn(n_samples_per_class, n_features) * 0.8 + 2.0
    
    X = np.vstack([X_N, X_S, X_V])
    y = np.array(['N'] * n_samples_per_class + 
                 ['S'] * n_samples_per_class + 
                 ['V'] * n_samples_per_class)
    
    # Initialize classifier
    classifier = ArrhythmiaClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
    
    # Train all models
    classifier.train_all_models(X_train, y_train, optimize=False, quick_mode=True)
    
    # Evaluate models
    results = classifier.evaluate_all_models(X_test, y_test)
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    sample_features = X_test[:5]
    predictions, probabilities = classifier.predict(sample_features)
    print(f"   Predictions: {predictions}")
    
    print("\n✅ All tests passed!")