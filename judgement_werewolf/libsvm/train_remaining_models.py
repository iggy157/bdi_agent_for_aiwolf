#!/usr/bin/env python3
"""
Simplified script to train remaining models (FastText and BERT)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class SimpleWerewolfTrainer:
    """Simplified werewolf model trainer"""
    
    def __init__(self, data_path: str, model_path: str, embedding_type: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.embedding_type = embedding_type
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Simplified models (faster training)
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'SVM': SVC(probability=True, random_state=42, C=1.0),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
    def load_libsvm_file(self, file_path: Path) -> Tuple[List[int], np.ndarray]:
        """Load libsvm file"""
        labels = []
        feature_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    
                    label = int(parts[0])
                    labels.append(label)
                    
                    features = {}
                    for part in parts[1:]:
                        if ':' in part:
                            try:
                                idx, value = part.split(':')
                                features[int(idx)] = float(value)
                            except ValueError:
                                continue
                    
                    feature_data.append(features)
            
            return labels, feature_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return [], []
    
    def aggregate_player_features(self, labels: List[int], feature_data: List[Dict[int, float]]) -> Tuple[int, np.ndarray]:
        """Aggregate player features"""
        if not labels or not feature_data:
            return 0, np.array([])
        
        player_label = 1 if sum(labels) > 0 else -1
        
        all_feature_indices = set()
        for features in feature_data:
            all_feature_indices.update(features.keys())
        
        if not all_feature_indices:
            return player_label, np.array([])
        
        max_feature_idx = max(all_feature_indices)
        
        feature_matrix = np.zeros((len(feature_data), max_feature_idx))
        for i, features in enumerate(feature_data):
            for idx, value in features.items():
                if idx <= max_feature_idx:
                    feature_matrix[i, idx-1] = value
        
        # Simple aggregation - just mean and std
        aggregated_features = []
        aggregated_features.extend(np.mean(feature_matrix, axis=0))
        aggregated_features.extend(np.std(feature_matrix, axis=0))
        aggregated_features.append(len(feature_data))  # number of utterances
        
        return player_label, np.array(aggregated_features)
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all player data"""
        print(f"Loading data from {self.data_path}")
        
        libsvm_files = list(self.data_path.glob("*.libsvm"))
        if not libsvm_files:
            raise ValueError(f"No libsvm files found in {self.data_path}")
        
        print(f"Found {len(libsvm_files)} player files")
        
        all_labels = []
        all_features = []
        
        for file_path in libsvm_files:
            labels, feature_data = self.load_libsvm_file(file_path)
            
            if labels and feature_data:
                player_label, aggregated_features = self.aggregate_player_features(labels, feature_data)
                
                if len(aggregated_features) > 0:
                    all_labels.append(1 if player_label == 1 else 0)
                    all_features.append(aggregated_features)
        
        if not all_features:
            raise ValueError("No valid features extracted from data")
        
        # Normalize feature lengths
        max_length = max(len(features) for features in all_features)
        normalized_features = []
        
        for features in all_features:
            if len(features) < max_length:
                padded_features = np.zeros(max_length)
                padded_features[:len(features)] = features
                normalized_features.append(padded_features)
            else:
                normalized_features.append(features[:max_length])
        
        X = np.array(normalized_features)
        y = np.array(all_labels)
        
        print(f"Loaded {len(X)} players")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution - Werewolf: {sum(y)}, Villager: {len(y) - sum(y)}")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train models"""
        print(f"\nTraining models for {self.embedding_type}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        best_score = 0
        best_model_name = None
        best_pipeline = None
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[model_name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'f1_score': f1,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{model_name} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model_name = model_name
                best_pipeline = pipeline
        
        print(f"\nBest model: {best_model_name} (F1: {best_score:.4f})")
        return results, best_model_name, best_pipeline
    
    def save_model(self, results: Dict[str, Any], best_model_name: str, best_pipeline):
        """Save models and results"""
        print(f"\nSaving models to {self.model_path}")
        
        # Save best model
        model_file = self.model_path / f"best_model_{self.embedding_type.lower()}.joblib"
        joblib.dump(best_pipeline, model_file)
        print(f"Best model saved: {model_file}")
        
        # Save all models
        for model_name, result in results.items():
            model_file = self.model_path / f"{model_name.lower()}_{self.embedding_type.lower()}.joblib"
            joblib.dump(result['model'], model_file)
        
        # Save results
        results_file = self.model_path / f"training_results_{self.embedding_type.lower()}.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Results for {self.embedding_type}\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {result['f1_score']:.4f}\n")
                f.write("\n")
                
                f.write(f"Classification Report for {model_name}:\n")
                f.write(classification_report(result['y_test'], result['y_pred']))
                f.write("\n" + "-" * 30 + "\n\n")
            
            f.write(f"Best Model: {best_model_name}\n")
        
        print(f"Results saved: {results_file}")
    
    def train(self):
        """Complete training pipeline"""
        try:
            X, y = self.load_all_data()
            results, best_model_name, best_pipeline = self.train_models(X, y)
            self.save_model(results, best_model_name, best_pipeline)
            print(f"\n{self.embedding_type} model training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

def main():
    
    for embedding_type in ['FastText', 'BERT']:
        print(f"\n{'='*60}")
        print(f"Training {embedding_type} Model")
        print(f"{'='*60}")
        
        try:
            # パス設定
            data_path = f"judgement_werewolf/libsvm/datasets/word_embeding/{embedding_type}"
            model_path = f"judgement_werewolf/libsvm/models/{embedding_type}"
            
            trainer = SimpleWerewolfTrainer(data_path, model_path, embedding_type)
            trainer.train()
            
        except Exception as e:
            print(f"Failed to train {embedding_type} model: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Model training completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()