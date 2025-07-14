#!/usr/bin/env python3
"""
人狼判定機械学習モデル訓練スクリプト

プレイヤーごとの発話データから人狼かどうかを判定するモデルを訓練する。
Word2Vec、FastText、BERTの3つの埋め込み手法に対応。

各プレイヤーファイルの全発話から特徴量を集約し、
プレイヤー単位での人狼判定を行う分類器を構築する。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


class WerewolfModelTrainer:
    """人狼判定モデルの訓練クラス"""
    
    def __init__(self, 
                 data_path: str,
                 model_path: str,
                 embedding_type: str):
        """
        初期化
        
        Args:
            data_path: libsvmファイルが格納されているディレクトリパス
            model_path: 訓練済みモデルを保存するディレクトリパス
            embedding_type: 埋め込み手法 ('Word2Vec', 'FastText', 'BERT')
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.embedding_type = embedding_type
        
        # 出力ディレクトリを作成
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # モデル候補
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # ハイパーパラメータグリッド
        self.param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        self.best_model = None
        self.best_pipeline = None
        self.scaler = StandardScaler()
        
    def load_libsvm_file(self, file_path: Path) -> Tuple[List[int], np.ndarray]:
        """
        libsvmファイルを読み込む
        
        Args:
            file_path: libsvmファイルのパス
            
        Returns:
            labels: ラベルのリスト
            features: 特徴量行列
        """
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
                    
                    # ラベルを取得
                    label = int(parts[0])
                    labels.append(label)
                    
                    # 特徴量を取得
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
        """
        プレイヤーの全発話から特徴量を集約
        
        Args:
            labels: 発話ごとのラベル
            feature_data: 発話ごとの特徴量辞書のリスト
            
        Returns:
            player_label: プレイヤーのラベル（多数決）
            aggregated_features: 集約された特徴量ベクトル
        """
        if not labels or not feature_data:
            return 0, np.array([])
        
        # プレイヤーラベル（多数決）
        player_label = 1 if sum(labels) > 0 else -1
        
        # 全特徴量のインデックスを収集
        all_feature_indices = set()
        for features in feature_data:
            all_feature_indices.update(features.keys())
        
        if not all_feature_indices:
            return player_label, np.array([])
        
        max_feature_idx = max(all_feature_indices)
        
        # 特徴量行列を作成
        feature_matrix = np.zeros((len(feature_data), max_feature_idx))
        for i, features in enumerate(feature_data):
            for idx, value in features.items():
                if idx <= max_feature_idx:
                    feature_matrix[i, idx-1] = value  # 1-indexedを0-indexedに変換
        
        # 複数の集約方法を使用
        aggregated_features = []
        
        # 統計的集約
        aggregated_features.extend(np.mean(feature_matrix, axis=0))      # 平均
        aggregated_features.extend(np.std(feature_matrix, axis=0))       # 標準偏差
        aggregated_features.extend(np.max(feature_matrix, axis=0))       # 最大値
        aggregated_features.extend(np.min(feature_matrix, axis=0))       # 最小値
        
        # パーセンタイル
        aggregated_features.extend(np.percentile(feature_matrix, 25, axis=0))  # 25パーセンタイル
        aggregated_features.extend(np.percentile(feature_matrix, 75, axis=0))  # 75パーセンタイル
        
        # 発話数関連の特徴量
        aggregated_features.append(len(feature_data))  # 発話数
        aggregated_features.append(np.sum(np.abs(feature_matrix)))  # 特徴量の絶対値の総和
        
        return player_label, np.array(aggregated_features)
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        全てのプレイヤーデータを読み込み
        
        Returns:
            X: 特徴量行列
            y: ラベルベクトル
        """
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
                    all_labels.append(1 if player_label == 1 else 0)  # 1: 人狼, 0: 村人
                    all_features.append(aggregated_features)
        
        if not all_features:
            raise ValueError("No valid features extracted from data")
        
        # 特徴量の長さを統一
        max_length = max(len(features) for features in all_features)
        normalized_features = []
        
        for features in all_features:
            if len(features) < max_length:
                # 不足分をゼロで埋める
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
        """
        複数のモデルを訓練し最適なものを選択
        
        Args:
            X: 特徴量行列
            y: ラベルベクトル
            
        Returns:
            results: 各モデルの結果辞書
        """
        print(f"\nTraining models for {self.embedding_type}...")
        
        # データを分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        best_score = 0
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # パイプラインを作成（前処理 + モデル）
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # グリッドサーチでハイパーパラメータ最適化
            param_grid = {}
            for param, values in self.param_grids[model_name].items():
                param_grid[f'model__{param}'] = values
            
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # テストデータで評価
            y_pred = grid_search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # クロスバリデーション
            cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='f1')
            
            results[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{model_name} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
            # 最良モデルを更新
            if f1 > best_score:
                best_score = f1
                self.best_model = model_name
                self.best_pipeline = grid_search.best_estimator_
        
        print(f"\nBest model: {self.best_model} (F1: {best_score:.4f})")
        return results
    
    def save_model(self, results: Dict[str, Any]):
        """
        モデルと評価結果を保存
        
        Args:
            results: 訓練結果辞書
        """
        print(f"\nSaving models to {self.model_path}")
        
        # 最良モデルを保存
        model_file = self.model_path / f"best_model_{self.embedding_type.lower()}.joblib"
        joblib.dump(self.best_pipeline, model_file)
        print(f"Best model saved: {model_file}")
        
        # 全モデルを保存
        for model_name, result in results.items():
            model_file = self.model_path / f"{model_name.lower()}_{self.embedding_type.lower()}.joblib"
            joblib.dump(result['model'], model_file)
        
        # 評価結果を保存
        results_file = self.model_path / f"training_results_{self.embedding_type.lower()}.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Results for {self.embedding_type}\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Best Parameters: {result['best_params']}\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {result['f1_score']:.4f}\n")
                f.write(f"  CV Mean: {result['cv_mean']:.4f}\n")
                f.write(f"  CV Std: {result['cv_std']:.4f}\n")
                f.write("\n")
                
                # 分類レポート
                f.write(f"Classification Report for {model_name}:\n")
                f.write(classification_report(result['y_test'], result['y_pred']))
                f.write("\n" + "-" * 30 + "\n\n")
            
            f.write(f"Best Model: {self.best_model}\n")
        
        print(f"Results saved: {results_file}")
    
    def generate_plots(self, results: Dict[str, Any]):
        """
        可視化プロットを生成
        
        Args:
            results: 訓練結果辞書
        """
        try:
            # モデル比較プロット
            model_names = list(results.keys())
            f1_scores = [results[name]['f1_score'] for name in model_names]
            accuracies = [results[name]['accuracy'] for name in model_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # F1スコア比較
            ax1.bar(model_names, f1_scores, color='skyblue', alpha=0.7)
            ax1.set_title(f'F1 Score Comparison - {self.embedding_type}')
            ax1.set_ylabel('F1 Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # 精度比較
            ax2.bar(model_names, accuracies, color='lightcoral', alpha=0.7)
            ax2.set_title(f'Accuracy Comparison - {self.embedding_type}')
            ax2.set_ylabel('Accuracy')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_file = self.model_path / f"model_comparison_{self.embedding_type.lower()}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 混同行列（最良モデル）
            best_result = results[self.best_model]
            cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Villager', 'Werewolf'],
                       yticklabels=['Villager', 'Werewolf'])
            plt.title(f'Confusion Matrix - {self.best_model} ({self.embedding_type})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_file = self.model_path / f"confusion_matrix_{self.embedding_type.lower()}.png"
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved: {plot_file}, {cm_file}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def train(self):
        """
        完全な訓練パイプラインを実行
        """
        try:
            # データ読み込み
            X, y = self.load_all_data()
            
            # モデル訓練
            results = self.train_models(X, y)
            
            # モデル保存
            self.save_model(results)
            
            # 可視化
            self.generate_plots(results)
            
            print(f"\n{self.embedding_type} model training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise


def main():
    """メイン実行関数"""
    # ベースパス
    base_path = "/home/bi23056/lab/aiwolf-nlp-agent-llm/judgement_werewolf/libsvm"
    
    # 各埋め込み手法に対してモデル訓練
    embedding_types = ['Word2Vec', 'FastText', 'BERT']
    
    for embedding_type in embedding_types:
        print(f"\n{'='*60}")
        print(f"Training {embedding_type} Model")
        print(f"{'='*60}")
        
        try:
            # パス設定
            data_path = f"{base_path}/datasets/word_embeding/{embedding_type}"
            model_path = f"{base_path}/models/{embedding_type}"
            
            # トレーナー初期化
            trainer = WerewolfModelTrainer(data_path, model_path, embedding_type)
            
            # 訓練実行
            trainer.train()
            
        except Exception as e:
            print(f"Failed to train {embedding_type} model: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All model training completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()