#!/usr/bin/env python3
"""
全てのワードエンベディング手法を実行する統合スクリプト

Word2Vec、FastText、BERTの3つの手法でエンベディング処理を順次実行
"""

import sys
import os
from pathlib import Path

# 現在のディレクトリをパスに追加
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 各プロセッサをインポート
try:
    from word2vec_processor import Word2VecProcessor
except ImportError as e:
    print(f"Error importing Word2VecProcessor: {e}")
    Word2VecProcessor = None

try:
    from fasttext_processor import FastTextProcessor
except ImportError as e:
    print(f"Error importing FastTextProcessor: {e}")
    FastTextProcessor = None

try:
    from bert_processor import BERTProcessor
except ImportError as e:
    print(f"Error importing BERTProcessor: {e}")
    BERTProcessor = None


def run_all_embeddings():
    """全てのエンベディング手法を実行"""
    
    print("="*60)
    print("Starting Word Embedding Processing for Werewolf Detection")
    print("="*60)
    
    # 1. Word2Vec処理
    if Word2VecProcessor:
        print("\n" + "="*40)
        print("1. Word2Vec Processing")
        print("="*40)
        try:
            processor = Word2VecProcessor()
            processor.process_all()
        except Exception as e:
            print(f"Error in Word2Vec processing: {e}")
    else:
        print("\nSkipping Word2Vec processing due to import error")
    
    # 2. FastText処理
    if FastTextProcessor:
        print("\n" + "="*40)
        print("2. FastText Processing")
        print("="*40)
        try:
            processor = FastTextProcessor()
            processor.process_all()
        except Exception as e:
            print(f"Error in FastText processing: {e}")
    else:
        print("\nSkipping FastText processing due to import error")
    
    # 3. BERT処理
    if BERTProcessor:
        print("\n" + "="*40)
        print("3. BERT Processing")
        print("="*40)
        try:
            processor = BERTProcessor()
            processor.process_all()
        except Exception as e:
            print(f"Error in BERT processing: {e}")
    else:
        print("\nSkipping BERT processing due to import error")
    
    print("\n" + "="*60)
    print("All Word Embedding Processing Completed!")
    print("="*60)
    
    # 結果の確認
    base_path = Path("judgement_werewolf/libsvm/datasets/word_embeding")
    
    for method in ["Word2Vec", "FastText", "BERT"]:
        method_path = base_path / method
        if method_path.exists():
            files = list(method_path.glob("*.libsvm"))
            print(f"{method}: {len(files)} files created")
        else:
            print(f"{method}: No output directory found")


if __name__ == "__main__":
    run_all_embeddings()