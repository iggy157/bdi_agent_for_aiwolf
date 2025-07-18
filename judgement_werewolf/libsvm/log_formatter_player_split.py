#!/usr/bin/env python3
"""
ログファイルをプレイヤーごとに分割整形するスクリプト

整形ルール:
1. 0列目を削除
2. status行: 4列目がWEREWOLFなら3列目を1に、それ以外は0に変換
3. talk行: status行の2列目と一致する4列目を持つtalk行を抽出し、
   該当status行の3列目の値をtalk行の5列目に挿入
4. talk行以外を削除
5. talk行の4列目の値ごとに別ファイルに保存
6. 各ファイルのtalk行の0~4列目を削除
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class LogFormatterPlayerSplit:
    """ログファイルをプレイヤーごとに分割整形するクラス"""
    
    def __init__(self, source_base_path: str, output_base_path: str):
        self.source_base_path = Path(source_base_path)
        self.output_base_path = Path(output_base_path)
        
        # 出力ディレクトリを作成
        self.output_base_path.mkdir(parents=True, exist_ok=True)
    
    def process_all_datasets(self):
        """3つのデータセットを処理"""
        datasets = ['jsai2025_truck5', 'jsai2025_truck13', 'nlp2025']
        
        for dataset in datasets:
            dataset_path = self.source_base_path / dataset
            if dataset_path.exists():
                print(f"Processing dataset: {dataset}")
                self.process_dataset(dataset_path, dataset)
            else:
                print(f"Dataset not found: {dataset_path}")
    
    def process_dataset(self, dataset_path: Path, dataset_name: str):
        """単一データセットの処理"""
        log_files = list(dataset_path.glob("*.log"))
        
        if not log_files:
            print(f"No log files found in {dataset_path}")
            return
        
        for log_file in log_files:
            print(f"  Processing: {log_file.name}")
            try:
                player_data = self.format_log_file(log_file)
                
                # プレイヤーごとにファイルを保存
                for player_id, talk_lines in player_data.items():
                    if talk_lines:  # 空でない場合のみ保存
                        output_filename = f"{dataset_name}_{log_file.stem}_player{player_id}.txt"
                        output_path = self.output_base_path / output_filename
                        self.save_player_data(talk_lines, output_path)
                
            except Exception as e:
                print(f"    Error processing {log_file.name}: {e}")
    
    def format_log_file(self, log_file_path: Path) -> Dict[str, List[str]]:
        """単一ログファイルの整形"""
        # ファイルを読み込み
        lines = self.read_log_file(log_file_path)
        
        # ステップ1: 0列目を削除し、行を分類
        status_lines, talk_lines, other_lines = self.parse_lines(lines)
        
        # ステップ2: status行の処理（3列目の値を変換）
        status_mapping = self.process_status_lines(status_lines)
        
        # ステップ3: talk行の処理（5列目に数値データを挿入）
        processed_talk_lines = self.process_talk_lines(talk_lines, status_mapping)
        
        # ステップ4: プレイヤーごとに分割
        player_data = self.split_by_player(processed_talk_lines)
        
        return player_data
    
    def read_log_file(self, log_file_path: Path) -> List[List[str]]:
        """ログファイルを読み込んで行に分割"""
        lines = []
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # カンマで分割
                    columns = line.split(',')
                    if len(columns) > 1:
                        lines.append(columns)
        
        return lines
    
    def parse_lines(self, lines: List[List[str]]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """行を分類し、0列目を削除"""
        status_lines = []
        talk_lines = []
        other_lines = []
        
        for line in lines:
            if len(line) < 2:
                continue
            
            # 0列目を削除（元の1列目が新しい0列目になる）
            processed_line = line[1:]
            
            if not processed_line:
                continue
            
            # 新しい0列目（元の1列目）で分類
            line_type = processed_line[0]
            
            if line_type == 'status':
                status_lines.append(processed_line)
            elif line_type == 'talk':
                talk_lines.append(processed_line)
            else:
                other_lines.append(processed_line)
        
        return status_lines, talk_lines, other_lines
    
    def process_status_lines(self, status_lines: List[List[str]]) -> Dict[str, str]:
        """status行を処理してマッピングを作成"""
        status_mapping = {}
        
        for line in status_lines:
            if len(line) < 4:
                continue
            
            # line[0] = 'status'
            # line[1] = 元の2列目（プレイヤーID）
            # line[2] = 元の3列目（役職）
            # line[3] = 元の4列目（状態）
            
            player_id = line[1]
            role = line[2] if len(line) > 2 else ''
            
            # WEREWOLFなら1、それ以外は0
            if role == 'WEREWOLF':
                status_mapping[player_id] = '1'
            else:
                status_mapping[player_id] = '0'
        
        return status_mapping
    
    def process_talk_lines(self, talk_lines: List[List[str]], status_mapping: Dict[str, str]) -> List[List[str]]:
        """talk行を処理（5列目に数値データを挿入）"""
        processed_lines = []
        
        for line in talk_lines:
            if len(line) < 4:
                continue
            
            # line[0] = 'talk'
            # line[1] = 元の2列目
            # line[2] = 元の3列目
            # line[3] = 元の4列目（プレイヤーID）
            # line[4以降] = 元の5列目以降（発話内容）
            
            player_id = line[3]
            
            # status_mappingでプレイヤーIDに対応する値を取得
            if player_id in status_mapping:
                status_value = status_mapping[player_id]
                
                # 新しい行を作成（5列目に数値データを挿入）
                new_line = line[:4] + [status_value] + line[4:]
                processed_lines.append(new_line)
        
        return processed_lines
    
    def split_by_player(self, talk_lines: List[List[str]]) -> Dict[str, List[str]]:
        """talk行をプレイヤーごとに分割"""
        player_data = defaultdict(list)
        
        for line in talk_lines:
            if len(line) < 6:  # 0:talk, 1:?, 2:?, 3:player_id, 4:status_value, 5以降:text
                continue
            
            player_id = line[3]
            
            # 0~4列目を削除（5列目以降のみ残す）
            remaining_content = line[5:] if len(line) > 5 else []
            status_value = line[4]  # 5列目に挿入された数値データ
            
            if remaining_content:
                # 数値データ,テキストの形式で保存
                text = ','.join(remaining_content)
                formatted_line = f"{status_value},{text}"
                player_data[player_id].append(formatted_line)
        
        return dict(player_data)
    
    def save_player_data(self, talk_lines: List[str], output_path: Path):
        """プレイヤーデータを保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in talk_lines:
                f.write(line + '\n')
        
        print(f"    Saved: {output_path.name} ({len(talk_lines)} lines)")


def main():
    """メイン実行関数"""
    # パスの設定
    source_base_path = "judgement_werewolf/libsvm/raw_data"
    output_base_path = "judgement_werewolf/libsvm/datasets/data"
    
    # フォーマッターを初期化
    formatter = LogFormatterPlayerSplit(source_base_path, output_base_path)
    
    # すべてのデータセットを処理
    print("Starting log file formatting with player split...")
    formatter.process_all_datasets()
    print("Log file formatting completed.")


if __name__ == "__main__":
    main()