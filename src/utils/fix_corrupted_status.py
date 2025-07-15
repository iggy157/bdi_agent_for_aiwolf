#!/usr/bin/env python3
"""
破損したstatus.ymlファイルを修正するスクリプト
"""

import os
import yaml
import re
from pathlib import Path

def fix_corrupted_status_file(file_path: Path):
    """破損したstatus.ymlファイルを修正"""
    try:
        # ファイル内容を読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # numpy関連の行を削除
        lines = content.split('\n')
        clean_lines = []
        skip_next_lines = 0
        
        for line in lines:
            if skip_next_lines > 0:
                skip_next_lines -= 1
                continue
                
            # numpy関連の行を検出
            if 'numpy' in line or '!!python/object' in line or '!!binary' in line:
                # この行と次の数行をスキップ
                skip_next_lines = 10  # 安全のため多めにスキップ
                continue
            
            # werewolf: で始まる行で numpy が含まれる場合、単純に False に置換
            if line.strip().startswith('werewolf:') and 'numpy' in line:
                clean_lines.append('    werewolf: false')
                continue
                
            clean_lines.append(line)
        
        # クリーンアップした内容を書き戻し
        clean_content = '\n'.join(clean_lines)
        
        # 最後に余分な空行を削除
        clean_content = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_content)
        clean_content = clean_content.rstrip() + '\n'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clean_content)
        
        print(f"Fixed: {file_path}")
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")

def main():
    """メイン関数"""
    status_base_path = Path("/home/bi23056/lab/aiwolf-nlp-agent-llm/info/status")
    
    # 破損したファイルを検索
    corrupted_files = []
    for status_file in status_base_path.glob("**/status.yml"):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'numpy' in content or '!!python/object' in content:
                    corrupted_files.append(status_file)
        except:
            pass
    
    print(f"Found {len(corrupted_files)} corrupted status files")
    
    # 修正実行
    for file_path in corrupted_files:
        fix_corrupted_status_file(file_path)
    
    print("Corruption fix completed!")

if __name__ == "__main__":
    main()