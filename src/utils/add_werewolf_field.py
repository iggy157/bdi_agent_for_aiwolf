#!/usr/bin/env python3
"""
既存のstatus.ymlファイルにwerewolfフィールドを追加するスクリプト
"""

import yaml
from pathlib import Path

def add_werewolf_field_to_status_file(file_path: Path):
    """単一のstatus.ymlファイルにwerewolfフィールドを追加"""
    try:
        # ファイル内容を読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            status_data = yaml.safe_load(f)
        
        if not status_data:
            return
        
        # 各ターンのデータを処理
        modified = False
        for turn_key, turn_data in status_data.items():
            if isinstance(turn_data, dict):
                for agent_key, agent_data in turn_data.items():
                    if isinstance(agent_data, dict) and agent_key.startswith('agent'):
                        # werewolfフィールドが存在しない場合はfalseを追加
                        if 'werewolf' not in agent_data:
                            agent_data['werewolf'] = False
                            modified = True
        
        # 変更があった場合のみファイルを更新
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(status_data, f, allow_unicode=True, default_flow_style=False)
            print(f"Added werewolf fields to: {file_path}")
        else:
            print(f"No changes needed for: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    """メイン関数"""
    status_base_path = Path("info/status")
    
    if not status_base_path.exists():
        print(f"Status base path does not exist: {status_base_path}")
        return
    
    # すべてのstatus.ymlファイルを検索
    status_files = list(status_base_path.glob("**/status.yml"))
    print(f"Found {len(status_files)} status files to process")
    
    # 各ファイルを処理
    for file_path in status_files:
        add_werewolf_field_to_status_file(file_path)
    
    print("Werewolf field addition completed!")

if __name__ == "__main__":
    main()