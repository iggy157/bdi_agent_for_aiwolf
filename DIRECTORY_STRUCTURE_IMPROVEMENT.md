# 分析システム ディレクトリ構造改善完了

## 📁 改善された新しいディレクトリ構造

### **変更前**
```
analysis_results/
├── 01JZ7JGKRBE0N8VGY4K5DM969Z/  # ゲームIDのみ（わかりにくい）
│   ├── status_day0.yml
│   ├── status_day1.yml
│   └── analysis.yml
```

### **変更後**
```
analysis_results/
├── 20250703_164220_ID_12345/     # 日時 + ゲームID（わかりやすい）
│   ├── game_summary.yml         # ゲーム概要（新規追加）
│   ├── status_day0.yml
│   ├── status_day1.yml
│   └── analysis.yml
├── 20250703_165030_A7B8C9/      # 次のゲーム
│   ├── game_summary.yml
│   ├── status_day0.yml
│   └── analysis.yml
```

## 🆕 新機能

### 1. **わかりやすいディレクトリ名**
- **フォーマット**: `YYYYMMDD_HHMMSS_ゲームID下8桁`
- **例**: `20250703_164220_ID_12345`
- **利点**: 一目でゲームの実行日時がわかる

### 2. **ゲーム概要ファイル（game_summary.yml）**
```yaml
game_id: TEST_GAME_ID_12345
start_time: '2025-07-03T16:42:20.957540'
end_time: '2025-07-03T16:42:20.959299'
total_days: 2
agents: ['Alice', 'Bob', 'Charlie', 'Dave']
final_status:
  Alice: ALIVE
  Bob: DEAD
  Charlie: ALIVE
  Dave: DEAD
analysis_version: '1.0'
```

### 3. **段階的ディレクトリ作成**
- ゲーム開始時に一意のディレクトリを作成
- ゲーム進行中はそのディレクトリに分析結果を保存
- ゲーム終了時に概要ファイルを生成

## 🛠️ 実装詳細

### **新規追加ファイル**
- `src/analysis/utils.py`: ディレクトリ名生成とユーティリティ関数

### **修正ファイル**
1. **`packet_analyzer.py`**:
   - ゲーム情報の追跡機能
   - ディレクトリ名の生成
   - 概要ファイルの作成

2. **`status_analyzer.py`** / **`analysis_analyzer.py`**:
   - ゲームディレクトリパラメータの追加
   - 保存先の動的変更対応

### **主要な改善点**

#### **1. ゲーム情報追跡**
```python
self.game_info: Dict[str, Dict] = {}  # game_id -> game_info
self.game_directories: Dict[str, Path] = {}  # game_id -> directory_path
```

#### **2. 動的ディレクトリ生成**
```python
dir_name = create_game_directory_name(game_id, start_time)
# → "20250703_164220_ID_12345"
```

#### **3. 概要ファイル自動生成**
- ゲーム開始・終了時刻
- 参加エージェント一覧
- 総日数
- 最終的な生存状況

## 📋 使用方法

### **通常のゲーム実行**
```bash
python src/main.py
```

### **結果の確認**
```bash
ls analysis_results/
# → 20250703_164220_ID_12345/
# → 20250703_165030_A7B8C9/

ls analysis_results/20250703_164220_ID_12345/
# → game_summary.yml
# → status_day0.yml
# → status_day1.yml
# → analysis.yml
```

### **概要の確認**
```bash
cat analysis_results/20250703_164220_ID_12345/game_summary.yml
```

## 🎯 利点

### **1. 整理しやすさ**
- ゲームごとに独立したディレクトリ
- 日時順でのソート可能
- 複数ゲーム実行時の混在防止

### **2. 情報の充実**
- ゲーム概要の一目把握
- 分析結果の体系的整理
- バックアップ・アーカイブの容易さ

### **3. デバッグ・分析の向上**
- 特定ゲームの結果の特定が容易
- 時系列での結果比較
- 異常ケースの追跡改善

これでゲームを複数回実行しても、結果が整理されて見返しやすくなりました！