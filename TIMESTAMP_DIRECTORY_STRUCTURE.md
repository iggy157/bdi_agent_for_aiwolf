# タイムスタンプベース ディレクトリ構造完成

## 📁 **最終的なディレクトリ構造**

### **ゲームごとのタイムスタンプディレクトリ**
```
analysis_results/
├── 20250703_170034/          # ゲーム1 (17:00:34開始)
│   ├── game_summary.yml      # ゲーム概要
│   ├── status_day0.yml       # 0日目のエージェント状態
│   ├── status_day1.yml       # 1日目のエージェント状態
│   └── analysis.yml          # コミュニケーション分析
├── 20250703_170036/          # ゲーム2 (17:00:36開始)
│   ├── game_summary.yml
│   ├── status_day0.yml
│   └── analysis.yml
└── 20250703_170038/          # ゲーム3 (17:00:38開始)
    ├── game_summary.yml
    ├── status_day0.yml
    └── analysis.yml
```

## 📋 **各ファイルの役割**

### **1. game_summary.yml - ゲーム概要**
```yaml
game_id: GAME_001_TEST
start_time: '2025-07-03T17:00:34.123456'
end_time: '2025-07-03T17:05:42.789012'
total_days: 2
agents: ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
final_status:
  Alice: ALIVE
  Bob: DEAD
  Charlie: ALIVE
  Dave: DEAD
  Eve: ALIVE
analysis_version: '1.0'
```

### **2. status_day*.yml - 日別エージェント状態**
```yaml
Alice:
  agent_name: Alice
  alive: true
  self_co: SEER          # 自己申告役職
  seer_co: null          # 占い師からの報告
Bob:
  agent_name: Bob
  alive: true
  self_co: null
  seer_co: HUMAN         # 占い師から「人間」と報告
```

### **3. analysis.yml - コミュニケーション分析**
```yaml
- day: 0
  from: Alice
  idx: 1
  to: []
  topic: みんな、おはよう！
  type: group_address     # 全体への呼びかけ

- day: 0
  from: Bob
  idx: 2
  to: [Charlie]
  topic: Charlieは人狼です！
  type: negative          # 否定的発言
```

## 🎯 **主な特徴**

### **1. 完全なタイムスタンプベース**
- ディレクトリ名: `YYYYMMDD_HHMMSS`
- 例: `20250703_170034` (2025年7月3日 17:00:34)
- ゲームIDに依存しない独立した命名

### **2. 自動的な整理**
- ゲーム開始時に新しいタイムスタンプディレクトリを作成
- すべての分析結果をそのディレクトリに集約
- 複数ゲーム実行時も自動で分離

### **3. 豊富な分析情報**
- **役職分析**: カミングアウト、占い結果の追跡
- **コミュニケーション分析**: 発言の相手、感情、タイプ分類
- **ゲーム概要**: 参加者、日数、最終結果

## 🚀 **使用方法**

### **通常のゲーム実行**
```bash
python src/main.py
```

### **結果の確認**
```bash
# ディレクトリ一覧
ls analysis_results/
# → 20250703_170034/  20250703_170036/  20250703_170038/

# 最新ゲームの確認
ls analysis_results/$(ls analysis_results/ | tail -1)/
# → analysis.yml  game_summary.yml  status_day0.yml  status_day1.yml

# ゲーム概要の確認
cat analysis_results/20250703_170034/game_summary.yml
```

### **複数ゲームの比較**
```bash
# 各ゲームの概要比較
for dir in analysis_results/*/; do
  echo "=== $(basename $dir) ==="
  grep "total_days\|agents" "$dir/game_summary.yml"
done
```

## 📊 **利点**

### **1. 明確な時系列整理**
- ゲーム実行順序が一目瞭然
- 日時でのソート・検索が容易
- バックアップ・アーカイブの効率化

### **2. 完全な独立性**
- ゲームIDの重複や衝突の心配なし
- 各ゲームが独立したディレクトリ
- 並行実行時の競合回避

### **3. 分析の体系化**
- ゲーム全体の流れを包括的に記録
- エージェントの行動パターン追跡
- コミュニケーション戦略の分析

これで、ゲームを何度実行しても、タイムスタンプベースで整理された分析結果が自動的に保存されます！