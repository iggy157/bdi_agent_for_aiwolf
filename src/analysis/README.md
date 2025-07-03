# AIWolf Game Analysis System

このモジュールは人狼ゲームのサーバーリクエスト情報を分析し、YAMLファイルに保存するシステムです。

## 概要

分析システムは以下の2種類のファイルを生成します：

### 1. status.yml（各日ごと）
各エージェントの状態を日単位で記録します：
- `agent_name`: エージェント名
- `self_co`: エージェントが自分で宣言した役職（null または役職名）
- `seer_co`: 占い師を名乗るエージェントから報告された役職（null または役職名）
- `alive`: 生存状態（boolean）

ファイル名例: `status_day0.yml`, `status_day1.yml`, ...

### 2. analysis.yml（ゲーム単位）
エージェント間のコミュニケーション分析結果をゲーム単位で記録します：
- `type`: メッセージタイプ（question, positive, negative, group_address）
- `from`: 発言者エージェント名
- `to`: 対象エージェント名のリスト
- `topic`: 発言内容
- `day`: 発言日
- `idx`: 発言インデックス

## 設定

`config/config.yml`で設定できます。分析用LLMは既存のLLM設定を使用します：

```yaml
llm:
  type: google  # または openai, ollama
  sleep_time: 3

# LLMプロバイダ固有の設定
google:
  model: gemini-2.0-flash-lite
  temperature: 0.7

openai:
  model: gpt-4o-mini
  temperature: 0.7

ollama:
  model: llama3.1
  temperature: 0.7
  base_url: http://localhost:11434

analysis:
  enabled: true                    # 分析機能の有効/無効
  output_dir: ./analysis_results   # 出力ディレクトリ
  
  prompts:
    # 各種分析プロンプトの設定
```

## 環境変数

使用するLLMプロバイダに応じて、必要なAPIキーを環境変数として設定してください：

### Google Gemini使用時:
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

### OpenAI使用時:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Ollama使用時:
環境変数は不要（ローカルサーバー使用）

## 出力構造

```
analysis_results/
├── game_id_1/
│   ├── status_day0.yml
│   ├── status_day1.yml
│   ├── ...
│   └── analysis.yml
├── game_id_2/
│   ├── status_day0.yml
│   ├── ...
│   └── analysis.yml
...
```

## 使用方法

分析システムは`starter.py`に自動的に統合されており、ゲーム実行時に自動的に動作します。手動で無効にする場合は設定で`analysis.enabled: false`を設定してください。

## LLMサポート

分析システムは既存のエージェントと同じLLMプロバイダをサポートしています：
- **Google Gemini** (推奨)
- **OpenAI GPT**
- **Ollama**（ローカル）

`config.yml`の`llm.type`設定で自動的に選択されます。

## 依存関係

- `langchain-google-genai>=2.1.0`: Google Gemini API接続
- `langchain-openai>=0.3.9`: OpenAI API接続
- `langchain-ollama>=0.3.0`: Ollama接続
- `pyyaml>=6.0.2`: YAML出力
- `jinja2>=3.1.6`: プロンプトテンプレート処理