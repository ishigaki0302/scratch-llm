# scratch-llm

LLMを作ってみる — 日本語の小型言語モデル（SLM）をスクラッチから学習するプロジェクトです。

- 計算資源: RTX A6000 48GB × 2
- 初号機: 約300Mパラメータ、context 4096、まずはbase model（SFT・対話化は後段）
- 方針の詳細: [`docs/training-prep.md`](docs/training-prep.md)、作業ログは [`docs/`](docs/README.md)

## ディレクトリ構成

- `configs/` - corpus・tokenizer・学習・評価の設定
- `scripts/` - データ準備・tokenizer学習・事前学習・生成のスクリプト
- `docs/` - 設計ドキュメントと日付付き作業ログ
- `src/scratch_llm/` - Pythonパッケージ（予約）
- `data/`, `artifacts/`, `checkpoints/`, `runs/`, `logs/` - 生成物（git管理外）

## 環境構築

環境管理には `uv` を使います。PyTorchはCUDA 12.4版を入れます。

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra tokenizer --extra training
```

## パイプライン

### 1. Corpus準備（日本語Wikipedia → cleaning → tokenizer学習）

HF `wikimedia/wikipedia` の `20231101.ja` を `data/raw/hf/wikipedia/` にダウンロードしたうえで:

```bash
scripts/run_corpus_v002.sh
```

中身は次の順に実行します。

1. `scripts/convert_wikipedia.py` - parquet → JSONL、manifest（`data/manifests/corpus-v002.jsonl`）を更新
2. `scripts/clean_corpus.py` - 正規化、短文・低日本語率・完全重複の除去
3. `scripts/build_tokenizer_corpus.py` - tokenizer学習用テキストを作成
4. `scripts/train_tokenizer.py` - SentencePiece（unigram, 48k, byte fallback）を学習

JSONLの文書形式:

```json
{"id":"source_doc_id","source_id":"source_name","text":"document text","meta":{}}
```

### 2. Tokenize

```bash
.venv/bin/python scripts/tokenize_corpus.py
```

`data/tokenized/corpus-v002-spm48k/{train,val}.bin`（uint16の連結token列）と `meta.json` を出力します。

### 3. 事前学習

```bash
.venv/bin/torchrun --nproc_per_node=2 scripts/pretrain.py \
  --config configs/training/dryrun-001-300m-ctx4k.json
# 再開する場合は --resume を付ける
```

- checkpoint: `checkpoints/<run_id>/step_XXXXXXX/`（HF形式 + optimizer等の `train_state.pt`）
- 学習ログ: `runs/<run_id>/metrics.jsonl`

### 4. 生成サンプル

```bash
.venv/bin/python scripts/generate_samples.py --checkpoint checkpoints/<run_id>/step_XXXXXXX
```

`configs/eval/prompt-set-v001.jsonl` のプロンプトに対する続きを生成します。

## 旧seed pipeline

`slides-text.txt`（企画スライドの抽出テキスト）を使った極小seed corpusでの動作確認用スクリプト（`prepare_seed_corpus.py` など）も残しています。`slides-text.txt` 自体は公開repoには含めていません。

## ライセンス

コードはMIT License。学習データ（日本語Wikipedia）は CC BY-SA 4.0 / GFDL に従います。
