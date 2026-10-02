# GPU初回作業ログ（2026-10-02）

`gpu-handoff.md` の「GPUが空いた後の順序」に沿って、tokenizer学習からdry runまでを進めた記録。

## 0. 決めたこと（handoffの「GPUを使う前に決めること」への暫定回答）

| 項目 | 暫定決定 | 理由 |
|---|---|---|
| 最初の本物corpus | 日本語Wikipedia（HF `wikimedia/wikipedia` `20231101.ja`） | ライセンス明確（CC BY-SA）、取得が容易、品質が高い。dry runに十分な量（約15億token） |
| ライセンス方針 | 当面はライセンスが明示されたsourceのみ | 公開repoで管理するため。web系は次段で検討 |
| model size | 313M（24層, d=1024, 16 heads / 8 KV heads, FFN 2560, tied embedding） | 「300M前後」の初期推奨に合わせた。GQAでKVを半分に |
| context length | 4096 | training-prepの初期推奨 |
| vocab size | 48,000（SentencePiece Unigram, byte fallback） | `tokenizer-v001.json` の通り |
| training stack | 素のPyTorch DDP + HF `LlamaForCausalLM`（SDPA attention） | 運用しやすさ優先。HF形式で保存するので生成・評価がそのまま使える |
| checkpoint保存先 | `checkpoints/<run_id>/step_XXXXXXX/`（HF形式 + `train_state.pt`） | gitignore済み |
| 実験ログ | `runs/<run_id>/metrics.jsonl`, `run_meta.json`, `config.json` | gitignore済み。要約はdocsに転記する |

## 1. 環境

- `requires-python >= 3.11`（venvは元々3.11）
- torch 2.6.0+cu124（driver 535 / CUDA 12.2 環境で動作確認、A6000×2 認識）
- `pyproject.toml` に PyTorch cu124 index を追加、`setuptools` を training extra に追加（tritonが要求）

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra tokenizer --extra training
```

## 2. Corpus v002（日本語Wikipedia）

一括実行: `scripts/run_corpus_v002.sh`（convert → clean → summarize → tokenizer入力 → tokenizer学習）

- 取得: 15 parquet, 3.7GB
- 変換: `scripts/convert_wikipedia.py` → `data/raw/wikipedia_ja/wikipedia-ja-20231101.jsonl`, manifest `data/manifests/corpus-v002.jsonl`
  - 1,389,467 文書 / 2,658,082,408 文字
- cleaning: `clean_corpus.py --min-chars 200 --min-ja-ratio 0.3`
  - 書き出し 1,283,826 文書 / 2,576,173,802 文字
  - too short 95,299 / low JA ratio 10,342 / exact dup 0

## 3. Tokenizer v001

- 入力: cleaned corpusから40万文書をランダム抽出（約940MB）、SentencePieceはさらに500万行をサンプル
- 設定: `configs/tokenizer/tokenizer-v001.json`（unigram, 48k, byte fallback, nmt_nfkc, split_digits）
- special: unk=0, bos=1, eos=2, pad=3。SFT用に `<|system|> <|user|> <|assistant|> <|end|>` を予約

（結果は下に追記）

## 4. スループット計測（合成データ）

313M / ctx4096 / global batch 128 seq（524,288 tok/step）、bf16 autocast、SDPA、2GPU DDP。

| 設定 | tok/s（2GPU合計） | peak mem/GPU |
|---|---|---|
| micro batch 8 | OOM | - |
| micro batch 4（grad accum 16） | 約38,200 | 36.4 GB |
| micro batch 2（grad accum 32） | 約36,100 | 21.5 GB |
| torch.compile（全体） | transformers 5.17 の出力キャプチャwrapperでdynamoが失敗 | - |
| torch.compile（layer単位） | CUDA illegal memory access | - |

→ 当面は eager, micro batch 4 で進める（dryrun config も 4 に修正済み。8 は OOM）。compileは torch/transformers のversion組み合わせを変えて後で再検討。

見積もり: 38k tok/s で 30B token ≈ 9.1日, 1.5B token（Wikipedia 1 epoch）≈ 11時間。

（dry run以降は下に追記）
