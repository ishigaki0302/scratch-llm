# GPU Handoff

このドキュメントは、GPUが空いた後に学習作業へ進むための引き継ぎメモです。

現在のスコープはGPUを使わない準備までです。ここでは、データパイプライン、設定、検証手順を揃えるところまでを完了条件にします。

## CPU準備の完了条件

- `uv.lock` がある
- seed corpus pipelineが通る
- cleaned corpusが作られている
- tokenizer training inputが作られている
- tokenizer configがある
- draft training configがある
- eval prompt setがある
- CPU-only validationが通る

検証コマンド:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_cpu_prep.py
```

## GPUを使う前に決めること

1. 本物のpretraining corpusをどこから始めるか
2. ライセンス方針をどこまで厳密にするか
3. 初号機の正確なmodel size
4. context length
5. tokenizer vocab size
6. training stack
7. checkpoint保存先
8. 実験ログの保存先

## GPUが空いた後の順序

1. `sentencepiece` を入れてtokenizerを学習する
2. tokenizerの日本語効率を評価する
3. 小さな外部corpusをcleaningする
4. tokenization throughputを測る
5. training stackを選ぶ
6. 10M-100M tokensのdry runを実施する
7. checkpoint resumeを確認する
8. 生成サンプルを確認する

## このセッションでやらないこと

- GPU memory確認
- CUDA / PyTorch GPU動作確認
- model training
- model checkpoint生成
- tokenizerの大規模学習
- external corpusの大規模tokenization

