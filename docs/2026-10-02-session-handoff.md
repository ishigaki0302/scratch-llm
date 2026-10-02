# セッション引き継ぎ（2026-10-02 16:50 JST時点）

前セッションはtmux外で作業していたため途中で終了。GPUジョブは `nohup` + init配下で裏で動き続けている。
次セッションはこのファイルを読んでから始めること。詳細ログは `2026-10-02-gpu-first-run.md`。

## 走っているジョブ（自動連結）

| 順 | ジョブ | 起動スクリプト | ログ | 完了条件 |
|---|---|---|---|---|
| 1 | dryrun-001（313M, 200 step ≈ 1.05億token）。100 stepで一旦停止 → `--resume` で200まで → 生成サンプル | `scripts/run_dryrun_001.sh` | `logs/dryrun-001.log` | ログ末尾に `DRYRUN_DONE`（16:43開始, 約50分） |
| 2 | run-002（Wikipedia 1 epoch: 2600 step ≈ 13.6億token, 約10時間） | `scripts/run_002_after_dryrun.sh`（dryrunの `DRYRUN_DONE` を待って自動起動。失敗時は起動しない） | `logs/run-002.log` | `runs/run-002-300m-ctx4k-wiki1ep/metrics.jsonl` に step 2600 の行。10/3 03:30頃完了見込み |

## 状況確認コマンド

```bash
cd /mnt/sda/ishigaki/scratch-llm
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ps -eo pid,etime,args | grep -E "pretrain.py|run_dryrun|run_002" | grep -v grep
grep -E "resumed|saved|DRYRUN_DONE|Error" logs/dryrun-001.log
tail -3 runs/dryrun-001-300m-ctx4k/metrics.jsonl
cat runs/dryrun-001-300m-ctx4k/samples.jsonl        # dryrunの生成サンプル
grep val_loss runs/run-002-300m-ctx4k-wiki1ep/metrics.jsonl | tail
cat checkpoints/run-002-300m-ctx4k-wiki1ep/latest
```

## run-002が止まっていた場合

`--resume` 付きなので、同じコマンドで最新checkpoint（250 stepごと）から再開できる。tmux内で実行すること。

```bash
tmux new -s scratch-llm
.venv/bin/torchrun --nproc_per_node=2 scripts/pretrain.py \
  --config configs/training/run-002-300m-ctx4k-wiki1ep.json --resume 2>&1 | tee -a logs/run-002.log
```

dryrunが失敗していた場合（`logs/run-002.log` に `dryrun failed`）: `logs/dryrun-001.log` のTracebackを見て修正 → `scripts/run_dryrun_001.sh` を再実行（`runs/` と `checkpoints/` の dryrun-001 を消してから）。

## 次セッションでやること

1. dryrun-001 の結果を `2026-10-02-gpu-first-run.md` に追記する
   - loss曲線（step 5/50/100/150/200, val_loss）、tok/s、peak mem
   - resume確認: `logs/dryrun-001.log` に `resumed from ... at step 100` があり、step 100前後でlossが連続していること
   - 生成サンプル（`runs/dryrun-001-300m-ctx4k/samples.jsonl`）。1億tokenなので崩れていて当然、形式が出ればOK
2. run-002 の経過を監視・記録（val_loss推移、生成サンプルを中間checkpointでも取る）
   - 生成: `.venv/bin/python scripts/generate_samples.py --checkpoint checkpoints/run-002-300m-ctx4k-wiki1ep/step_XXXXXXX`
   - 注意: 生成はGPUを使う。学習中は空きメモリ約6GB/GPUなので、bf16の313Mなら載るが、念のため学習完了後か小さく
3. run-002 の裏で（CPU作業）次のcorpus準備
   - 30B token目標に対しWikipediaは約14億token。web系日本語corpusが必要
   - 候補: FineWeb-2 `jpn_Jpan`（ODC-By）, CC-100 ja, llm-jp-corpus のライセンス明示部分。ライセンス方針を決めてから取得
   - English/code を混ぜる場合は tokenizer v002 を再学習（v001はURL・英語が細切れ）
4. 未解決: torch.compile（transformers 5.17 + torch 2.6 で失敗）。torch 2.7+/cu126 などで再挑戦すれば高速化の余地

## 前セッションでやったこと（要約）

- `uv` 環境: torch 2.6.0+cu124, transformers 5.17, sentencepiece（`uv sync --extra tokenizer --extra training`）
- 日本語Wikipedia取得 → corpus-v002（128万文書, 25.8億文字）→ tokenizer v001（unigram 48k, 1.86文字/token）→ tokenize（train 13.7億 / val 272万 token）
- `scripts/pretrain.py`: DDP + HF LlamaForCausalLM, checkpoint/resume, metrics.jsonl
- 313M / ctx4096 で 38.2k tok/s（2GPU合計）, micro batch 4 で 36.4GB/GPU
- GitHub: `git@github.com:ishigaki0302/scratch-llm.git`（public, main）。`slides-text.txt` は社内資料扱いで除外。READMEは日本語
- コミットは `git -c user.name=ishigaki0302 -c user.email=...` で作成（このマシンにgit global設定なし）
