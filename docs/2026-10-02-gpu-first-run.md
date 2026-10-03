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

学習時間: 約8分（CPU 20 threads）。出力 `artifacts/tokenizers/tokenizer-v001-spm48k.{model,vocab}`（gitignore）。

評価（cleaned corpusから1,000文書を等間隔抽出）:

| 指標 | 値 |
|---|---|
| 文字/token | 1.864 |
| byte fallback率 | 0.38% |
| unk | 0 |

分割例:

- `東京都は、日本の首都であり、` → `▁東京都 / は / 、 / 日本 / の首都 / であり / 、`（14文字→7token）
- `2026年10月2日に…` → 数字は1桁ずつ（split_digits）
- URL・英語・コードは細切れ（`https://example.com` → `http / s / :// / ex / amp / le …`）。日本語Wikipediaのみで学習したため。英語/code混入を始めたらtokenizer v002で再学習する

## 3.5 Tokenize

`scripts/tokenize_corpus.py`（20 worker, 約5分）→ `data/tokenized/corpus-v002-spm48k/`

| split | 文書 | token |
|---|---|---|
| train | 1,281,279 | 1,372,611,766 |
| val（doc id hashで0.2%） | 2,547 | 2,716,105 |

各文書は `<bos> ... <eos>` で連結。学習時はランダム位置から4096 token窓を切り出す（文書境界のattention maskはしない）。

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

## 5. dryrun-001（2026-10-02 16:43〜17:30）

`scripts/run_dryrun_001.sh`: 313M, 200 step（約1.05億token）。100 stepで一旦止めて `--resume` で200まで続け、最後に生成サンプルを出す。

| step | train loss | val_loss |
|---|---|---|
| 5 | 9.47 | - |
| 50 | 7.17 | 7.16 |
| 100 | 6.27 | 6.31 |
| 105（再開直後） | 6.19 | - |
| 150 | 5.84 | 5.91 |
| 200 | 5.72 | 5.68（ppl 292） |

- tok/s 中央値 38,040（2GPU合計）、peak mem 36.4 GB/GPU。スループット計測と一致
- **resume確認OK**: `resumed from checkpoints/dryrun-001-300m-ctx4k/step_0000100 at step 100`。step 95→100→105 で loss が 6.48→6.27→6.19 と連続しており、optimizer stateとdata RNGの復元に問題なし
- 生成サンプル（`runs/dryrun-001-300m-ctx4k/samples.jsonl`）: 年月日・カタカナ・記号の羅列で文になっていない。1億tokenの段階なので想定どおりで、生成パイプライン（HF形式でのload → sentencepieceでのdecode）が動くことは確認できた

## 6. run-002（Wikipedia 1 epoch, 2026-10-02 17:32〜10-03 03:33）

`configs/training/run-002-300m-ctx4k-wiki1ep.json`: 2600 step × 524,288 tok = 13.6億token（corpus-v002 train をほぼ1 epoch）。lr 6e-4, warmup 200, cosineで下限6e-5まで下げる。dryrunの `DRYRUN_DONE` を受けて自動起動し、止まらずに完走した（所要10.0時間、`resumed` なし、NaNなし）。

| step | 100 | 300 | 500 | 1000 | 1500 | 1800 | 2000 | 2200 | 2400 | 2500 | 2600 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val_loss | 6.64 | 4.76 | 3.82 | 3.24 | 3.02 | 2.89 | 2.92 | 2.84 | 2.80 | 2.79 | **2.82** |
| val ppl | 764 | 117 | 45.5 | 25.5 | 20.5 | 17.9 | 18.5 | 17.0 | 16.4 | 16.3 | **16.7** |

- train lossの100 step平均: step 1700〜1800 で 2.95 → 2000〜2100 で 2.85 → 2500〜2600 で 2.81。終盤はlrが下限付近にあり、下がり方が緩い
- val_lossは1回あたり40 step分の評価で、上下±0.03程度ぶれる（2000, 2600で小さく上がるのはこのぶれの範囲内）
- tok/s 中央値 37,994、peak mem 36.4 GB/GPU、grad_norm 最大 2.6（序盤だけ）
- checkpointは `step_0002000 / 2250 / 2500 / 2600`（keep 4）

### 生成サンプル（2026-10-03取得）

`scripts/generate_samples.py`（temperature 0.8, top_p 0.95, repetition_penalty 1.1, 最大128 token）。
出力: `runs/run-002-300m-ctx4k-wiki1ep/samples-step{2000,2600}.jsonl`

step 2600 の抜粋:

- `東京都は、日本の首都であり、` → 「首都でもある。 概要 東京都は、1948年に東京特別区(現在の豊島区)と東京市の区境となる東京特別区から分割され成立し、…」
- `織田信長は、戦国時代から安土桃山時代にかけての武将で、` → 「信長政権下の中国や東南アジア方面の覇権を巡っていた。 出自について 本記事では、南朝と室町時代の間を取り持つ説や、…」
- `気象庁によると、今週末は西日本を中心に` → 「、冬期間は日中も気温が厳しくなってきます。九州は夏日で、秋は暑くなるのです。 歴史 2016年(平成28年)3月 - 気象庁は、…」
- `小型言語モデルでtokenizerの品質が重要になる理由を説明してください。` → 「例: 普通は に から に  と が に対応する。 は、 が に対応できる …」

所見:

1. **日本語として文法的に読める文が出るようになった**。dryrun（1億token）と比べて明らかに改善。「概要」「生涯」「歴史」「出自について」といったWikipediaの節見出しや、年表の箇条書きをよく再現する
2. **内容は事実と合わない**（例: 信長の長子、東京都が大阪府に分割）。313M / 13.6億tokenの段階では想定どおり
3. **どのプロンプトもWikipedia調になる**。ニュース・日常文・敬体を指定しても、数文で「歴史」の節や年表に流れる。corpusがWikipediaだけなことがそのまま出ている → web系corpusを混ぜる理由になる
4. **数式の位置が空白になる**。`ja_explain_001` や `ja_list_001` で「 は、 が に対応できる」のように、数式があった場所が空白のまま並ぶ。HFの `wikimedia/wikipedia` のtextは数式（`<math>`）が除去済みで、モデルが空白の穴まで学習している。corpus側でも確認済み: `corpus-v002-cleaned.jsonl` の先頭20万文書のうち、「線形・関数・写像」を含み、かつ助詞の前後が空白になっている文書が1,378件あった（例: 「ここで、 と は共通のアルファベットから構成される言語であるとする」）。対策の候補は、cleaningで「助詞の前に空白が残った文」を検出して落とすか、数式を含む記事の比率を下げること
5. 敬体プロンプトでは同じ文を繰り返す（`ja_style_001`）。sampling設定の問題というより、このプロンプトに対応する分布をまだ学習できていない
6. step 2000 と 2600 の差は小さい（val_lossの差は0.1程度）

### run-002 のまとめ

Wikipediaだけ・1 epochでの土台ができた。val_loss 2.82 はtokenizer v001（1.86文字/token）での値なので、tokenizerが変わると比較できない点に注意。次の伸びしろはモデルではなくデータにある（量: 30B目標に対して1.4B、多様性: Wikipedia調から抜けられない）ので、次はweb系corpusの準備を優先する。
