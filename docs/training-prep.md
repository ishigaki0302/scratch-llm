# 学習準備の基本方針

## 目的

A6000 48GB x 2 の現実的な計算資源で、日本語として自然に読める小型base modelをスクラッチから作る。

初号機の狙いは、既存0.5B-1B級instruct modelの再現ではなく、次の3点を成立させること。

- 日本語base modelとして破綻しない文章生成
- tokenizer、corpus、training recipeを自前で管理できる状態
- 後段のSFTで対話化できる土台

## 初号機の前提

- Model size: 200-350M parameters
- Training tokens: 30-60B tokens
- Phase 1: tokenizer設計
- Phase 2: pretraining corpus構築
- Phase 3: base pretraining
- Phase 4: evaluation
- Phase 5: SFTによる対話化

## 進め方の原則

1. まずbase modelを作る。
   chat能力や安全性はpretrainingではなく、SFT以降で足す。

2. データを先に固める。
   小型モデルほど、architecture差よりtokenizerとcorpus品質の影響が大きい。

3. 各工程をmanifestで記録する。
   後から比較できるように、データ、tokenizer、学習設定、評価結果を分けて保存する。

4. 最初から巨大化しない。
   1B級を無理に回すより、200-350Mを十分なtokenで育てて、失敗要因を観察する。

5. 評価を後回しにしない。
   pretraining途中のcheckpointごとに、perplexity、日本語生成、タスク評価を確認する。

## 成果物

最初に揃える成果物は次の通り。

- tokenizer training recipe
- corpus manifest
- cleaned corpus shards
- deduplication report
- model config
- training config
- evaluation script
- checkpoint管理ルール
- SFT用データ設計メモ

