# 学習準備ロードマップ初版

## ゴール

スクラッチから日本語SLMを学習するために、まず「何を決め、何を作り、どの順番で検証するか」を整理する。

この段階では学習を開始しない。先に、失敗したときに原因を切り分けられる準備を整える。

## 全体の流れ

1. 仕様を固定する
2. corpus候補を集める
3. cleaningとdedupの基準を作る
4. tokenizerを設計する
5. 小規模dry runを行う
6. 本学習用のtraining recipeを固める
7. 評価セットと評価手順を用意する
8. checkpointと実験ログの運用を決める
9. SFT用データ設計に進む

## 1. 仕様を固定する

最初に、初号機で狙う範囲を明文化する。

決めること:

- model size: 200M, 300M, 350Mのどれを初号機にするか
- context length: 2k, 4k, 8kのどれを採用するか
- tokenizer vocabulary size
- pretraining token target
- corpusの日本語比率
- code、英語、数式、表データをどこまで入れるか
- license上使えるデータだけに限定するか、研究用途として広めに集めるか

初期推奨:

- model size: 300M前後
- context length: 4096
- vocab size: 32k-64k
- target tokens: 30Bから開始
- Japanese-heavy corpus
- SFT前提なので、pretrainingではinstruction形式に寄せすぎない

## 2. Corpus候補を集める

pretrainingの主材料を決める。

候補:

- Japanese web text
- Wikipedia
- 青空文庫などの権利確認済みテキスト
- technical docs
- FAQ
- PDF由来テキスト
- codeを少量混ぜるかどうか
- English textを少量混ぜるかどうか

必要な管理情報:

- source name
- source URL or dataset ID
- license
- raw size
- estimated token count
- language ratio
- cleaning status
- dedup status
- inclusion decision

出力物:

- `corpus-manifest.csv` または `corpus-manifest.jsonl`
- source別のraw/cleaned保存ルール

## 3. CleaningとDedupの基準を作る

学習前に、品質の低いテキストを落とす。

最低限やること:

- HTML boilerplate除去
- 重複行、重複document除去
- 文字化け除去
- 極端に短いdocument除去
- 記号率が高すぎるdocument除去
- 日本語比率が低すぎるdocument除去
- 同一テンプレート文の過剰混入チェック
- 個人情報や危険な漏洩データの確認

判断を保留すること:

- 掲示板口調やSNS口調をどの程度残すか
- 低品質web文を完全に落とすか、少量残して頑健性を取るか
- English/codeを混ぜる割合

出力物:

- cleaning rules
- removed sample report
- before/after token count
- dedup rate

## 4. Tokenizerを設計する

日本語SLMではtokenizerが効くため、早めに比較する。

候補:

- SentencePiece BPE
- SentencePiece Unigram
- Hugging Face tokenizers BPE

比較観点:

- 日本語1文字あたりのtoken効率
- ひらがな、カタカナ、漢字、英数字混在の扱い
- technical termsの分割
- URL、コード、数式の扱い
- unknownやbyte fallbackの扱い
- SFT時のchat templateとの相性

初期推奨:

- SentencePiece Unigram or BPE
- vocab size 48k前後から比較
- byte fallbackあり
- normalization設定を明示

出力物:

- tokenizer config
- tokenizer model
- tokenizer evaluation report

## 5. 小規模Dry Runを行う

本学習前に、すべてのパイプラインを小さく回す。

目的:

- データローダが詰まらないこと
- tokenizationが破綻しないこと
- lossが正常に下がること
- checkpoint保存とresumeが動くこと
- 生成サンプルを取れること
- GPU memoryが見積もり内に収まること

規模:

- 10M-100M tokens程度
- 1-3時間で終わる範囲

出力物:

- dry-run config
- loss curve
- sample generations
- memory usage
- throughput

## 6. Training Recipeを固める

dry run後に、本学習用設定を決める。

決めること:

- architecture
- hidden size
- number of layers
- attention heads
- context length
- optimizer
- learning rate schedule
- warmup steps
- global batch size
- gradient accumulation
- precision
- checkpoint interval
- evaluation interval

検討項目:

- Hugging Face Transformersで行くか
- Megatron/DeepSpeed系を使うか
- FSDPを使うか
- FlashAttentionを使えるか

初期方針:

- まず運用しやすい実装を優先する
- 性能最適化は、dry runでボトルネックを見てから入れる

## 7. 評価セットを用意する

pretraining中から評価できるようにする。

最低限の評価:

- held-out Japanese corpus perplexity
- Wikipedia系held-out perplexity
- web text held-out perplexity
- 手作りpromptでの生成確認
- 日本語QAの簡易評価
- instruction追従はSFT後に評価

生成確認prompt例:

- 日本語のニュース風文章を続ける
- 技術解説文を続ける
- 箇条書きを続ける
- 敬体と常体の文章を続ける
- 不自然な反復が出ないか確認する

出力物:

- eval dataset
- eval script
- prompt set
- checkpoint別評価ログ

## 8. Checkpointと実験ログの運用

後から原因分析できるように、実験管理を固定する。

保存するもの:

- git commit hash
- model config
- training config
- tokenizer version
- corpus manifest version
- data shard checksums
- loss logs
- eval logs
- sample generations
- hardware info
- runtime and throughput

命名例:

- `run-001-300m-ctx4k-30btok`
- `tokenizer-v001-spm48k`
- `corpus-v001-ja-heavy`

## 9. SFT用データ設計に進む

base modelがある程度自然な日本語を出せるようになってから、SFTに進む。

SFTで足すもの:

- instruction following
- chat format
- safety refusal
- tone control
- Japanese assistant behavior
- domain-specific behavior

pretrainingと混ぜないこと:

- chat template依存の振る舞い
- refusal policy
- assistant persona
- 長いsystem prompt前提の挙動

## 直近のTODO

1. 初号機のmodel sizeを決める
2. corpus manifestの形式を決める
3. tokenizer比較用の小さな日本語corpusを作る
4. cleaning ruleの初版を書く
5. dry run用training stackを選ぶ
6. evaluation prompt setを作る

## 次に書くべきドキュメント

- `corpus-design.md`
- `tokenizer-design.md`
- `training-stack.md`
- `evaluation-plan.md`

