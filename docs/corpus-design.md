# Corpus Design

## 目的

日本語base modelのpretrainingに使うcorpusを、sourceごとに追跡可能な形で管理する。

初期段階では、巨大な一枚岩のデータセットを作らない。sourceごとのraw、cleaned、report、manifestを分けて、後から品質差を比較できるようにする。

## JSONL Document Format

すべてのsource converterは次の形式を出力する。

```json
{"id":"source_doc_id","source_id":"source_name","text":"document text","meta":{}}
```

必須フィールド:

- `id`
- `source_id`
- `text`
- `meta`

## Manifest Fields

`data/manifests/*.jsonl` は、cleaning対象のsource fileを列挙する。

推奨フィールド:

- `source_id`
- `name`
- `path`
- `format`
- `language`
- `license`
- `include`
- `document_count`
- `character_count`
- `created_at`
- `notes`

## Source Categories

初期候補:

- Japanese Wikipedia
- Japanese web text
- technical documentation
- public domain books
- FAQ / help-center style text
- small amount of English
- small amount of code

## 初期Mix方針

初号機は日本語baseとして成立させることを優先する。

- Japanese prose: high priority
- technical Japanese: high priority
- English: low but non-zero
- code: low but non-zero
- noisy web text: controlled amount
- instruction/chat data: pretrainingには混ぜすぎない

## Cleaning Baseline

最低限のfilter:

- too short
- low Japanese ratio
- exact duplicate
- control characters
- broken encoding
- excessive symbols
- boilerplate-heavy documents

今後追加するfilter:

- near dedup
- URL-heavy filter
- template repetition filter
- PII risk filter
- source-specific blocklist


## ライセンス方針（2026-10-03 暫定）

公開repoでcorpusの作り方を管理し、将来的にモデルを公開する可能性も残すため、次の基準でsourceを選ぶ。
最終決定ではなく、モデルを公開する前に見直す。

| 区分 | 扱い | 例 |
|---|---|---|
| 再配布・改変・商用利用を許可する明示ライセンス | 採用 | CC0, パブリックドメイン, CC BY, ODC-By, Apache-2.0, MIT |
| 継承条件付き（SA） | 採用。manifestに明記する | CC BY-SA（Wikipedia） |
| 非営利・改変禁止 | 不採用 | CC BY-NC, CC BY-ND |
| ライセンスが明示されていない・不明 | 保留（使わない） | ライセンス表記のないcrawl dump |
| データセットごとに条件が違う集合 | ライセンスが明示されたsubsetだけ採用 | llm-jp-corpus |

運用ルール:

- manifestの `license` に、元のライセンス文字列をそのまま書く。Common Crawl由来なら「Common Crawl Terms of Use に従う」も併記する
- 1 sourceごとに `source_id` を分け、後から特定のsourceだけ除外して再構築できるようにする
- raw / cleaned / tokenized データ自体は公開repoに含めない（gitignore済み）。公開するのは変換スクリプトとmanifestだけ

候補sourceの判断:

| source | ライセンス | 判断 | メモ |
|---|---|---|---|
| 日本語Wikipedia（`wikimedia/wikipedia`） | CC BY-SA 4.0 / GFDL | 採用済み（corpus-v002） | |
| FineWeb-2 `jpn_Jpan`（`HuggingFaceFW/fineweb-2`） | ODC-By 1.0（Common Crawl ToUに従う） | **採用**（corpus-v003） | 全体で717GB（約148シャード × 4.84GB、1シャード約276万文書・約40億文字）。言語判定・dedup・品質filterは配布元で適用済み |
| CC-100 ja | 明示ライセンスなし（Common Crawl ToU） | 保留 | FineWeb-2と中身の多くが重なるうえ、ライセンスがFineWeb-2ほど明確でない |
| llm-jp-corpus v3 | subsetごとに異なる | 一部採用候補 | 科研費報告書（kaken）などライセンス明示のsubsetを個別に確認してから |
| 青空文庫 | 著作権切れ作品はパブリックドメイン | 採用候補 | 旧仮名遣い・文体の偏りがあるので比率は少なめに |
| English / code | sourceによる | 未着手 | 混ぜる場合はtokenizer v002を学習し直す |
