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

