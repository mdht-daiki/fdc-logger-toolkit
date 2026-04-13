Title: Child - GET /charts/history

Parent

- Parent - DB API Minimum Contract Draft (Phase 1)

Background

- chart 閾値変更履歴を dashboard/ops が監査可能に参照するための endpoint。

Scope

- GET /charts/history の実装
- 任意クエリのフィルタ実装
  - chart_id, chart_set_id, change_source, from_ts, to_ts, limit, offset
- ページング実装（default/max の契約遵守）

Acceptance Criteria

- 200: ok=true, data=list を返す
- history レコードに最小フィールドを含む
  - history_id, chart_id, chart_set_id, change_source, change_reason
  - before, after, changed_by, changed_at
- limit default=100, max=500 を満たす
- 400/422: クエリ不正時に契約どおりエラー
- changed_at が UTC ISO 8601 ミリ秒固定精度

Tests

- 契約テスト: 200
- 契約テスト: 主要 4xx
- 契約テスト: timestamp 形式
- 契約テスト: limit/offset 境界
- 回帰確認: 既存 ingest write endpoint へ非影響

Docs

- docs/db-api-endpoints.md の status/tracking 更新
- 必要に応じて docs/db-api-minimum-contract.md の差分更新
