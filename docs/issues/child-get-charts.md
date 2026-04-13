Title: Child - GET /charts

Parent

- Parent - DB API Minimum Contract Draft (Phase 1)

Background

- dashboard/judge が chart 定義一覧を db_api 経由で取得するための基礎 endpoint。

Scope

- GET /charts の実装
- 任意クエリのフィルタ実装
  - tool_id, chamber_id, recipe_id, parameter, step_no, feature_type, active_only
- レスポンス envelope を契約どおり実装

Acceptance Criteria

- 200: 正常時に ok=true, data=list を返す
- 400/422: クエリ不正時に契約どおりエラーを返す
- timestamp が UTC ISO 8601 ミリ秒固定精度
- dashboard/judge から参照可能な最小フィールドを返す
  - chart_id, chart_set_id, tool_id, chamber_id, recipe_id, parameter, step_no, feature_type
  - warning_lcl, warning_ucl, critical_lcl, critical_ucl, updated_at, version, is_active

Tests

- 契約テスト: 200
- 契約テスト: 主要 4xx
- 契約テスト: timestamp 形式
- 回帰確認: 既存 ingest write endpoint へ非影響

Docs

- docs/db-api-endpoints.md の status/tracking 更新
- 必要に応じて docs/db-api-minimum-contract.md の差分更新
