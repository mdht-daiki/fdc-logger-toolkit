Title: Child - GET /charts/active

Parent

- Parent - DB API Minimum Contract Draft (Phase 1)

Background

- active chart set と有効閾値の取得を dashboard/judge で共通化する。

Scope

- GET /charts/active の実装
- 任意クエリのフィルタ実装
  - tool_id, chamber_id, recipe_id
- active_chart_set_id と charts 配列の返却

Acceptance Criteria

- 200: ok=true, data=object を返す
- data に active_chart_set_id, activated_at, charts を含む
- charts の各要素に最小閾値フィールドを含む
  - chart_id, parameter, step_no, feature_type, warning_lcl, warning_ucl, critical_lcl, critical_ucl
- 400/422: クエリ不正時に契約どおりエラー
- activated_at が UTC ISO 8601 ミリ秒固定精度

Tests

- 契約テスト: 200
- 契約テスト: 主要 4xx
- 契約テスト: timestamp 形式
- 回帰確認: 既存 ingest write endpoint へ非影響

Docs

- docs/db-api-endpoints.md の status/tracking 更新
- 必要に応じて docs/db-api-minimum-contract.md の差分更新
