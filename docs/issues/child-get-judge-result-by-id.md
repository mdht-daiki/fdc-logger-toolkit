Title: Child - GET /judge/results/{result_id}

Parent

- Parent - DB API Minimum Contract Draft (Phase 1)

Background

- dashboard のドリルダウン表示で、単一判定結果の詳細参照が必要。

Scope

- GET /judge/results/{result_id} の実装
- not found 時の 404 エラー envelope 実装

Acceptance Criteria

- 200: ok=true, data=object を返す
- data に最小フィールドを含む
  - result_id, chart_id, process_id, lot_id, wafer_id, tool_id, chamber_id
  - recipe_id, parameter, step_no, feature_type, feature_value
  - warning_lcl, warning_ucl, critical_lcl, critical_ucl
  - level, judged_at, process_start_ts, stop_api_called, stop_api_status
- 404: 契約どおり NOT_FOUND エラーを返す
- 422: path parameter 不正時に契約どおりエラー
- judged_at, process_start_ts が UTC ISO 8601 ミリ秒固定精度

Tests

- 契約テスト: 200
- 契約テスト: 404
- 契約テスト: 422
- 契約テスト: timestamp 形式
- 回帰確認: 既存 ingest write endpoint へ非影響

Docs

- docs/db-api-endpoints.md の status/tracking 更新
- 必要に応じて docs/db-api-minimum-contract.md の差分更新
