Title: Child - GET /judge/results

Parent

- Parent - DB API Minimum Contract Draft (Phase 1)

Background

- dashboard で判定結果一覧を表示するため、judge 結果の read endpoint を提供する。

Scope

- GET /judge/results の実装
- 任意クエリのフィルタ実装
  - chart_id, process_id, lot_id, recipe_id, level, from_ts, to_ts, limit, offset
- ページング実装（default/max の契約遵守）

Acceptance Criteria

- 200: ok=true, data=list を返す
- result レコードに最小フィールドを含む
  - result_id, chart_id, process_id, lot_id, wafer_id, recipe_id
  - step_no, feature_type, feature_value, level, judged_at, process_start_ts
- level は OK/WARN/NG のみ許可
- limit default=200, max=1000 を満たす
- 400/422: クエリ不正時に契約どおりエラー
- judged_at, process_start_ts が UTC ISO 8601 ミリ秒固定精度

Tests

- 契約テスト: 200
- 契約テスト: 主要 4xx
- 契約テスト: timestamp 形式
- 契約テスト: level フィルタ妥当性
- 回帰確認: 既存 ingest write endpoint へ非影響

Docs

- docs/db-api-endpoints.md の status/tracking 更新
- 必要に応じて docs/db-api-minimum-contract.md の差分更新
