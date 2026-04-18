# DB API Minimum Contract Draft (Phase 1)

## Goal

dashboard read-only baseline と judge 最小実装に必要な API 契約を、実装可能な最小範囲で固定する。

- 対象フェーズ: Phase 1（read + judge result read）
- 対象モジュール: db_api / dashboard / judge
- 非対象: governance write endpoint、しきい値編集フローの完全実装

## Contract Scope

本ドラフトで固定する endpoint:

1. `GET /charts`
2. `GET /charts/active`
3. `GET /charts/history`
4. `GET /judge/results`
5. `GET /judge/results/{result_id}`

## Common Rules

### Response Envelope

成功時は以下の形式を基本とする。

```json
{
  "ok": true,
  "data": []
}
```

詳細取得 (`/judge/results/{result_id}`) は `data` を object とする。

### Error Envelope

失敗時は以下の形式を基本とする。

```json
{
  "ok": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "judge result not found",
    "details": {}
  }
}
```

### Timestamp Format

時刻文字列は UTC / ISO 8601 / ミリ秒固定精度を必須とする。

- 形式: `YYYY-MM-DDTHH:mm:ss.SSSZ`
- 例: `2026-04-14T00:00:00.000Z`
- ミリ秒化ルール: マイクロ秒以下は切り捨て（round half-up ではない）

### Status Code Policy

- 200: 正常
- 400: クエリ不正
- 404: 対象なし（ID 指定）
- 422: バリデーション不正
- 500: 予期しない内部エラー

## Endpoint Contracts

### 1) GET /charts

用途:

- chart 定義の一覧取得（dashboard 表示、judge 参照準備）

クエリ（任意）:

- `tool_id`
- `chamber_id`
- `recipe_id`
- `parameter`
- `step_no`
- `feature_type`
- `active_only` (bool)

クエリ検証ルール:

- `step_no` は `>= 0`
- 文字列フィルタ（`tool_id`, `chamber_id`, `recipe_id`, `parameter`, `feature_type`）は `1..128` 文字
- 文字列フィルタは `^[A-Za-z0-9_./:-]+$` のみ許可（空白や制御文字は不可）

フィールド意味（互換注記）:

- `lcl` / `ucl` は後方互換のための互換フィールドであり、現在は `critical_lcl` / `critical_ucl` と同値を返す
- 新規実装では `warning_*` と `critical_*` の明示フィールド参照を優先する

成功レスポンス例:

```json
{
  "ok": true,
  "data": [
    {
      "chart_id": "CHART_001",
      "chart_set_id": "SET_20260414",
      "tool_id": "TOOL_A",
      "chamber_id": "CH1",
      "recipe_id": "RECIPE_A",
      "parameter": "dc_bias",
      "step_no": 1,
      "feature_type": "mean",
      "lcl": 1.2,
      "ucl": 2.8,
      "warning_lcl": 1.4,
      "warning_ucl": 2.6,
      "critical_lcl": 1.2,
      "critical_ucl": 2.8,
      "updated_at": "2026-04-14T00:00:00.000Z",
      "version": 3,
      "is_active": true
    }
  ]
}
```

### 2) GET /charts/active

用途:

- active chart set と有効閾値の取得

クエリ（任意）:

- `tool_id`
- `chamber_id`
- `recipe_id`

成功レスポンス例:

```json
{
  "ok": true,
  "data": {
    "active_chart_set_id": "SET_20260414",
    "activated_at": "2026-04-14T00:00:00.000Z",
    "charts": [
      {
        "chart_id": "CHART_001",
        "parameter": "dc_bias",
        "step_no": 1,
        "feature_type": "mean",
        "warning_lcl": 1.4,
        "warning_ucl": 2.6,
        "critical_lcl": 1.2,
        "critical_ucl": 2.8
      }
    ]
  }
}
```

### 3) GET /charts/history

用途:

- chart 閾値変更履歴の参照（dashboard/ops）

クエリ（任意）:

- `chart_id`
- `chart_set_id`
- `change_source`
- `from_ts`
- `to_ts`
- `limit` (default: 100, max: 500)
- `offset` (default: 0)

クエリ検証ルール:

- `from_ts` / `to_ts` は timezone-aware な ISO 8601 datetime のみ受け付ける
- `chart_id` フィルタは現在 `ChartsV2` に存在する chart のみ解決対象とする
- 削除済み chart の履歴は `chart_id` では取得不可で、`chart_set_id` など別条件で参照する

成功レスポンス例:

```json
{
  "ok": true,
  "data": [
    {
      "history_id": "HIS_001",
      "chart_id": "CHART_001",
      "chart_set_id": "SET_20260414",
      "change_source": "normal_pr",
      "change_reason": "spc drift adjustment",
      "before": {
        "warning_ucl": 2.5
      },
      "after": {
        "warning_ucl": 2.6
      },
      "changed_by": "ops_reviewer",
      "changed_at": "2026-04-14T00:00:00.000Z"
    }
  ]
}
```

### 4) GET /judge/results

用途:

- 判定結果一覧の参照（dashboard）

クエリ（任意）:

- `chart_id`
- `process_id`
- `lot_id`
- `recipe_id`
- `level` (`OK|WARN|NG`)
- `from_ts`
- `to_ts`
- `limit` (default: 200, max: 1000)
- `offset` (default: 0)

成功レスポンス例:

```json
{
  "ok": true,
  "data": [
    {
      "result_id": "JR_001",
      "chart_id": "CHART_001",
      "process_id": "P_20260414_0001",
      "lot_id": "LOT_001",
      "wafer_id": "W01",
      "recipe_id": "RECIPE_A",
      "step_no": 1,
      "feature_type": "mean",
      "feature_value": 2.72,
      "level": "WARN",
      "judged_at": "2026-04-14T00:00:00.000Z",
      "process_start_ts": "2026-04-14T00:00:00.000Z"
    }
  ]
}
```

### 5) GET /judge/results/{result_id}

用途:

- 判定結果の詳細参照（ドリルダウン）

パス:

- `result_id`（必須）

入力バリデーション:

- 形式が `JR_[0-9]+` に一致しない場合は `422 VALIDATION_ERROR`
- 形式は一致するが数値部が `1` 未満（例: `JR_0`）または int64 範囲外の場合は `400 Invalid result_id`
- 先頭ゼロ付き形式（例: `JR_01`）は正規化されて `JR_1` として扱われる

成功レスポンス例:

```json
{
  "ok": true,
  "data": {
    "result_id": "JR_001",
    "chart_id": "CHART_001",
    "process_id": "P_20260414_0001",
    "lot_id": "LOT_001",
    "wafer_id": "W01",
    "tool_id": "TOOL_A",
    "chamber_id": "CH1",
    "recipe_id": "RECIPE_A",
    "parameter": "dc_bias",
    "step_no": 1,
    "feature_type": "mean",
    "feature_value": 2.72,
    "warning_lcl": 1.4,
    "warning_ucl": 2.6,
    "critical_lcl": 1.2,
    "critical_ucl": 2.8,
    "level": "WARN",
    "judged_at": "2026-04-14T00:00:00.000Z",
    "process_start_ts": "2026-04-14T00:00:00.000Z",
    "stop_api_called": false,
    "stop_api_status": "NOT_CALLED"
  }
}
```

404 エラー例:

```json
{
  "ok": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "judge result not found",
    "details": {
      "result_id": "JR_UNKNOWN"
    }
  }
}
```

## Minimum Test Matrix

実装 PR ごとに最低限以下を追加する。

1. 契約テスト: 各 endpoint の 200 と主要 4xx（400/404/422 の該当ケース）
2. 契約テスト: timestamp 形式（UTC ISO 8601 ミリ秒固定）
3. 統合テスト: しきい値変更後の判定結果参照が反映される連鎖
4. 回帰判定: 既存 ingest write endpoint が非影響であること

## Implementation Order (Recommended)

1. `GET /charts` と `GET /charts/active` を先行実装
2. `GET /judge/results` と `GET /judge/results/{result_id}` を実装
3. `GET /charts/history` を実装
4. dashboard read-only baseline を API 接続で有効化
5. judge の最小保存ループ（結果 write）を別 PR で接続

## Out of Scope for This Draft

- 閾値更新の write 契約（`expected_version` / emergency write）
- governance 承認ワークフロー endpoint の完全契約
- 認証/認可方式の詳細（別ドキュメントで固定）
