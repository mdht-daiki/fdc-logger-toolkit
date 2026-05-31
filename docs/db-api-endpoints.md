# DB API Endpoint Catalog

## Goal

`db_api` のエンドポイントを一覧化し、実装済み/計画中/廃止予定を同じ基準で追跡する。
本ドキュメントは API 契約のインデックスとして扱う。

## Update Policy

1. `src/portfolio_fdc/db_api/app.py` のエンドポイント追加・変更・削除を行う PR では、本ドキュメントを同一 PR で更新する。
2. `status` は `implemented` / `planned` / `deprecated` のいずれかを使う。
3. `planned` には必ず追跡先（Issue/Discussion/PR）を記載する。
4. `deprecated` には移行先と廃止予定日（または条件）を記載する。

## Endpoint Table

| Method   | Path                                                | Area           | Status      | Consumer            | Tracking                | Notes                                        |
| -------- | --------------------------------------------------- | -------------- | ----------- | ------------------- | ----------------------- | -------------------------------------------- |
| `POST`   | `/processes`                                        | ingest write   | implemented | ingest              | source: `db_api/app.py` | `ProcessInfo` を1件保存                      |
| `DELETE` | `/processes/{process_id}`                           | ingest write   | implemented | ingest/ops          | source: `db_api/app.py` | 推奨削除 API                                 |
| `DELETE` | `/processes`                                        | ingest write   | deprecated  | ingest/ops          | source: `db_api/app.py` | 旧互換 API。`/processes/{process_id}` へ移行 |
| `POST`   | `/step_windows/bulk`                                | ingest write   | implemented | ingest              | source: `db_api/app.py` | `StepWindow` を一括保存                      |
| `POST`   | `/parameters/bulk`                                  | ingest write   | implemented | ingest              | source: `db_api/app.py` | `Parameter` を一括保存                       |
| `POST`   | `/aggregate/write`                                  | ingest write   | implemented | ingest              | source: `db_api/app.py` | Process/StepWindow/Parameter を原子的に保存  |
| `GET`    | `/charts`                                           | dashboard read | implemented | dashboard/judge     | source: `db_api/app.py` | chart 定義一覧                               |
| `GET`    | `/charts/active`                                    | dashboard read | implemented | dashboard/judge     | source: `db_api/app.py` | active chart set と閾値                      |
| `GET`    | `/charts/history`                                   | dashboard read | implemented | dashboard/ops       | source: `db_api/app.py` | chart 変更履歴                               |
| `GET`    | `/judge/results`                                    | dashboard read | implemented | dashboard           | Issue #132              | 判定結果一覧                                 |
| `GET`    | `/judge/results/{result_id}`                        | dashboard read | implemented | dashboard           | Issue #133              | 判定詳細（トレース情報含む）                 |
| `POST`   | `/governance/change-requests`                       | governance     | implemented | dashboard/ops       | source: `db_api/app.py` | 通常変更の申請作成                           |
| `POST`   | `/governance/change-requests/{request_id}/approve`  | governance     | implemented | dashboard/ops       | source: `db_api/app.py` | 変更申請の承認                               |
| `POST`   | `/governance/change-requests/{request_id}/apply`    | governance     | implemented | dashboard/ops       | source: `db_api/app.py` | 承認済み申請の反映                           |
| `POST`   | `/governance/emergency-changes`                     | governance     | implemented | dashboard/ops       | source: `db_api/app.py` | 緊急変更の即時反映                           |
| `POST`   | `/governance/emergency-changes/{request_id}/ratify` | governance     | implemented | ops                 | source: `db_api/app.py` | 緊急変更の事後追認                           |
| `GET`    | `/governance/change-requests`                       | governance     | implemented | dashboard/ops/audit | source: `db_api/app.py` | 変更申請の検索                               |
| `GET`    | `/governance/audit-events`                          | governance     | implemented | ops/audit           | source: `db_api/app.py` | 監査イベントの検索                           |
| `GET`    | `/governance/notifications/failed`                  | governance     | implemented | dashboard/ops/audit | source: `db_api/app.py` | 通知失敗レコードの検索                       |
| `POST`   | `/governance/notifications/{event_id}/retry`        | governance     | implemented | dashboard/ops       | source: `db_api/app.py` | 通知失敗時の再送                             |

## Notes

- dashboard の read path は `db_api` 経由のみとする（Discussion #93）。
- dashboard 実装前提の read endpoint は Issue #98 で追跡している。
- 変更ガバナンス endpoint は Issue #102 の合意を前提に設計・実装する。
- Phase 1 の最小 read 契約は `docs/db-api-minimum-contract.md` を実装基準とする。
- `GET /charts` の文字列クエリ（`tool_id` 等）は `^[A-Za-z0-9_./:-]+$`（1..128 文字）を許可形式とする。
- API の timestamp 正規化は UTC ISO 8601 ミリ秒固定で、マイクロ秒以下は切り捨てとする。
- `GET /charts/history` の `from_ts` / `to_ts` は timezone-aware datetime のみ受け付ける。
- `GET /charts/history` の `chart_id` フィルタは現在 `ChartsV2` に存在する chart にのみ有効で、削除済み chart 履歴の `chart_id` 解決は行わない。
- `GET /charts/history` は `is_emergency`（bool）フィルタを受け付ける。`true` は緊急変更履歴のみ、`false` は通常変更履歴のみを返す。省略時は両方を返す。

## Read-only Endpoint Access Pattern

- `GET /charts*` と `GET /governance/change-requests` / `GET /governance/audit-events` は read-only のため `_connect_readonly(MAIN_DB)` で直接接続し、`DBTaskRunner` 経由は不要
- `DBTaskRunner` は write タスク直列化・排他制御専用で、read では並行接続をサポート

## Consumer Permission Scope

モジュール境界（dashboard -> api のみ、dashboard -> judge 禁止、judge -> dashboard 禁止）を維持するため、
各 consumer の許可範囲を以下で定義する。

| Consumer  | Allowed Scope                                                                                                                                                                                                                                                                                                                                                        | Disallowed Scope                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ingest    | `POST /processes`, `POST /step_windows/bulk`, `POST /parameters/bulk`, `POST /aggregate/write`, `DELETE /processes/*`                                                                                                                                                                                                                                                | dashboard 向け read endpoint、governance endpoint                                                       |
| judge     | `GET /charts`, `GET /charts/active`, judge 結果の write endpoint（実装時）                                                                                                                                                                                                                                                                                           | `GET /charts/history`（dashboard/ops 専用）、dashboard 専用集計 read endpoint、governance endpoint      |
| dashboard | `GET /charts*`, `GET /charts/history`, `GET /judge/results*`, `GET /governance/change-requests`, `POST /governance/change-requests`, `POST /governance/change-requests/{id}/approve`, `POST /governance/change-requests/{id}/apply`, `POST /governance/emergency-changes`, `GET /governance/notifications/failed`, `POST /governance/notifications/{event_id}/retry` | `GET /governance/audit-events`, `POST /governance/emergency-changes/{id}/ratify`, ingest write endpoint |
| ops/audit | `GET /governance/change-requests`, `GET /governance/audit-events`, `GET /governance/notifications/failed`, `POST /governance/change-requests`, `POST /governance/change-requests/{id}/approve`, `POST /governance/change-requests/{id}/apply`, `POST /governance/emergency-changes/{id}/ratify`                                                                      | ingest の通常データ投入 endpoint                                                                        |

検証方針:

1. import 境界は `import-linter` で pre-commit と CI の両方で機械検出する（Issue #108 対応）。
2. endpoint 権限は API スキーマ実装時にテスト（認可・認証）で検証する。

## Dashboard Tab Access Baseline（Issue #222）

`dashboard` のタブ表示制御は endpoint の Consumer 範囲に従う。

| Dashboard Tab        | Main Endpoint Group                                                                       | Access Baseline                                 |
| -------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `charts`             | `GET /charts`                                                                             | Engineer/Ops: readonly, Audit: hidden           |
| `active`             | `GET /charts/active`                                                                      | Engineer/Ops: readonly, Audit: hidden           |
| `history`            | `GET /charts/history`                                                                     | Engineer/Ops: readonly, Audit: hidden           |
| `judge`              | `GET /judge/results*`                                                                     | Engineer/Ops: readonly, Audit: hidden           |
| `change_requests`     | `GET/POST /governance/change-requests*`                                                   | Engineer/Ops: execute, Audit: hidden            |
| `emergency:apply`     | `POST /governance/emergency-changes`                                                      | Engineer: execute, Ops: readonly, Audit: hidden |
| `emergency:ratify`    | `POST /governance/emergency-changes/{request_id}/ratify`                                  | Ops: execute, Audit: readonly, Engineer: hidden |
| `notification_retry` | `GET /governance/notifications/failed`, `POST /governance/notifications/{event_id}/retry` | Ops: execute, Audit: readonly, Engineer: hidden |

補足:

1. `readonly` は閲覧のみ許可し、更新系操作は UI で無効化する。
2. `execute` の最終可否は API 側認可で判定し、権限外は 403 を返す。

## Must-Test Cases for API Contract

しきい値更新 API を実装する際、以下のケースを最小必須テストとする.

### Normal Update

1. `expected_version` 一致で更新成功する
2. 成功時に `version` が 1 増加する
3. `ChartsHistory` に 1 件追加される
4. 成功レスポンスに含まれる `current.updated_at` が UTC の ISO 8601 かつミリ秒固定精度（`YYYY-MM-DDTHH:mm:ss.SSSZ`）であることを検証する

### Conflict Handling

1. stale `expected_version` で更新失敗する
2. `409 Conflict` を返す
3. 競合時は `ChartsHistory` を追加しない
4. `409` レスポンスに最新状態（`current.version`, `current.updated_at` など）を含める
5. `409` レスポンスの `current.updated_at` が UTC の ISO 8601 かつミリ秒固定精度（`YYYY-MM-DDTHH:mm:ss.SSSZ`）であることを検証する

### Idempotent Retry

1. 同一 `Idempotency-Key` で再送した場合、二重更新しない
2. `ChartsHistory` が二重記録されない

### Emergency Update Flow

#### POST /governance/emergency-changes (即時反映)

1. 正常系: `reason` は任意（後追い可）とし、即座に ChartsV2 を更新し version を 1 増加させる
2. 成功時に ChartsHistory に 1 件追加される（`change_source='emergency_manual'`, `is_emergency=true`）
3. AuditEvents に `emergency_changed` タイプのイベントが記録される
4. NotificationOutbox に `pending` 状態で登録され、通知ポーラー対象になる
5. 空 payload（`{}`）やタイポのみの payload は 422 を返す
6. 対象 chart が存在しない場合は 404 を返す
7. しきい値整合性チェック失敗は 422 を返す
8. 変更なし更新（同一値送信）は no-op として成功応答を返し、ChartsHistory を追加しない

#### POST /governance/emergency-changes/{request_id}/ratify (事後追認)

1. 正常系: 緊急変更を追認し、`related_issue_or_pr` を GovernanceEmergencyChanges に埋め込む
2. AuditEvents に `emergency_ratified` タイプのイベントが記録される（`actor` は `ratified_by` 入力値を使用）
3. 同一 request_id での重複追認は 409 で拒否する

### Boundary and Edge Cases

1. 対象 chart が存在しない場合のエラー
2. archived/disabled chart の更新挙動
3. 変更なし更新（no-op）の扱い
4. system job 更新との競合

#### Threshold consistency (Issue #109)

以下は API 契約としての必須テスト（mandatory contract tests）とする。

1. `UCL < LCL` を拒否し、validation error を返す
2. `UCL == LCL`（equal 不許可ルール時）を拒否し、validation error を返す
3. 負のしきい値（仕様で不許可の項目）を拒否し、validation error を返す
4. 数値以外（文字列/NaN/inf 等）を拒否し、validation error を返す
5. 許容範囲外（min/max を超過）のしきい値を拒否し、validation error を返す
6. 変更なし更新（同一値送信）は no-op として扱う（履歴を増やさないことを確認）

期待挙動:

- validation error の場合は 4xx（入力不正）として返し、更新を実行しない
- no-op の場合は成功応答を返しつつ、`ChartsHistory` を追加しない

注記:

- 競合検出と冪等性は別責務として検証する。
- 競合検出は `expected_version` 比較、冪等性は `Idempotency-Key` で担保する。
