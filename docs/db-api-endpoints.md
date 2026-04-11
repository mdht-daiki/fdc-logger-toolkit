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

| Method   | Path                                                | Area           | Status      | Consumer        | Tracking                | Notes                                        |
| -------- | --------------------------------------------------- | -------------- | ----------- | --------------- | ----------------------- | -------------------------------------------- |
| `POST`   | `/processes`                                        | ingest write   | implemented | ingest          | source: `db_api/app.py` | `ProcessInfo` を1件保存                      |
| `DELETE` | `/processes/{process_id}`                           | ingest write   | implemented | ingest/ops      | source: `db_api/app.py` | 推奨削除 API                                 |
| `DELETE` | `/processes`                                        | ingest write   | deprecated  | ingest/ops      | source: `db_api/app.py` | 旧互換 API。`/processes/{process_id}` へ移行 |
| `POST`   | `/step_windows/bulk`                                | ingest write   | implemented | ingest          | source: `db_api/app.py` | `StepWindow` を一括保存                      |
| `POST`   | `/parameters/bulk`                                  | ingest write   | implemented | ingest          | source: `db_api/app.py` | `Parameter` を一括保存                       |
| `POST`   | `/aggregate/write`                                  | ingest write   | implemented | ingest          | source: `db_api/app.py` | Process/StepWindow/Parameter を原子的に保存  |
| `GET`    | `/charts`                                           | dashboard read | planned     | dashboard/judge | Issue #98               | chart 定義一覧                               |
| `GET`    | `/charts/active`                                    | dashboard read | planned     | dashboard/judge | Issue #98               | active chart set と閾値                      |
| `GET`    | `/charts/history`                                   | dashboard read | planned     | dashboard/ops   | Issue #98               | chart 変更履歴                               |
| `GET`    | `/judge/results`                                    | dashboard read | planned     | dashboard       | Issue #98               | 判定結果一覧                                 |
| `GET`    | `/judge/results/{result_id}`                        | dashboard read | planned     | dashboard       | Issue #98               | 判定詳細（トレース情報含む）                 |
| `POST`   | `/governance/change-requests`                       | governance     | planned     | dashboard/ops   | Issue #102              | 通常変更の申請作成                           |
| `POST`   | `/governance/change-requests/{request_id}/approve`  | governance     | planned     | ops             | Issue #102              | 変更申請の承認                               |
| `POST`   | `/governance/change-requests/{request_id}/apply`    | governance     | planned     | ops             | Issue #102              | 承認済み申請の反映                           |
| `POST`   | `/governance/emergency-changes`                     | governance     | planned     | dashboard/ops   | Issue #102              | 緊急変更の即時反映                           |
| `POST`   | `/governance/emergency-changes/{request_id}/ratify` | governance     | planned     | ops             | Issue #102              | 緊急変更の事後追認                           |
| `GET`    | `/governance/change-requests`                       | governance     | planned     | ops/audit       | Issue #102              | 変更申請の検索                               |
| `GET`    | `/governance/audit-events`                          | governance     | planned     | ops/audit       | Issue #102              | 監査イベントの検索                           |
| `POST`   | `/governance/notifications/{event_id}/retry`        | governance     | planned     | ops             | Issue #102              | 通知失敗時の再送                             |

## Notes

- dashboard の read path は `db_api` 経由のみとする（Discussion #93）。
- dashboard 実装前提の read endpoint は Issue #98 で追跡している。
- 変更ガバナンス endpoint は Issue #102 の合意を前提に設計・実装する。

## Consumer Permission Scope

モジュール境界（dashboard -> api のみ、dashboard -> judge 禁止、judge -> dashboard 禁止）を維持するため、
各 consumer の許可範囲を以下で定義する。

| Consumer  | Allowed Scope                                                                                                         | Disallowed Scope                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| ingest    | `POST /processes`, `POST /step_windows/bulk`, `POST /parameters/bulk`, `POST /aggregate/write`, `DELETE /processes/*` | dashboard 向け read endpoint、governance endpoint                                                    |
| judge     | `GET /charts`, `GET /charts/active`, `GET /charts/history`, judge 結果の write endpoint（実装時）                     | dashboard 専用集計 read endpoint、governance endpoint                                                |
| dashboard | `GET /charts*`, `GET /judge/results*`, `POST/GET /governance/*`                                                       | ingest write endpoint（`/processes*`, `/step_windows/bulk`, `/parameters/bulk`, `/aggregate/write`） |
| ops/audit | `GET /governance/*`, `POST /governance/*`（approve/apply/ratify/retry）                                               | ingest の通常データ投入 endpoint                                                                     |

検証方針:

1. import 境界は `import-linter` で CI と pre-commit で機械検出する。
2. endpoint 権限は API スキーマ実装時にテスト（認可・認証）で検証する。
