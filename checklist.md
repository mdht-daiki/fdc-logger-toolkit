# #141 Definition of Done

---

## Portfolio 公開前チェックリスト（PR用）

このセクションは、公開直前の PR でそのままコピペして使うためのテンプレートです。

### 1) 公開情報

- [ ] リポジトリ名・説明文がポートフォリオ向けになっている
- [ ] README 冒頭に 30 秒サマリがある
- [ ] README に最短デモ手順（5 分以内）がある
- [ ] README にスクリーンショットがあり、表示崩れがない
- [ ] ライセンス（`LICENSE`）が存在し、`pyproject.toml` と整合している

### 2) 実行性・再現性

- [ ] 新規環境で `./tasks.ps1 install` または `make install-dev` が通る
- [ ] `db_api` が起動できる（`./tasks.ps1 demo-db-api`）
- [ ] `demo-data` が成功し、DB にデータ投入できる
- [ ] `dashboard` が起動し、`http://localhost:8050` で画面確認できる
- [ ] 主要ページ遷移とフォーム送信（change request/emergency/retry）を確認した

### 3) 品質ゲート

- [ ] `ruff check .` が通る
- [ ] `mypy src` が通る
- [ ] `pytest` または最低 `tests/integration/test_smoke.py` が通る
- [ ] CI がグリーン（必須ジョブ全通過）

### 4) アーキテクチャ/ガバナンス整合

- [ ] モジュール境界に反する依存を追加していない（import-linter で確認）
- [ ] しきい値/ガバナンス変更がある場合、監査可能性（source/reason/history）を担保
- [ ] モジュール境界を変更した場合、`docs/architecture.md` を更新
- [ ] ガバナンス運用変更がある場合、`docs/decision-log.md` を更新

### 5) PR 記載事項

- [ ] 目的（なぜこの変更が必要か）を 3 行以内で記載
- [ ] 変更範囲（ingest/dashboard/judge/api/docs/tests）を明記
- [ ] 検証コマンドと結果を記載
- [ ] リスクとロールバック手順を記載
- [ ] 画面変更がある場合、Before/After スクショを添付

### PR 本文テンプレート（公開前チェック用）

```md
## Summary
-

## Scope
- ingest:
- dashboard:
- judge:
- api:
- docs/tests:

## Validation
- [ ] ruff check .
- [ ] mypy src
- [ ] pytest
- [ ] demo-data
- [ ] demo-dashboard

## Risks / Rollback
- Risk:
- Rollback:

## Screenshots
- Before:
- After:
```

## POST/GET /governance/change-requests 実装

### 🔧 設計確定（実装前に合意）

- [x] POST リクエストボディの必須/任意フィールドを確定（`chart_id`, `proposed_by`, `change_payload`, `expected_version`, `idempotency_key`）
- [x] `change_payload` のバリデーション範囲を確定（A: JSON パース可能性のみ）
- [x] `idempotency_key` 重複時の HTTP ステータスを確定（409 + B: error envelope）
- [x] POST 時に `chart_id` の存在確認を行うかどうかを確定（B: POST 時は確認しない）
- [x] POST レスポンス形式を確定（A: ok/data 形式で固定、data は request_id と status を含む）
- [x] 初期 `status` 値を確定（`"pending"`）
- [x] GET フィルタ仕様を確定（`status`, `chart_id`, `from_ts` / `to_ts`、期間対象は `proposed_at` 固定、`proposed_by` は Phase 1 で除外）
- [x] GET の `limit` / `offset` デフォルト値と上限を確定（`limit`: default 100, max 500 / `offset`: default 0, 上限なし）
- [x] GET の DB 接続方式を確定（read-only 直接接続、`DBTaskRunner` 経由なし）
- [x] app 実装での依存注入パターンを確定（POST は `RunnerDep` / GET は read-only 直接接続）
- [x] GET テスト観点を確定（0件時、`status`/`chart_id`/`from_ts`/`to_ts` フィルタ、`limit`/`offset` 境界）
- [x] POST 時の監査イベント `event_type` 値を確定（`"change_requested"`）

### 🗂️ Repository 層

- [x] `GovernanceChangeRequestRepository.list()` メソッドを追加（status / chart_id / from_ts / to_ts / limit / offset でフィルタ）
- [x] `list()` で 0件時に空リストを返す（例外なし）

### 📐 Pydantic スキーマ（`schemas.py`）

- [x] `ChangeRequestIn` — POST リクエストボディモデル追加
- [x] `ChangeRequestsQuery` — GET クエリパラメータモデル追加（フィルタ + limit/offset）
- [x] timestamp フィールドは `datetime` 型で受け取り、UTC ISO 8601 ミリ秒に正規化

### 🌐 エンドポイント実装（`app.py`）

- [x] `POST /governance/change-requests` 実装
  - [x] 正常時: `request_id` と初期 `status` を `{ok: true, data: {...}}` で返す
  - [x] `idempotency_key` 重複時: 409 を返す
  - [x] バリデーション不正時: 422 を返す（Pydantic / FastAPI 既存ハンドラで処理）
  - [x] 申請作成直後に `AuditEventWriter` で監査イベントを記録する
  - [x] write は `DBTaskRunner` 経由で実行する
- [x] `GET /governance/change-requests` 実装
  - [x] status / chart_id / from_ts / to_ts / limit / offset フィルタを受け付ける
  - [x] 0件時は `{ok: true, data: []}` を返す
  - [x] read-only のため `DBTaskRunner` 経由なしで直接接続する

### ✅ テスト（`tests/db_api/` に追加）

#### POST テスト

- [x] POST 正常系: 200, レスポンスに `request_id`（int）と `status: "pending"` が含まれる
- [x] POST envelope 契約: `ok: true`, `data` キーが存在する
- [x] POST 422: 必須フィールド欠落でバリデーションエラー
- [x] POST 422: `chart_id` が整数でない場合
- [x] POST 409: 同一 `idempotency_key` で2回 POST すると 409 が返る
- [x] POST 監査イベント: POST 成功後に `GovernanceAuditEvents` に 1件追加される
- [x] POST 監査イベント: `event_type` が `"change_requested"` である

#### GET テスト

- [x] GET 正常系: 0件のとき `{ok: true, data: []}` を返す
- [x] GET envelope 契約: `ok: true`, `data` がリスト型である
- [x] GET フィルタ `status`: 該当するレコードのみ返る
- [x] GET フィルタ `chart_id`: 該当するレコードのみ返る
- [x] GET フィルタ `from_ts` / `to_ts`: 期間外レコードが除外される
- [x] GET `limit` 境界: `limit=1` のとき最大1件のみ返る
- [x] GET `offset` 境界: `offset` で先頭 n 件をスキップできる

### 📄 ドキュメント更新（同一 PR で必須）

- [x] `docs/db-api-endpoints.md`: `POST /governance/change-requests` と `GET /governance/change-requests` のステータスを `planned` → `implemented` に変更
- [x] `docs/chart-governance-playbook.md`: 通常変更フロー（POST/GET の動作仕様）を追記
