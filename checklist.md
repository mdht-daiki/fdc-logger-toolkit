## 確定チェックリスト（#140 実装前）

### 📋 テーブル設計・スキーマ

- [x] **change_requests** 列定義（request_id, chart_id, status, proposed_by, proposed_at, ...）
- [x] **approvals** 列定義（approval_id, request_id, approved_by, approved_at, comment）
- [x] **apply_results** 列定義（apply_id, request_id, applied_at, success, error_message）
- [x] **emergency_changes** 列定義（ec_id, chart_id, changed_by, changed_at, reason, is_ratified）
- [x] **ratifications** 列定義（ratification_id, ec_id, ratified_by, ratified_at, comment）
- [x] **audit_events** 列定義（event_id, event_type, target_id, changed_by, changed_at, before_json, after_json）
- [x] **notification_outbox** 列定義（outbox_id, event_id, status, retry_count, next_retry_at, last_error）
- [x] 各テーブルの主キー型を確定（UUID / ULID / INTEGER autoincrement）
- [x] 外部キー制約の有無と `ON DELETE` 挙動を確定

### 🔄 状態遷移

- [x] change_request の status 一覧（例: `draft` → `approved` → `applied` / `rejected`）
- [x] 二重 approve の扱い（409 か冪等成功か）
- [x] 未承認 apply の扱い（400 か 409 か）
- [x] ratify の重複追認の扱い（409 か冪等成功か）

### 🔍 監査イベント契約

- [x] `event_type` 一覧を固定（`change_requested`, `approved`, `applied`, `emergency_changed`, `ratified`, `notified`, `retry_succeeded`, `retry_failed` など）
- [x] `before_json` / `after_json` の差分形式（changed fields のみ or フル snapshot）
- [x] audit writer の配置責務（共通 service か repository 内か）

### ⏱️ Timestamp 正規化

- [x] 正規化処理の責務先を確定（repository 共通 util か service 層か）
- [x] マイクロ秒以下の切り捨てを共通関数で一元化することを明文化

### 🔧 Migration / 初期化

- [x] 冪等初期化（`CREATE TABLE IF NOT EXISTS` or migration ツール）の方針確定
- [x] 既存 DB への適用順（既存テーブルへの影響なし宣言）
- [x] forward-only 前提の明文化

### 📬 Notification Outbox Retry モデル

- [x] 初期 `status` 値（例: `pending` / `failed` / `sent`）
- [x] `retry_count` 上限値
- [x] `next_retry_at` の算出方式（固定間隔 or 指数バックオフ）
- [x] 最終失敗時の監査イベント連携（`retry_failed` として audit_events に残すか）

### 🗂️ Repository インターフェース

- [x] 各テーブルの最小 CRUD スコープ（create/read のみか update/delete も含むか）
- [x] トランザクション境界（apply = ChartsHistory書き込み + audit_event が 1 atomic か）

### ✅ テスト受け入れ条件の具体化

- [ ] schema 初期化の冪等性テスト（2回実行してもエラーなし）
- [ ] repository 正常 CRUD テスト（最小1件ずつ）
- [ ] repository 異常系テスト（存在しない ID、制約違反）
- [ ] audit event の自動必須項目テスト（changed_by / changed_at / before / after が全て記録されること）
- [x] timestamp 正規化テスト（ミリ秒固定、マイクロ秒切り捨て）

### 📄 ドキュメント更新（同一 PR で必須）

- [x] db-api-endpoints.md の governance endpoint tracking 更新
- [x] decision-log.md にスキーマ/状態遷移確定を記録
- [x] 必要なら architecture.md の db_api 責務記述を補完

---

使い方: チェックが全部入った状態を Issue #140 の DoD に追加し、未確定の行は実装 PR を開く前に Discussion か Issue コメントで方針を決める、という流れが最もシンプルです。
