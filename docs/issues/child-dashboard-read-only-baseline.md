Title: Child - Dashboard Read-only Baseline (Phase 1)

Parent

- #102 [Governance] 変更ガバナンス方針の最終確定（通常PR必須 + 緊急変更例外）
- 参照: #98 feat(db_api): add read endpoints for dashboard support

Background

- Phase 1 の read endpoint（/charts, /charts/active, /charts/history, /judge/results, /judge/results/{result_id}）が実装済みとなったため、dashboard の read-only baseline を着手可能な状態。
- 編集系（governance approve/apply/emergency）は #102 配下の後続実装とし、本 Issue では参照表示に責務を限定する。

Scope

- dashboard の read-only baseline 画面実装
  - chart 一覧表示（/charts）
  - active chart set 表示（/charts/active）
  - chart 履歴表示（/charts/history）
  - judge 結果一覧/詳細ドリルダウン（/judge/results, /judge/results/{result_id}）
- 表示ルールの適用
  - 優先順位: NG > WARN > OK
  - color band: Center/Warning/Critical
- URL パラメータ経由の画面遷移最小対応（dashboard playbook 準拠）

Out of Scope

- しきい値編集・承認・適用操作
- emergency change 実行 UI
- 通知再送 UI

Acceptance Criteria

- read-only 画面から上記 5 endpoint の主要ユースケースを操作できる
- API エラー時に envelope を解釈してユーザー向けメッセージを表示できる
- judged_at / process_start_ts / changed_at など時刻表示が UTC ISO 8601 ミリ秒精度の入力を正しく扱う
- NG/WARN/OK の優先表示と color band が playbook 仕様に一致する
- dashboard から DB へ直接接続せず、db_api 経由のみで取得する

Tests

- 受け入れテスト: 主要表示フロー（一覧/詳細/履歴/ドリルダウン）
- 契約テスト: 必須フィールド欠落や 4xx/5xx 応答時の表示劣化なし
- 回帰確認: 既存 db_api read endpoint 契約への非影響

Docs

- docs/dashboard-architecture-playbook.md（必要に応じて表示仕様追記）
- docs/db-api-endpoints.md（consumer/tracking 注記が必要なら更新）

Dependency Notes

- Must: #128 完了（read endpoint 実装済み）
- Must: #102 governance 実装とは分離し、read-only 範囲で先行
- Follow-up: 編集系 UI は #102 配下 endpoint 実装後に別Issueで対応
