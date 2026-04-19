Title: Child - Judge MVP (Phase 1)

Parent

- #102 [Governance] 変更ガバナンス方針の最終確定（通常PR必須 + 緊急変更例外）
- 参照: #85 [Architecture] judge モジュールの設計前提確定
- 参照: #109 [API Contract] しきい値更新APIの競合制御・監査・必須テスト仕様を確定

Background

- 判定ロジックの最小実装（MVP）を先行し、dashboard read-only baseline へ結果を供給できる状態を作る。
- governance の編集承認フロー実装前でも、read path と運用監視の基盤を進める。

Scope

- judge MVP の実装
  - warning / critical 判定ロジック
  - 同一 run 内の評価対象データ取得（db_api read path 準拠）
  - 判定結果の保存（JudgementResults）
  - 通知送信（warning/critical）
  - critical 時の停止 API 呼び出しフック
- 監査可能性の担保
  - stop_api_called / stop_api_status の記録
  - 判定時刻・入力特徴量・参照 chart のトレース情報保持

Out of Scope

- suppression 高度化（重複通知最適化の拡張）
- 複雑な運転モード別ポリシー
- governance 承認状態を加味した編集反映制御

Acceptance Criteria

- warning/critical のしきい値判定が docs 方針どおり動作する
- warning/critical いずれも通知処理が実行される
- critical で停止 API 呼び出しが行われ、結果が記録される
- stop API 失敗時でも judge 全体が異常終了せず、結果に失敗状態を残せる
- db_api read endpoint の契約変更なしで judge が必要データを取得できる

Tests

- 単体テスト: 判定境界（LCL/UCL 付近、WARN/NG 遷移）
- 単体テスト: 通知分岐（OK/WARN/NG）
- 単体テスト: 停止 API 成功/失敗/タイムアウト
- 統合テスト: 入力 -> 判定 -> JudgementResults 記録 -> db_api /judge/results 参照
- 回帰確認: 既存 ingest/db_api の write/read フローに非影響

Docs

- docs/decision-log.md（判定/通知/停止 API 実装判断を追記）
- docs/dashboard-architecture-playbook.md（judge 結果表示契約に変更があれば更新）

Dependency Notes

- Must: #128 完了（judge 結果参照 endpoint が利用可能）
- Must: #98 の read path 方針準拠（dashboard/judge は db_api 経由）
- Should: #102 の governance endpoint 実装と独立に MVP を先行し、後で編集フローと統合
