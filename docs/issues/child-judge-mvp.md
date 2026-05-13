Title: Child - Judge MVP (Phase 1)

Parent

- #102 [Governance] 変更ガバナンス方針の最終確定（通常PR必須 + 緊急変更例外）
- 参照: #85 [Architecture] judge モジュールの設計前提確定
- 参照: #109 [API Contract] しきい値更新APIの競合制御・監査・必須テスト仕様を確定

Background

- 判定ロジックの最小実装（MVP）を先行し、dashboard read-only baseline へ結果を供給できる状態を作る。
- governance の編集承認フロー実装前でも、read path と運用監視の基盤を進める。

Scope

- judge MVP の最小実装を checkpoint として固定する
  - warning / critical 判定ロジック
  - 同一 run 内の評価対象データ取得（db_api read path 準拠）
  - 判定結果の保存（JudgementResults）
  - 通知送信フック（warning/critical、実送信アダプタは Phase 2）
  - critical 時の停止 API 呼び出しフック（MES POST アダプタは Phase 2）
  - Windows Task Scheduler から呼べる CLI / task エントリポイント
- 監査可能性の担保
  - stop_api_called / stop_api_status の記録
  - 判定時刻・入力特徴量・参照 chart のトレース情報保持

Out of Scope

- Windows Task Scheduler への本番ジョブ登録（30 分周期設定・運用 runbook 確定）
- メール通知の実送信実装（SMTP / API / 再送戦略 / 宛先決定）
- MES API への実 POST 実装（認証 / timeout / retry / idempotency）
- suppression 高度化（重複通知最適化の拡張）
- 複雑な運転モード別ポリシー
- governance 承認状態を加味した編集反映制御

Acceptance Criteria

- warning/critical のしきい値判定が docs 方針どおり動作する
- warning/critical いずれも通知フックが実行される
- critical で停止 API フックが呼ばれ、結果が記録される
- stop API 失敗時でも judge 全体が異常終了せず、結果に失敗状態を残せる
- db_api read endpoint の契約変更なしで judge が必要データを取得できる
- Task Scheduler から呼ぶための実行入口が固定される

Tests

- 単体テスト: 判定境界（LCL/UCL 付近、WARN/NG 遷移）
- 単体テスト: 通知分岐（OK/WARN/NG）
- 単体テスト: 停止 API 成功/失敗/タイムアウト
- 統合テスト: 入力 -> 判定 -> JudgementResults 記録 -> db_api /judge/results 参照
- 回帰確認: 既存 ingest/db_api の write/read フローに非影響

Roadmap

1. Phase 1: 現在の checkpoint

- JudgeEngine.run_once と CLI を固定
- 通知 / 停止 API は hook 注入で置換可能に留める
- `tasks.ps1 judge-run-once` を scheduler の呼び出し口にする

2. Phase 2: 運用接続

- Windows Task Scheduler に 30 分周期ジョブを登録する
- warning / critical 時のメール通知アダプタを実装する
- critical 時に MES API へ POST するアダプタを実装する

3. Phase 3: 運用強化

- 重複通知抑制、再送、dead-letter 相当の失敗管理
- 停止 API の認証・idempotency・運転モード別ポリシー
- dashboard での状態可視化と runbook 整備

Docs

- docs/decision-log.md（判定/通知/停止 API 実装判断を追記）
- docs/dashboard-architecture-playbook.md（judge 結果表示契約に変更があれば更新）

Dependency Notes

- Must: #128 完了（judge 結果参照 endpoint が利用可能）
- Must: #98 の read path 方針準拠（dashboard/judge は db_api 経由）
- Should: #102 の governance endpoint 実装と独立に MVP を先行し、後で編集フローと統合
