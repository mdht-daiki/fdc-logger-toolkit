# アーキテクチャ概要

## 目的

Logger Toolkit は装置データを処理し、監視・可視化・アラート判定・DB アクセス制御を支える仕組みを提供する。

## モジュール一覧

パスの対応:

- ingest: `src/portfolio_fdc/main`, `src/portfolio_fdc/core`
- dashboard: `src/portfolio_fdc/dashboard`
- judge: `src/portfolio_fdc/judge`
- api: `src/portfolio_fdc/db_api`
- docs: `docs`
- tests: `tests`

### ingest

責務:

- 生データの取得
- 論理単位への分割
- 特徴量の計算
- 特徴量データの保存

担当しないこと:

- UI の描画
- アラート判定の直接実行

### dashboard

責務:

- SPC チャートの可視化
- 生データへのドリルダウン表示
- しきい値の参照と編集

担当しないこと:

- 生データ取込の所有
- アラート最終発報

### judge

責務:

- チャート条件の評価
- しきい値逸脱の検出
- アラートアクションのトリガー

担当しないこと:

- チャート描画の所有
- 生装置データの直接パース（明示的に設計した場合を除く）

### api

責務:

- DB アクセスの仲介
- SQLite の並行処理・ロック制御

## 依存方向の規約

許可:

- ingest -> api
- judge -> api
- dashboard -> api
- main -> core

禁止:

- dashboard -> judge への直接依存
- judge -> dashboard への依存
- dashboard/judge 間のビジネスロジックの漏れ
- core ドメインロジック内への web/API フレームワーク依存

理由:

- ランタイム処理と UI 処理を疎結合に保つ
- アラート評価を UI なしで単体テスト可能にする
- 再利用可能なセグメンテーション/特徴量ロジックを独立させる

## データ所有権の規約

- chart の runtime 正本は DB（`ChartsV2` + `ActiveChartSet`）。
- `charts_seed.yaml` は初回投入・復旧用の入力と export 対象フォーマットであり、runtime の正本ではない。
- chart の変更履歴・監査ログは `ChartsHistory` に保持する。

緊急対応時の規約:

- 運用安全上、緊急 DB 直変更は許容する
- アプリ/API 層は source/reason メタデータを必須とし、変更履歴を必ず記録しなければならない
- DB レイヤでの厳格な制約（NOT NULL/CHECK 等）の適用は、別フォローアップ Issue で管理する

## アーキテクチャルール

1. 各モジュールの責務を明示する
2. 依存は一方向を基本とする
3. UI 層にビジネスロジックを埋め込まない
4. アラート判定ロジックは説明可能な状態に保つ
5. 主要な設計決定は docs または Discussions に記録する
6. モジュール境界や依存規約を変更した場合はこのドキュメントを更新する
7. ガバナンス決定を変更した場合は `docs/decision-log.md` を更新する

## PR 作成者向けチェックリスト

- この変更が必要な理由
- 影響するモジュール
- 依存方向に変化があるか
- ドキュメント更新が含まれているか（`docs/architecture.md`、`docs/decision-log.md`）
- ロールバックと検証の方針
