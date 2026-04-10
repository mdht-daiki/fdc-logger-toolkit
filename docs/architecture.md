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
- db_api の公開インターフェース経由でのデータ参照

担当しないこと:

- 生データ取込の所有
- アラート最終発報
- DB への直アクセス
- judge の直接呼び出し

実装方針（Issue #86）:

- 初期実装フレームワークは Plotly Dash とする
- read-only baseline を先行し、編集機能は段階導入する
- 将来の代替 UI（例: C# Windows アプリ）は同一の db_api 契約を利用する
- read path は db_api 経由に完全固定し、直 DB 読み込みは採用しない
- API 境界は機能単位を基本とし、read 性能要件に応じて集約 endpoint を許容する
- API バージョニング（例: `/v1`）は当面導入せず、必要時に再評価する
- Chart 表示は半透明しきい値レンジ（緑/黄/赤）で状態把握を補助し、優先度は `NG > WARN > OK` を適用する
- 点詳細では判定結果に加え、処理開始時間・ロットID・（取得可能なら）ウェハIDを参照可能にする

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
- 通常変更は PR レビューを必須とし、緊急時のみ例外ルートを許容する
- 緊急変更の実行者・追認者は個人名ではなく役割で定義する
- 緊急変更後は 24 時間以内または翌営業日内の事後 PR と追認を運用目標とする
- 緊急例外ルートの対象は chart 閾値・chart set 運用に限定し、DB スキーマ変更や大量一括変更は含めない
- アプリ/API 層は最低限の監査情報（変更者/時刻/対象/差分）を自動記録する
- reason/承認コメント/関連 Issue・PR などは後追い入力を許可する
- 緊急変更イベント発生時は運用通知（メール等）を送信する
- DB レイヤでの厳格な制約（NOT NULL/CHECK 等）の適用は、別フォローアップ Issue で管理する

## Seed Recovery and Conflict Resolution

Seed（`charts_seed.yaml`）は disaster recovery に限定される。Runtime 正本は常に DB である。

### Seed Re-initialization Triggers

Seed から復旧する条件は以下の 3 つに限定する：

1. **DB 喪失** (DB ファイル削除、ストレージ故障など)
2. **DB 破損** (チェックサム不整合、テーブル破損)
3. **DB 利用不可スナップショット** (バックアップから復旧時に古いスナップショット状態)

### Priority Rules

**Runtime（平時）**

- DB が利用可能な場合、常に DB を正本とする
- seed を読む処理は許可しない（初期化以外）
- Seed YAML の update は DB 反映に影響しない

**Recovery（復旧フェーズ）**

- DB が利用不可の場合、seed からロードして DB を再構築する
- Seed ロード後、ChartsHistory + PR 履歴で事後変更を再適用する
- 復旧完全性の確認は ChartsHistory 件数とソースコード履歴の整合で検証する

### Audit and Post-Recovery Verification

- Seed ロード時は ChartsHistory に `change_source='seed_recovery'` として記録する
- 復旧後の変更再適用は PR 履歴から判定し、emergency change も含めて再実行する
- 復旧完全性は以下で検証する：
  1. DB チャート件数 vs seed YAML + PR 履歴の期待件数
  2. ChartsHistory の最新 timestamp が復旧前のスナップショット を超えているか
  3. Active set ID は復旧時の既定値から PR 履歴で更新されているか

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
- API 契約を変更した場合、`docs/db-api-endpoints.md` が更新されているか
- ロールバックと検証の方針
