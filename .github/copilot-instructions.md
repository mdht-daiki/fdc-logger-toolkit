# Logger Toolkit Copilot Instructions

このリポジトリは、ロギング・特徴量抽出・可視化・アラート判定・DB アクセスのためのモジュール型ツールキットです。

## アーキテクチャ

システムは以下の領域に分割されます:

- ingest:
  生装置データを収集し、分割・特徴量抽出・保存を行う
- dashboard:
  SPC チャート・生データ・しきい値・ドリルダウンを表示する
- judge:
  チャートルールを評価し、アラートをトリガーする
- api:
  SQLite へのアクセスを制御し、並行処理を管理する
- docs:
  アーキテクチャ・運用ルール・設計根拠
- tests:
  ユニットテストと統合テスト

## プロジェクト構成の取り決め

クロスモジュール変更に対する明示的な契約として使用してください。

### モジュール所有権（ソースパス）

- ingest: `src/portfolio_fdc/main`, `src/portfolio_fdc/core`
- dashboard: `src/portfolio_fdc/dashboard`
- judge: `src/portfolio_fdc/judge`
- api: `src/portfolio_fdc/db_api`
- docs: `docs`
- tests: `tests`

### 依存方向

- ingest -> api: 明示的な API 境界経由で許可
- judge -> api: しきい値・特徴量の読み取りと結果書き込みで許可
- dashboard -> api: API が明示的に公開する読み書き操作で許可
- dashboard -> judge の直接呼び出しは禁止
- judge -> dashboard への依存は禁止
- core は再利用可能な状態を保ち、dashboard や db_api の web 層に依存してはならない

### データ所有権と runtime 正本

- chart の runtime 正本は DB（`ChartsV2` + `ActiveChartSet`）
- `charts_seed.yaml` は初回投入・復旧用の可搬アーティファクトであり、runtime 正本ではない
- 緊急手動変更は source/reason フィールドと履歴レコードで監査可能な状態を保たなければならない

### ドキュメント更新ポリシー

- モジュール境界を変更する場合は、同じ PR で `docs/architecture.md` を更新する
- ガバナンス決定を変更する場合は、同じ PR で `docs/decision-log.md` を更新する
- アーキテクチャやガバナンスを変更する PR の説明には、理由・影響範囲・ロールバック計画を含める

## 運用フロー規約（Git / Issue / Discussion）

### ブランチと PR

- `main` への直接コミットは原則禁止。変更は必ず作業ブランチで行い、PR レビューを経由して取り込む。
- docs のみの変更でも同様に PR を作成する。
- 誤って `main` にコミットした場合は、コミットを作業ブランチへ移してからローカル `main` を `origin/main` に戻す。

### Discussion の分割方針

- 論点が複数モジュール（judge / api / dashboard）に派生する場合は、責務ごとに Discussion または Issue を分割する。
- judge 本体の設計論点と、db_api endpoint 要件、dashboard URL/表示要件は同一スレッドに混在させない。
- 分割した場合は、元のトラッキング Issue にリンクを集約し、どこで最終決定するかを明記する。

### 決定事項の反映

- Discussion 分割や責務境界の合意を行った場合は、同一 PR で `docs/decision-log.md` に記録する。
- dashboard 連携契約（例: URL スキーマ、judge 結果表示契約）を変更した場合は、同一 PR で `docs/dashboard-architecture-playbook.md` も更新する。

## 設計原則

- モジュールごとに責務を分離する
- dashboard・judge・ingest の密結合を避ける
- 暗黙のクロスモジュール依存より明示的なインターフェースを優先する
- SQLite アクセスは慎重に扱う（並行処理とロックが重要）
- アラートロジックと特徴量計算の説明可能性を維持する
- 変更は小さく・テスト可能・ドキュメント化された状態を保つ

## コーディング規約

- 明確で保守しやすい Python コードを書く
- 明示的な名前の小さな関数を優先する
- 振る舞いが変わる場合はテストを追加または更新する
- 隠れた副作用を避ける
- 自明でない設計決定は PR または Discussions に記録する

## レビュー規約

レビュー時に注意すること:

- モジュール間の境界の明確さ
- ingest/dashboard/judge/api 間の結合リスク
- DB ロックやトランザクションのリスク
- テスト可能性
- 命名の明確さ
- 実装が PR に記載された目的と一致しているか

## コンテキストルール

レビュー時は PR 説明・関連 Issue・関連 Discussion を必ず確認する。
実装の妥当性判断は、コードだけでなく上記コンテキストとの整合を必須チェックとして行う。
