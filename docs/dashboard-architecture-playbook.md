# Dashboard Architecture Playbook

## Goal

dashboard モジュールの設計前提を統一し、read-only baseline から編集機能までを
安全に段階導入できる状態を作る。

- Primary UI: Plotly Dash
- Future alternative UI: C# Windows アプリ（後続検討）
- Data source of truth: DB（ChartsV2 + ActiveChartSet / judge 結果）
- Access policy: dashboard は db_api 経由のみ（直 DB アクセス禁止）

## Scope

本プレイブックは以下を対象とする。

1. フレームワーク選定
2. データ取得方式
3. judge 結果の参照方式
4. read-only baseline から編集機能への移行方針

## Architecture Principles

1. dashboard は UI クライアントとして実装し、ビジネスロジックを埋め込まない
2. dashboard -> api の依存のみ許可し、dashboard -> judge の直接依存を禁止する
3. DB 直読み込みを禁止し、すべて db_api の契約経由で取得する
4. UI 実装を差し替えても API 契約は維持する（Dash/C# 併存可能性を確保）

## Framework Decision

- 初期実装は Plotly Dash を採用する
- 理由:
  - read-only baseline を短期間で構築しやすい
  - 既存 Python 資産との統合コストが低い
  - 可視化（SPC チャート、ドリルダウン）に必要な機能を早期に提供しやすい
- C# Windows アプリは後続フェーズで検討する

## Data Access Policy

dashboard のデータ取得・更新経路は以下に限定する。

1. chart 定義、active set、履歴: db_api の chart 系エンドポイント
2. judge 判定結果: judge が DB に保存した結果を db_api 経由で参照
3. 緊急運用情報（source/reason など）: db_api が返す監査用フィールドを利用

API 設計方針（Discussion #93）:

1. read path は db_api 経由に完全固定する
2. API 境界は機能単位で定義する（画面単位分割は採用しない）
3. 通信往復削減のため、db_api 側に集約 read endpoint を提供する
4. API バージョニング（例: `/v1`）は当面導入しない

禁止事項:

- dashboard から SQLite ファイルへ直接接続
- dashboard から judge モジュールを直接 import/call

実装前提（Issue #98）:

db_api は以下の read endpoint を実装する必要があります。
Phase 1 の契約詳細は `docs/db-api-minimum-contract.md` を参照します。
実装済み/計画中 endpoint の一覧は `docs/db-api-endpoints.md` を参照します。

### Phase 1 Endpoint Contract Summary

dashboard および judge の実装者は以下の契約を基準として実装・テストする。
契約内容を変更する場合は `docs/db-api-minimum-contract.md` を同一 PR で更新し、
下記 Contract Test 要件を再実行すること。
レビュー時は ingest / dashboard / judge / api 間の結合リスクを評価すること。

#### 対象 Endpoint（Phase 1）

| Endpoint                         | 用途                         | Consumer          |
| -------------------------------- | ---------------------------- | ----------------- |
| `GET /charts`                    | chart 定義一覧               | dashboard / judge |
| `GET /charts/active`             | active chart set と有効閾値  | dashboard / judge |
| `GET /charts/history`            | 閾値変更履歴                 | dashboard / ops   |
| `GET /judge/results`             | 判定結果一覧                 | dashboard         |
| `GET /judge/results/{result_id}` | 判定結果詳細（ドリルダウン） | dashboard         |

#### Key Contract Points

- **Response Envelope**: 成功時 `{"ok": true, "data": ...}`、失敗時 `{"ok": false, "error": {"code": ..., "message": ..., "details": ...}}`。`/judge/results/{result_id}` の `data` は object。
- **Timestamp**: 全フィールドで UTC / ISO 8601 / ミリ秒固定精度（例: `2026-04-14T00:00:00.000Z`）。
- **Status codes**: 200 正常 / 400 クエリ不正 / 404 対象なし / 422 バリデーション不正 / 500 内部エラー。
- **Payload 例**: `docs/db-api-minimum-contract.md` の各 endpoint セクションを参照。

#### Contract Test Requirements

dashboard または judge の実装者は、上記 5 endpoint の契約を変更する PR において
`docs/db-api-minimum-contract.md` の **Minimum Test Matrix** に委譲し、以下を最低限実施すること。

1. 各 endpoint の 200 と主要 4xx（400/404/422 の該当ケース）の契約テストを追加・更新する
2. timestamp 形式（UTC ISO 8601 ミリ秒固定精度）の検証を含める
3. しきい値変更後の判定結果参照が反映される連鎖の統合テストを追加・更新する
4. 既存 ingest write endpoint への非影響を回帰確認する

## Judge Result Integration

judge 連携は「直接実行」ではなく「結果参照」を原則とする。

- dashboard は最新判定と履歴判定を db_api から取得して表示する
- 判定ロジック自体の変更や実行トリガーは judge 側責務として分離する
- 表示要件追加時は、judge 変更より先に db_api のレスポンス契約を拡張する

表示規約（Discussion #96 論点1-4）:

1. Chart 一覧は半透明レンジ表示を採用する（Center=緑、Warning=黄、Critical=赤）
2. 視認性の優先順位は `NG > WARN > OK` とし、色分け主体 + 最小ラベル/凡例で補助する
3. judge 情報は直近 N 件チャートに重ね、詳細は hover/click で段階表示する
4. 点詳細には判定結果に加え、処理開始時間・ロットID・（取得可能なら）ウェハIDを表示する

URL 契約トラッキング（Discussion #94）:

1. judge 通知から dashboard へ遷移する直リンクの URL スキーマは #94 で確定する
2. 最小要件は `chart_id` 必須 + `lot_id`/`segment_id`/`judged_at` 任意を候補とする
3. 互換方針は「未知クエリを無視し既存リンクを壊さない」を前提に検討する

注記:

- 論点5（Phase 1/2/3 の最終境界）は継続議論とし、本規約では固定しない

## Rollout Plan

### Phase 1: Read-only Baseline

- SPC チャート表示
- 生データドリルダウン表示
- active chart set / chart 閾値の参照
- judge 判定結果の参照表示

### Phase 2: Limited Edit Flow

- emergency change 実行 UI（`POST /governance/emergency-changes`）
- emergency ratify UI（`POST /governance/emergency-changes/{request_id}/ratify`）
- role/認可結果に応じた 403 表示
- 実行結果と履歴（`/charts/history`）の確認導線

#### Phase 2 Endpoint Contract Detail（Change Request 系）

**Change Request Workflow Endpoints**（approval flow）

1. `POST /governance/change-requests`
   - Request JSON: `chart_id`(int), `proposed_by`(string), `change_payload`(string), `expected_version`(int), `idempotency_key`(string)
   - 認可: dashboard/ops が利用可
   - 役割: change request（申請）を作成

2. `POST /governance/change-requests/{request_id}/approve`
   - Request JSON: `approved_by`(string), `approved_by_role`(string), `comment`(string|null)
   - 認可: dashboard/ops が利用可
   - 役割: pending → approved ステータス遷移、承認決定の記録
   - エラー: 409（既に承認済み）、422（無効な request_id）、5xx

3. `POST /governance/change-requests/{request_id}/apply`
   - Request JSON: `applied_by`(string), `applied_by_role`(string), `reason`(string|null)
   - 認可: dashboard/ops が利用可
   - 役割: approved → applied ステータス遷移、chart 反映実行、履歴記録
   - エラー: 409（既に適用済み/状態不正）、422（バリデーション失敗）、5xx

#### Phase 2 Endpoint Contract Detail（Emergency 系）

1. `POST /governance/emergency-changes`

- Request JSON（必須/任意）
  - 必須: `chart_id`(int), `changed_by`(string), `changed_by_role`(string), `change_payload`(string, JSON)
  - 任意: `reason`(string|null)
- 認証・認可境界（重要）
  - `changed_by` / `changed_by_role` は監査メモ用途として受け付ける（必須の入力契約は維持）
  - 認可判定はサーバ側で認証トークン/セッションのクレームから確定した actor/role のみを使用する
  - 役割確定に利用するクレーム（例: `sub`, `roles`, `scope`）はサーバ側設定で固定し、入力 JSON の role 値は認可根拠に使わない
  - サーバ側で確定した actor/role は監査ログへ記録し、必要に応じて入力値との差分も監査可能にする
- Success Response JSON
  - `{"ok": true, "data": {"request_id": int, "status": "applied", "resulting_version": int, "noop": bool}}`
- Error Response JSON（envelope）
  - 共通: `{"ok": false, "error": {"code": string, "message": string, "details": object}}`
  - 想定 status: `400`, `403`, `422`, `500`
- 認可要件
  - emergency 実行権限を持つ role のみ許可
  - 権限外 role は `403`（`code=FORBIDDEN` 系）
- テスト要件
  - 正常: apply 成功、`request_id/status/resulting_version/noop` 検証
  - 異常: 422（入力不正）, 403（認可失敗）, 5xx（内部異常）

2. `POST /governance/emergency-changes/{request_id}/ratify`

- Request JSON（必須/任意）
  - 必須: `ratified_by`(string), `ratified_by_role`(string)
  - 任意: `ratification_comment`(string|null), `related_pr`(string|null)
- 認証・認可境界（重要）
  - `ratified_by` / `ratified_by_role` は監査メモ用途として受け付ける（必須の入力契約は維持）
  - 認可判定はサーバ側で認証トークン/セッションのクレームから確定した actor/role のみを使用する
  - 役割確定に利用するクレーム（例: `sub`, `roles`, `scope`）はサーバ側設定で固定し、入力 JSON の role 値は認可根拠に使わない
  - サーバ側で確定した actor/role は監査ログへ記録し、必要に応じて入力値との差分も監査可能にする
- Success Response JSON
  - `{"ok": true, "data": {"request_id": int, "status": "ratified"}}`
- Error Response JSON（envelope）
  - 共通: `{"ok": false, "error": {"code": string, "message": string, "details": object}}`
  - 想定 status: `400`, `403`, `404`, `409`, `422`, `500`
- 認可要件
  - ratify 権限を持つ role のみ許可
  - 権限外 role は `403`（`code=FORBIDDEN` 系）
- テスト要件
  - 正常: ratify 成功、`request_id/status` 検証
  - 異常: 404（対象なし）, 409（多重 ratify）, 403（認可失敗）, 422（入力不正）

3. 実行結果導線 `GET /charts/history`

- 用途
  - apply 実行後に `change_source`/`changed_by`/`changed_at` を確認する
- Response JSON
  - `{"ok": true, "data": [{"history_id": ..., "chart_id": ..., "change_source": ..., "changed_by": ..., "changed_at": ...}]}`
- Error Response JSON（envelope）
  - 共通: `{"ok": false, "error": {"code": string, "message": string, "details": object}}`
  - 想定 status: `400`, `403`, `500`
- テスト要件
  - 正常: emergency apply 後に履歴が取得できる
  - 異常: 履歴取得失敗時でも apply 成功表示は維持し、履歴失敗メッセージを表示する
  1. 通常変更 UI（change request / approve / apply）
  - 用途
    - dashboard から `GET /governance/change-requests` で一覧/詳細を表示し、`POST /governance/change-requests` / `POST /governance/change-requests/{id}/approve` / `POST /governance/change-requests/{id}/apply` を実行する
  - 表示契約
    - 一覧は `status`, `chart_id`, `proposed_at`, `expected_version`, `idempotency_key` を表示する
    - 詳細は選択された request の `change_payload` を含むフル情報を表示する
    - 409 conflict 時は current 情報または失敗理由をそのまま表示し、`expected_version` と `idempotency_key` の再確認を促す
    - 422/4xx/5xx envelope はユーザー向けメッセージへ正規化して表示する
  - テスト要件
    - 作成 -> 承認 -> 適用の正常系を dashboard から通せること
    - 409 conflict, 422 validation, 5xx の各エラー表示を確認すること
    - 一覧/詳細の表示とフィルタ（status, chart_id, from_ts, to_ts, limit, offset）が動作すること

### Phase 3: Governed Activation Flow

- draft set のレビュー連携
- マージ後 active 化 API 実行
- 変更履歴と判定結果の事後確認

## Release and Rollback Policy（論点10）

1. 機能フラグは運用トグルが必要な dashboard 機能に限定し、設定ファイル（YAML）で制御する
2. 新規 API endpoint 連携の通常リリースは原則フラグなしとし、契約テスト/統合テストを受け入れ条件とする
3. ロールバック単位は「API のみ」「設定のみ」「API+設定」の 3 区分を基本とし、障害原因に応じて最小影響単位で実施する
4. dashboard 変更で設定切り戻しが必要な場合は、Git 管理された設定差分の復元を優先し、復元後に表示契約（`NG > WARN > OK`、color band）を再検証する
5. 引き継ぎチェックは「事前確認」「実行中監視」「ロールバック判定/実行」の 3 フェーズで実施し、実施記録をトラッキング Issue に残す
6. 実行中監視では、API エラー率、DB lock timeout 傾向、judge 判定表示の異常増加を監視し、閾値超過時は即時に切り戻し判定へ移行する
7. ロールバック実行後は、dashboard の主要画面（一覧、詳細、ドリルダウン、履歴）の表示契約を確認し、復旧完了を記録する

## Test Strategy

dashboard 関連テストは以下の 4 層で管理する。

1. 契約テスト（db_api 境界）

- `GET /charts*`, `GET /judge/results*`, `GET /charts/history` のレスポンス契約を固定する
- 必須項目、エラー形式、timestamp 形式、互換要件（deprecated endpoint ヘッダ）を検証する

2. 統合テスト（しきい値変更フロー）

- しきい値更新 -> 履歴記録 -> active set 反映 -> judge 参照結果の連鎖を検証する
- 更新後に dashboard 参照結果が契約どおり変化することを確認する

3. 受け入れテスト（表示ルール）

- 優先順位 `NG > WARN > OK` が表示判定に反映されること
- color band（Center/Warning/Critical）が仕様どおり表示されること
- hover/click で判定結果 + 処理開始時間 + ロット ID（取得可能ならウェハ ID）を表示できること

4. 回帰テスト

- 既存 ingest/db_api 連携の最小成功フローを壊さないこと
- 新規 UI 機能追加時に既存表示規約が後退しないこと

PR で dashboard 契約や表示仕様を変更する場合は「契約テスト 1 件 + 統合テスト 1 件 + 回帰影響判定（影響なしの場合は根拠を明記）」を DoD として同一 PR に含める。
例: color band 仕様変更時は、契約テスト（`GET /judge/results` 必須項目）1 件 + 統合テスト（しきい値変更後の表示更新）1 件 + 既存 `NG > WARN > OK` ルールへの回帰影響判定を PR 説明に記載する。

## Non-Goals

- dashboard から judge の直接実行
- UI 側での独自判定ロジック実装
- DB スキーマへ直接依存したクエリ実装

## Change Management

設計前提を変更する場合は同一 PR で以下を更新する。

1. docs/architecture.md
2. docs/decision-log.md
3. 本ドキュメント
4. docs/db-api-endpoints.md（API 契約に変更がある場合）

dashboard 連携契約（read path、URL スキーマ、表示契約、judge 結果参照契約）を変更する場合は、
本ドキュメント（`docs/dashboard-architecture-playbook.md`）の更新を同一 PR で必須とする。

PR 説明には理由、影響範囲、ロールバック方針を含める。

レビュー時チェック:

- PR 説明・関連 Issue・関連 Discussion が実装内容と整合しているか
