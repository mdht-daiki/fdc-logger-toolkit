# Chart Governance Playbook

## Goal

Chart 閾値変更を「レビュー必須」で安全に運用する。

- Source of truth: DB (`ChartsV2` + `ActiveChartSet`)
- Portable artifact: `src/portfolio_fdc/configs/charts_seed.yaml`（初回投入・復旧用。正本ではない）
- Emergency change: dashboard/API からの緊急変更を許可。後追い PR は努力義務（24h 以内または翌営業日内目標）。

通常運用と緊急運用:

- 通常変更: PR 必須（レビュー承認後に反映）
- 緊急変更: 即時反映を許可（例外ルート）
- 緊急変更後: 事後 PR と追認は努力義務として運用する

緊急変更の権限と範囲:

- 実行者と追認者は個人名ではなく役割で定義する
- 緊急変更で許可する対象は chart 閾値・chart set 運用に限定する
- DB スキーマ変更や大量一括変更は緊急例外ルートの対象外とする

緊急変更時の監査と通知:

- 自動必須記録: 変更者、変更時刻、変更対象、変更差分
- 後追い入力可: reason、承認コメント、関連 Issue/PR
- 緊急変更イベント発生時は運用通知（メール等）を送信する

注記（強制の実装境界）:

- 自動必須記録（変更者/時刻/対象/差分）はアプリ/API 層で強制する（実装対象: `src/portfolio_fdc/db_api/*`、スキーマ定義は `src/portfolio_fdc/db_api/db.py`）。
- reason や承認コメントは緊急時の後追い入力を許可する（運用努力義務）。
- DB レイヤの制約（NOT NULL/CHECK/trigger など）での厳格強制は別途フォローアップ Issue で管理する。

## Branch Strategy

- 方針整理: `chore/chart-governance-*`
- 閾値変更: `chart/threshold-<ticket>-<short-desc>`
- API/実装変更: `feature/chart-<short-desc>`

例:

- `chart/threshold-42-recipe-a-step2`
- `feature/chartset-import-export`

## GitHub Discussions Usage

カテゴリを以下で作る。

1. `Chart-Proposals`
2. `Chart-Operations`
3. `Chart-Postmortems`

使い分け:

- 変更案の相談: `Chart-Proposals`
- 運用手順や定常作業: `Chart-Operations`
- 誤判定/見逃しの振り返り: `Chart-Postmortems`

Discussion テンプレートに最低限含める項目:

- 背景
- 影響範囲 (`tool/chamber/recipe/parameter/step/feature_type`)
- 期待効果
- リスク
- 検証方法

## GitHub Projects Usage

単一プロジェクト `Chart Threshold Management` を作成し、以下のカスタムフィールドを追加する。

1. `Type` (`proposal`, `implementation`, `ops`, `incident`)
2. `Target Tool` (text)
3. `Target Recipe` (text)
4. `Risk` (`low`, `medium`, `high`)
5. `Review Required` (`yes`, `no`)

ステータス列:

1. `Backlog`
2. `Proposed`
3. `Ready for PR`
4. `In Review`
5. `Approved`
6. `Deployed`
7. `Verified`

運用ルール:

- Discussion で合意したら Project に起票
- PR 作成時に Project アイテムを `In Review` へ移動
- 本番反映後 `Deployed`、監視完了後 `Verified`

## Pull Request Rules

- `src/portfolio_fdc/configs/charts_seed.yaml` の変更は必ずレビュー 1 名以上（初回投入・復旧用 YAML として管理）
- `ChartsV2` の閾値変更 PR は `change_reason` を必ず本文に含める
- `CODEOWNERS` の approver を必須化
- CI で `tests/test_charts_seed.py` を必須チェックにする
- 緊急変更は例外ルートとして即時反映を許可し、事後 PR の作成と追認を行う

注記:

- 現時点では上記ルールは「方針」であり、GitHub 側の branch protection / required checks への適用は未完了。
- 実運用で強制する場合は、別 PR またはリポジトリ設定で required reviewers と required status checks を有効化する。

PR 本文に最低限含める項目:

1. 変更理由
2. 変更点サマリ
3. 影響範囲
4. ロールバック手順
5. 検証結果

## Seed Recovery and Conflict Resolution

### Seed Re-initialization Procedure

Disaster recovery フローは以下に従う：

1. **事象検知**：DB 利用不可を確認
2. **復旧開始決定**：役割ベース（例: Infrastructure Lead）が承認
3. **Seed ロード**：
   - `src/portfolio_fdc/configs/charts_seed.yaml` から ChartsV2 を初期化
   - レコード作成時に ChartsHistory に `change_source='seed_recovery'` と timestamp を記録
4. **Post-restore Reconciliation**：
   - PR 履歴をトレース（マージ時刻の新しい順）
   - Seed ロード後の差分を再適用（主に emergency change と通常 PR による変更）
5. **検証**：
   - ChartsHistory 件数と期待値の照合
   - Active set ID の確認
   - Dashboard での表示確認

### DB-Seed Conflict Rules

| 状況                          | 優先度    | 動作                                        |
| ----------------------------- | --------- | ------------------------------------------- |
| DB 利用可（Platform time）    | DB 100%   | Seed は参照しない。DB を読む                |
| DB 利用不可（Recovery phase） | Seed 100% | Seed からロード。PR 履歴で再適用            |
| Seed YAML 更新（通常）        | -         | DB には影響しない。Documentation 的価値のみ |
| Export at active state        | -         | DB → YAML export。次回復旧時の seed になる  |

### Role-Based Recovery Approval

- **Recovery Initiator**：Infrastructure Lead が復旧開始を判定・決定
- **Recovery Executor**：API または admin tool で seed ロードを実行
- **Reconciliation Lead**：PR 履歴から post-restore 差分を検証・再適用指示

いずれの役割も個人名ではなく role で定義し、複数人が兼任可能な状態を保つ。

## Backlog

| Issue | 内容                                                                   |
| ----- | ---------------------------------------------------------------------- |
| #71   | implementation/migration/removal: 旧 Charts 参照コード撤去・データ移行 |
| #72   | feat(tool): seed import ツール（初回投入・復旧用）                     |
| #73   | feat(tool): export ツール（active set 既定、set 指定対応）             |
| #74   | feat(db_api): chart_sets/charts_v2/history エンドポイント復元          |
| #75   | test: chart 管理フロー統合テスト                                       |
| #76   | feat(dashboard): read-only baseline                                    |

## Definition of Done (Threshold Change)

1. Discussion 合意済み
2. Project アイテムが `Approved`
3. YAML 変更 PR がマージ済み
4. DB へ反映済み (`chart_set_id` 記録)
5. リリース後の判定結果を確認済み
