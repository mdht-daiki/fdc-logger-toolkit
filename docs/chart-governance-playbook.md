# Chart Governance Playbook

## Goal

Chart 閾値変更を「レビュー必須」で安全に運用する。

- Source of truth: DB (`ChartsV2` + `ActiveChartSet`)
- Portable artifact: `src/portfolio_fdc/configs/charts_seed.yaml`（初回投入・復旧用。正本ではない）
- Emergency change: DB 直変更は許可。`change_source='manual'` 必須、`ChartsHistory` 記録必須。後追い PR は努力義務（24h 以内目標）。

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

注記:

- 現時点では上記ルールは「方針」であり、GitHub 側の branch protection / required checks への適用は未完了。
- 実運用で強制する場合は、別 PR またはリポジトリ設定で required reviewers と required status checks を有効化する。

PR 本文に最低限含める項目:

1. 変更理由
2. 変更点サマリ
3. 影響範囲
4. ロールバック手順
5. 検証結果

## Backlog

| Issue | 内容                                                          |
| ----- | ------------------------------------------------------------- |
| #71   | design: Charts テーブル廃止・ChartsV2 正本確定                |
| #72   | feat(tool): seed import ツール（初回投入・復旧用）            |
| #73   | feat(tool): export ツール（active set 既定、set 指定対応）    |
| #74   | feat(db_api): chart_sets/charts_v2/history エンドポイント復元 |
| #75   | test: chart 管理フロー統合テスト                              |
| #76   | feat(dashboard): read-only baseline                           |

## Definition of Done (Threshold Change)

1. Discussion 合意済み
2. Project アイテムが `Approved`
3. YAML 変更 PR がマージ済み
4. DB へ反映済み (`chart_set_id` 記録)
5. リリース後の判定結果を確認済み
