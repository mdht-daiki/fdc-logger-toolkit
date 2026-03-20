# Chart Governance Playbook

## Goal

Chart 閾値変更を「レビュー必須」で安全に運用する。

- Source of truth: Git (`src/portfolio_fdc/configs/charts_seed.yaml`)
- Runtime source: DB (`ChartsV2` + `ActiveChartSet`)
- Emergency change: DB 直変更は許可。ただし必ず後追いで Git PR に反映する。

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

- `src/portfolio_fdc/configs/charts_seed.yaml` の変更は必ずレビュー 1 名以上
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

## Initial Backlog

1. `charts_seed.yaml` -> `ChartsV2` import ツールを作る
2. `ChartsV2` -> YAML export ツールを作る
3. `ActiveChartSet` 切替 API を app 本体に戻す
4. dashboard 編集を draft set 作成フローへ寄せる

## Definition of Done (Threshold Change)

1. Discussion 合意済み
2. Project アイテムが `Approved`
3. YAML 変更 PR がマージ済み
4. DB へ反映済み (`chart_set_id` 記録)
5. リリース後の判定結果を確認済み
