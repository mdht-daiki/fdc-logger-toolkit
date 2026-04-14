# CodeRabbit Findings Memo (PR #135)

更新日: 2026-04-14 (JST)
対象PR: #135 (`feature/get-charts-endpoint-129`)

## 目的

このメモは、CodeRabbit から指摘された論点を「どのファイルのどの種類の問題か」で追跡するための作業メモです。

## Findings 一覧（現時点）

### 1. docs 系

1. `docs/db-api-minimum-contract.md` 変更に対して `docs/dashboard-architecture-playbook.md` 側の同期不足

- 指摘内容:
  - Phase 1 の 5 endpoint の明示
  - response envelope / timestamp / status code / payload 例の同期
  - Minimum Test Matrix 実行の注記
- 現状: 対応済み

2. `docs/issues/parent-db-api-minimum-contract-phase1.md` の子Issue表記がプレースホルダ

- 指摘内容:
  - `Child - ...` 文字列を `#129`-`#133` へ置換
  - tracking note の文言更新
- 現状: 対応済み

3. `docs/dashboard-architecture-playbook.md` の詳細契約重複とMD040

- 指摘内容:
  - 設計書に詳細JSONを重複記載しすぎ
  - code fence に language 未指定
  - 結合リスク評価の注意を追記
- 現状: 対応済み

### 2. db_api 実装系

1. `src/portfolio_fdc/db_api/app.py` の 500 detail に内部例外文字列を露出

- 指摘内容:
  - ログには詳細、レスポンスは generic message にする
- 現状: 対応済み

2. `src/portfolio_fdc/db_api/chart_repository.py` の read path で `_init_schema` 実行

- 指摘内容:
  - repository の hidden side effect を削除
  - schema 初期化は app startup へ移動
- 現状: 対応済み

3. 422 の error envelope が FastAPI 既定フォーマット依存

- 指摘内容:
  - `/charts` の invalid `step_no` で契約フォーマットを返すべき
- 現状: 対応済み
- 補足:
  - `RequestValidationError` 共通ハンドラを導入し `ok/error/details` へ統一

4. パフォーマンスリスク: `version` 算出で行ごと相関 `COUNT(*)`

- 指摘内容:
  - 大量取得時にCPU負荷増大の懸念
- 現状: 対応済み
- 補足:
  - 相関サブクエリから `filtered_charts` + `history_versions` 集約JOINへ変更

### 3. test 系（GET /charts）

1. `tests/test_db_api_charts_endpoint.py` の filter 検証が部分集合チェック

- 指摘内容:
  - 全レスポンス配列を対象に検証する
  - `all(...)` の前に非空確認を入れる
- 現状: 対応済み

2. 同ファイルで query filter の追加カバレッジ不足

- 指摘内容:
  - `chamber_id`, `recipe_id`, `parameter`, 正の `step_no`, `feature_type` を追加
- 現状: 対応済み

3. `updated_at` と閾値系フィールドの検証が緩い

- 指摘内容:
  - regex ではなく期待値一致（UTC正規化結果）
  - `warning_lcl/ucl`, `lcl/ucl`, `critical_lcl/ucl` を fixture 実値と厳密比較
- 現状: 対応済み

4. `version` テストの `any(...)` が重複混入を見逃す

- 指摘内容:
  - 単一件を明示 (`len(data) == 1`) して `data[0]["version"] == 3` 検証
- 現状: 対応済み

5. history seed timestamp の生成が脆い

- 指摘内容:
  - `"0{index}"` 連結ではなく ISO 8601 生成へ
- 現状: 対応済み

### 4. test ヘルパー・fixture 構成

1. `assert_validation_error_envelope` の重複定義

- 指摘内容:
  - 共通 helper へ抽出
- 現状: 対応済み
- 反映先:
  - `tests/test_utils.py`
  - `tests/test_db_api_charts_endpoint.py`
  - `tests/test_db_api_integration.py`

2. `seeded_chart_rows_for_get_charts` の setup/teardown を fixture 化

- 指摘内容:
  - `yield` fixture で ActiveChartSet 復元を明示
- 現状: 対応済み

3. `seeded_chart_rows_for_filter_tests` も同様に fixture 化

- 指摘内容:
  - DB状態変更を fixture の setup/teardown に閉じ込める
- 現状: 対応済み

4. 2つの seed fixture で重複する共通処理

- 指摘内容:
  - 共有 helper（共通フロー）へ抽出
- 現状: 対応済み

5. `tests/test_utils.py` で `details["issues"]` 検証が緩い

- 指摘内容:
  - 非空、各issueの `loc` / `msg` キーと型を必須化
  - `issue.get` ではなく direct access で fail-loud にする
- 現状: 対応済み

## 現状サマリ

- CodeRabbit 指摘のうち、ここまでに提示された項目はすべて対応済み。
- 以降の差分で新たな指摘が出た場合は、このファイルに追記する。
