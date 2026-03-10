# FDC Logger Toolkit (Portfolio Edition)

製造装置ログを対象にした、ポートフォリオ向け FDC (Fault Detection & Classification) ツールキットです。

このプロジェクトは社内監視ツールを再構成した公開版で、機密情報や固有ロジックは削除し、以下に置き換えています。

- 疑似 logger データ生成機能
- 設定ファイルベースのマッピング/ルール
- ローカル SQLite パイプライン

---

## このプロジェクトで示していること

- 大規模 CSV ログ（1秒サンプリング）への対応
- 30分間隔の増分取り込み
- プロセス区間切り出し（edge-based / step-peak-based）
- ステップ単位特徴量（mean / max / min / std）抽出
- SPC 風しきい値監視（warn / crit）
- ダッシュボードでの可視化としきい値編集
- DB API 経由の SQLite 同時実行制御

---

## 構成コンポーネント

本リポジトリは主に 4 つのプログラムで構成されます。

1. **main**

- `scrape`: logger/device ログの増分読み取り
- `aggregate`: 区間切り出しと特徴量計算、DB への保存
- 主な保存先:
  - `ProcessInfo`（プロセス情報 + detail CSV パス）
  - `Parameters`（特徴量）
  - `StepWindows`（可視化用ステップ境界）
  - `ChartsV2`（ダッシュボードしきい値）

2. **dashboard**（Plotly Dash）

- 条件フィルタ（tool/chamber/recipe/parameter/step/feature-type）
- SPC チャート + しきい値表示
- 点クリックで生波形ドリルダウン（ステップ色分け）
- しきい値編集結果を DB に保存（ChartsV2）

3. **judge**

- 最新特徴量としきい値の取得
- warn/crit 判定
- アラート送信（メール）
- （公開版では stub）装置停止コマンド連携

4. **db_api**（FastAPI）

- SQLite 読み書きゲートウェイ（シリアライズ write queue）
- main/dashboard/judge 向け REST エンドポイント
- 読み取り中書き込み向け `Temp.db` スワップ戦略（任意）

---

## アーキテクチャ概要

```text
Synthetic Logger CSV / Equipment Logs
        |
      scrape  (incremental read)
        |
    aggregate (segmentation + features + detail csv)
        |
      db_api   (SQLite gateway)
     /   |   \
dashboard judge  (future: exporter)
```

---

## データフロー（概要）

- Logger raw CSV:
  - 1秒サンプリングの大容量ストリーム
  - `timestamp,value01,value02,...` 形式
  - 異常注入用の内部マスクを使う場合があるが CSV には出力しない
  - ヘッダー部と `DATA` マーカー行を含む

- scrape:
  - 前回実行以降の新規行のみ抽出（約30分）
  - 巨大 CSV 全体は読まず、必要に応じて末尾読み取り
  - `tool_id` / `chamber_id` 付与
  - マッピングファイルでチャンネル名を論理名へ変換

- aggregate:
  - プロセス区間切り出し
    - edge-based: キーチャンネルの ON 区間検出
    - step-peak-based: 複数ステップを 1 プロセスとして束ねる
  - ダッシュボード用 detail 波形を長形式 CSV で保存
  - `ProcessInfo` / `Parameters` / `StepWindows` を書き込み

---

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

---

## クイックスタート（ローカル）

### 1) db_api を起動

```bash
python -m portfolio_fdc.db_api.app
# or uvicorn portfolio_fdc.db_api.app:app --host 0.0.0.0 --port 8000
```

現時点の実装スコープ（PR を小さく保つため）:

- `aggregate` 連携エンドポイントのみ実装
  - `POST /processes`
  - `DELETE /processes`
  - `POST /step_windows/bulk`
  - `POST /parameters/bulk`
- 先送りしたエンドポイント（charts/judge/chart_sets/charts_v2）は
  `src/portfolio_fdc/db_api/app.py.backup_non_aggregate_endpoints.py` に退避
- 退避ファイルは `.gitignore` でレビュー対象外

### 2) 疑似 logger CSV を生成

```bash
python -m portfolio_fdc.tools.generate_logger_csv --out data/raw/logger_raw.csv --seconds 86400 --scenario mix
```

### 3) main パイプラインを実行（scrape + aggregate）

```bash
python -m portfolio_fdc.main.run_once --tool TOOL_A --raw data/raw/logger_raw.csv --db-api http://localhost:8000
```

DB API 未起動時は `aggregate` の dry-run（ローカル処理のみ、POST なし）も可能です。

```bash
python -m portfolio_fdc.main.aggregate \
  --input data/scrape/scrape_TOOL_A.csv \
  --config src/portfolio_fdc/configs/aggregate_tools.yaml \
  --detail-out data/detail \
  --dry-run
```

Makefile 版（Mac/Linux）:

```bash
make aggregate-dry-run
# optional override
make aggregate-dry-run AGG_INPUT=data/scrape/scrape_TOOL_B.csv AGG_DETAIL_OUT=data/detail_tmp
```

PowerShell タスク版（Windows）:

```powershell
.\tasks.ps1 aggregate-dry-run
# optional override
.\tasks.ps1 aggregate-dry-run -AggInput data/scrape/scrape_TOOL_B.csv -AggDetailOut data/detail_tmp
```

### 4) dashboard を起動

```bash
python -m portfolio_fdc.dashboard.app
# open http://localhost:8050
```

### 5) judge を手動実行

```bash
python -m portfolio_fdc.judge.run_once --db-api http://localhost:8000
```

---

## 開発コマンド

```bash
make fmt
make lint
make type
make test
make all
```

---

## CI 品質ゲート

Pull Request では以下の通過が必要です。

- Ruff（lint + format）
- MyPy（型チェック）
- Pytest
- （任意）CodeRabbit review

---

## 免責

本リポジトリはポートフォリオ向け簡易版です。
本番向けの機密ロジック、固有プロセス条件、社内インフラ詳細は含みません。
