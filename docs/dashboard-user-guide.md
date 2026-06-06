# Dashboard 使い方ガイド（初見ユーザー向け）

このガイドは、初めてこのリポジトリを触る人が dashboard を迷わず動かすための最短手順です。

## 対象

- dashboard をローカルで起動し、主要タブの意味と基本操作を把握したい人
- Change Requests / Emergency / Notification Retry の最小操作を確認したい人

## 事前準備

1. 依存関係をインストールする

```powershell
.\tasks.ps1 install
```

2. db_api を起動する

```powershell
.\tasks.ps1 demo-db-api
```

3. サンプルデータを投入する

```powershell
.\tasks.ps1 demo-data
```

4. dashboard を起動する

```powershell
.\tasks.ps1 demo-dashboard
```

5. ブラウザで `http://localhost:8050` を開く

## 画面の見方（最初に見る場所）

### 上部フィルタ

- `db_api base URL`: 通常は `http://localhost:8000`
- `recipe_id` / `chart_id` / `chart_name` / `result_id`: 各タブの絞り込み条件
- `Load`: 現在のタブを再読み込み

最初は base URL だけ確認して `Load` を押せば十分です。

### タブ構成

- `Charts`: チャート定義一覧。監視対象を俯瞰するタブ
- `Active`: 有効な閾値セットと最新点群。点クリックで生波形ドリルダウン
- `History`: 閾値変更履歴（誰がいつ変更したか）
- `Judge`: 判定結果（NG/WARN/OK）と詳細
- `Change Requests`: 通常変更フロー（申請/承認/適用）
- `Emergency`: 緊急変更と追認（Ratify）
- `Notification Retry`: 失敗通知の再送

## 5分で触る最小操作

### 1. チャートを確認する（Charts）

1. `Charts` タブを開く
2. `Load` を押す
3. 行数と各 `chart_id` が表示されることを確認する

期待結果:

- 一覧に `warning` / `critical` の範囲が見える
- `open` リンクで `Active` タブへ遷移できる

### 2. Active で点と生波形を確認する

1. `Active` タブを開く
2. 上段グラフの点を1つクリックする
3. 下段にドリルダウン波形が表示されることを確認する

期待結果:

- `Focused chart` が表示される
- 下段グラフに対象点の波形が描画される

### 3. 通常変更を試す（Change Requests）

1. `Change Requests` タブを開く
2. `Create Change Request` で以下を入力する
   - `chart_id`
   - `proposed_by`
   - `warn_low` / `warn_high`（threshold fields）
3. `Create Change Request` を押す

期待結果:

- payload preview にJSONが表示される
- 結果欄に `ok: true` のレスポンスが表示される

補足:

- `advanced change_payload JSON` は上級者向けです。
- raw JSON を入力した場合は、threshold fields より raw JSON が優先されます。

### 4. 緊急変更を試す（Emergency）

1. `Emergency` タブを開く
2. `Emergency Change` で以下を入力する
   - `chart_id`
   - `changed_by`
   - `changed_by_role`
   - `warn_high` / `crit_high` など必要な閾値
3. `Apply Emergency Change` を押す

期待結果:

- 結果欄に `request_id` や `resulting_version` が表示される
- 下部 `History Preview` に履歴反映が出る

### 5. 失敗通知を再送する（Notification Retry）

1. `Notification Retry` タブを開く
2. `Refresh Failed Notifications` を押す
3. `event_id` を入力して `Retry Notification` を実行する

期待結果:

- failed レコード一覧が更新される
- 再送結果が結果欄に表示される

## よくあるつまずき

### 一覧が空のまま

- `db_api` が起動しているか確認する
- `demo-data` を実行済みか確認する
- `db_api base URL` が `http://localhost:8000` になっているか確認する

### 4xx / 5xx エラーが出る

- dashboard は API の error envelope をそのまま表示します
- 入力値（ID、role、payload JSON）の形式を見直す
- 競合時（409）は最新状態を再読込して再実行する

### payload preview と送信内容が合っているか不安

- preview に表示された JSON が API に送られる内容です
- raw JSON を埋めた場合は raw が優先されます

## 関連ドキュメント

- endpoint 契約一覧: `docs/db-api-endpoints.md`
- dashboard 設計方針: `docs/dashboard-architecture-playbook.md`
- 全体設計: `docs/architecture.md`
