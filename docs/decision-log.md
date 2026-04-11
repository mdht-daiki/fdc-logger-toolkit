# Decision Log

## 2026-04-11: #106 モジュール境界の機械的担保と API 許可範囲の運用ルール実装

### Context

Discussion #90 と PR #92 / #100 で、モジュール境界方針
（dashboard -> api のみ許可、dashboard -> judge 禁止、judge -> dashboard 禁止）は
合意済みだったが、コードベースで違反を検知する機械的仕組みは未実装だった。
また、ingest/judge/dashboard/ops が api を呼ぶ際の許可範囲が
運用手順として明文化されておらず、将来の境界逸脱リスクが残っていた。

### Decision

システムとして以下を実装する。

1. import 境界は `import-linter` で機械検証する
2. `pyproject.toml` に禁止依存契約を定義する
   - dashboard must not import judge
   - judge must not import dashboard
   - main ingest must not import dashboard or judge
3. `.pre-commit-config.yaml` に import-linter フックを追加する
4. `scripts/run_import_linter.py` を追加し、`src` 配下を安定的に解析可能にする
5. `docs/db-api-endpoints.md` に `Consumer Permission Scope` を追加し、
   ingest/judge/dashboard/ops-audit の許可/禁止範囲を明記する

### Why

- 方針だけでなく機械検証を導入することで、実装拡張時の境界逸脱を早期に検出できる
- pre-commit フックによりローカル段階で違反を止められる
- API 許可範囲を文書化することで、モジュール責務を運用と実装の両面で一致できる
- 将来の judge/dashboard 実装拡張時にも境界ルールを継続適用しやすい

### Consequence

- モジュール依存違反は import-linter 契約違反として検出される
- 開発フローに pre-commit での境界チェックが追加される
- endpoint ごとの consumer 許可範囲が docs 上で参照可能になる
- CI 必須化（GitHub Actions で import-linter 実行）はフォローアップタスクとして管理する

## 2026-04-10: #104 正本データの扱い（DB正本 + seed復旧用）

### Context

Issue #102 で変更ガバナンスが確定したことに伴い、次のステップとして
「正本データの扱い」を明確化する必要があった。
Chart の runtime 正本は DB（ChartsV2 + ActiveChartSet）であり、
seed（charts_seed.yaml）は災害復旧用のアーティファクトと位置づけられていたが、
seed 初期化トリガーと DB-seed 競合時の優先ルールが明示的でなかった。

### Decision

正本データとして以下の方針を採用する：

1. **Runtime 正本は DB のみ** → Seed は参照しない（初期起動時以外）
2. **Seed 再初期化トリガーは 3 条件に限定** → DB 喪失、破損、利用不可スナップショット
3. **DB-Seed 優先ルール**
   - Runtime（平時）：DB を 100% 優先。Seed は読まない
   - Recovery（復旧フェーズ）：Seed をロード後、PR 履歴で差分を再適用
4. **Seed ロード時の監査記録** → `change_source='seed_recovery'` で ChartsHistory に記録
5. **復旧完全性検証** → ChartsHistory 件数 + PR 履歴 + Active set ID で整合確認

### Why

- Runtime を DB のみに統一することで、緊急変更・API 更新・UI 編集が混在する運用を一本化できる
- Seed 再初期化を 3 条件に限定することで、乱開始を防ぎ、計算可能な復旧フローを確立できる
- 優先ルールを明示することで、復旧フェーズでの実装判断を機械的にできる
- ChartsHistory + PR 履歴で復旧後の状態を検証できるため、復旧ミスを早期に検出できる

### Consequence

- docs: architecture.md に「Seed Recovery and Conflict Resolution」セクションを追加
- docs: chart-governance-playbook.md に「Seed Recovery and Conflict Resolution」セクションを追加
- DB: ChartsHistory に `change_source` カラムに `'seed_recovery'` 値を許可する実装
- API: Seed ロード endpoint（Issue #72 の実装対象）で change_source を自動設定する
- Ops: Disaster recovery playbook でロール定義と手順を明記する

## 2026-04-09: #102 変更ガバナンス方針の最終確定（通常PR必須 + 緊急変更例外）

### Context

Issue #102 で、変更ガバナンスの運用ルールを実装前に確定する必要があった。
既存方針では緊急変更時の監査項目が厳格で、運用負荷が高くなる懸念があった。
一方で、緊急対応の即応性を維持しつつ、監査可能性を落とさない設計が必要だった。

### Decision

変更ガバナンスは以下の方針で運用する。

1. 通常変更は PR 必須とする
2. 緊急変更は dashboard/API から即時反映を許可する
3. 緊急変更後の事後 PR と追認は努力義務とし、24時間以内または翌営業日内を目標とする
4. 緊急変更の実行者/追認者は個人名ではなく役割で定義する
5. 緊急変更で許可する対象は chart 閾値・chart set 運用に限定し、DB スキーマ変更や大量一括変更は対象外とする
6. 監査項目は「自動必須」と「後追い入力可」に分離する
   - 自動必須: 変更者、変更時刻、変更対象、変更差分
   - 後追い入力可: reason、承認コメント、関連 Issue/PR
7. 緊急変更イベント発生時は運用通知（メール等）を送信する

### Why

- 平時は PR レビューで品質を担保し、緊急時は即応性を確保できる
- 監査の最低限を自動記録することで、後追い入力が遅れてもトレーサビリティを維持できる
- 役割ベースにすることで、個人依存を避けた運用にできる
- 対象範囲を限定することで、緊急例外ルートの乱用を防げる

### Consequence

- docs 上のガバナンス定義は #102 を基準に更新する
- db_api では自動必須監査項目の保存と緊急通知を優先実装対象にする
- reason/承認コメントなどの後追い入力フローは運用手順と API 設計で追補する

## 2026-04-04: 論点1 Charts テーブルの扱い

### Context

Discussion #68 で、Charts と ChartsV2 が並存している状態を解消し、
chart 閾値の正本をどちらに寄せるかを決める必要があった。
dashboard/judge の本格実装前であり、互換維持コストを最小化できるタイミングだった。

### Decision

Charts は廃止し、ChartsV2 + ActiveChartSet を正本として一本化する。
本決定はターゲット状態であり、現行コードベースには legacy Charts テーブルが残っているため、
移行タスクで段階的に実装へ反映する（Charts テーブル定義の削除/マイグレーションが必要）。

### Why

- 実装前段階なので、破壊的変更の影響が最小
- set 管理と履歴監査を標準フローにできる
- legacy 互換層を将来はがすコストを先に回避できる
- データモデルが単純化され、保守性が上がる

### Consequence

旧 Charts を前提にした実装は採用しない。
judge/dashboard/db_api は ChartsV2 系を前提に実装する。
既存DB移行が必要な場合は migration で対応する。

## 2026-04-04: 論点2 judge の参照先

### Context

Discussion #68 の論点2として、judge が参照する chart 閾値の参照先を
ChartsV2 + ActiveChartSet にするか、旧 Charts を使うかを決める必要があった。
論点1で正本を ChartsV2 に一本化する方針を採ったため、judge 側の参照先も整合が必要だった。

### Decision

judge は常に ChartsV2 + ActiveChartSet を参照する（案 X）。
旧 Charts は judge の参照先として採用しない。

### Why

- ActiveChartSet による切替運用に対応できる
- ChartsHistory と整合した監査可能な運用になる
- 正本定義と実行系の参照先を一致させられる
- 将来の閾値運用機能（set単位切替、履歴追跡）と矛盾しない

### Consequence

judge 実装では、実行時に ActiveChartSet から chart_set_id を解決して
ChartsV2 を参照する設計を前提とする。
旧 Charts を読む分岐や互換レイヤは追加しない。

## 2026-04-04: 論点3 ChartsHistory の記録粒度

### Context

Discussion #68 の論点3として、ChartsHistory をどの更新経路で記録するか、
および「変更なし UPSERT」を履歴化するかを決める必要があった。
運用上は dashboard 編集、seed import、API 経由更新が混在するため、
記録経路の偏りがあると監査上の抜け漏れが発生する懸念があった。

### Decision

ChartsHistory は dashboard 編集、seed import、API 経由の任意更新の
すべてで記録する。
ただし、値変化のない UPSERT は記録しない。

### Why

- 更新チャネルを限定しないことで監査ログの一貫性を担保できる
- seed import を含めることで初期投入/復旧時の変更履歴も追跡できる
- API 経由更新を含めることで運用時の手動/自動更新を同じ基準で管理できる
- 変更なし UPSERT を除外することで履歴ノイズを抑え、実変更を追いやすくできる

### Consequence

実装では update_source を保持しつつ、値差分がある場合のみ
ChartsHistory にレコードを追加する。
無変更の再投入は成功扱いにしつつ、履歴件数は増やさない。

## 2026-04-04: 論点4 緊急 DB 直変更のルール

### Context

judge は装置停止を伴う監視システムであり、chart 設定誤りの影響が大きい。
運用上は正常フロー（PR承認 → デプロイ）が基本だが、
装置の異常停止など緊急時には本来の承認フローをすっ飛ばして
即座に chart を修正する必要が生じる可能性がある。
このときの DB 直変更をどのレベルで制御・記録するかの方針が必要だった。

### Decision

以下の3点ルールで緊急対応を許容する：

1. **change_source = 'manual' を必須** → DB 直変更は必ずこの値を指定
2. **follow-up PR は義務ではなく努力義務** → 事業判断で事後承認パターンを許容
3. **ChartsHistory に記録を必須** → 事後検証と監査のため全件記録

### Why

- change_source を必須にすることで「いつ誰が何を変更したか」を明確化できる
- PR 強制ではなく「実質的強制」に留めることで、本当の緊急時のブロッキングを避ける
- History 記録は全チャネル原則なため、直変更も例外なく監査対象にする
- 事後 PR への心理的プレッシャーが事実上、承認フロー遵守を促す効果がある

### Consequence

緊急対応の流れ：

1. 装置異常を検知 → DB 直変更（`ChartsV2.update_source='manual'` 必須、`ChartsV2.update_reason` に理由記述）
2. 履歴監査として `ChartsHistory.change_source='manual'` と `ChartsHistory.change_reason` を必ず記録する
3. 営業日内に事後 PR で根拠を説明（実質的強制）

技術的には PR なしでも緊急変更は成立するが、
運用ルールとして「事後 PR 作成 24h以内」を明文化し、
PR レビューで「なぜこれが必要だったか」を記録する。

## 2026-04-04: 論点5 seed / export の位置づけ

### Context

Chart 閾値運用では DB を正本にする方針を採用している一方で、
初期投入や障害復旧に使えるファイルベース資産も必要だった。
そのため、seed と export の責務、対象範囲、フォーマット互換性を
明確に定義する必要があった。

### Decision

seed / export は以下の方針とする。

1. charts_seed.yaml は正本 YAML ではなく、初回投入と復旧用の入力アーティファクトとする
2. export の既定対象は active set のみとし、必要時は set 指定で個別 export を可能にする
3. export YAML と seed YAML は同一フォーマットとし、round-trip 可能にする

### Why

- 正本を DB に一元化することで、緊急更新や API 更新を含む実運用状態との乖離を防げる
- active set 既定にすることで日常運用の操作を単純化できる
- set 指定 export を許可することで監査や再現用途をカバーできる
- 同一フォーマットに統一すると import/export テストが容易で、運用ミスを減らせる

### Consequence

運用上の真実は ChartsV2 + ActiveChartSet（DB）となる。
YAML は可搬・復旧用途として扱い、
export -> import の往復で同等状態を再構築できることを品質条件に含める。
将来的に複数 set 一括 export が必要になった場合は、別機能として追加検討する。

## 2026-04-05: #90 dashboard のフレームワーク選定

### Context

Issue #86 で dashboard モジュールの設計前提を確定するにあたり、
read-only baseline（Issue #76）から段階的に拡張できる実装方針と、
将来の別 UI 実装（Windows ネイティブ）を両立できる方針が必要だった。

### Decision

dashboard の初期実装は Plotly Dash を採用する。
将来の代替 UI として C# による Windows アプリ実装を後続で検討する。
ただし両者ともデータ取得は db_api の公開 API を経由し、
dashboard から DB 直読みや judge 直接呼び出しは行わない。

### Why

- read-only baseline を最短で形にしやすい
- 既存 Python スタックとの接続コストが低い
- API 境界を固定することで UI 実装を差し替え可能にできる
- 将来の C# UI を追加しても judge/db の責務分離を維持できる

### Consequence

dashboard 実装は「UI クライアント」として扱い、
データ参照・更新は db_api の契約に集約する。
judge 結果は DB に保存された結果を db_api 経由で参照する。
機能拡張は read-only baseline 完了後、編集フロー（draft set など）を
API 契約を保ったまま段階導入する。

## 2026-04-05: #93 dashboard のデータ取得方式と API 契約

### Context

Issue #86 の論点分割として Discussion #93 で、dashboard の read path と
API 契約の境界を具体化した。
将来の C# クライアント追加を見据えつつ、工場内無線通信 + ノートPC での
応答遅延課題に対応できる構成が必要だった。

### Decision

1. dashboard の read path は db_api 経由に完全固定する（直 DB 読み込みは採用しない）
2. API 境界は画面単位ではなく機能単位で設計する
3. 集約レスポンスは db_api 側で提供する（dashboard 側の多重合成を前提にしない）
4. API バージョニング（例: `/v1`）は当面導入しない

### Why

- C# を含む複数クライアント化に備えて API 契約を単一化できる
- 画面単位 API は UI 分割変更の影響を受けやすく、境界が不安定になりやすい
- 通信往復回数を減らすことで、ネットワーク・端末制約下の体感性能を改善しやすい
- 現時点は単一チーム・単一プロダクト運用であり、早期のバージョニング導入コストが高い

### Consequence

dashboard 実装は db_api の read endpoint を唯一の参照経路とする。
db_api は機能単位 endpoint を基本としつつ、dashboard の read 性能要件に対して
集約 endpoint を提供する。
将来、互換維持要件が顕在化した時点で API バージョニング導入を再評価する。

## 2026-04-05: #96 Chart表示中心UIにおける judge 結果表示（論点1-4）

### Context

Discussion #94/#96 で、dashboard の主機能を「Chart 表示中心」で再整理した。
運用上は SPC の直近 N 件表示を基準に、点の異常把握と製造トレース情報への
即時アクセスを両立する表示設計が必要だった。

### Decision

1. Chart 一覧の状態認識は半透明レンジ表示を採用する（Center=緑、Warning=黄、Critical=赤）
2. 視認性ルールは `NG > WARN > OK` を優先し、色分け主体 + 最小ラベル/凡例で補助する
3. 点クリックまたはマウスオーバーで、判定結果に加えて処理開始時間・ロットID・（取得可能なら）ウェハIDを表示する
4. judge 情報の重ね方は「直近 N 件チャートを基本表示、詳細は hover/click で段階表示」とする

### Why

- SPC として成立する連続点表示を維持しつつ、異常点の認知を速くできる
- 装置別/Chamber別/Recipe別で比較する運用に対して、過剰な文字情報を抑えられる
- 現場の一次切り分けで必要なトレース情報（ロット・ウェハ）へ即時に到達できる
- 画面常時表示の情報量を抑え、必要時のみ詳細確認できる

### Consequence

dashboard の表示仕様は color band + 優先度ルールを基準として実装する。
db_api の judge 系レスポンスは、判定結果に加えてトレース用メタデータを
UI が参照できる形で提供する。
論点5（Phase 境界の最終確定）は別途継続し、本決定には含めない。

## 2026-04-05: #85 judge 設計前提の論点スコープ分離

### Context

Issue #85 の議論で、judge 本体の設計論点から
db_api 実装要件（snapshot 整合）と dashboard 実装要件（直リンク URL）が派生した。
同一スレッドで扱うと責務境界が曖昧になり、DoD 判定が難しくなる懸念があった。

### Decision

論点を以下の3系統に分離する。

1. judge 本体の設計前提は #85 で扱う（入力取得、実行トリガー、run 内整合、JudgeResults 最小項目）
2. snapshot を成立させる db_api read 要件は #98 で扱う
3. 通知本文からの dashboard 直リンク URL 契約は Discussion #94 で扱う

### Why

- モジュール責務（judge/api/dashboard）と議論単位を一致させられる
- 実装順序が異なる論点を独立に進められる
- レビュー観点と完了条件をスレッドごとに明確化できる

### Consequence

judge 実装タスクは #85 の範囲で先行し、
db_api endpoint の詳細仕様は #98 で確定する。
dashboard URL スキーマは #94 で確定後、judge 通知テンプレートに取り込む。
