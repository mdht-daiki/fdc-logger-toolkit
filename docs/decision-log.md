# Decision Log

## 2026-04-04: 論点1 Charts テーブルの扱い

### Context

Discussion #68 で、Charts と ChartsV2 が並存している状態を解消し、
chart 閾値の正本をどちらに寄せるかを決める必要があった。
dashboard/judge の本格実装前であり、互換維持コストを最小化できるタイミングだった。

### Decision

Charts は廃止し、ChartsV2 + ActiveChartSet を正本として一本化する。

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

1. 装置異常を検知 → DB 直変更（change_source='manual' 必須、change_reason に理由記述）
2. ChartsHistory に記録される
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
