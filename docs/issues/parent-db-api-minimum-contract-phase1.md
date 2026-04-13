Title: Parent - DB API Minimum Contract Draft (Phase 1)

Background

- dashboard read-only baseline と judge 最小実装に必要な read API を Phase 1 として固定し、実装を分割して進める。
- 契約基準は docs/db-api-minimum-contract.md と docs/db-api-endpoints.md を正とする。

Goal

- db_api の read 契約 5 endpoint を実装可能な粒度で子Issueに分割し、順次クローズする。

Scope

- 子Issueの管理と依存関係の整理
- 仕様差分発生時のドキュメント同期
- 実装完了条件とテスト完了条件の統一

Out of Scope

- 閾値更新 write 契約
- governance 承認ワークフローの実装
- 認証認可方式の詳細設計

Child Issues

- [ ] Child - GET /charts
- [ ] Child - GET /charts/active
- [ ] Child - GET /charts/history
- [ ] Child - GET /judge/results
- [ ] Child - GET /judge/results/{result_id}

Definition of Done

- 子Issue 5件がクローズされている
- docs/db-api-minimum-contract.md と実装が整合している
- docs/db-api-endpoints.md の status/tracking が更新されている
- 契約テストが追加され、既存 ingest 系回帰が確認されている

Notes for Tracking

- 子Issue作成後、上記チェックボックスを #番号付きリンクに置換する。
- 最終決定や仕様変更が出た場合は docs/decision-log.md に同一PRで反映する。
