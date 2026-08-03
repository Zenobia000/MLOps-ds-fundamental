# ADR-005：部署管線的守門員必須基於真實訊號

> **版本:** v2.0 | **更新:** 2026-08-03 | **狀態:** 已接受（v1 的佔位問題已解除）
> **Owner:** Capstone 維護者
> **實例:** 每決策一份

## 脈絡

`pipelines/deployment_pipeline.py` 的 canary 流程：
`resolve_model → build_image → canary_probe → promote_or_rollback`。

**v1（2026-08-03 之前）的狀態**：`canary_probe` 是佔位，直接回傳 `1.0`：

```python
logger.info("[佔位] canary 探測 %s 假定成功。", image_tag)
return 1.0
```

於是 `promote_or_rollback` 拿它跟門檻（0.95）比，**永遠判定 promote**——
rollback 那條邊在程式碼裡存在，但執行時走不到。

一個回傳固定成功率的探測，是最危險的那種假象：管線看起來綠燈、有 log、跑得通，
但實際上沒有任何保護作用。複製這份骨架到真實專案的人，等於部署了一個永遠說 OK 的守門員。

## 決策

**實作真實探測**（`src/serving/healthcheck.py`），並確立三條規則：

### 1. `degraded` 算失敗

服務的 `/healthz` 有三種結果：`ok`、`degraded`（活著但有模型沒載到）、連不上。

canary 要回答的**不是**「服務活著嗎」，而是**「這個版本能承接生產流量嗎」**。
一個少了影像模型的版本顯然不行。所以只有 `status == "ok"` 算成功。

> 這與 liveness 探測的判準不同：liveness 該容忍 degraded（不要重啟一個還能服務一半的實例），
> canary 不該。同一個端點，兩種讀法——這個區別值得在課堂上停一下。

### 2. 預設值站在保守的一邊

探測不到、模組載不到、拋例外——**一律回 `0.0`**，觸發 rollback。

「連不上」與「健康」是相反的結論。預設值站錯邊，管線就會在最不該推的時候推上線。

### 3. 連打多次，不採信單次結果

預設探 5 次取成功率。單次成功不足以判斷穩定性。

## 為什麼不是「保留佔位並標示清楚」（v1 的決策）

v1 認為真實探測需要可導流量的環境（K8s / service mesh），超出「本機 compose 跑完」的教學邊界。

**那個判斷過度了。** 一個 HTTP 健康探測只需要標準函式庫的 `urllib` 與一個位址，
位址由 `CANARY_BASE_URL` 環境變數提供。它不需要 K8s，也不需要雲端帳號——
本機起服務就能驗證兩條分支。實作成本遠低於當初的估計。

保留一個永遠說 OK 的守門員，代價則是任何複製這份骨架的人都繼承了一個假的保護。

## 後果

**正面**

- rollback 分支**走得到了**。服務未啟動時探測回 `0.0` → 判定 rollback，已由測試覆蓋
- `degraded` 與 `ok` 的差別成為可執行的教材，不只是文件上的說明
- 只用標準函式庫，推論映像不需要為此多一個依賴

**負面 / 代價**

- `image_tag` **推導不出服務位址**。探測位址得另外由環境變數或設定提供——
  這是真實部署的普遍問題（建置產物與執行位址是兩件事），課堂上要講清楚
- 探測的是「服務健康」，不是「模型品質」。一個載入成功但預測很爛的模型仍會通過 canary。
  模型品質由訓練期的品質門檻負責（見 [`../../qa/test_plan.md` §4](../../qa/test_plan.md#4-品質門檻自動擋交付)），
  兩道關卡守不同的東西

## 仍未解除的部分

`resolve_model` 的後援 URI `models:/smart-factory/latest` 仍是硬編字串。
`src/serving/registry.py` 的 `resolve_latest` 已實作（alias → stage → 最新版本號三層解析），
但 registry 完全不可用時仍會落到那個佔位字串。真實部署應在此處失敗而非猜測。

## 更正紀錄

v1 曾聲稱「`infra/terraform/` 只有 README，沒有可用的 IaC」。**那是錯的**——
`main.tf` / `variables.tf` / `outputs.tf` 一直都在，且 `terraform validate` 通過。
該結論來自一次只搜尋 `*.md` 的檔案掃描，是推論錯誤而非事實。詳見
[`../sad.md` §7](../sad.md#7-部署視圖)。

## 相關

- [`../sad.md` §5.3 部署決策](../sad.md#9-風險與演進)
- [`../../design/lld.md` §5.3](../../design/lld.md#5-狀態機設計契約)
- [`../../ops/runbook-serving-degraded.md`](../../ops/runbook-serving-degraded.md)
