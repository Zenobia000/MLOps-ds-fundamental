# ADR-005：部署管線保留佔位，不假裝可用於生產

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 已接受（暫時性）
> **Owner:** Capstone 維護者
> **實例:** 每決策一份

## 脈絡

`pipelines/deployment_pipeline.py` 有完整的 canary 流程骨架：
`resolve_model → build_image → canary_probe → promote_or_rollback`。

但 `canary_probe` 目前是**佔位**：

```python
logger.info("[佔位] canary 探測 %s 假定成功。", image_tag)
return 1.0
```

它不會真的送流量、不會真的量成功率，回傳固定的 `1.0`。
於是 `promote_or_rollback` 拿它跟門檻（預設 0.95）比，**永遠判定 promote**。

`infra/terraform/` 同樣只有 README，沒有可用的 IaC。

## 決策

**保留佔位，但在程式碼、SAD 與本 ADR 三處明確標示它不可信**，
而不是：(a) 刪掉整個流程，或 (b) 讓它看起來像能用。

## 理由

**為什麼不刪掉**
這個骨架有教學價值——它示範了部署管線該有哪些步驟、canary 的決策點長什麼樣、
門檻該從哪裡讀（`config['deploy']['canary_threshold']`）。刪了學員就看不到形狀。

**為什麼不補成真的**
真實的 canary 探測需要：可導流量的部署環境（K8s 或 service mesh）、真實請求來源、
指標收集管線。這些的前提是 IaC 與雲端環境，超出「本機 compose 跑完」的教學邊界。
硬做會讓專案從「能在筆電上跑完的教材」變成「需要雲端帳號才能開始的專案」。

**為什麼一定要標示**
一個回傳固定 `1.0` 的成功率探測，是**最危險的那種假象**：
它讓整條管線看起來綠燈、跑得通、有 log，但實際上沒有任何保護作用。
如果有人把這份骨架複製到真實專案而沒注意到，等於部署了一個「永遠說 OK 的守門員」。

## 後果

**正面**

- 學員看得到完整的部署管線形狀，也學到「佔位要標示」這件事本身
- 不引入雲端依賴，教材維持可在筆電跑完

**負面 / 代價**

- `make` 沒有對應的部署 target，deployment pipeline 只能手動觸發
- **任何人要把這份骨架用於生產，必須先實作 `canary_probe`**。這是硬性前提，不是建議

## 待辦（解除此 ADR 的條件）

1. 實作真實 `canary_probe`：對 canary 版本送實際流量，量錯誤率 / 延遲 / 業務指標
2. 補 `infra/terraform/` 的實際 IaC
3. 兩者完成後，把本 ADR 狀態改為「已取代」，並更新 [`../sad.md` §9 風險](../sad.md#9-風險與演進)

## 相關

- [`../sad.md` §9 風險與演進](../sad.md#9-風險與演進)
- [`../../README.md` §4 已知的教學簡化](../../README.md)
