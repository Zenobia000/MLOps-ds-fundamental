# Runbook：模型未通過品質門檻

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **實例:** 每症狀一份

## 症狀

訓練跑完，log 出現：

```
品質門檻：f1=0.0000 vs 門檻 0.7000（maximize）→ 未通過
未通過品質門檻，跳過 registry 註冊。
```

模型**沒有**被註冊，服務端的 `models:/<name>/Production` 仍指向舊模型。

---

## 先建立正確心態

**這不是故障，是門檻在做它該做的事。**

品質門檻的存在意義就是「不夠好的模型上不了線」。看到它擋下來，
第一反應應該是「為什麼這次訓練不夠好」，而不是「怎麼讓它過」。

> **禁止**：為了讓流程走完而調低 `min_threshold` 或繞過註冊。
> 那等於拆掉煞車來解決煞車在響的問題。真要調門檻，必須有業務理由並記錄。

---

## 診斷

### 步驟 1：確認門檻設定與實際指標

```bash
# 目前門檻
grep -A3 "^evaluation:" conf/train/default.yaml

# 這次跑出什麼（MLflow）
mlflow ui --backend-store-uri sqlite:///mlflow.db   # 開瀏覽器看最近一次 run
```

判定規則（`src/training/evaluate.py`）：

| 主指標 | 判定方向 |
| :--- | :--- |
| `rmse` / `mae` / `mse` | `value <= threshold`（越小越好） |
| 其他（f1、auc…） | `value >= threshold` |
| **主指標不存在** | **直接判不通過**（fail-safe） |

### 步驟 2：分辨是哪一類問題

| 指標樣態 | 最可能的原因 |
| :--- | :--- |
| **f1 = 0.0** | 模型把所有樣本預測成同一類。通常是玩具/合成資料太小或標籤退化 |
| 指標略低於門檻 | 真的是模型品質問題（特徵、超參、資料量） |
| **主指標根本沒出現在 metrics** | 評估流程沒算出它——設定的 `primary_metric` 名稱與實際算出的指標不一致 |
| 分數異常高（例如 1.0） | **反而要警戒**：可能是資料洩漏，見下 |

### 步驟 3：檢查資料

```bash
# 資料契約測試會抓出多數資料面問題
make test
```

特別注意這兩個測試（`tests/data/test_expectations.py`）：

- `test_label_not_degenerate` — 標籤全 0 或全 1，模型學不到東西
- `test_no_duplicate_entity_timestamp` — 重複的 (設備, 時刻) 會讓 join 產生歧義

### 步驟 4：分數太高也要查

如果指標突然接近完美，先懷疑**時間穿越**：特徵是不是用到了標籤時刻之後的資訊。
本專案靠 Feast 的 `get_historical_features` 保證 point-in-time 正確性
（[ADR-002](../architecture/adr/ADR-002-feast-as-feature-store.md)），
但若有人繞過 Feast 自己算特徵，保證就失效了。

對照示範：洩漏版 AUC 0.940 vs 正確版 0.791
（[`02_leakage_viz.ipynb`](../../../../modules/m3-feature-store/sandbox/02_leakage_viz.ipynb)）。

---

## 處置

| 原因 | 動作 |
| :--- | :--- |
| **玩具資料**（capstone 預設狀態） | **這是預期行為**，不必處理。vision 模型在合成資料下 f1 常為 0，門檻正確擋下註冊 |
| 標籤退化 / 資料品質問題 | 修上游資料，不要調門檻 |
| `primary_metric` 名稱寫錯 | 對齊 `conf/train/default.yaml` 與 `src/training/evaluate.py` 實際輸出的指標名 |
| 模型真的不夠好 | 調特徵 / 超參（`make tune` 跑 Optuna）/ 增加資料量，重訓 |
| 業務上確認較低門檻可接受 | 修改 `min_threshold` **並在 commit message 寫清楚理由**；同步更新 Model Card 的適用邊界 |

---

## 完成後

- [ ] 指標達標且模型成功註冊
- [ ] 確認服務端取得的 `model_version` 已更新（見 [serving runbook](runbook-serving-degraded.md)）
- [ ] 若調整過門檻，更新 [`../../governance/model_cards/`](../../governance/model_cards/) 的適用邊界與限制
- [ ] 若根因是資料，補一條資料契約測試，讓下次更早被擋下

---

## 相關

- [`../qa/test_plan.md` §4 品質門檻](../qa/test_plan.md#4-品質門檻自動擋交付)
- [`../architecture/sad.md` §5.1 訓練 → 註冊](../architecture/sad.md#51-訓練--註冊含品質門檻)
- [`../../governance/model_cards/`](../../governance/model_cards/)
