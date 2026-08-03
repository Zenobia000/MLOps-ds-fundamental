# Runbook：資料漂移告警

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **實例:** 每症狀一份

## 症狀

Evidently 漂移報告顯示某些欄位 `drift_detected = true`，或排程的漂移檢查失敗。

設定來源：`monitoring/evidently/drift_config.yaml`
偵測邏輯：`src/monitoring/drift.py`

---

## 先確認這不是假警報

**漂移告警最常見的成因不是模型出問題，是比較的基準本身就不對。** 依序排除：

### 1. reference 切法對不對？

reference 應該是「**上線當時的資料分布**」，是個**時間**的切法。
如果 reference 是按列序或按設備切出來的，你比較的是「不同設備」而不是「不同時間」——
各設備的感測基準天生就差好幾度，會**在還沒有任何真實漂移時就報 drift**。

```bash
# 檢查 reference 快照的組成
python3 -c "
import pandas as pd
df = pd.read_parquet('monitoring/evidently/reference/tabular_reference.parquet')
print('列數:', len(df))
print('設備組成:', sorted(df.machine_id.unique()) if 'machine_id' in df else '(已排除)')
"
```

兩邊的設備組成應該一致。不一致 → 重建 reference 快照，不要調門檻。

### 2. current 視窗夠不夠大？

`drift_config.yaml` 的 `schedule.current_window_hours` 預設 24 小時。
樣本太少時統計檢定不穩定，會忽紅忽綠。

### 3. 是不是只有單欄漂移、但整體旗標是 False？

`dataset_drift_share_threshold: 0.5` 表示**要超過半數欄位漂移**整體才亮燈。
本專案只監控 3 欄（temperature / vibration / current），推一欄 = 1/3 < 0.5 → 整體仍是 `False`。

> **只看整體旗標會漏掉單欄漂移。** 一個關鍵欄位漂掉往往就足以讓模型失效。
> 請以**逐欄結果**為準。

---

## 確認是真漂移之後

### 步驟 1：判斷是哪一種漂移

| 類型 | 表現 | 本專案抓得到嗎 |
| :--- | :--- | :---: |
| **covariate drift** | 輸入特徵分布變了（感測器老化、換料） | ✓ Evidently 直接抓 |
| **label drift** | 不良率本身變了 | △ 需標籤回流 |
| **concept drift** | 同樣的輸入，對應的結果變了 | ✗ **抓不到** |

**concept drift 是最危險的**：特徵分布看起來完全正常，但 X→y 的關係已經改變，
模型悄悄失效而所有監控都是綠的。本專案**沒有**標籤回流機制，所以偵測不到它。
若業務端回報「預測不準」但漂移報告全綠，優先懷疑這個。

### 步驟 2：評估業務影響，不要只看 p-value

K-S 檢定在樣本數大時**非常敏感**——平移 0.25 個標準差就會被標記。
統計上顯著 ≠ 業務上重要。先問：

- 這個幅度的變化，模型的預測真的變差了嗎？（有標籤的話直接算指標）
- 業務指標（誤報率、漏檢率）有動嗎？

沒有業務影響的顯著漂移，該做的是**調整靈敏度**（改用 PSI / Wasserstein，或加「連續 N 天才告警」），
而不是重訓模型。

### 步驟 3：決定處置

| 判斷 | 動作 |
| :--- | :--- |
| 感測器故障 / 資料管線壞掉 | **修上游**，不要重訓。用壞資料重訓會把錯誤學進模型 |
| 真實的環境變化，模型指標下降 | 觸發重訓（下方步驟 4） |
| 真實變化但指標沒掉 | 更新 reference 快照，記錄一次分布基準的變更 |
| 告警太敏感 | 調整 `drift_config.yaml` 的門檻或改檢定方法，**並記錄理由** |

### 步驟 4：觸發重訓（CT）

```bash
# 1) 重跑資料 → 特徵 → 訓練 → 評估
make pipeline            # dvc repro

# 2) 看品質門檻有沒有過
#    未過 → 不會註冊，服務端仍指向舊模型（這是保護，不是失敗）
```

重訓後**務必確認新模型真的比舊的好**再放行。漂移不等於「新資料訓出來的一定更好」。

---

## 完成後

- [ ] 在 MLflow 記下這次事件（哪幾欄漂、幅度、處置）
- [ ] 若更新了 reference，記錄新基準的時間範圍與理由
- [ ] 若調整了門檻，更新 `drift_config.yaml` 的註解說明為什麼
- [ ] 若是 concept drift 的疑慮，評估是否要建標籤回流管線

---

## 相關

- [`../architecture/sad.md` §9 風險與演進](../architecture/sad.md#9-風險與演進)
- 課程 m6 模組（Evidently 與監控四層）
- [`../../monitoring/evidently/README.md`](../../monitoring/evidently/README.md)
