# datasets/ — 共用玩具資料（Layer 1 沙盒專用）

> 本資料夾放「最小、可重現」的玩具資料，給每個工具的**初次接觸沙盒**使用。
> 設計依據：[teaching-progression.md](../../docs/teaching-progression.md) 三定律之一 ——**玩具資料先行**。

---

## 為什麼玩具資料先行？

學一個新工具時，認知頻寬有限。如果同時要「理解複雜資料」又要「搞懂工具怎麼用」，注意力會被資料吃掉，學不到工具本身。

所以這裡的資料**刻意小、欄位刻意少、特徵與標籤的關係刻意直白**：

- 你一眼就能看懂資料長什麼樣，不用花時間做 EDA。
- 工具一定跑得快（資料小），縮短「改一下→看結果」的回饋迴圈。
- 沙盒之間零依賴，每份資料都可以「丟掉重來」，不怕弄壞。

> 真實、複雜的智慧工廠資料，留到最後的 `capstone/`（Layer 3）才登場。
> 那時候所有工具都熟了，重點變成「怎麼設計、為什麼這樣組」，而非「工具怎麼用」。

---

## 三個資料集

| 檔案 | 用途 | 對應沙盒 | 列數 |
| :--- | :--- | :--- | :--- |
| `iris.csv` | **分類沙盒** | m1 baseline、m2 MLflow / Optuna | 150 |
| `diabetes.csv` | **Feast 沙盒** | m3 Feast 特徵存儲 | 768 |
| `toy_sensors.csv` | **時序 / 漂移沙盒** | m3 Feast 時序、m6 Evidently 漂移偵測 | 750 |

### `iris.csv` — 分類沙盒
sklearn 內建鳶尾花。4 個數值特徵、3 個花種、無缺值、類別平衡。
最經典的玩具分類資料，用來「先把模型跑起來、把工具接上去」。

| 欄位 | 說明 |
| :--- | :--- |
| `sepal_length` / `sepal_width` | 花萼長 / 寬（cm） |
| `petal_length` / `petal_width` | 花瓣長 / 寬（cm） |
| `target` | 類別標籤（0 / 1 / 2，建模用） |
| `target_name` | 類別名稱（setosa / versicolor / virginica，人眼閱讀用） |

### `diabetes.csv` — Feast 沙盒
皮馬印第安人糖尿病資料（沿用既有檔，**非**程式重產）。
表格型、有明確 `Outcome` 標籤，欄位數適中，適合練「定 entity + feature view → 取 historical / online 特徵」。

### `toy_sensors.csv` — 時序 / 漂移沙盒
自製的極簡感測時序：5 台機台 × 150 個時間點 = 750 列，每台每小時一筆。

| 欄位 | 說明 |
| :--- | :--- |
| `machine_id` | 機台代號（Feast 的 **entity** 鍵） |
| `event_timestamp` | 事件時間（Feast 的**時間**鍵） |
| `temperature` | 溫度（°C） |
| `vibration` | 振動（mm/s） |
| `current` | 電流（A） |
| `failure` | 是否故障（0 / 1 標籤） |

故障規則刻意簡單：**溫度高 + 振動大 → 故障機率上升**，學生能一眼看出「特徵 → 標籤」的關係。
這份資料同時帶有時間鍵，所以既能餵 Feast（時序特徵），也能餵 Evidently（漂移偵測）。

---

## 怎麼重新產生

`iris.csv` 與 `toy_sensors.csv` 由腳本產生；`diabetes.csv` 是既有檔，沿用不重產。

```bash
cd datasets
python make_datasets.py
```

腳本特性：
- **可重現**：固定 `SEED = 42`（numpy `default_rng`），同一份程式每次重跑得到**完全相同**的資料。
- **路徑無關**：輸出一律寫到本資料夾，不管你從哪裡呼叫。
- **零外部依賴**：只用 `numpy` / `pandas` / `scikit-learn`。

> 想做「資料漂移」實驗？`make_datasets.py` 裡 `make_toy_sensors` 的 docstring
> 有一段「可注入的漂移選項」註解，示範如何對後半段時間的特徵動手腳，
> 製造參考期 vs 當前期的分布差異，給 m6 Evidently 沙盒使用。
