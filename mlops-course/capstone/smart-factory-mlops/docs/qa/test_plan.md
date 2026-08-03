# 測試計畫 — Smart Factory MLOps

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **語域:** L2（橋接）
>
> **定位**：測什麼、測到什麼程度、什麼情況下擋下交付。
> **實例:** 單例

## 目錄

- [1. 為什麼 ML 系統的測試不一樣](#1-為什麼-ml-系統的測試不一樣)
- [2. 四層測試](#2-四層測試)
- [3. 現有測試盤點](#3-現有測試盤點)
- [4. 品質門檻（自動擋交付）](#4-品質門檻自動擋交付)
- [5. 執行方式](#5-執行方式)
- [6. 覆蓋缺口](#6-覆蓋缺口)

---

## 1. 為什麼 ML 系統的測試不一樣

一般軟體只有程式碼會錯；ML 系統有**三個**會錯的東西，而且後兩個不會拋例外：

| 會錯的東西 | 症狀 | 傳統測試抓得到嗎 |
| :--- | :--- | :---: |
| 程式碼 | 例外、錯誤結果 | ✓ |
| **資料** | schema 變了、分布飄了、標籤退化 | ✗ |
| **模型** | 指標下降、對特定切片失效 | ✗ |

所以本專案除了單元 / 整合測試，還有**資料契約測試**與**品質門檻**兩層——
它們的目的不是驗證程式跑得動，而是**擋下「跑得動但結果是錯的」**。

---

## 2. 四層測試

| 層 | 位置 | 測什麼 | 失敗代表 |
| :--- | :--- | :--- | :--- |
| **單元** | `tests/unit/` | 特徵計算、指標計算的數學正確性 | 程式邏輯錯 |
| **整合** | `tests/integration/` | 訓練管線能端到端跑完並產出合理結果 | 元件接不起來 |
| **資料契約** | `tests/data/` | 資料的 schema、範圍、標籤健康度 | 上游資料變了 |
| **品質門檻** | `src/training/evaluate.py` | 模型指標是否達標 | 模型不夠好，**不准註冊** |

前三層由 pytest 在 CI 跑；第四層在訓練流程內即時判定。

---

## 3. 現有測試盤點

**共 26 個測試**，全部使用合成資料（`tests/conftest.py` 的 `_synthetic_sensors`），
不依賴外部資料檔——所以 CI 不需要準備資料就能跑。

### 3.1 單元：特徵（`tests/unit/test_features.py`，4 個）

| 測試 | 守住什麼 |
| :--- | :--- |
| `test_rolling_mean_does_not_leak_across_machines` | **最重要的一個**：滾動特徵不可跨設備汙染。groupby 漏掉就會讓 machine_01 的歷史混進 machine_02 |
| `test_rolling_mean_window_values` | 視窗算出來的數值正確 |
| `test_rolling_std_is_non_negative` | 標準差不可為負（數值穩定性） |
| `test_src_features_build_contract` | 特徵建構的介面契約沒被改壞 |

### 3.2 單元：指標與門檻（`tests/unit/test_metrics.py`，6 個）

| 測試 | 守住什麼 |
| :--- | :--- |
| `test_perfect_classifier_scores_are_one` | 完美分類器指標應為 1（邊界情況） |
| `test_random_classifier_rocauc_near_half` | 亂猜的 ROC AUC 應接近 0.5 |
| `test_pr_auc_rewards_minority_ranking` | PR-AUC 對少數類排序敏感（不平衡資料下比 ROC 更有意義） |
| `test_f1_is_harmonic_mean_of_precision_recall` | F1 定義正確 |
| `test_src_evaluate_classification_contract` | 評估函式介面契約 |
| `test_src_quality_gate_blocks_low_metric` | **品質門檻真的會擋**——低於門檻必須判定不通過 |

### 3.3 整合（`tests/integration/test_training_pipeline.py`，4 個）

| 測試 | 守住什麼 |
| :--- | :--- |
| `test_mini_training_beats_random` | 小規模訓練的結果要優於亂猜——**擋下「跑得動但沒學到東西」** |
| `test_src_training_entrypoint_signature` | 訓練進入點簽章穩定（DVC stage 靠它呼叫） |
| `test_orchestration_training_flow_end_to_end` | Prefect flow 能端到端跑完 |
| `test_feature_flow_produces_features` | 特徵管線真的產出特徵 |

### 3.4 資料契約（`tests/data/test_expectations.py`，10 個）

這一層對應課程 m5 教的「ML 專屬測試」，是傳統測試沒有的：

| 測試 | 守住什麼 |
| :--- | :--- |
| `test_required_columns_present` | 必要欄位存在 |
| `test_entity_and_timestamp_naming` | **entity 與時間欄命名對齊**（`machine_id` / `event_timestamp`）——這是跨 Feast、訓練、服務的契約 |
| `test_no_nulls_in_core_columns` | 核心欄位無缺值 |
| `test_label_is_binary` | 標籤是二元 |
| `test_label_not_degenerate` | **標籤未退化**——全 0 或全 1 的資料能訓練出「高準確率」的廢模型 |
| `test_sensor_values_in_range` | 感測值在物理合理範圍（與 API schema 的範圍一致） |
| `test_timestamp_parseable_and_monotonic_per_machine` | 每台設備的時間戳單調遞增——時序特徵的前提 |
| `test_no_duplicate_entity_timestamp` | 同一 (設備, 時刻) 不重複——重複會讓 point-in-time join 產生歧義 |

---

## 4. 品質門檻（自動擋交付）

`src/training/evaluate.py` 的 `quality_gate()`，設定在 `conf/train/default.yaml`：

```yaml
evaluation:
  primary_metric: f1
  min_threshold: 0.70
```

**判定規則**

| 情況 | 行為 |
| :--- | :--- |
| 指標屬 `{rmse, mae, mse}` | 用 `value <= threshold`（越小越好） |
| 其他指標 | 用 `value >= threshold` |
| **主指標不存在於結果中** | **直接判不通過**（fail-safe） |

最後一條是刻意的：忘了算指標不該被當成通過。**沉默的缺席不等於成功。**

**未通過的後果**：指標照樣記進 MLflow（保留證據），但**跳過模型註冊**——
於是服務端的 `models:/<name>/Production` 仍指向舊模型，壞模型上不了線。

> 玩具資料下 vision 模型的 f1 常為 0，門檻會正確擋下註冊。
> **那是門檻在正常工作，不是壞掉。**

---

## 5. 執行方式

```bash
make test          # 全部測試 + 覆蓋率
make lint          # ruff + black（不改檔）
```

CI（`.github/workflows/ci.yml` 的 `capstone` job）會：

1. 用 `uv sync --extra dev --frozen` 從 `uv.lock` 建環境（版本與本機一致）
2. 跑 26 個測試
3. 跑 3 本 notebook（`--nbmake`，確認教材可從頭跑到尾）

任一失敗 → job 變紅 → 壞的東西進不了 main。

---

## 6. 覆蓋缺口

**誠實列出目前沒測到的**，以免誤以為有保障：

| 缺口 | 風險 | 為什麼還沒補 |
| :--- | :--- | :--- |
| **服務層端點沒有測試** | schema 改動、降級邏輯、503 行為都沒被自動驗證 | 需要起 BentoML 服務或 mock，工作量較大。**這是最該優先補的一項** |
| **前處理一致性沒有跨側比對** | 訓練用 torchvision transform、服務用 numpy，兩邊算不一樣會靜默降準（[ADR-004](../architecture/adr/ADR-004-onnx-for-vision-serving.md) 已標為最大風險） | 需要固定測試影像與期望張量 |
| **模型行為測試（不變性 / 方向性）** | 例：溫度升高時故障機率不該下降 | 課程 m5 有提到概念，capstone 未實作 |
| **部署管線沒有測試** | `canary_probe` 本身就是佔位（[ADR-005](../architecture/adr/ADR-005-deployment-placeholders.md)），測了也沒意義 | 待實作真實探測後一併補 |
| **時序 / 影像模型只有 smoke 級覆蓋** | 玩具資料下指標無意義 | 接真實資料集屬 Capstone 練習範圍 |
