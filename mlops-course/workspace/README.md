# workspace/ — 你的漸進整合主線（Layer 2）

> 這裡住著**整門課唯一、跨模組累積長大的主線專案**。
> 每個模組的「整合任務（B 段）」都往這裡加**一個**剛學會的工具，看著它一步步長成真正的 MLOps pipeline。

---

## 這跟 sandbox 有什麼不同？

| | `modules/mN/sandbox/` | `workspace/`（這裡） |
| :--- | :--- | :--- |
| 性質 | 孤立、玩具資料、可丟可重來 | 唯一、持續累積、不能砍掉 |
| 目的 | 「我在學這一個工具怎麼用」 | 「我把學會的工具接到我的專案」 |
| 心態 | Layer 1 單點精熟 | Layer 2 漸進整合 |

> 規則：**先在 sandbox 用玩具資料把工具玩熟，再回 workspace 接到你的主線。** 不要在還沒玩熟工具前就硬接。

---

## 隨課程長大的預期結構（會逐模組長出來）

```
workspace/
├── train.py            # M1 起點：純 sklearn baseline
├── conf/               # M1+：參數抽出來，不寫死
├── （M2 後）           # + MLflow 追蹤、Optuna 調參、DVC 版本化
├── feature_repo/       # （M3 後）+ Feast 特徵
├── service.py          # （M4 後）+ BentoML 服務
├── flow.py             # （M5 後）+ Prefect 編排、CI gate
└── monitoring/         # （M6 後）+ Evidently 漂移、Model Card
```

| 模組結束 | workspace 新增了什麼 | 技能階梯 |
| :--- | :--- | :--- |
| M1 | `train.py` baseline + Git | 階 0 |
| M2 | MLflow 追蹤 + Optuna 調參 + DVC | 階 1–3 |
| M3 | Feast 特徵接進訓練 | 階 4 |
| M4 | 模型包成可呼叫 API | 階 5–8 |
| M5 | Prefect 編排 + GitHub Actions | 階 9–10 |
| M6 | Evidently 監控 + 治理（收束到 capstone） | 階 11 |

---

## 卡住了怎麼辦？

每個模組結束時 workspace 的「已知良好狀態」都存在 `../checkpoints/after-mN/`。卡住就重置：

```bash
cp -r ../checkpoints/after-m2/. ./        # 例：重置到 M2 結束狀態
```

## 從哪開始？

去 `../modules/m1-foundations/README.md` 讀「整合任務」段，在這裡建出你的第一支 `train.py`。
