# workspace/ — 你的漸進整合主線（Layer 2）

> 這裡住著**整門課唯一、跨模組累積長大的主線專案**。
> 每個模組先在 sandbox 練工具，再 `make workspace-mN` **解鎖填空模板**，對照 sandbox 把 `TODO(MN-` 填完。

---

## 怎麼開始（不要從空白檔硬寫）

在 `mlops-course/`：

```bash
make m1                 # 看本模組 README
# …先跑完 sandbox…
make workspace-m1       # 把填空骨架解鎖進這裡（不覆蓋已有檔）
# 打開 train.py，搜尋 TODO(M1- 依序填
python train.py
```

之後每個模組同樣：`make workspace-m2` … `make workspace-m6`。

| 指令 | 解鎖進 workspace 的內容 |
| :--- | :--- |
| `make workspace-m1` | `train.py` |
| `make workspace-m2` | `train_tracking.py`、`conf/params.yaml` |
| `make workspace-m3` | `prepare_features.py`、`feature_repo/` |
| `make workspace-m4` | `services/`（FastAPI + Dockerfile） |
| `make workspace-m5` | `flow.py`、`tests/` |
| `make workspace-m6` | `monitoring/`（漂移報告 + Model Card） |

> **規則**：模板只在檔案尚不存在時複製；你填過的進度不會被 `make workspace-mN` 蓋掉。

---

## 這跟 sandbox 有什麼不同？

| | `modules/mN/sandbox/` | `workspace/`（這裡） |
| :--- | :--- | :--- |
| 性質 | 孤立、玩具資料、可丟可重來 | 唯一、持續累積、不能砍掉 |
| 目的 | 「我在學這一個工具怎麼用」 | 「我把學會的工具接到我的專案」 |
| 心態 | Layer 1 單點精熟 | Layer 2 填空整合 |

> 規則：**先在 sandbox 用玩具資料把工具玩熟，再 `make workspace-mN` 接到主線。**

---

## 隨課程長大的預期結構

```
workspace/
├── train.py              # M1：純 sklearn baseline（填空）
├── train_tracking.py     # M2：MLflow + Optuna（填空）
├── conf/params.yaml      # M2：參數抽出
├── prepare_features.py   # M3
├── feature_repo/         # M3：Feast
├── services/             # M4：API + Docker
├── flow.py               # M5：Prefect
├── tests/                # M5：CI
└── monitoring/           # M6：Evidently + Model Card
```

---

## 卡住了怎麼辦？

每個模組結束時的「已知良好狀態」在 `../checkpoints/after-mN/`：

```bash
cp -r ../checkpoints/after-m2/. ./        # 例：重置到 M2 結束狀態
```

若只是想重拿**空白填空模板**（會覆蓋你自己的檔——慎用）：

```bash
rm workspace/train.py    # 只刪你要重來的那個檔
make workspace-m1        # 再解鎖一次
```

## 從哪開始？

`../modules/m1-foundations/README.md` → sandbox → `make workspace-m1`。
