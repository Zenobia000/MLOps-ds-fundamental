# checkpoints/ — 各模組結束的已知良好快照（救援 / 補課）

> 每個 `after-mN/` 是 `workspace/` 在**該模組結束時**的「已知良好狀態」快照。
> 用途有三：**卡住重置、缺課補進度、對照「我接對了沒」**。

---

## 用法

**卡住或弄壞了 → 一鍵重置 workspace：**
```bash
cp -r checkpoints/after-m2/. workspace/        # 重置到 M2 結束狀態
```

**缺了某一模組 → 直接從上一個 checkpoint 接著做：**
```bash
cp -r checkpoints/after-m3/. workspace/        # 跳到 M4 前的起點
```

**想知道自己接對沒 → 跟 checkpoint 比對：**
```bash
diff -r workspace/ checkpoints/after-m2/
```

---

## 各快照接上了哪些工具

| 快照 | workspace 狀態 | 技能階梯 |
| :--- | :--- | :--- |
| `after-m1/` | 純 sklearn baseline + Git | 階 0 |
| `after-m2/` | + MLflow 追蹤 + Optuna 調參 + DVC | 階 1–3 |
| `after-m3/` | + Feast 特徵 | 階 4 |
| `after-m4/` | + BentoML 服務 / PyTorch 模型 | 階 5–8 |
| `after-m5/` | + Prefect 編排 + GitHub Actions CI | 階 9–10 |

> 沒有 `after-m6/`：M6 的整合會收束到 `../capstone/`（Layer 3 完整專案），不再回寫 workspace。

---

## 維護者守則

- 每次更新教材，**必須**同步更新對應 checkpoint，並確認「乾淨環境跑得起來」。
- checkpoint 內**不放**產物（mlruns/、*.db、*.parquet 等已被 .gitignore 忽略）。
- 目前各 `after-mN/` 為占位骨架；建置教材時用 `make checkpoint-mN` 從通過的 workspace 產生。
