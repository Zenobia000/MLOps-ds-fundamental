# m1 沙盒（Layer 1）

本沙盒只做一件事：用純 sklearn 訓一個**乾淨 baseline**（iris + LogisticRegression），固定 seed 確保可重現，**不碰任何 MLOps 工具**。

| 編號 | 檔案 | 學到的最小動詞 |
| :--- | :--- | :--- |
| 01 | `01_baseline_iris.py` | `read_csv` → `train_test_split(seed)` → `fit` → `accuracy` |

跑法：

```bash
python 01_baseline_iris.py
```

每次跑都應印出**一模一樣**的 accuracy——這就是「可重現」的第一個成功經驗。
