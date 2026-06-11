# capstone/ — 完整智慧工廠 MLOps（Layer 3，最後才解鎖）

> ⚠️ **這個資料夾到 M6 才解鎖。** 前面的模組請先別進來——一開始就看到完整骨架會造成認知負荷。

到了 M6，你對每個工具都已親手用過。此時才在這裡組出**生產級的端到端 pipeline**：
資料版本化 → Feast 特徵 → MLflow 訓練/Optuna 調參/註冊 → BentoML 服務 → GitHub Actions 自動化 → Evidently 監控 → Model Card。

完整生產級資料夾結構見：[`../../docs/project-structure.md`](../../docs/project-structure.md)
（`capstone/smart-factory-mlops/` 內部 = 該文件描述的結構）

教學的終點 = 生產的起點：你在這裡第一次看到生產結構時，每個零件都已會用，不再是負擔。
