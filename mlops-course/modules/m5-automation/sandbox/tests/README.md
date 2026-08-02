# sandbox/tests — pytest 最小範例（階 10）

> 完整說明（怎麼用 / 怎麼寫 / 怎麼接 CI）：[`../../pytest-introduction.md`](../../pytest-introduction.md)  
> 概念圖：[`../../assets/m5-pytest-ci.png`](../../assets/m5-pytest-ci.png)

## 怎麼跑

```bash
# 在 m5-automation/sandbox/ 下
pip install pytest
python -m pytest tests/ -v
```

## 這個資料夾有什麼

| 檔案 | 內容 |
| :--- | :--- |
| `test_sample.py` | `accuracy` 單元測試 + 玩具資料 schema/值域驗證 |

通過後，對照 [`../github-actions/ci.yml`](../github-actions/ci.yml)：CI 最後一步跑的是同一套 pytest。
