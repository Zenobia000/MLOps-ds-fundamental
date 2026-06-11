# sandbox/github-actions — push 觸發 pytest 的最小 CI（階 10）

> 一次一動詞：這個沙盒只教 GitHub Actions 的最小骨架 `on: push` -> 跑 `pytest`。
> matrix、cache、OIDC、self-hosted runner **全部延後**。

## 這個沙盒在教什麼

GitHub Actions 是 GitHub 內建的 CI/CD 工具。它的最小心智模型只有三個欄位：

| 欄位 | 問題 | 本範例的答案 |
| :--- | :--- | :--- |
| `on` | **什麼時候**跑？ | `push`（任何 push 都觸發） |
| `runs-on` | 在**哪台機器**跑？ | `ubuntu-latest`（GitHub 提供的雲端機） |
| `steps` | 跑**哪些指令**？ | checkout -> 裝 Python -> 裝 pytest -> 跑 pytest |

核心觀念：**測試失敗 = workflow 變紅 = 這次提交「沒通過」**。
這就是 CI 的價值——把「壞 code 進不來」這件事自動化，不靠人記得手動跑測試。

## 真實使用時：檔案要放哪裡？

GitHub **只**會偵測 repo 根目錄下 `.github/workflows/` 裡的 YAML 檔。
本沙盒的 `ci.yml` 放在這裡只是給你「看與改」的範本。要讓它真的生效：

```bash
# 在你的 repo 根目錄執行
mkdir -p .github/workflows
cp modules/m5-automation/sandbox/github-actions/ci.yml .github/workflows/ci.yml

git add .github/workflows/ci.yml
git commit -m "ci: add pytest workflow on push"
git push
```

push 之後，打開 GitHub 上的 **Actions** 分頁，就會看到這條 workflow 自動跑起來：
綠勾 = 測試通過，紅叉 = 有測試失敗（點進去看哪一條 assert 掛了）。

> 注意：本範本裡的測試路徑寫死成 `modules/m5-automation/sandbox/tests/`。
> 搬到自己 repo 後，把它改成你實際放測試的資料夾（例如 `tests/`）。

## 怎麼在本機先驗證（push 前自測）

CI 上跑的就是 pytest，所以你可以在本機先跑一模一樣的指令，確認綠了再 push：

```bash
pip install pytest
python -m pytest modules/m5-automation/sandbox/tests/ -v
```

## 明確延後（先不要學）

- **matrix**：同時在多個 Python 版本 / OS 上跑。
- **cache**：快取 pip 依賴，加速 CI。
- **OIDC / secrets**：安全地連雲端做部署。
- **self-hosted runner**：用自己的機器當 runner。

## 自我檢核

1. GitHub 偵測 workflow 檔的「唯一」資料夾是哪個？放錯地方會怎樣？
2. 為什麼「pytest 失敗 -> step 回傳非 0 -> workflow 變紅」能達到品質門檻 gate？
3. push 前想先確認會綠，你會在本機跑哪一行指令？
