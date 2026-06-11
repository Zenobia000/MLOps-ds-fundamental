# DVC 沙盒：把「一個 CSV」做版本控制（技能階梯 階 3）

> 最小可用動詞：`init` / `add` / `push` / `checkout`。
> 目標：讓「同一個 Git commit」永遠對應「同一份資料」——程式碼與資料一起被版本化。
> 玩具資料：`datasets/iris.csv`（不複製到模組內，只在沙盒裡做一份可隨意改的副本來練）。

DVC 解決什麼問題：Git 適合存「小的程式碼」，不適合存「大的資料/模型」。
DVC 把資料**內容**放到一個「儲存空間（remote）」，只在 Git 裡留一個小小的指標檔
（`xxx.csv.dvc`）。於是 `git checkout` 一個舊 commit，再 `dvc checkout`，
就能把**那個時間點的資料**精準拉回來。

---

## 0. 前置：建一個乾淨的練習資料夾

我們在沙盒裡開一個獨立的小 Git repo 來玩，不影響課程主 repo。

```bash
# 進到本沙盒資料夾
cd mlops-course/modules/m2-tracking-tuning-versioning/sandbox/dvc

# 開練習區（dvc 需要一個 git repo 才能運作）
mkdir -p dvc-playground && cd dvc-playground
git init

# 從共用 datasets 複製一份 iris.csv 進來當「我們要管的資料」
cp ../../../../../datasets/iris.csv data.csv

# 確認 dvc 裝好了（沒裝就 `pip install dvc`）
dvc version
```

---

## 1. `dvc init`：在這個 repo 啟用 DVC

```bash
dvc init
git add .dvc .dvcignore
git commit -m "chore: init dvc"
```

`dvc init` 會建立 `.dvc/` 設定資料夾。把它 commit 進 Git，DVC 才算正式啟用。

---

## 2. `dvc add`：開始追蹤這個 CSV

```bash
dvc add data.csv
```

這一步做了三件事：

1. 算出 `data.csv` 的內容雜湊（md5），存進本機 cache（`.dvc/cache/`）。
2. 產生指標檔 `data.csv.dvc`（很小的純文字，裡面記著雜湊）。
3. 自動把 `data.csv` 加進 `.gitignore`（資料本體不進 Git，只進 DVC）。

把「指標檔」交給 Git 管：

```bash
git add data.csv.dvc .gitignore
git commit -m "data: track iris.csv v1 with dvc"
```

> 重點：Git 裡現在只有 `data.csv.dvc`（幾行字），真正的資料內容由 DVC 管。

---

## 3. `dvc remote` + `dvc push`：把資料推到「儲存空間」

教學用一個「本地資料夾」當 remote（正式環境會換成 S3 / GCS / MinIO 等）。

```bash
# 用沙盒外的一個本地資料夾當 remote 儲存（-d 設為預設 remote）
mkdir -p /tmp/dvc-remote-iris
dvc remote add -d localremote /tmp/dvc-remote-iris

git add .dvc/config
git commit -m "chore: add local dvc remote"

# 把資料內容推到 remote
dvc push
```

`dvc push` 把 cache 裡的資料內容上傳到 remote。之後別人（或 CI、或換台機器）
只要 `git pull` 拿到指標檔，再 `dvc pull` 就能下載到同一份資料。

---

## 4. 製造「第二版」資料，體會版本切換

改動資料，做出 v2，並提交對應的 commit：

```bash
# 故意改資料：只留前 100 列，做出一個不同版本
head -n 101 data.csv > data_v2.csv && mv data_v2.csv data.csv

dvc add data.csv          # 重新計算雜湊，更新 data.csv.dvc
dvc push                  # 把 v2 內容也推到 remote

git add data.csv.dvc
git commit -m "data: shrink iris.csv to 100 rows (v2)"
```

現在 Git 有兩個 commit：v1（150 列）與 v2（100 列），各自對應一份資料指標。

---

## 5. `git checkout` + `dvc checkout`：同一 commit 拉到同一份資料

這是 DVC 的核心價值——**程式碼回到過去，資料也跟著回到過去**。

```bash
# 看一下現在是 v2（資料列數 = 100 + 1 標題列）
wc -l data.csv          # -> 101

# 回到 v1 的那個 commit（先用 git log 找到 v1 的 commit id 或訊息）
git checkout HEAD~1      # 程式碼/指標檔回到 v1，但 data.csv 本體還沒換

# 用 dvc checkout 讓「資料本體」對齊現在的指標檔
dvc checkout

wc -l data.csv          # -> 151（v1 的 150 列 + 標題列）資料真的回到 v1 了
```

> 驗收標準：**只憑 commit，就能還原出當時那一份資料**。
> 回到最新版同理：`git checkout -` 之後再 `dvc checkout`，資料就回到 v2。

---

## 你學會的四個動詞

| 動詞 | 作用 |
| :--- | :--- |
| `dvc init` | 在 git repo 裡啟用 DVC |
| `dvc add <file>` | 開始追蹤一份資料，產生 `.dvc` 指標檔 |
| `dvc push` / `dvc pull` | 把資料內容推到 / 拉回 remote 儲存空間 |
| `dvc checkout` | 讓資料本體對齊「目前 commit 的指標檔」 |

## 明確延後（之後需要時再回來）

- `dvc.yaml` 多階段 pipeline（把資料處理→訓練串成可重跑的流程）
- 雲端 remote 細節（S3/GCS 認證、版本控管）
- `dvc exp` 實驗管理（與 MLflow 的分工取捨）

## 檢核（做完問自己）

1. Git 裡被 commit 的是「資料本體」還是「`.dvc` 指標檔」？為什麼這樣設計？
2. `dvc push` 把資料推去哪？沒有它，別人 `git pull` 後能拿到資料嗎？
3. 為什麼 `git checkout` 之後還要再 `dvc checkout`，資料才真的回到那一版？
