# MLOps 元件入門：初學者應用面速讀

這份文章整理 `modules/` 六個模組提到的主要工具與概念。目標不是背工具名，而是先知道：它解決什麼問題、什麼情境會用到、在本課程扮演哪個位置。

一句話總覽：MLOps 是把模型從「我電腦上跑得動」推進到「團隊可重現、可追蹤、可部署、可監控、可治理」的一整套工程流程。

## 先看全局：一條模型生命週期

| 階段 | 你在解決的問題 | 本課元件 |
| :--- | :--- | :--- |
| 基礎開發 | 程式與結果能不能重現 | Python、scikit-learn、Git、固定 seed |
| 實驗管理 | 哪次訓練效果最好、參數是什麼 | MLflow、Optuna |
| 資料版本 | 同一份程式碼能不能拿回同一份資料 | DVC |
| 特徵管理 | 訓練與推論是否讀同一套特徵 | Feast |
| 模型服務 | 模型如何變成 API 給別人呼叫 | FastAPI、Docker、BentoML、ONNX、PyTorch |
| 自動化 | 流程能不能自動跑、失敗能不能擋 | Prefect、pytest、GitHub Actions |
| 監控治理 | 上線後有沒有壞、能不能被稽核 | Evidently、Model Card、EU AI Act、Prometheus/Grafana |

## M1 基礎：先讓結果可重現

### Python

Python 是本課所有範例的主要語言。資料讀取、模型訓練、API 服務、監控報告都會用 Python 串起來。

應用面：

- 資料處理：讀 CSV、Parquet，清理欄位，切 train/test。
- 模型訓練：呼叫 scikit-learn、PyTorch 等套件。
- 自動化腳本：把 notebook 裡確認過的流程改成可重複執行的 `.py`。
- 服務化：用 FastAPI 或 BentoML 把模型包成 HTTP API。

初學者要抓的重點：Python 在 MLOps 裡不是只拿來「做分析」，也拿來定義可重複執行的工程流程。

### scikit-learn

scikit-learn 是傳統機器學習最常見的 Python 套件之一。本課用它訓練 iris baseline，讓你先掌握乾淨、可重現的訓練流程。

應用面：

- 快速建立 baseline，例如 LogisticRegression、RandomForest、SVM。
- 做 train/test split、cross validation、metric 評估。
- 在結構化資料場景快速驗證模型是否有基本訊號。

什麼時候適合用：資料是表格型、模型不用深度學習、你想快速建立一個可比較的基準線。

### Git

Git 是程式碼版本控制工具。它回答的是：「這份程式碼是什麼時候、為什麼、由誰改的？」

應用面：

- 用 branch 隔離不同任務，避免直接污染 `main`。
- 用 commit 保存可回溯的變更點。
- 讓團隊 code review、回滾、協作開發。
- 和 DVC 搭配，形成「程式碼版本 + 資料版本」的可重現組合。

初學者要抓的重點：Git 管的是程式碼與小型文字檔，不適合直接管理大型資料或模型檔。

### 固定 seed

固定 seed 不是工具，而是可重現性的基本習慣。模型訓練常有隨機性，例如資料切分、初始化、抽樣。

應用面：

- 讓同一支訓練腳本重跑後得到相同結果。
- 降低除錯成本，避免不知道是程式改壞還是隨機性造成分數變動。
- 讓實驗追蹤工具記下來的結果有比較意義。

初學者要抓的重點：如果沒有固定 seed，你很難判斷模型變好是因為方法變好，還是只是這次運氣好。

## M2 實驗追蹤、調參、資料版本

### MLflow

MLflow 是實驗追蹤與模型紀錄工具。它回答的是：「這次訓練用了什麼參數？分數是多少？模型檔在哪？」

應用面：

- 記錄參數：例如 `C=1.0`、`max_depth=5`。
- 記錄指標：例如 accuracy、f1、AUC。
- 記錄模型 artifact：把訓練好的模型與環境資訊存起來。
- 用 UI 比較不同 run，找出最好的實驗。

什麼時候需要：只要你開始訓練多次模型，就需要 MLflow 或同類工具。否則結果會散落在終端機、檔名、筆記裡，後面很難追。

本課定位：先讓每次手動訓練都被記錄，再把 Optuna 產生的大量 trial 也記成 MLflow run。

### Optuna

Optuna 是超參數搜尋工具。它回答的是：「哪些參數組合可能讓模型更好？」

應用面：

- 自動搜尋超參數，例如 learning rate、regularization strength、tree depth。
- 用 trial 記錄每一次嘗試。
- 用 pruning 提早停止看起來沒希望的 trial，節省訓練成本。
- 透過視覺化看哪些參數最影響結果。

什麼時候需要：模型 baseline 已經可重現，但你不想手動亂試參數時。

本課定位：每個 Optuna trial 都對應一個 MLflow run。Optuna 負責產生實驗，MLflow 負責保存與比較實驗。

### DVC

DVC 是資料與模型檔案的版本控制工具。它回答的是：「這個 Git commit 對應哪一版資料？」

應用面：

- 把大型資料檔放進 DVC cache 或 remote，Git 只保存小型 `.dvc` 指標檔。
- 切回舊版程式碼時，也能切回當時使用的資料版本。
- 讓團隊、CI、訓練機器拿到一致的資料。
- 管理模型檔、資料集、特徵輸出等大型 artifact。

什麼時候需要：資料會變、檔案大到不適合進 Git、或你需要重現過去某次訓練。

初學者要抓的重點：Git 管程式碼，DVC 管資料內容。兩者一起用，才有完整可重現性。

### Plotly

Plotly 是互動式視覺化工具。本課在 Optuna 視覺化 notebook 裡用它觀察搜尋過程。

應用面：

- 看 optimization history，判斷搜尋是否還有進步。
- 看 parameter importance，判斷哪些超參數值得繼續調。
- 用 parallel coordinate 或 slice plot 分析好 trial 的參數範圍。

什麼時候需要：你不只想知道最佳分數，也想知道搜尋過程為什麼得到這個結果。

## M3 特徵商店

### Feast

Feast 是 feature store。它回答的是：「訓練和線上推論能不能使用同一份特徵定義？」

應用面：

- 定義 Entity：特徵掛在哪個主體上，例如 `patient_id`、`machine_id`、`customer_id`。
- 定義 FeatureView：一組特徵的 schema、來源、ttl、是否可上線查詢。
- 做 point-in-time join：訓練時只取當下已知的特徵，避免偷看到未來資料。
- materialize 到 online store：讓線上服務可以用 entity key 快速查最新特徵。

什麼時候需要：你的特徵被多個模型重複使用，或訓練與服務之間容易出現特徵不一致。

本課定位：用 diabetes 資料建立 `patient` entity、`predictors_feature_view`、`target_feature_view`，示範 `get_historical_features` 與 `get_online_features`。

### Entity

Entity 是特徵的主體。你可以把它想成「這筆特徵屬於誰」。

應用面：

- 病患模型：entity 可能是 `patient_id`。
- 工廠設備模型：entity 可能是 `machine_id`。
- 電商推薦模型：entity 可能是 `user_id`、`item_id`。

初學者要抓的重點：沒有 entity，就不知道 feature 要 join 到哪個樣本上。

### FeatureView

FeatureView 是一組特徵的契約。它定義欄位名稱、型別、來源、ttl、是否可以進 online store。

應用面：

- 把一組常用特徵變成可重用資產。
- 讓訓練與推論使用同一套 schema。
- 用 ttl 控制特徵有效期限，避免拿太舊的值做判斷。

初學者要抓的重點：FeatureView 不是資料本體，而是「怎麼取用這組特徵」的定義。

### FileSource、Parquet、Offline Store

FileSource 是 Feast 的離線資料來源；本課用 Parquet 檔當來源。

應用面：

- 訓練時從離線資料產生歷史特徵。
- 用 `event_timestamp` 表示這筆特徵何時被觀測到。
- 用 Parquet 儲存表格資料，兼顧壓縮與讀取效率。

初學者要抓的重點：如果資料沒有 `event_timestamp`，就很難做 point-in-time join，也很容易發生資料洩漏。

### Registry

Feast registry 是特徵定義的登記簿。`feast apply` 會把 entity、feature view 等定義寫進 registry。

應用面：

- 保存 feature contracts。
- 讓訓練與服務端知道有哪些特徵可以查。
- 作為團隊共同理解 feature store 狀態的來源。

### Online Store

Online store 是線上推論查特徵的地方。本課用 SQLite 示範，正式環境常見 Redis 或雲端低延遲儲存。

應用面：

- 線上 API 收到 `patient_id` 或 `machine_id` 後，快速查最新特徵。
- 降低推論延遲。
- 讓服務端不用重新跑昂貴的特徵工程。

初學者要抓的重點：offline store 服務訓練，online store 服務即時推論；兩者靠同一份 Feast 定義維持一致。

## M4 模型服務化

### FastAPI

FastAPI 是 Python web API 框架。本課用它把模型包成 `POST /predict`。

應用面：

- 建立 `/health` 健康檢查端點。
- 建立 `/predict` 推論端點。
- 用 Pydantic schema 驗證輸入格式。
- 把模型嵌進既有後端服務。

什麼時候適合用：服務需求簡單、需要高度客製化、或團隊本來就熟 Python web 開發。

初學者要抓的重點：FastAPI 只負責 API 框架；模型版本管理、批次最佳化、打包策略通常要自己補。

### Pydantic

Pydantic 是資料驗證與 schema 定義工具。FastAPI 常用它描述 request/response。

應用面：

- 確認使用者傳進來的欄位存在且型別正確。
- 讓 API 文件自動產生清楚的輸入格式。
- 在模型推論前先擋掉格式錯誤的請求。

什麼時候需要：只要模型 API 對外收資料，就應該明確定義輸入 schema。

### Uvicorn

Uvicorn 是 ASGI server，用來啟動 FastAPI app。

應用面：

- 本機開發時用 `uvicorn app:app --port 8000` 啟動服務。
- 容器裡作為 API process 的入口。

初學者要抓的重點：FastAPI 是你的應用程式，Uvicorn 是把它跑起來的伺服器。

### Docker

Docker 是容器化工具。它回答的是：「我的服務能不能在別人的機器上用同樣環境跑起來？」

應用面：

- 把程式碼、依賴、啟動指令包成 image。
- 用 container 在不同機器上得到一致環境。
- 讓部署系統可以標準化啟動服務。
- 用 `HEALTHCHECK` 讓平台知道服務是否健康。

什麼時候需要：模型要交給別人跑、要部署到伺服器、或你不想再處理「我這台能跑你那台不能跑」。

### BentoML

BentoML 是 ML 原生服務化框架。它比 FastAPI 更偏向「模型服務產品化」。  
**專頁**：[`m4-serving/bentoml-introduction.md`](./m4-serving/bentoml-introduction.md)（Model Store、Service、Bento、工作流與選型）。

應用面：

- 管理模型版本，放進 Model Store。
- 定義 ML service，產生互動式 API 文件。
- 用 `bentofile.yaml` 描述打包內容。
- 建立 Bento，再 containerize 成 Docker image。
- 支援多模型、多版本、批次推論等 ML 服務需求。

什麼時候適合用：你的重點是模型服務，而不是一般 web app；你希望模型管理與部署流程更標準化。

### pickle / joblib

pickle 與 joblib 是 Python 常見序列化方式，常用來保存 scikit-learn 模型。

應用面：

- 快速保存訓練好的傳統 ML 模型。
- 在同樣 Python 與套件版本下讀回模型推論。

限制：

- 綁定 Python 生態與套件版本。
- 不適合跨語言部署。
- 反序列化不可信檔案有安全風險。

初學者要抓的重點：它很方便，但不是最通用的模型交換格式。

### PyTorch

PyTorch 是深度學習框架。本課在影像子場景用預訓練 ResNet 示範服務化。

應用面：

- 訓練影像、語音、文字等深度學習模型。
- 做 transfer learning：凍結 backbone，只訓最後一層。
- 將模型匯出為 TorchScript 或 ONNX，以便部署。

什麼時候需要：資料型態複雜，例如影像瑕疵檢測、語音辨識、自然語言處理。

### ResNet

ResNet 是常見的卷積神經網路架構。本課把它當成預訓練影像模型的代表。

應用面：

- 影像分類。
- 產線瑕疵檢測的 baseline。
- transfer learning：用既有 ImageNet 權重當特徵抽取器。

初學者要抓的重點：預訓練模型讓你不用從零訓練整個深度網路，特別適合資料量不大的教學或原型。

### TorchScript

TorchScript 是 PyTorch 模型的序列化與部署格式。

應用面：

- 把 PyTorch 模型轉成較不依賴原始 Python 程式碼的形式。
- 在 PyTorch 生態內部署模型。

限制：仍然偏 PyTorch 生態；如果你要跨框架或跨硬體，ONNX 通常更通用。

### ONNX

ONNX 是開放的模型交換格式。它回答的是：「模型能不能脫離原框架，到不同環境推論？」

應用面：

- 把 PyTorch 模型匯出到 ONNX。
- 用 ONNX Runtime 在不同硬體上推論。
- 做量化，降低模型大小與延遲。
- 把模型交給非 Python 系統使用。

什麼時候需要：你要跨語言、跨框架、跨硬體部署，或希望推論環境更輕量。

### GPU 服務關鍵概念

GPU 服務不是「把模型丟到 GPU 就結束」。本課提到三個初學者必懂概念。

| 概念 | 解決什麼問題 | 應用場景 |
| :--- | :--- | :--- |
| dynamic batching | 把多個請求湊成一批，提高 GPU 吞吐 | 高流量影像或 NLP 推論 |
| quantization | 降低模型精度與大小，換取速度與成本 | 邊緣部署、低延遲服務 |
| warmup | 避免第一個真實請求很慢 | 模型剛啟動、GPU kernel 尚未初始化 |

## M5 自動化與 CI

### Prefect

Prefect 是 workflow orchestration 工具。它回答的是：「一串訓練流程能不能穩定、自動、可觀測地跑？」

應用面：

- 用 `@task` 標記 pipeline 裡的步驟，例如 load、validate、train、evaluate。
- 用 `@flow` 把多個 task 串成有狀態的流程。
- 追蹤每個 task 成功或失敗。
- 後續可接排程、重試、遠端部署、告警。

什麼時候需要：你的訓練不再是一支單檔腳本，而是一條多步驟流程，且需要定期跑或失敗可追蹤。

本課定位：先在本機跑 Prefect flow，建立 CT 的心智模型。

### pytest

pytest 是 Python 測試框架。本課用它建立 CI 會執行的測試。

應用面：

- 測 metric 函式是否算對。
- 測資料 schema 是否符合預期。
- 測 pipeline 的關鍵步驟是否能跑。
- 作為 GitHub Actions 的品質門檻。

初學者要抓的重點：ML 專案不只測程式碼，也要測資料與模型品質。

### GitHub Actions

GitHub Actions 是 GitHub 內建 CI/CD 工具。它回答的是：「每次 push 後，能不能自動確認專案還是好的？」

應用面：

- push 或 pull request 時自動跑測試。
- 安裝 Python 依賴，執行 `pytest`。
- 測試失敗就讓 workflow 變紅，阻止壞變更合併。
- 後續可延伸到 build Docker image、部署服務。

什麼時候需要：多人協作、開 PR、或你不想靠人手動記得跑測試。

### CI / CD / CT

CI、CD、CT 是自動化流程的三個層次。

| 名稱 | 觸發來源 | 產出 | 初學者理解 |
| :--- | :--- | :--- | :--- |
| CI | 程式碼 push | 測試通過的程式碼 | 壞 code 不要進來 |
| CD | CI 通過 | 可交付或已部署的服務 | 好 code 能穩定上線 |
| CT | 新資料、漂移、排程 | 重新訓練的模型 | 模型會因資料變化而需要重訓 |

MLOps 比一般軟體多 CT，因為模型品質會隨資料分布改變而退化，即使程式碼沒有變。

### 品質門檻 gate

品質門檻 gate 是「不夠好就不能繼續往下走」的規則。

應用面：

- accuracy 低於門檻就不部署。
- 資料缺值比例太高就停止訓練。
- 漂移程度超過門檻就觸發告警或重訓。

初學者要抓的重點：模型不是能跑就能上線，而是要通過品質門檻才准進下一步。

### Canary 與 Blue-Green

這兩個是安全部署策略，目的都是降低新模型上線風險。

| 策略 | 做法 | 適合場景 |
| :--- | :--- | :--- |
| Canary | 先讓少量流量用新模型，觀察沒問題再放大 | 想小流量試水溫 |
| Blue-Green | 同時準備舊版與新版，切換流量，出事快速切回 | 需要快速回滾 |

初學者要抓的重點：部署模型不是一次全換掉，而是要控制風險。

## M6 監控與治理

### Evidently

Evidently 是資料與模型監控工具。本課用它產生資料漂移報告。

應用面：

- 比較 reference data 與 current data。
- 偵測特徵分布是否改變。
- 產生 HTML drift report。
- 把漂移結果接到 CI 或 Prefect，作為告警或重訓條件。

什麼時候需要：模型已上線，資料會隨時間改變，你需要知道模型是否可能開始失效。

初學者要抓的重點：離線測試分數好，只代表過去資料上表現好；上線後資料變了，模型就可能壞。

### Reference Data 與 Current Data

Reference data 是你認為「正常」或訓練時期的資料；current data 是上線後新進來的資料。

應用面：

- 用 reference 當基準，觀察 current 是否偏移。
- 按時間切資料，避免監控基準本身就有偏差。
- 用逐欄結果找出哪個特徵開始異常。

初學者要抓的重點：監控一定要有比較對象。沒有 reference，就很難說 current 到底有沒有變。

### Drift

Drift 是資料或資料關係隨時間改變。

| 類型 | 什麼變了 | 例子 |
| :--- | :--- | :--- |
| covariate drift | 輸入特徵分布 `P(X)` 變了 | 感測器讀值整體偏移 |
| label drift | 標籤分布 `P(y)` 變了 | 不良率突然升高 |
| concept drift | `X -> y` 的關係變了 | 同樣溫度以前正常，現在代表故障 |

初學者要抓的重點：covariate drift 最容易用特徵分布抓；concept drift 通常需要真實標籤回流才能確認。

### Prometheus / Grafana

Prometheus 與 Grafana 是常見系統監控組合。本課在 M6 監控四層中提到它們作為系統層工具。

應用面：

- Prometheus 收集 metrics，例如 QPS、延遲、錯誤率、CPU、記憶體。
- Grafana 做 dashboard，把服務狀態視覺化。
- 幫助你先確認「服務是否活著、是否變慢」。

初學者要抓的重點：Prometheus/Grafana 多半看系統健康；Evidently 更偏資料與模型行為。

### Great Expectations

Great Expectations 是資料品質驗證工具。本課把它列為資料品質層的典型工具之一。

應用面：

- 驗證欄位是否存在。
- 驗證型別是否正確。
- 驗證值域、缺值比例、唯一性等資料規則。

什麼時候需要：你想把「資料應該長什麼樣」寫成可執行規格，而不是只靠人工看。

### Model Card

Model Card 是模型文件，像模型的身分證與使用說明書。

應用面：

- 說明模型用途、適用範圍、不適用範圍。
- 記錄訓練資料、評估結果、限制、偏誤風險。
- 寫清楚維運責任與重訓/下線條件。
- 讓模型能被交接、稽核、維護。

什麼時候需要：模型要交給別人用、要上線、要做合規或風險審查。

初學者要抓的重點：Model Card 不是漂亮文件，而是降低誤用與維運風險的工程文件。

### EU AI Act 風險分級

EU AI Act 是以風險為基礎的 AI 監管框架。本課用它建立治理心智：不同風險等級，對應不同義務。

應用面：

- 先判斷系統用途是否高風險。
- 盤點資料治理、技術文件、人工監督、紀錄保存等要求。
- 幫助團隊在模型設計早期就思考合規與問責。

初學者要抓的重點：治理不是上線後補文件，而是從資料、訓練、評估、部署、監控一路留下可追溯紀錄。

## 工具怎麼選：初學者判斷表

| 如果你的問題是... | 優先看 |
| :--- | :--- |
| 我怎麼確保同一支程式重跑結果一樣？ | 固定 seed、Git、scikit-learn baseline |
| 我訓了很多次模型，結果散落各處怎麼辦？ | MLflow |
| 我不想手動試超參數怎麼辦？ | Optuna |
| 資料版本跟程式碼版本對不起來怎麼辦？ | DVC |
| 訓練和推論用的特徵不一致怎麼辦？ | Feast |
| 我要把模型變成 HTTP API 怎麼辦？ | FastAPI |
| 我要讓服務在別台機器也能穩定跑？ | Docker |
| 我要標準化 ML 模型服務與版本？ | BentoML |
| 我要把深度學習模型跨環境部署？ | ONNX、TorchScript |
| 我要把訓練流程變成自動 pipeline？ | Prefect |
| 我要每次 push 都自動跑測試？ | GitHub Actions、pytest |
| 我要知道上線後資料有沒有變？ | Evidently |
| 我要讓模型可交接、可稽核？ | Model Card、治理文件 |

## 最小學習路線

1. 先跑出一個可重現 baseline：Python + scikit-learn + Git + seed。
2. 開始記錄實驗：MLflow。
3. 自動找更好的參數：Optuna，並把每個 trial 記進 MLflow。
4. 把資料版本納管：DVC。
5. 把特徵定義標準化：Feast。
6. 把模型包成 API：FastAPI，再用 Docker 包環境。
7. 服務需求變複雜後升級：BentoML、ONNX、PyTorch。
8. 把流程自動化：Prefect + pytest + GitHub Actions。
9. 上線後持續盯資料與模型：Evidently。
10. 補齊交接與問責文件：Model Card + 風險分級。

## 一句話收斂

初學 MLOps 不要先追工具數量。每個元件都只是在回答一個工程問題：能不能重現、能不能追蹤、能不能版本化、能不能一致地取特徵、能不能安全部署、能不能自動化、能不能監控與問責。能把問題講清楚，再選工具才不會迷路。
