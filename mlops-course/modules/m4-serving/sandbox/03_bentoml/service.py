"""
這個檔示範什麼:
    階 7 BentoML 的「最小可用單元」——把跟 01 同一個 iris 模型,
    改用 BentoML 的 service 來服務,體會「ML 原生服務框架」和 FastAPI 的差異。

    FastAPI vs BentoML 的核心差別:
        - FastAPI 是「通用 web 框架」,你要自己處理模型載入、輸入解析、打包。
        - BentoML 是「ML 服務框架」,內建模型管理(Model Store)、
          自動 OpenAPI、可一鍵打包成 Bento、並支援 adaptive batching 等
          推論專屬最佳化。你只描述「模型 + API」,瑣事它幫你做。

    本檔用新版 BentoML(1.2+)的 @bentoml.service 寫法:
        一個 class = 一個 service,方法上加 @bentoml.api 就是一個端點。

怎麼跑:
    pip install -r requirements.txt

    # 第一次要先把模型存進 BentoML 的 Model Store
    python service.py        # 直接執行會觸發底部的「存模型」流程

    # 起服務(--reload 開發時自動重載)
    bentoml serve service:IrisClassifier --reload

    # 測試(見同層 README.md)
    curl -X POST http://localhost:3000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
"""

import bentoml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
MODEL_NAME = "iris_clf"


def save_model_to_store() -> None:
    """現訓 iris 並把模型存進 BentoML 的 Model Store。

    Model Store 是 BentoML 內建的「模型倉庫」,服務啟動時用名稱載回——
    這正是 BentoML 比 FastAPI 多出來的 ML 原生能力:模型有版本、可追溯。
    真實情境會改成從 MLflow / 既有 pickle 載入後再 save_model。
    """
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
    model.fit(X, y)
    saved = bentoml.sklearn.save_model(MODEL_NAME, model)
    print(f"模型已存入 BentoML Model Store: {saved.tag}")


def _ensure_model() -> "bentoml.Model":
    """確保 Model Store 內有模型;不存在就現訓現存。

    為什麼需要這個:class 屬性 ``bento_model = bentoml.models.get(...)`` 會在
    「class 定義時(即 import 時)」就執行,早於 ``__main__`` 的存模型。若 store
    內還沒有模型,import / serve 都會直接 NotFound 崩潰。這裡讓服務「自我 bootstrap」:
    第一次起服務若沒模型,就自動建立,學生不必先手動跑一次存模型。
    """
    try:
        return bentoml.models.get(f"{MODEL_NAME}:latest")
    except bentoml.exceptions.NotFound:
        save_model_to_store()
        return bentoml.models.get(f"{MODEL_NAME}:latest")


@bentoml.service(
    name="iris_classifier",
    # resources/traffic 是 BentoML 的「服務描述」,FastAPI 沒有這層抽象。
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    """一個 BentoML service:啟動時從 Model Store 載入最新版 iris 模型。"""

    # 在 class 屬性宣告「我要用哪個模型」,BentoML 啟動時自動注入。
    # 用 _ensure_model() 而非裸 get：store 內沒有時自動建立,避免 import/serve 崩潰。
    bento_model = _ensure_model()

    def __init__(self) -> None:
        # 把存好的模型載成可呼叫的 runner/物件
        self.model = bentoml.sklearn.load_model(self.bento_model)

    @bentoml.api
    def predict(self, features: list[float]) -> dict:
        """接收一筆 4 維特徵,回傳類別與機率。

        注意:相比 FastAPI 要自己寫 Pydantic model,BentoML 直接用
        Python type hint(list[float] / dict)就自動生出 schema 與 OpenAPI。
        """
        x = np.array([features])
        class_index = int(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        return {
            "class_index": class_index,
            "class_name": CLASS_NAMES[class_index],
            "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, proba)},
        }


if __name__ == "__main__":
    # 直接 `python service.py` 時,先把模型存進 Model Store,
    # 之後才能用 `bentoml serve` 起服務。
    save_model_to_store()
