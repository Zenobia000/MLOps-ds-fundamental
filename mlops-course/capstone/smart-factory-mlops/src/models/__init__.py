"""模型層（models）。

三種資料型態各一個模型子套件，共用「fit / predict / save / load」一致介面，
讓上層 :mod:`src.training` 能以 config 的 ``active_model`` 動態切換而不改程式：

- :mod:`src.models.tabular`     XGBoost 預測性維護（結構化 + 時序特徵 → 故障）。
- :mod:`src.models.timeseries`  LSTM 產能需求預測（單變量序列 → 未來需求）。
- :mod:`src.models.vision`      ResNet transfer learning 瑕疵檢測（影像 → good/defect）。

設計原則：每個模型自帶序列化格式，與訓練 / 服務端對齊；
深度學習模型一律含「無 GPU 後援」（自動退回 CPU）。
"""
