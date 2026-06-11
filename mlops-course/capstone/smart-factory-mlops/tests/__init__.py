"""測試套件根。

由 CI/CD agent 建立。整個專案以 repo 根為 import 起點，
測試一律以 `from src.xxx import ...` 取用正式原始碼（pytest pythonpath=["."]）。
"""
