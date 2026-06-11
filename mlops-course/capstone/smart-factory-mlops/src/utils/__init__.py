"""共用工具層（utils）。

與業務無關的橫切關注點，全專案重用：

- :mod:`src.utils.config`：讀 ``conf/`` 的 YAML，組成型別化設定（dataclass）。
- :mod:`src.utils.logging`：結構化 logger，統一格式與等級。
- :mod:`src.utils.seed`：一次設定 numpy / torch / random 的亂數種子，確保可重現。

設計原則：零業務依賴、純函式優先、不在此層 import 任何模型或資料模組，
避免循環依賴（高內聚、低耦合）。
"""

from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

__all__ = ["AppConfig", "load_config", "get_logger", "set_seed"]
