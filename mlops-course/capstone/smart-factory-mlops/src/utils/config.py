"""設定載入（config-driven 的單一入口）。

職責：
    把 ``conf/`` 底下的 YAML 組成「一份完整設定」並回傳，全專案共用。
    遵守跨 agent 設定檔契約：
        conf/config.yaml         project / seed / paths / mlflow / active_model / defaults
        conf/model/<name>.yaml   name / params{...}
        conf/data/<name>.yaml    name / source / 欄位契約
        conf/train/<name>.yaml   訓練超參與切分
        conf/hpo/<name>.yaml     n_trials / direction / pruner / search_space{...}

設計：
    - ``load_config()`` 預設回傳 ``dict``（與既有 src.data / src.serving 消費端相容，
      它們以 ``cfg.get(...)`` / ``cfg["data"]`` 取值）。
    - 需要型別化存取時用 ``load_app_config()`` 取 :class:`AppConfig` dataclass。
    - 支援 ``${oc.env:VAR,default}`` 形式的環境變數插值（不依賴 OmegaConf，
      以最小實作覆蓋契約所需語法），避免硬編碼密鑰 / URI。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# repo 根 = 本檔上溯三層（src/utils/config.py → 專案根，含 conf/）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_DIR = _REPO_ROOT / "conf"

# 比對 ${oc.env:VAR} 或 ${oc.env:VAR,default} 的環境變數插值語法。
_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """遞迴解析設定值中的 ``${oc.env:VAR,default}`` 環境變數插值。"""
    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var.strip(), (default or "").strip())

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, Mapping):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    """讀單一 YAML 檔為 dict；缺檔時拋出清晰錯誤（fail-fast）。"""
    if not path.exists():
        raise FileNotFoundError(f"設定檔不存在：{path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"設定檔頂層必須是對應（mapping）：{path}")
    return data


def _load_group(group: str, name: str | None) -> dict[str, Any]:
    """載入 ``conf/<group>/<name>.yaml`` 子設定；name 為 None 時回傳空 dict。"""
    if not name:
        return {}
    return _read_yaml(_CONF_DIR / group / f"{name}.yaml")


def load_config(
    conf_dir: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """載入完整設定（根 + defaults 指定的 data / model / train 子群組）。

    Args:
        conf_dir: 設定根目錄，預設為 repo 的 ``conf/``（測試可注入暫存目錄）。
        overrides: 淺層覆蓋根設定的鍵（如測試時改 ``active_model``）。

    Returns:
        合併後的設定 dict，含 ``data`` / ``model`` / ``train`` 三個子區段，
        並已完成環境變數插值。

    Raises:
        FileNotFoundError: 根設定或被引用的子設定不存在。
    """
    global _CONF_DIR
    base_dir = Path(conf_dir) if conf_dir else _CONF_DIR
    root = _read_yaml(base_dir / "config.yaml")
    if overrides:
        root = {**root, **dict(overrides)}

    defaults = root.get("defaults", {}) or {}
    # active_model 優先於 defaults.model，確保「切模型只改 active_model」成立。
    model_name = root.get("active_model") or defaults.get("model")

    # 暫時切換 _CONF_DIR 以支援測試注入的 conf_dir。
    prev_dir = _CONF_DIR
    _CONF_DIR = base_dir
    try:
        merged: dict[str, Any] = {
            **root,
            "data": _load_group("data", defaults.get("data")),
            "model": _load_group("model", model_name),
            "train": _load_group("train", defaults.get("train") or "default"),
        }
    finally:
        _CONF_DIR = prev_dir

    return _interpolate_env(merged)


@dataclass(frozen=True)
class AppConfig:
    """型別化根設定（不可變）。

    僅涵蓋跨模組契約欄位；模型 / 資料 / 訓練子設定以原始 dict 保留在
    :attr:`raw`，避免為每個模型型態維護不同 schema。
    """

    project: str
    seed: int
    paths: Mapping[str, str]
    mlflow: Mapping[str, str]
    active_model: str
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "AppConfig":
        """從 :func:`load_config` 的 dict 建立型別化設定。"""
        return cls(
            project=str(cfg["project"]),
            seed=int(cfg.get("seed", 42)),
            paths=dict(cfg.get("paths", {})),
            mlflow=dict(cfg.get("mlflow", {})),
            active_model=str(cfg.get("active_model", "")),
            raw=cfg,
        )


def load_app_config(conf_dir: str | Path | None = None) -> AppConfig:
    """載入設定並包成型別化 :class:`AppConfig`（需要屬性存取時用）。"""
    return AppConfig.from_dict(load_config(conf_dir))
