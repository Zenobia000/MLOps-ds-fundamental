"""Optuna 超參搜尋（config-driven HPO）。

流程：
    讀 conf/hpo/<model>.yaml → 建 sampler / pruner / study →
    objective 內依 search_space 採樣超參、訓練、評估 →
    每個 trial 包成一個 MLflow nested run（掛在 parent run 下）→
    回傳最佳超參與最佳值。

執行（從 repo 根）::

    python -m src.tuning.optuna_tuning --model xgboost
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import optuna
import yaml

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

_CONF_DIR = Path(__file__).resolve().parents[2] / "conf"


def load_hpo_config(model_name: str) -> dict[str, Any]:
    """讀 ``conf/hpo/<model_name>.yaml``（契約：n_trials/direction/pruner/search_space）。"""
    path = _CONF_DIR / "hpo" / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到 HPO 設定：{path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def suggest_from_space(
    trial: optuna.Trial, search_space: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """依 search_space 的 ``type`` 呼叫對應的 ``trial.suggest_*``。

    支援 type：float（含 log）/ int（含 step）/ categorical。
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec.get("type")
        if kind == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=int(spec.get("step", 1))
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"不支援的 search_space type：{kind}（{name}）")
    return params


def _build_pruner(pruner_cfg: Mapping[str, Any]) -> optuna.pruners.BasePruner:
    """依設定建立 pruner（median / hyperband / nop）。"""
    name = str(pruner_cfg.get("name", "median")).lower()
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 5)))
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pruner_cfg.get("min_resource", 1)),
            max_resource=int(pruner_cfg.get("max_resource", 5)),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
        )
    return optuna.pruners.NopPruner()


def _make_objective(cfg: Mapping[str, Any], hpo: Mapping[str, Any]):
    """建立 objective：採樣超參 → 用 _trainers 訓練 → 回主指標。"""
    from src.training._trainers import train_one  # 延遲匯入避免循環依賴

    model_name = str(cfg.get("active_model"))
    metric = str(hpo.get("metric", "f1"))
    search_space = hpo.get("search_space", {})

    def objective(trial: optuna.Trial) -> float:
        import mlflow

        sampled = suggest_from_space(trial, search_space)
        # 把採樣到的超參覆蓋進 model.params（不可變：建新 dict）。
        model_cfg = {**cfg.get("model", {})}
        model_cfg["params"] = {**model_cfg.get("params", {}), **sampled}
        trial_cfg = {**cfg, "model": model_cfg}

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            art = train_one(model_name, trial_cfg)
            mlflow.log_params(sampled)
            mlflow.log_metrics(dict(art.metrics))
            if metric not in art.metrics:
                raise optuna.TrialPruned(f"指標 {metric} 不存在於 {list(art.metrics)}")
            return float(art.metrics[metric])

    return objective


def run_study(cfg: Mapping[str, Any]) -> optuna.Study:
    """對 ``active_model`` 執行一場 Optuna 搜尋，回傳 study（含 best_params）。"""
    set_seed(int(cfg.get("seed", 42)))
    model_name = str(cfg.get("active_model"))
    hpo = load_hpo_config(model_name)

    sampler = optuna.samplers.TPESampler(seed=int(hpo.get("sampler", {}).get("seed", 42)))
    study = optuna.create_study(
        direction=str(hpo.get("direction", "maximize")),
        sampler=sampler,
        pruner=_build_pruner(hpo.get("pruner", {})),
    )

    import mlflow

    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg.get("experiment", "smart-factory"))

    with mlflow.start_run(run_name=f"hpo-{model_name}"):
        study.optimize(
            _make_objective(cfg, hpo),
            n_trials=int(hpo.get("n_trials", 20)),
            timeout=hpo.get("timeout"),
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_value", study.best_value)

    logger.info("HPO 完成：best_value=%.5f best_params=%s", study.best_value, study.best_params)
    return study


def main() -> None:
    """CLI 進入點：對指定（或 active）模型做 HPO。

    支援 ``--model xgboost`` 與 Hydra 風格 ``model=xgboost data=sensors``
    （與 src.training.train 一致，Makefile 採後者）。
    """
    parser = argparse.ArgumentParser(description="Smart Factory Optuna 調參")
    parser.add_argument("--model", default=None, help="覆蓋 active_model（xgboost/lstm/resnet）")
    args, extras = parser.parse_known_args()

    overrides: dict[str, Any] = {}
    for token in extras:
        if "=" in token and not token.startswith("-"):
            key, value = token.split("=", 1)
            overrides[key.strip()] = value.strip()
    if args.model:
        overrides["model"] = args.model

    cfg = load_config(overrides=overrides or None)
    run_study(cfg)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
