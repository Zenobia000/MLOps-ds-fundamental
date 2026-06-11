"""部署管線（Prefect flow）。

階段：load registered model → build bento/image（佔位）→ canary 部署（佔位）
      → smoke probe → promote / rollback 決策。
真正的 build/部署在教學環境多半靠 CI（deploy.yml）執行，這裡提供可程式
驅動的骨架與決策邏輯，並對 ``src.serving`` 軟相依。
"""

from __future__ import annotations

import logging

from pipelines import flow, task
from pipelines._helpers import load_config, soft_import

logger = logging.getLogger("pipelines.deployment")

# canary 放行門檻：smoke 探測成功率須達標才全量
DEFAULT_CANARY_THRESHOLD = 0.95


@task
def resolve_model(config: dict, model_uri: str | None) -> str:
    """解析要部署的模型版本（None 時取 registry latest 佔位）。"""
    resolver = soft_import("src.serving.registry", "resolve_latest")
    if model_uri:
        return model_uri
    if resolver is not None:
        try:
            return resolver(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析 latest 模型失敗（%s），用佔位 URI。", exc)
    return "models:/smart-factory/latest"


@task
def build_image(model_uri: str, config: dict) -> str:
    """建置服務映像（佔位）。真正 build 由 docker/Dockerfile.serve + CI 處理。"""
    tag = f"{config.get('project', 'smart-factory')}:serve-canary"
    logger.info("[佔位] 以模型 %s 建置映像 %s（實作見 docker/、deploy.yml）。", model_uri, tag)
    # TODO: 接 BentoML `bentoml build` 或 `docker build -f docker/Dockerfile.serve`。
    return tag


@task
def canary_probe(image_tag: str) -> float:
    """對 canary 實例做 smoke 探測，回傳成功率（佔位回傳 1.0）。"""
    prober = soft_import("src.serving.healthcheck", "probe")
    if prober is not None:
        try:
            return float(prober(image_tag))
        except Exception as exc:  # noqa: BLE001
            logger.warning("健康探測失敗（%s），視為 0.0。", exc)
            return 0.0
    logger.info("[佔位] canary 探測 %s 假定成功。", image_tag)
    return 1.0


@task
def promote_or_rollback(success_rate: float, config: dict) -> str:
    """依 canary 成功率決定 promote 或 rollback。"""
    threshold = config.get("deploy", {}).get("canary_threshold", DEFAULT_CANARY_THRESHOLD)
    decision = "promote" if success_rate >= threshold else "rollback"
    logger.info("canary 成功率 %.3f vs 門檻 %.3f -> %s", success_rate, threshold, decision)
    return decision


@flow(name="deployment-pipeline")
def deployment_flow(config_path: str | None = None, model_uri: str | None = None) -> dict:
    """部署管線主流程，回傳含 image_tag / success_rate / decision 的結果。"""
    config = load_config(config_path)
    resolved = resolve_model(config, model_uri)
    image_tag = build_image(resolved, config)
    rate = canary_probe(image_tag)
    decision = promote_or_rollback(rate, config)
    return {"image_tag": image_tag, "success_rate": rate, "decision": decision}


def main() -> None:
    """CLI 入口：``python -m pipelines.deployment_pipeline``。"""
    logging.basicConfig(level=logging.INFO)
    result = deployment_flow()
    print(f"[deployment-pipeline] 決策={result['decision']}；映像={result['image_tag']}")


if __name__ == "__main__":
    main()
