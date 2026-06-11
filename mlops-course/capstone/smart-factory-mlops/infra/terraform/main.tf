# 主要基礎設施定義（M5 IaC）— ILLUSTRATIVE / 示意性
#
# ⚠️ 本檔為「教學示意」，刻意使用通用、不綁定特定雲商的資源語意，
#    示範如何以 IaC 部署 MLflow tracking + 模型服務 + artifact 儲存。
#    實務上請替換為對應雲商 provider（aws / google / azurerm）的真實資源型別，
#    並通過 `terraform plan` 驗證後再 apply。
#
# 設計原則：
#   - 全變數化、無硬編碼帳密（秘密由 TF_VAR_* 環境變數注入）。
#   - 命名加 project / environment 前綴，避免跨環境衝突。
#   - state 建議存遠端（S3 + DynamoDB lock 或 GCS），見 README。

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    # 示意：以 local provider 佔位，避免教學環境需要真實雲端憑證。
    # 真實部署時改為 aws / google / azurerm 並設定對應 provider 區塊。
    local = {
      source  = "hashicorp/local"
      version = ">= 2.4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.0"
    }
  }

  # 遠端 state 後端（示意，預設註解；正式環境請啟用）。
  # backend "s3" {
  #   bucket         = "smart-factory-mlops-tfstate"
  #   key            = "infra/terraform.tfstate"
  #   region         = "ap-northeast-1"
  #   dynamodb_table = "smart-factory-mlops-tflock"
  #   encrypt        = true
  # }
}

locals {
  name_prefix = "${var.project}-${var.environment}"
  common_tags = merge(var.tags, { environment = var.environment })
}

# --- Artifact 儲存（示意 MLflow artifact bucket） ----------------------------
# 真實環境：替換為 aws_s3_bucket / google_storage_bucket。
resource "random_id" "artifact_suffix" {
  byte_length = 4
}

resource "local_file" "artifact_store_descriptor" {
  filename = "${path.module}/.generated/artifact_store.json"
  content = jsonencode({
    bucket      = "${var.mlflow_artifact_bucket_name}-${random_id.artifact_suffix.hex}"
    region      = var.region
    versioning  = true
    encryption  = "AES256"
    tags        = local.common_tags
  })
}

# --- MLflow Tracking Server（示意容器服務） ----------------------------------
# 真實環境：替換為 aws_ecs_service / google_cloud_run_v2_service / azurerm_container_app。
resource "local_file" "mlflow_service_descriptor" {
  filename = "${path.module}/.generated/mlflow_service.json"
  content = jsonencode({
    name  = "${local.name_prefix}-mlflow"
    image = var.mlflow_image
    port  = 5000
    env = {
      MLFLOW_BACKEND_STORE_URI    = "postgresql://mlflow@db:5432/mlflow"
      MLFLOW_DEFAULT_ARTIFACT_ROOT = "s3://${var.mlflow_artifact_bucket_name}-${random_id.artifact_suffix.hex}"
      # 密碼不寫入描述檔，僅標記由 secret manager 注入。
      MLFLOW_DB_PASSWORD_REF = "secret-manager://${local.name_prefix}/db_password"
    }
    cpu       = var.serving_cpu
    memory_mb = var.serving_memory_mb
    tags      = local.common_tags
  })

  # 確保密碼變數確實被提供（雖不寫入檔案，但驗證注入路徑存在）。
  lifecycle {
    precondition {
      condition     = length(var.db_password) > 0
      error_message = "必須透過 TF_VAR_db_password 提供資料庫密碼，勿硬編碼。"
    }
  }
}

# --- 模型服務（示意 BentoML serving 容器） -----------------------------------
resource "local_file" "serving_service_descriptor" {
  filename = "${path.module}/.generated/serving_service.json"
  content = jsonencode({
    name      = "${local.name_prefix}-serving"
    image     = var.serving_image
    port      = 8080
    cpu       = var.serving_cpu
    memory_mb = var.serving_memory_mb
    # 服務暴露 /metrics 供 Prometheus 抓取（對齊 src/monitoring/metrics.py）。
    metrics_path = "/metrics"
    tags         = local.common_tags
  })
}
