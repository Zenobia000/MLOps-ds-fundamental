# 變數定義（M5 IaC）
# 設計原則：全變數化、無硬編碼帳密。秘密一律走環境變數 / secret manager，
# 不在 .tf 或 .tfvars 提交真實憑證。

variable "project" {
  description = "專案識別名稱，作為資源命名前綴"
  type        = string
  default     = "smart-factory-mlops"
}

variable "environment" {
  description = "部署環境（dev / staging / prod）"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment 必須是 dev、staging 或 prod 之一。"
  }
}

variable "region" {
  description = "雲端區域（示意，以 AWS 命名為例）"
  type        = string
  default     = "ap-northeast-1"
}

variable "mlflow_artifact_bucket_name" {
  description = "MLflow artifact 儲存桶名稱（須全域唯一）"
  type        = string
}

variable "mlflow_image" {
  description = "MLflow tracking server 容器映像"
  type        = string
  default     = "ghcr.io/mlflow/mlflow:latest"
}

variable "serving_image" {
  description = "模型服務（BentoML）容器映像"
  type        = string
  default     = "smart-factory/serving:latest"
}

variable "serving_cpu" {
  description = "服務容器 CPU 配額（vCPU 單位數，示意值）"
  type        = number
  default     = 1
}

variable "serving_memory_mb" {
  description = "服務容器記憶體（MB）"
  type        = number
  default     = 2048
}

# 秘密：宣告為 sensitive 且無 default，強制由環境變數（TF_VAR_db_password）或
# secret manager 注入，避免硬編碼。
variable "db_password" {
  description = "MLflow backend store 資料庫密碼（由環境變數注入，勿提交）"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "套用到所有資源的共用標籤"
  type        = map(string)
  default = {
    managed_by = "terraform"
    project    = "smart-factory-mlops"
  }
}
