# Terraform IaC（M5）— Illustrative

> ⚠️ **示意性（illustrative）**：本模組為教學示範，預設使用 `local`/`random` provider
> 以避免在課程環境需要真實雲端憑證。它展示「如何用 IaC 變數化部署 MLflow tracking、
> 模型服務與 artifact 儲存」的結構與良好實踐，**不可直接用於正式生產**。
> 正式部署時請將 `local_file` 資源替換為對應雲商的真實資源（見下方對照）。

## 設計原則

- **全變數化**：所有可變參數集中於 `variables.tf`，切環境只改 `.tfvars`。
- **無硬編碼帳密**：秘密（如 `db_password`）宣告為 `sensitive` 且**無 default**，
  強制由環境變數 `TF_VAR_db_password` 或 secret manager 注入，描述檔僅留 secret 參照。
- **遠端 state**：正式環境啟用 `main.tf` 中註解的 `backend "s3"`（含鎖），勿用本地 state。
- **命名前綴**：所有資源加 `project-environment` 前綴，避免跨環境衝突。

## 檔案

| 檔案 | 用途 |
| :--- | :--- |
| `main.tf` | 資源定義：artifact 儲存、MLflow server、模型服務（示意）。 |
| `variables.tf` | 輸入變數與驗證規則。 |
| `outputs.tf` | 部署後輸出（端點、資源名；不含秘密）。 |

## 真實雲商資源對照

| 示意資源 | AWS | GCP | Azure |
| :--- | :--- | :--- | :--- |
| artifact 儲存 | `aws_s3_bucket` | `google_storage_bucket` | `azurerm_storage_account` |
| MLflow server | `aws_ecs_service` | `google_cloud_run_v2_service` | `azurerm_container_app` |
| 模型服務 | `aws_ecs_service` | `google_cloud_run_v2_service` | `azurerm_container_app` |
| state 後端 | S3 + DynamoDB | GCS | azurerm backend |

## 使用方式

```bash
# 1. 設定秘密（勿寫入檔案）
export TF_VAR_db_password="$(your-secret-manager get mlflow-db-password)"

# 2. 提供必填變數（建立 dev.tfvars，勿提交真實值）
cat > dev.tfvars <<'EOF'
environment                 = "dev"
mlflow_artifact_bucket_name = "smart-factory-mlflow-artifacts"
EOF

# 3. 標準流程
terraform init
terraform plan  -var-file=dev.tfvars
terraform apply -var-file=dev.tfvars   # 教學示意會在 .generated/ 產出描述檔
```

## 安全提醒

- `.tfvars`（可能含環境細節）與 `.generated/`、`*.tfstate` 應列入 `.gitignore`。
- 真實憑證一律走 secret manager，輪換任何疑似外洩的密鑰。
- apply 前務必 `terraform plan` review；正式環境啟用遠端 state 與鎖。
