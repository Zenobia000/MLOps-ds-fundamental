# 輸出值（M5 IaC）
# 供 CI/CD 與其他模組引用部署後的端點與資源識別。
# 注意：不輸出任何秘密（密碼 / token）。

output "name_prefix" {
  description = "資源命名前綴（project-environment）"
  value       = local.name_prefix
}

output "mlflow_artifact_bucket" {
  description = "MLflow artifact 儲存桶名稱（含隨機後綴，全域唯一）"
  value       = "${var.mlflow_artifact_bucket_name}-${random_id.artifact_suffix.hex}"
}

output "mlflow_service_name" {
  description = "MLflow tracking server 服務名稱"
  value       = "${local.name_prefix}-mlflow"
}

output "mlflow_tracking_port" {
  description = "MLflow tracking server 連接埠（示意）"
  value       = 5000
}

output "serving_service_name" {
  description = "模型服務名稱"
  value       = "${local.name_prefix}-serving"
}

output "serving_metrics_endpoint" {
  description = "服務 Prometheus 指標端點路徑（供 monitoring/grafana 抓取設定參考）"
  value       = "/metrics"
}

output "region" {
  description = "部署區域"
  value       = var.region
}
