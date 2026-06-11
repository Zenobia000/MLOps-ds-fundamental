# Grafana + Prometheus 監控（M6）

線上服務指標的視覺化與告警設定。資料流：

```
src/monitoring/metrics.py（服務內匯出 /metrics）
   → Prometheus（抓取 + 記錄/告警規則：prometheus_rules.yml）
   → Grafana（dashboard.json 視覺化）
   → Alertmanager（路由告警到 Slack / Email）
```

## 檔案

| 檔案 | 用途 |
| :--- | :--- |
| `dashboard.json` | Grafana 儀表板：QPS、延遲分位數、錯誤率、預測分布。可直接 Import。 |
| `prometheus_rules.yml` | Prometheus 記錄規則（預聚合）+ 告警規則（錯誤率/延遲/流量/漂移）。 |

## 匯入 Grafana 儀表板

1. Grafana → Dashboards → **Import** → 上傳 `dashboard.json`。
2. 系統會要求選擇 `DS_PROMETHEUS` 資料來源變數，選你的 Prometheus 即可。
3. 儀表板 UID 為 `smartfactory-serving`。

## 載入 Prometheus 規則

在 `prometheus.yml` 加入：

```yaml
rule_files:
  - /etc/prometheus/rules/prometheus_rules.yml

scrape_configs:
  - job_name: smartfactory-serving
    metrics_path: /metrics
    static_configs:
      - targets: ["serving:8080"]   # BentoML / FastAPI 服務的 /metrics 端點
```

## 指標契約

儀表板與規則中的指標名稱必須與 `src/monitoring/metrics.py` 的 `namespace="smartfactory"` 一致：

| 指標 | 型別 | 說明 |
| :--- | :--- | :--- |
| `smartfactory_inference_requests_total` | Counter | 請求數，label：`model`, `status` |
| `smartfactory_inference_errors_total` | Counter | 錯誤數，label：`model`, `error_type` |
| `smartfactory_inference_latency_seconds` | Histogram | 延遲，label：`model` |
| `smartfactory_model_prediction` | Histogram | 預測值分布，label：`model` |

> 修改 `metrics.py` 的 namespace 或 label 時，務必同步更新本目錄的 JSON 與 rules，否則面板會無資料。
