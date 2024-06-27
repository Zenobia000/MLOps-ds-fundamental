from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from MLflowManager import MLflowManager
import mlflow
import os

import pandas as pd


def main():
    # 數據處理
    data_processor = DataProcessor("../basic/data/red-wine-quality.csv")
    data_processor.load_data()
    data_processor.split_data()
    train_x, train_y, test_x, test_y = data_processor.get_train_test_data()

    # model 創建
    alpha = 0.9
    l1_ratio = 0.9
    model_trainer = ModelTrainer(alpha=alpha, l1_ratio=l1_ratio)

    # MLflow 管理
    mlflow_manager = MLflowManager("experiment_custom_metrics", model_trainer.model, model_trainer.baseline_model)
    mlflow_manager.set_tracking_uri()  # 設定 Log 存檔位置
    mlflow_manager.start_experiment()  # 設定 mlflow 當前實驗名稱

    # 模型訓練和評估
    model_trainer.train(train_x, train_y)
    predicted, baseline_predicted = model_trainer.predict(test_x)
    metrics = model_trainer.evaluate(test_y, predicted)
    baseline_metrics = model_trainer.evaluate(test_y, baseline_predicted)

    print(f"compare_model: RMSE: {metrics[0]}  MAE: {metrics[1]}  R2: {metrics[2]}")
    print(f"base_model: RMSE: {baseline_metrics[0]}  MAE: {baseline_metrics[1]}  R2: {baseline_metrics[2]}")

    # MLflow 管理
    # exp tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    # params info
    log_params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }

    # metrics info
    metrics_params = {
        "rmse": metrics[0],
        "mae": metrics[1],
        "r2": metrics[2]}

    base_metrics_params = {
        "baseline rmse": baseline_metrics[0],
        "baseline mae": baseline_metrics[1],
        "baseline r2": baseline_metrics[2]}

    mlflow_manager.set_tags(tags)

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False)

    mlflow_manager.set_log_params_metrics(log_params, metrics_params)
    mlflow_manager.set_log_params_metrics(log_params, base_metrics_params)

    # model save
    model_instance = "sklearn_model.pkl"
    model_name = "ElasticNetModel"
    base_model_instance = "baseline_sklearn_model.pkl"
    base_model_name = "BaseModel"

    mlflow_manager.save_model(model_trainer.model, model_instance, model_name)
    mlflow_manager.save_model(model_trainer.baseline_model, base_model_instance, base_model_name)

    data_dir = 'red-wine-data'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    artifacts = {
        model_name: model_instance,
        "data": data_dir
    }
    baseline_artifacts = {base_model_name: base_model_instance}

    mlflow_manager.set_pyfunc_model(model_instance, model_name, artifacts)
    mlflow_manager.set_pyfunc_model(base_model_instance, base_model_name, baseline_artifacts)

    artifacts_uri = mlflow.get_artifact_uri()
    print("", end="\n")
    print("The artifact path is", artifacts_uri)

    # end_run
    mlflow_manager.end_experiment()


if __name__ == "__main__":
    main()

    # data_processor = DataProcessor.DataProcessor("./basic/data/red-wine-quality.csv")
    # data_processor.load_data()
    # data_processor.split_data()
    # train_x, train_y, test_x, test_y = data_processor.get_train_test_data()


