from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from MLflowManager import MLflowManager
import mlflow
import os
import logging
from pathlib import Path

# 設置日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentWorkflow:
    def __init__(self, raw_data_path, clean_data_path,alpha, l1_ratio, experiment_name, model_instance, model_name,
                 base_model_instance, base_model_name):
        self.raw_data_path = raw_data_path
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.experiment_name = experiment_name
        self.clean_data_path = clean_data_path

        self.params_metrics = {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio
        }
        self.model_instance = model_instance
        self.model_name = model_name
        self.base_model_instance = base_model_instance
        self.base_model_name = base_model_name

    def run(self):
        # 數據處理
        data_processor = DataProcessor(self.raw_data_path)
        data_processor.load_data()
        data_processor.split_data()
        train_x, train_y, test_x, test_y = data_processor.get_train_test_data()

        # Model 創建
        model_trainer = ModelTrainer(alpha=self.alpha, l1_ratio=self.l1_ratio)

        # MLflow 管理
        mlflow_manager = MLflowManager(self.experiment_name, model_trainer.model, model_trainer.baseline_model)
        mlflow_manager.set_tracking_uri()
        mlflow_manager.start_experiment()

        # 模型訓練和評估
        model_trainer.train(train_x, train_y)
        predicted, baseline_predicted = model_trainer.predict(test_x)
        metrics = model_trainer.evaluate(test_y, predicted)
        baseline_metrics = model_trainer.evaluate(test_y, baseline_predicted)

        logging.info(f"compare_model: RMSE: {metrics[0]}  MAE: {metrics[1]}  R2: {metrics[2]}")
        logging.info(f"base_model: RMSE: {baseline_metrics[0]}  MAE: {baseline_metrics[1]}  R2: {baseline_metrics[2]}")

        # MLflow 設定評估指標
        metrics_metrics = {
            "rmse": metrics[0],
            "mae": metrics[1],
            "r2": metrics[2],
            "baseline_rmse": baseline_metrics[0],
            "baseline_mae": baseline_metrics[1],
            "baseline_r2": baseline_metrics[2]
        }

        mlflow_manager.set_log_params_metrics(self.params_metrics, metrics_metrics)

        # Model 儲存
        mlflow_manager.save_model_artifacts(model_trainer.model, self.model_instance, self.model_name)
        mlflow_manager.save_model_artifacts(model_trainer.baseline_model, self.base_model_instance, self.base_model_name)

        # Model artifact 紀錄
        model_artifacts_uri = mlflow.get_artifact_uri(self.model_name)
        base_model_artifacts_uri = mlflow.get_artifact_uri(self.base_model_name)
        logging.info(f"The compare_model artifact path is {model_artifacts_uri}")
        logging.info(f"The base model artifact path is {base_model_artifacts_uri}")

        if not os.path.exists(self.clean_data_path):
            os.makedirs(self.clean_data_path)

        artifacts = {
            self.model_name: self.model_instance,
            "clean_data": self.clean_data_path
        }

        baseline_artifacts = {self.base_model_name: self.base_model_instance}

        mlflow_manager.set_pyfunc_model(self.model_instance, self.model_name, artifacts)
        mlflow_manager.set_pyfunc_model(self.base_model_instance, self.base_model_name, baseline_artifacts)

        # end_run
        mlflow_manager.end_experiment()

def main():
    raw_data_path = "../basic/data/red-wine-quality.csv"
    clean_data_path = 'red-wine-data'
    alpha = 0.9
    l1_ratio = 0.9
    experiment_name = "experiment_validation"

    model_instance = "sklearn_model.pkl"
    model_name = "ElasticNetModel"

    base_model_instance = "baseline_sklearn_model.pkl"
    base_model_name = "BaseModel"


    # 使用配置文件或環境變量來設定這些值可以提高彈性
    workflow = ExperimentWorkflow(raw_data_path, clean_data_path,
                                  alpha, l1_ratio,
                                  experiment_name,
                                  model_instance, model_name,
                                  base_model_instance, base_model_name)
    workflow.run()

if __name__ == "__main__":
    main()
