import cloudpickle
import mlflow
import mlflow.sklearn
import joblib
import sklearn
import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, artifacts_name):
        self.artifacts_name = artifacts_name

    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts[self.artifacts_name])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)

class MLflowManager:
    """
    A class for managing MLflow experiments.

    Attributes:
        experiment_name (str): The name of the MLflow experiment.
        model (ElasticNet): The trained ElasticNet model.
        baseline_model (DummyRegressor): The trained baseline model.
    """


    def __init__(self, experiment_name, model, baseline_model):
        """
        Initializes the MLflowManager with the experiment name and models.

        Args:
            experiment_name (str): The name of the MLflow experiment.
            model (ElasticNet): The trained ElasticNet model.
            baseline_model (DummyRegressor): The trained baseline model.
        """
        self.experiment_name = experiment_name
        self.model = model
        self.baseline_model = baseline_model
        self.conda_env = {
            "channels": ["defaults"],
            "dependencies": [
                "python={}".format(3.10),
                "pip",
                {
                    "pip": [
                        "mlflow=={}".format(mlflow.__version__),
                        "scikit-learn=={}".format(sklearn.__version__),
                        "cloudpickle=={}".format(cloudpickle.__version__),
                    ],
                },
            ],
            "name": "sklearn_env",
        }

    # 模型相關資訊儲存位置
    def set_tracking_uri(self, uri=""):
        # 全路徑寫法 file:xxxx
        # mlflow.set_tracking_uri(uri=r"file:C:\Users\xdxd2\Sunny_VS_worksapce\Sunny_python\ML\mytracks")

        mlflow.set_tracking_uri(uri)
        print("The set tracking uri is ", mlflow.get_tracking_uri())

    def start_experiment(self):
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        # Set the experiment name
        exp = mlflow.set_experiment(experiment_name=self.experiment_name)

        # Get the current timestamp and format it as a string
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Start the MLflow run with the run name including the timestamp
        run_name = "run_" + timestamp
        mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name)
        print("**" *14 + "實驗開始" + "**" *14)
        print("Experiment_Name: {}".format(exp.name))
        print("Experiment_id: {}".format(exp.experiment_id))
        print("Artifact Location: {}".format(exp.artifact_location))
        print("Tags: {}".format(exp.tags))
        print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
        print("Creation timestamp: {}".format(exp.creation_time))
        print("**" * 28)


    # mlflow 實驗狀態的 tags
    def set_tags(self, tags):
        mlflow.set_tags(tags)

    # 參數, 預測指標紀錄
    def set_log_params_metrics(self, params, metrics):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


    # 儲存模型, 實體模型, 模型路徑, 模型config key(artifact)
    def save_model_artifacts(self, model, model_instance, model_name):
        joblib.dump(model, model_instance)
        mlflow.sklearn.log_model(model, model_name)

    def set_pyfunc_model(self, model_instance, model_name, artifacts):
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=SklearnWrapper(model_name),
            artifacts=artifacts,
            code_path=["main.py"],
            conda_env=self.conda_env
        )

    # 結束實驗
    def end_experiment(self):
        mlflow.end_run()


