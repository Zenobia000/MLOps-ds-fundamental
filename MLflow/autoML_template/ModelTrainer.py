import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.dummy import DummyRegressor

class ModelTrainer:
    """
    A class for training and evaluating models.

    Attributes:
        alpha (float): The constant that multiplies the penalty terms.
        l1_ratio (float): The ElasticNet mixing parameter.
    """

    def __init__(self, alpha=0.9, l1_ratio=0.9):
        """
        Initializes the ModelTrainer with model parameters.

        Args:
            alpha (float): The constant that multiplies the penalty terms.
            l1_ratio (float): The ElasticNet mixing parameter.
        """
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        self.baseline_model = DummyRegressor()

    def train(self, train_x, train_y):
        """
        Trains the ElasticNet model and a baseline model using the provided training data.

        Args:
            train_x (DataFrame): The training features.
            train_y (DataFrame): The training labels.
        """
        self.model.fit(train_x, train_y)
        self.baseline_model.fit(train_x, train_y)

    def predict(self, test_x):
        """
        Makes predictions using the trained models on the provided test features.

        Args:
            test_x (DataFrame): The testing features.

        Returns:
            tuple: A tuple containing the predictions from the ElasticNet model and the baseline model.
        """
        return self.model.predict(test_x), self.baseline_model.predict(test_x)

    def evaluate(self, actual, pred):
        """
        Evaluates the performance of the model using RMSE, MAE, and R2 metrics.

        Args:
            actual (DataFrame): The actual labels.
            pred (DataFrame): The predicted labels.

        Returns:
            tuple: A tuple containing the RMSE, MAE, and R2 score.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2
