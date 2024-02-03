import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    A class for loading and splitting data.

    Attributes:
        filepath (str): The path to the CSV data file.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
    """

    def __init__(self, filepath, test_size=0.25, random_state=40):
        """
        Initializes the DataProcessor with a file path and split parameters.

        Args:
            filepath (str): The path to the CSV data file.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.train = None
        self.test = None

    def load_data(self):
        """Loads data from the CSV file specified in the filepath."""
        self.data = pd.read_csv(self.filepath)

    def split_data(self):
        """Splits the data into training and testing datasets."""
        self.train, self.test = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state)

    def get_train_test_data(self):
        """
        Returns the training and testing datasets, excluding the 'quality' column from the features.

        Returns:
            tuple: A tuple containing the training features, training labels, testing features, and testing labels.
        """
        train_x = self.train.drop(["quality"], axis=1)
        test_x = self.test.drop(["quality"], axis=1)
        train_y = self.train[["quality"]]
        test_y = self.test[["quality"]]
        return train_x, train_y, test_x, test_y
