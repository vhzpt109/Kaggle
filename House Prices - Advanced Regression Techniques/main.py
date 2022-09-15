import pandas as pd
import numpy as np

from dataprocessing import DataProcess

from models import MLP


if __name__ == "__main__":
    data_path = "D:/Kaggle/Titanic - Machine Learning from Disaster/"
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")

    train, test = DataProcess(train, test)