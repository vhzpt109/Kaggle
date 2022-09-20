import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataprocessing import DataProcess

from models import MLP


if __name__ == "__main__":
    data_path = "D:/Kaggle/House Prices - Advanced Regression Techniques/"
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")

    # print(train.head())
    # train, test = DataProcess(train, test)
    print("train is null :", train.isnull().sum())
    print("test is null :", test.isnull().sum())


    # prediction = []
    #
    # submission = pd.DataFrame({
    #     "Id": test["Id"],
    #     "SalePrice": prediction
    # })
    #
    # submission.to_csv('submission.csv', index=False)