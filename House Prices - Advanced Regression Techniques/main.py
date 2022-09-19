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

    print(train.head())
    # train, test = DataProcess(train, test)

    train.set_index('Id', inplace=True)
    test.set_index('Id', inplace=True)
    len_train_df = len(train)
    len_test_df = len(test)

    corrmat = train.corr()
    top_corr_features = corrmat.index[abs(corrmat["SalePrice"]) >= 0.3]

    # heatmap
    plt.figure(figsize=(13, 10))
    g = sns.heatmap(train[top_corr_features].corr(), annot=True, cmap="RdYlGn")

    de = 10
    de = 20


    # prediction = []
    #
    # submission = pd.DataFrame({
    #     "Id": test["Id"],
    #     "SalePrice": prediction
    # })
    #
    # submission.to_csv('submission.csv', index=False)