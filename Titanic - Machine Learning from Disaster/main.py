import pandas as pd

if __name__ == "__main__":
    data_path = "D:/Kaggle/Titanic - Machine Learning from Disaster/train.csv"
    df = pd.read_csv(data_path)
    print(df)