import pandas as pd
import numpy as np

from dataprocessing import DataProcess

from models import MLP


if __name__ == "__main__":
    data_path = "D:/Kaggle/Titanic - Machine Learning from Disaster/"
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")

    train, test = DataProcess(train, test)

    train_data = np.asarray(train.drop('Survived', axis=1))
    target = np.asarray(pd.Categorical(train['Survived']))

    model = MLP()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, target, epochs=50)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = model.predict(test_data)
    prediction = np.around(prediction, decimals=0)
    prediction = np.squeeze(prediction, axis=1)
    prediction = prediction.astype(np.int)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)