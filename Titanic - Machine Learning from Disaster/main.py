import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots

from models import MLP

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy as np


def bar_chart(feature):
    survived_count = train[train['Survived'] == 1][feature].value_counts()
    dead_count = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived_count, dead_count])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))


if __name__ == "__main__":
    data_path = "D:/Kaggle/Titanic - Machine Learning from Disaster/"
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")

    train_test_data = [train, test]  # combining train and test dataset

    for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                     "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
    for dataset in train_test_data:
        dataset['Title'] = dataset['Title'].map(title_mapping)

    # delete unnecessary feature from dataset
    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    sex_mapping = {"male": 0, "female": 1}
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    # fill missing age with median age for each title (Mr, Mrs, Miss, Others)
    train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
    test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


    train[['Age']] = MinMaxScaler().fit_transform(train[['Age']])
    test[['Age']] = MinMaxScaler().fit_transform(test[['Age']])

    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    train[['Fare']] = MinMaxScaler().fit_transform(train[['Fare']])
    test[['Fare']] = MinMaxScaler().fit_transform(test[['Fare']])

    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].str[:1]

    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    # fill missing Fare with median fare for each Pclass
    train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
    test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

    train[['FamilySize']] = MinMaxScaler().fit_transform(train[['FamilySize']])
    test[['FamilySize']] = MinMaxScaler().fit_transform(test[['FamilySize']])

    features_drop = ['Ticket', 'SibSp', 'Parch']
    train = train.drop(features_drop, axis=1)
    test = test.drop(features_drop, axis=1)
    train = train.drop(['PassengerId'], axis=1)

    train_data = np.asarray(train.drop('Survived', axis=1))
    target = np.asarray(pd.Categorical(train['Survived']))

    # model = RandomForestClassifier()
    model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
                ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, target, epochs=10)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = model.predict(test_data)
    prediction = np.ndarray.round(decimals=1)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)