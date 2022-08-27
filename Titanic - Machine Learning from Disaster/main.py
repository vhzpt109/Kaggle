import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots

import torch
from torch import nn, optim
from models import MLP

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('CUDA:', torch.cuda.is_available(), '     Use << {} >>'.format(device.upper()))
    print('PyTorch Version:', torch.__version__)

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

    train_data = train.drop('Survived', axis=1)
    target = train['Survived']

    # model = RandomForestClassifier()
    model = MLP().to(device)

    lr = 1e-3
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=3407)
    k = 1
    for train_index, valid_index in k_fold.split(train_data, target):
        train, valid = train_data.iloc[train_index], train_data.iloc[train_index]
        target_train, target_valid = target.iloc[train_index], target.iloc[train_index]

        pred = model(train)
        loss = loss_fn(pred, target_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # model.fit(train, target_train)

        # Predicting y for X_val
        # pred = model.predict(valid)

        # print("%d번째 fold, accuracy : %f" %(k, accuracy_score(pred, target_valid)))
        # k += 1

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = model.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)