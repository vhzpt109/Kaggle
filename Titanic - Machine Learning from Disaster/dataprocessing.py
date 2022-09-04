from sklearn.preprocessing import MinMaxScaler

def DataProcess(train, test):
    train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                     "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
    train['Title'] = train['Title'].map(title_mapping)
    test['Title'] = test['Title'].map(title_mapping)

    # delete unnecessary feature from dataset
    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    sex_mapping = {"male": 0, "female": 1}
    train['Sex'] = train['Sex'].map(sex_mapping)
    test['Sex'] = test['Sex'].map(sex_mapping)

    # fill missing age with median age for each title (Mr, Mrs, Miss, Others)
    train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
    test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

    train[['Age']] = MinMaxScaler().fit_transform(train[['Age']])
    test[['Age']] = MinMaxScaler().fit_transform(test[['Age']])

    train['Embarked'] = train['Embarked'].fillna('S')
    test['Embarked'] = test['Embarked'].fillna('S')

    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    train['Embarked'] = train['Embarked'].map(embarked_mapping)
    test['Embarked'] = test['Embarked'].map(embarked_mapping)

    train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

    train[['Fare']] = MinMaxScaler().fit_transform(train[['Fare']])
    test[['Fare']] = MinMaxScaler().fit_transform(test[['Fare']])

    train['Cabin'] = train['Cabin'].str[:1]
    test['Cabin'] = test['Cabin'].str[:1]

    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
    train['Cabin'] = train['Cabin'].map(cabin_mapping)
    test['Cabin'] = test['Cabin'].map(cabin_mapping)

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

    return train, test