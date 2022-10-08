import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def DataProcess(train, test):
    train.set_index('Id', inplace=True)
    test.set_index('Id', inplace=True)

    train_y_label = train['SalePrice']
    train.drop(['SalePrice'], axis=1, inplace=True)

    boston_df = pd.concat((train, test), axis=0)
    boston_df_index = boston_df.index

    # check null
    check_null = boston_df.isna().sum() / len(boston_df)

    # columns of null ratio >= 0.5
    remove_cols = check_null[check_null >= 0.5].keys()
    boston_df = boston_df.drop(remove_cols, axis=1)

    # split object & numeric
    boston_obj_df = boston_df.select_dtypes(include='object')
    boston_num_df = boston_df.select_dtypes(exclude='object')

    boston_dummy_df = pd.get_dummies(boston_obj_df, drop_first=True)
    boston_dummy_df.index = boston_df_index

    imputer = SimpleImputer(strategy='mean')
    imputer.fit(boston_num_df)
    boston_num_df_ = imputer.transform(boston_num_df)
    boston_num_df = pd.DataFrame(boston_num_df_, columns=boston_num_df.columns, index=boston_df_index)

    boston_df = pd.merge(boston_dummy_df, boston_num_df, left_index=True, right_index=True)

    train = boston_df[:len(train)]
    test = boston_df[len(train):]

    train['SalePrice'] = train_y_label

    return train, test


def SplitTrainValid(train):
    x_train = train.drop(['SalePrice'], axis=1)
    y_train = train['SalePrice']

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

    return x_train, x_val, y_train, y_val