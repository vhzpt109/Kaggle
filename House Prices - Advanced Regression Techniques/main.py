import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataprocessing import DataProcess
from sklearn.impute import SimpleImputer


from models import MLP


if __name__ == "__main__":
    data_path = "D:/Kaggle/House Prices - Advanced Regression Techniques/"
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")

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

    from sklearn.model_selection import train_test_split

    X_train = train.drop(['SalePrice'], axis=1)
    y_train = train['SalePrice']

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    X_test = test
    test_id_idx = test.index

    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb

    param = {
        'max_depth': [2, 3, 4],
        'n_estimators': range(550, 700, 50),
        'colsample_bytree': [0.5, 0.7, 1],
        'colsample_bylevel': [0.5, 0.7, 1],
    }
    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param, cv=5,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1)

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    # pred_train = grid_search.predict(X_train)
    # pred_val = grid_search.predict(X_val)

    test_y_pred = grid_search.predict(X_test)
    id_pred_df = pd.DataFrame()
    id_pred_df['Id'] = test_id_idx
    id_pred_df['SalePrice'] = test_y_pred
    id_pred_df.to_csv('submission.csv', index=False)