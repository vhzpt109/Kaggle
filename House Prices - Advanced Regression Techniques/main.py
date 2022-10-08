import pandas as pd
import xgboost as xgb

from dataprocessing import DataProcess
from dataprocessing import SplitTrainValid
from sklearn.model_selection import GridSearchCV

pd.set_option('mode.chained_assignment',  None)

if __name__ == "__main__":
    data_path = "D:/Kaggle/House Prices - Advanced Regression Techniques/"
    train_csv = pd.read_csv(data_path + "train.csv")
    test_csv = pd.read_csv(data_path + "test.csv")

    train, test = DataProcess(train_csv, test_csv)

    x_train, x_val, y_train, y_val = SplitTrainValid(train)

    model = xgb.XGBRegressor()
    param = {
        'max_depth': [2, 3, 4],
        'n_estimators': [550, 600, 650],
        'colsample_bytree': [0.5, 0.7, 1],
        'colsample_bylevel': [0.5, 0.7, 1],
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    test_pred = grid_search.predict(test)

    submission = pd.DataFrame()
    submission['Id'] = [i for i in range(1461, 2920)]
    submission['SalePrice'] = test_pred
    submission.to_csv('submission.csv', index=False)