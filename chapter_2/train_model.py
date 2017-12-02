

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def fit_linreg_model(df, df_labels):

    lin_reg = LinearRegression()
    fitted = lin_reg.fit(df, df_labels)

    return fitted

def fit_treereg_model(df, df_labels):

    tree_reg = DecisionTreeRegressor()
    fitted = tree_reg.fit(df, df_labels)

    return fitted

def fit_rf_model(df, df_labels):

    rf = RandomForestRegressor()
    fitted = rf.fit(df, df_labels)

    return fitted

def implement_cross_val_score(model, df, df_labels):

    scores = cross_val_score(model, df, df_labels,
            scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)

    return tree_rmse_scores, tree_rmse_scores.mean(), tree_rmse_scores.std()

def evaluate_rmse(model, X, labels):

    preds = model.predict(X)
    lin_mse = mean_squared_error(labels, preds)
    lin_rmse = np.sqrt(lin_mse)

    return lin_rmse

def grid_search(X, labels):

    param_grid = [{'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
        {'bootstrap': [False], 'n_estimators': [3,10], 'max_features' : [2,3,4]}
    ]

    forest_reg = RandomForestRegressor()

    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
        scoring='neg_mean_squared_error')

    grid_search.fit(X, labels)

    #print(grid_search.best_estimator_.feature_importances_)

    cvres = grid_search.cv_results_

    #for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #    print(np.sqrt(-mean_score), params)

    return grid_search.best_estimator_

def random_grid_search(X, labels):

    forest_reg = RandomForestRegressor()

    param_grid = {
        'bootstrap':[True, False],
        'n_estimators':[20, 60],
        'max_features':[2,8]
    }

    n_iter_search = 8
    random_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv = 8,
        scoring='neg_mean_squared_error'
    )

    random_search.fit(X, labels)

    return random_search.best_estimator_

def fit_test(X, y, est):

    preds = est.predict(X)
    test_mse = mean_squared_error(y, preds)

    return np.sqrt(test_mse)
