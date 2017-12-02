"""
Author: Nicholas Kinsey (kinsey40)

Date: 01/12/2017

Description: Chapter 3 exercises on MNIST dataset, end of chapter
"""

import numpy as np
from exercise import grab_data, split_train_test, implement_standard_scalar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.ndimage.interpolation import shift

def data_grabbing():

    X, y = grab_data()
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    return X_train, y_train, X_test, y_test

def preprocessing(X_train, X_test):

    X_train_alt = implement_standard_scalar(X_train)
    X_test_alt = implement_standard_scalar(X_test)

    return X_train_alt, X_test_alt

def knn_model(X, y, best_estimator):

    X_trimmed = X[:60000,:]
    y_trimmed = y[:60000,]

    best_estimator.fit(X_trimmed, y_trimmed)

    return best_estimator

def eval_acc(model, X_test, y_test):

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc

def grid_searching(X, y):

    X_trimmed = X[:10000, :]
    y_trimmed = y[:10000,]

    knn_clf = KNeighborsClassifier(n_jobs=-1)
    param_grid = {'n_neighbors': np.arange(1, 21, 2),
                    'weights': ["uniform", "distance"]}

    rand_search = RandomizedSearchCV(knn_clf, param_grid,
        scoring='accuracy')

    rand_search.fit(X_trimmed, y_trimmed)

    return rand_search.best_estimator_

def shift_pixels(X):

    shifting_vals = [[1,0], [-1,0], [0,1], [0,-1]]
    for row in X:
        for shift_vals in shifting_vals:
            image = row.reshape(28, 28)
            new_row = shift(image, shift_vals, cval=0)
            new_row = new_row.reshape(1, 784)
            X = np.vstack((X, new_row))

    return X

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = data_grabbing()
    X_train, X_test = preprocessing(X_train, X_test)

    X_train = shift_pixels(X_train[:1000])
    X_test = shift_pixels(X_test[:1000])

    print(X_train.shape, X_test.shape)

    """
    best_estimator = grid_searching(X_train, y_train)
    print(best_estimator)

    model = knn_model(X_train, y_train, best_estimator)
    accuracy = eval_acc(model, X_test, y_test)

    print(accuracy)
    """
