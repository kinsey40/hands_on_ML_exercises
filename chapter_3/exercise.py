"""
Author: Nicholas Kinsey (kinsey40)

Date: 28/11/2017

Description: Chapter 3 exercise on MNIST dataset
"""

import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
import itertools
from random import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=FutureWarning, append=1)

class Always_No_Classifier():
    def __init__(self, X):
        pass

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X),), dtype=np.int)

def grab_data():
    """ Grab the data from sklearn library """

    mnist = fetch_mldata('MNIST original')
    y_values = mnist["target"]
    x_values = mnist["data"]

    return x_values, y_values

def plot_image(save_loc, X, row_index=36000):
    """ Plot an individual data image """

    save_fig_loc = os.path.join(os.getcwd(), save_loc, "first_image.jpg")
    row = X[row_index,:]
    row_r = row.reshape((int(math.sqrt(len(row))), int(math.sqrt(len(row)))))

    plt.imshow(row_r, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig(save_fig_loc)
    plt.close()

    return row_r

def split_train_test(X, y, train_values=60000):
    """ Split the data into train/test as determined in MNIST dataset """

    X_train, X_test = X[:train_values], X[train_values:]
    y_train, y_test = y[:train_values], y[train_values:]

    train_data = list(zip(list(X_train), list(y_train)))
    shuffle(train_data)

    X_train_s, y_train_s = zip(*train_data)

    X_train_s_np = np.asarray(X_train_s, dtype=np.float32)
    y_train_s_np = np.asarray(y_train_s, dtype=np.int16)

    return X_train_s_np, y_train_s_np, X_test, y_test

def binary_classification_feed(y_train, y_test, number=5):
    """ This alters the labels into a binary format """

    y_train_new = np.where(y_train == number, 1, 0)
    y_test_new = np.where(y_test == number, 1, 0)

    return y_train_new, y_test_new

def make_and_fit_sgd_model(X, y, rand_state=42):
    """ Make and fit SGD model """

    sgd_clf = SGDClassifier(random_state=rand_state)
    sgd_clf.fit(X, y)

    return sgd_clf

def implement_cv(model, X_train, y_train, rand_state=42):
    """ CV on training, looking at accuracy of prediction of model """

    skfolds = StratifiedKFold(n_splits=3, random_state=rand_state)

    for train_index, valid_index in skfolds.split(X_train, y_train):
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_valid_folds = X_train[valid_index]
        y_valid_folds = y_train[valid_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_valid_folds)
        n_correct = sum(y_valid_folds == preds)
        acc = n_correct / len(preds)

def implement_cv_score(model, X_train, y_train, cv=3):
    """ Does the above, in a condensed fashion, returning an array """

    model_c = clone(model)
    return cross_val_score(
        estimator=model_c,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="accuracy"
    )

def plot_conf_mat(
        cm,
        classes,
        save_loc,
        normalize=True,
        title='Confusion matrix',
        cmap=plt.cm.Blues
    ):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_loc)
    plt.close()

def conf_mat(model, X, y, plot_save_loc, classes, cv=3):

    y_train_pred = cross_val_predict(model, X, y, cv=cv)

    if not classes:
        classes = list(range(np.amin(y), np.amax(y)+1))
        conf_mat_table = confusion_matrix(y, y_train_pred)
        plot_conf_mat(conf_mat_table, classes, plot_save_loc)

        return conf_mat_table

    else:
        plot_save_loc = os.path.join(os.getcwd(), plot_save_loc, "conf_mat.jpg")
        conf_mat_table = confusion_matrix(y, y_train_pred)
        plot_conf_mat(conf_mat_table, classes, plot_save_loc)

        recall = recall_score(y, y_train_pred)
        precision = precision_score(y, y_train_pred)
        f_score = f1_score(y, y_train_pred)

        return conf_mat_table, recall, precision, f_score

def inv_decision_function(model, X, y, digit=16000, threshold=0):

    y_scores = model.decision_function(X[digit,:].reshape(1, -1))
    y_pred_digit = (y_scores > threshold)

    return y_pred_digit

def plot_precision_recall_curve(model, X, y, save_loc, cv=3):

    thresh_plot_save_loc = os.path.join(os.getcwd(),
                                    save_loc,
                                    "precision_recall_against_threshold.jpg")

    recall_pres_plot_save_loc = os.path.join(os.getcwd(),
                                    save_loc,
                                    "precision_recall_curve.jpg")


    y_scores = cross_val_predict(model, X, y, cv=cv, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_b,
                                                                y_scores)

    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "r-", label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Precision, recall against threshold")
    plt.legend(loc=7)
    plt.savefig(thresh_plot_save_loc)
    plt.close()

    plt.figure()
    plt.plot(recalls, precisions, "g-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision against Recall")
    plt.savefig(recall_pres_plot_save_loc)
    plt.close()

    return y_scores

def plot_roc_curve(X, y_true, y_scores, save_loc, cv=3, rand_state=42):
    """
        Plot the ROC curve for the data, used when we care more about
        false negatives or when positive class has a high proportion.
        Can only be used on binary data.
    """

    roc_plot_save_loc = os.path.join(os.getcwd(),
                                    save_loc,
                                    "ROC_curve.jpg")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    forest_clf = RandomForestClassifier(random_state=rand_state)
    y_probas_forest = cross_val_predict(forest_clf, X, y_true, cv=cv,
                                                method="predict_proba")


    # First col. in array gives prob. of positive class
    y_scores_forest_p = y_probas_forest[:,1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_true, y_scores_forest_p)

    plt.plot(fpr, tpr, linewidth=2, label="SGD", color="b")
    plt.plot(fpr_forest, tpr_forest, linewidth=2, label="RF", color="r")
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])

    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.savefig(roc_plot_save_loc)
    plt.close()

    return roc_auc_score(y_true, y_scores), \
            roc_auc_score(y_true, y_scores_forest_p)

def multi_class(X, y, output_loc, number=16000, rand_state=42, cv=3):
    """ Model fitted, does a OvA, hence shown by desc_func output """

    sgd_clf = SGDClassifier(random_state=rand_state)
    sgd_clf.fit(X, y)

    # Can train on the OvO strategy
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=rand_state))

    ovo_clf.fit(X, y)

    forest_clf = RandomForestClassifier(random_state=rand_state)
    forest_clf.fit(X, y)

    # Index of the highest score is the given class
    scores = sgd_clf.decision_function(X[number,:].reshape(1, -1))

    """
    assert int(sgd_clf.predict(X[number,:].reshape(1, -1))[0,]) == \
        int(np.argmax(scores))

    assert int(ovo_clf.predict(X[number,:].reshape(1, -1))[0,]) == \
        int(sgd_clf.predict(X[number,:].reshape(1, -1))[0,])

    assert int(sgd_clf.predict(X[number,:].reshape(1, -1))[0,]) == \
        int(forest_clf.predict(X[number,:].reshape(1, -1))[0,])

    assert int(np.argmax(
        forest_clf.predict_proba(
            X[number,:].reshape(1, -1))[0,]).flatten()) == \
        int(forest_clf.predict(X[number,:].reshape(1, -1))[0,])
    """

    assert len(ovo_clf.estimators_) == 45

    sgd_score = cross_val_score(sgd_clf, X, y, cv=cv, scoring="accuracy")
    ovo_score = cross_val_score(ovo_clf, X, y, cv=cv, scoring="accuracy")
    rf_score = cross_val_score(forest_clf, X, y, cv=cv, scoring="accuracy")

    plot_save_loc = os.path.join(os.getcwd(), output_loc, "conf_mat_multi.jpg")
    conf_mat_normal = conf_mat(sgd_clf, X, y, plot_save_loc, None, cv=3)

    return np.mean(sgd_score), np.mean(rf_score), np.mean(ovo_score)

def implement_standard_scalar(X):
    """ Implement a scaling method (zero mean; unit variance for each column)"""

    scaler = StandardScaler()
    X_fitted = scaler.fit_transform(X.astype(np.float64))

    return X_fitted

"""
def multilabel_classification(X, y, digit=16000, cv=3):

        Combine binary operations, here a multilabel
        procedure is used to identify targets that are odd and large
    

    y_large = (y >= 7)
    y_odd = (y % 2 == 1)

    y_multilabel = np.c_[y_large, y_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X, y_multilabel)

    print(knn_clf.predict(X[digit,:].reshape(1, -1)), y[digit,])

    y_pred = cross_val_predict(knn_clf, X, y, cv=cv)
    f_s = f1_score(y, y_pred, average="weighted")

    return f_s
"""

if __name__ == "__main__":

    output_loc = "chapter_3/outputs"

    X, y = grab_data()
    rand_row = plot_image(output_loc, X)

    X_train, y_train, X_test, y_test = split_train_test(X, y)
    y_train_b, y_test_b = binary_classification_feed(y_train, y_test)

    X_train = implement_standard_scalar(X_train)

    sgd_model = make_and_fit_sgd_model(X_train, y_train_b)

    implement_cv(sgd_model, X_train, y_train_b)
    cv_score = implement_cv_score(sgd_model, X_train, y_train_b)

    never_model = Always_No_Classifier(y_train_b)
    implement_cv(never_model, X_train, y_train_b)

    conf_mat_tab, recall, precision, f1 = conf_mat(sgd_model,
                                                X_train,
                                                y_train_b,
                                                output_loc,
                                                ["Not Five", "Five"])

    single_dig_output = inv_decision_function(sgd_model, X_train, y_train_b)
    y_s = plot_precision_recall_curve(sgd_model, X_train, y_train_b, output_loc)

    sgd_roc_auc_s, RF_roc_auc_s = plot_roc_curve(X_train,
                                                y_train_b,
                                                y_s,
                                                output_loc)

    sgd_score, ovo_score, rf_score = multi_class(X_train, y_train, output_loc)

    #f_s = multilabel_classification(X_train, y_train)
    #print(f_s)
