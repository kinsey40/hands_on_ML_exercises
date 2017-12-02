"""
Date: 22/11/2017

Author: Nicholas Kinsey (kinsey40)

Description:
This is the read and plot functions file for Chapter 2 in Hands on Machine Learning
with Scikit Learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit

def read_data(data_loc, header_no=0):
    """
    Reads in the Housing csv data
    """

    df = pd.read_csv(data_loc, header=header_no, delimiter=',')

    return df

def copy_data(df):
    """
    Return a copy of the df
    """

    return df.copy()

def plot_data(df, path, bins=50):
    """
    Creates various plots of the data
    """

    for var_name in list(df):
        plt.figure()
        df[var_name].hist(bins=bins)
        plt.title(var_name)
        plt.savefig(os.path.join(path, var_name + "_hist.jpg"))
        plt.close()

def categorize_income(df, path, cat_value=1.5, bins=50):
    """
    Categorizes the income col.
    """

    df["income_cat"] = np.ceil(df["median_income"] / 1.5)
    df["income_cat"].where(df["income_cat"] < 5.0, 5.0, inplace=True)

    bins = [0, 1.0, 2.0, 3.0, 4.0, 5.0]

    fig = plt.figure()
    plt.hist(df["income_cat"], bins=bins)
    plt.title("income_cat")
    plt.savefig(os.path.join(path, "income_cat" + "_hist.jpg"))
    plt.close()

    return df

def create_train_test(df, train_test_split=0.2, random_state=42):
    """
    Creates the train and test sets
    """

    X = df.drop(["income_cat"], axis=1)
    y = df["income_cat"].astype('category')

    sss = StratifiedShuffleSplit(n_splits=1,
                                test_size=train_test_split,
                                random_state=42)

    for train_index, test_index in sss.split(X,y):

        train, test = X.iloc[train_index], X.iloc[test_index]

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, test

def form_scat_plots(df, path):
    """
    Call the scatter plot function
    """
    create_scatter_plots(df, "longitude", "latitude", "population", path)

def create_scatter_plots(df, var1, var2, var3, path):
    """
    Create the scatter plot, 3 variables
    """
    save_loc = os.path.join(path, var1 + "_" + var2 + "_scatter.jpg")

    fig, ax = plt.subplots()
    scatters = plt.scatter(x=df[var1].values,
                y=df[var2].values,
                c=((df[var3].values/100) - min(df[var3].values/100) / max(df[var3].values/100)),
                alpha=0.4,
                cmap='magma')
    plt.colorbar(scatters)
    plt.title(var1 + "_against_" + var2 + "_colour_" + var3)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend()
    plt.savefig(save_loc)
    plt.close()

def calc_corrs(df, target_var):
    """
    Calc. the PCC on the numeric cols
    """
    corr_values = pd.DataFrame()
    for col_name in list(df):

        if col_name == target_var:
            continue

        try:
            value = df[target_var].corr(df[col_name])
            df_value = pd.DataFrame([[col_name, value]])
            corr_values = corr_values.append(df_value)
        except TypeError:
            print("TypeError", col_name)
            continue

    corr_values.columns = ["var", "value"]
    corr_values_s = corr_values.sort_values("value", ascending=False)

    return corr_values_s

def plot_scatter_mat(df, attributes, path):
    """
    Plots the scatter mat. for the data
    """
    save_loc = os.path.join(path, "scatter_mat.jpg")

    plt.figure()
    scatter_matrix(df[attributes], figsize=(12,8))
    plt.savefig(save_loc)
    plt.close()

    plot_two_vars(df, "median_income", "median_house_value", path)

def plot_two_vars(df, var1, var2, path):

    save_loc = os.path.join(path, var1 + "_against_" + var2 + "_scatter_big.jpg")

    plt.figure(figsize=(12,8))
    plt.scatter(x=df[var1].values, y=df[var2].values, alpha=0.1)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(var1 + "_against_" + var2 + ".jpg")
    plt.savefig(save_loc)
    plt.close()
