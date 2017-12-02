"""
Can define classes which alter the dataset accordingly in a specific way. The
BaseEstimator and TransformerMixin are needed to define our own transformer
methods, with __init__, fit and transform methods, can also add more as needed
for more complex processes.

Then use the pipeplines to apply the each transform to the data in turn.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    ROOMS_IX = 3
    BEDROOMS_IX = 4
    POPULATION_IX = 5
    HOUSEHOLD_IX = 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.ROOMS_IX] / X[:, self.HOUSEHOLD_IX]
        population_per_household = X[:, self.POPULATION_IX] / X[:, self.HOUSEHOLD_IX]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.BEDROOMS_IX] / X[:, self.ROOMS_IX]

            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DataFrame_selector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class PCA_selector(BaseEstimator, TransformerMixin):

    def __init__(self, n_comps):
        self.n_comps = n_comps
    def fit(self, X):
        return self
    def transform(self, X):
        print(X.shape)
        pca = PCA(n_components=self.n_comps,  svd_solver='full')
        out = pca.fit_transform(X)
        print(out.shape)
        return out

def feature_engineering_norm(df):

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    corr_vals = calc_corrs(df, "median_house_value")
    print(corr_vals)

    return corr_vals

def convert_to_numeric(df):

    non_numeric = df.select_dtypes(exclude=[np.number])
    df.drop(non_numeric.columns, axis=1, inplace=True)
    encoder = LabelEncoder()
    housing_cat_encoded = encoder.fit_transform(non_numeric.values.reshape(len(df),))
    housing_cat_encoded_df = pd.DataFrame(housing_cat_encoded, columns=["housing_cat_encoded"])

    values, counts = stats.mode(housing_cat_encoded)
    index = np.argmax(counts)
    modal_val = values[index]

    housing_cat_encoded_df.fillna(value=modal_val, inplace=True)
    final_df = pd.concat([df, housing_cat_encoded_df], axis=1)

    return final_df

def one_hot_encode(df, var):

    df_s = df[[var]]

    encoded_array = np.zeros([len(df_s), len(np.unique(df_s.values))])

    for row, value in df_s.iterrows():
        encoded_array[row, value] = 1

    return encoded_array

def using_binarizer(df, var):
    """
    Performs the one hot encoding in one step (does not require numerical)
    """

    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(df[var])

    return housing_cat_1hot

def sort_missing_data(df, drop_vars):

    imputer = Imputer(strategy="median")
    new_df = df.drop(drop_vars, axis=1)

    # imputer.statistics_ holds the median values
    # fits and transforms in one go
    train_set_array = imputer.fit_transform(new_df)

    train_df = pd.DataFrame(train_set_array, columns=new_df.columns)

    train_df_all = pd.concat([df[drop_vars], new_df], axis=1)
    return train_df_all

def use_attr_class(df):

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    extra_attrbs = attr_adder.transform(df.values)

    return extra_attrbs

def using_pipeline(df, target_var):

    df.drop(target_var, axis=1, inplace=True)

    num_attribs = list(df.select_dtypes(include=[np.number]).columns)
    cat_attribs = list(df.select_dtypes(exclude=[np.number]).columns)

    n_components = 9

    num_pipeline = Pipeline([
            ('selector', DataFrame_selector(num_attribs)),
            ('imputer', Imputer(strategy='median')),
            ('attrbs_attr', CombinedAttributesAdder()),
            ('std_scalar', StandardScaler()),
            #('pca_analysis', PCA_selector(n_components))
        ])

    cat_pipeline = Pipeline([
            ('selector', DataFrame_selector(cat_attribs)),
            ('label_binarizer', LabelBinarizerPipelineFriendly())
        ])

    full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ])

    full_pip_activated = full_pipeline.fit_transform(df)

    return full_pip_activated
