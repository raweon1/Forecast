from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Hands-On Machine Learning, Aurelien Geron, Chapter 2
# Takes a Pandas DataFrame and returns selected columns as Numpy Array
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attr_names = attr_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attr_names].values


class PrefixDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attr_names = attr_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[col for col in X if col.startswith((*self.attr_names,))]].values


class PrefixMeanDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attr_names = attr_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = []
        for prefix in self.attr_names:
            columns = [col for col in X if col.startswith(prefix)]
            if columns.__len__() > 1:
                values.append(X[columns].mean(axis=1).values)
            else:
                values.append(X[columns].values[:, 0])
        return np.column_stack(values)
