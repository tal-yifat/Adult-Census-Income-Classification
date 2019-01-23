import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


class NominalCategoryMerger(BaseEstimator, TransformerMixin):
    """merges nominal categories that have similar proportions of positive target.

    Parameters
    ----------
    nominal_features : a list of nominal features for category combining
    label : the dataframe column name of the label
    alpha : p value threshold for deciding whether categories should be merged. Default = 0.1
    n_threshold : a threshold for the number of samples in the category above which it will not be merged
                    with a larger category. Default = None

    Attributes
    ----------
    merged_nominal_features_map_ : a dictionary that maps each original category to the category to with which
                                        it was merged.

    """

    def __init__(self, nominal_features, label, alpha=.1, n_threshold=None):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("alpha={0} must be between 0 and 1.".format(alpha))

        self.nominal_features = nominal_features
        self.label = label
        self.alpha = alpha
        if type(n_threshold) == list:
            if len(n_threshold) != len(nominal_features):
                raise ValueError("The length of n_threshold={0} and nominal_features={1} must be similar."
                                 .format(len(n_threshold), (nominal_features)))
            self.n_threshold = n_threshold
        else:
            self.n_threshold = [n_threshold] * len(nominal_features)
        self.merged_nominal_features_map_ = {}

    def aggregate_categorical_feature(self, var, df):
        # Aggregate the target variable with respect to the categorical feature
        cat_explore = pd.concat([df[var].value_counts(), df.groupby(var)[self.label].sum()],
                                axis=1, keys=['Trials', 'Successes'], sort=True)
        cat_explore.index.name = var
        return cat_explore

    def get_best_nominal_value_pair(self, nominal_values, trials, successes, value_pairs):
        # Return the pair of categories that are the most likely (in terms of p value) to be sampled from
        # the same distribution.
        p_vals = np.asarray([0.] * len(value_pairs))

        for i in range(len(value_pairs)):
            trials_i = [trials[value_pairs[i][0]]] + [trials[value_pairs[i][1]]]
            successes_i = [successes[value_pairs[i][0]]] + [successes[value_pairs[i][1]]]
            z, p_vals[i] = proportions_ztest(nobs=trials_i, count=successes_i)

        best_value_pair = sorted([nominal_values[value_pairs[p_vals.argmax()][0]],
                                  nominal_values[value_pairs[p_vals.argmax()][1]]],
                                 key=lambda x: trials[list(nominal_values).index(x)], reverse=True)

        return best_value_pair, np.amax(p_vals)

    def fit(self, X):
        """Fit NominalCategoryMerger to X.
        Parameters
        ----------
        X : a dataframe.
        Returns
        -------
        self
        """
        # Binarize label for proportion calculations
        label_values = X[self.label].unique()
        if len(label_values) != 2:
            raise ValueError("Number of classes={0} must be 2".format(len(label_values)))
        X[self.label] = X[self.label].replace({label_values[0]: 1, label_values[1]: 0})

        for var in self.nominal_features:
            # Get aggregate stats for pairs of categories and pick the best one.
            ag_results = self.aggregate_categorical_feature(var, X)
            trials = ag_results['Trials'].values
            successes = ag_results['Successes'].values
            nominal_values = ag_results.index
            value_pairs = [[i, j] for i in range(len(nominal_values)) for j in range(len(nominal_values))
                           if j > i]
            best_value_pair, p_val = self.get_best_nominal_value_pair(nominal_values, trials, successes,
                                                                      value_pairs)
            self.merged_nominal_features_map_[var] = {val: val for val in nominal_values}

            while p_val > self.alpha:
                # Unite best pair
                indices_to_change = [i for i, x in enumerate(list(self.merged_nominal_features_map_[var]
                                                                  .values())) if x == best_value_pair[1]]
                values_to_change = [list(self.merged_nominal_features_map_[var].keys())[i]
                                    for i in indices_to_change]
                for val in values_to_change:
                    self.merged_nominal_features_map_[var][val] = best_value_pair[0]
                    X[var] = X[var].apply(lambda x: best_value_pair[0] if x == val else x)

                # Find the best pair among the new set of categories.
                ag_results = self.aggregate_categorical_feature(var, X)
                trials = ag_results['Trials'].values
                successes = ag_results['Successes'].values
                nominal_values = ag_results.index
                value_pairs = [[i, j] for i in range(len(nominal_values)) for j in range(len(nominal_values))
                               if j > i]
                best_value_pair, p_val = self.get_best_nominal_value_pair(nominal_values, trials, successes,
                                                                          value_pairs)

        return self

    def transform(self, X):
        check_is_fitted(self, 'merged_nominal_features_map_')

        for var in self.nominal_features:
            X[var] = X[var].replace(self.merged_nominal_features_map_[var])
        return X


class OrdinalCategoryMerger(BaseEstimator, TransformerMixin):
    """Merges adjuscent ordinal categories that have similar proportions of positive target.

    Parameters
    ----------
    ordinal_features : a list of ordinal features for category combining
    label : the dataframe column name of the label
    alpha : p value threshold for deciding whether categories should be merged. Default = 0.1
    n_threshold : a threshold for the number of samples in the category above which it will not be merged
                    with a larger category. Default = None

    Attributes
    ----------
    merged_ordinal_features_map_ : a dictionary that maps each original category to the category to with which
                                        it was merged.

    """

    def __init__(self, ordinal_features, label, alpha=.1, n_threshold=None):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("alpha={0} must be between 0 and 1.".format(alpha))

        self.ordinal_features = ordinal_features
        self.label = label
        self.alpha = alpha
        if type(n_threshold) == list:
            if len(n_threshold) != len(ordinal_features):
                raise ValueError("The length of n_threshold={0} and ordinal_features={1} must be similar."
                                 .format(len(n_threshold), (ordinal_features)))
            self.n_threshold = n_threshold
        else:
            self.n_threshold = [n_threshold] * len(ordinal_features)
        self.merged_ordinal_features_map_ = {}

    def aggregate_categorical_feature(self, var, df):
        # Aggregate the target variable with respect to the categorical feature
        cat_explore = pd.concat([df[var].value_counts(), df.groupby(var)[self.label].sum()],
                                axis=1, keys=['Trials', 'Successes'])
        cat_explore.index.name = var
        return cat_explore.sort_index()

    def unite_ordinal_categories(self, ordinal_series, category_1, category_2):
        return ordinal_series.apply(lambda x: category_1 if x == category_2 else x)

    def get_categories_to_collapse(self, ordinal_values, trials, successes, p_vals=[None]):
        # Calculate P-values for adjacent categories
        if p_vals[0] is None:
            p_vals = [0.] * (len(ordinal_values) - 1)
        p_vals = np.asarray(p_vals)
        for i in range(len(ordinal_values) - 1):
            if p_vals[i] == 0:
                z, p = proportions_ztest(nobs=trials[i:i + 2], count=successes[i:i + 2])
                p_vals[i] = p
        return p_vals

    def fit(self, X):
        """Fit OrdinalCategoryMerger to X.
        Parameters
        ----------
        X : a dataframe.
        Returns
        -------
        self
        """
        # Binarize label for proportion calculations
        label_values = X[self.label].unique()
        if len(label_values) != 2:
            raise ValueError("Number of classes={0} must be 2".format(len(label_values)))
        X[self.label] = X[self.label].replace({label_values[0]: 1, label_values[1]: 0})

        for var in self.ordinal_features:
            # Get aggregate stats for adjacent pairs of categories and calculate P-values.
            ag_results = self.aggregate_categorical_feature(var, X)
            trials = ag_results['Trials'].values
            successes = ag_results['Successes'].values
            ordinal_values = ag_results.index
            self.merged_ordinal_features_map_[var] = {val: val for val in ordinal_values}
            pvals = self.get_categories_to_collapse(ordinal_values=ordinal_values, trials=trials,
                                                    successes=successes)

            while np.amax(pvals) > self.alpha:
                # Unite best pair
                category_1, category_2 = ordinal_values[pvals.argmax()], ordinal_values[pvals.argmax() + 1]
                self.merged_ordinal_features_map_[var][category_2] = category_1
                X[var] = self.unite_ordinal_categories(X[var], category_1, category_2)

                # Find the best pair among the new set of categories.
                ag_results = self.aggregate_categorical_feature(var=var, df=X)
                trials = ag_results['Trials'].values
                successes = ag_results['Successes'].values
                ordinal_values = ag_results.index
                pvals = self.get_categories_to_collapse(ordinal_values=ordinal_values, trials=trials,
                                                        successes=successes)
            # Recode ordinal features as consecutive integers
            values = sorted(list(set(self.merged_ordinal_features_map_[var].values())))
            for key in self.merged_ordinal_features_map_[var].keys():
                self.merged_ordinal_features_map_[var][key] = values.index(self.merged_ordinal_features_map_
                                                                           [var][key]) + 1
        # print(self.merged_ordinal_features_map_)
        return self

    def transform(self, X):
        check_is_fitted(self, 'merged_ordinal_features_map_')

        for var in self.ordinal_features:
            X[var] = X[var].replace(self.merged_ordinal_features_map_[var])

        return X


class UnknownValuesDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, unknown_value='?'):
        self.unknown_value = unknown_value
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X
        for feat in self.features:
            X_transformed = X_transformed[X_transformed[feat] != self.unknown_value]
        print("Dropped {} rows with unknown values.".format(X.shape[0] - X_transformed.shape[0]))
        return X_transformed


class CharacterStripper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_strip, character_to_strip='.'):
        self.character_to_strip = character_to_strip
        self.features_to_strip = features_to_strip

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X
        for feat in self.features_to_strip:
            X_transformed[feat] = X_transformed[feat].str.strip(self.character_to_strip)
        return X_transformed
