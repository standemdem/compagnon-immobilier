from sklearn.base import BaseEstimator, TransformerMixin


class CommuneSalesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.commune_counts_ = None
        self.median_ = None

    def fit(self, X, y=None):
        counts = X.groupby("nom_commune").size()
        self.commune_counts_ = counts
        self.median_ = counts.median()
        return self

    def transform(self, X):
        X = X.copy()
        X["nb_ventes_commune"] = X["nom_commune"].map(
            self.commune_counts_
        )
        X["nb_ventes_commune"] = X["nb_ventes_commune"].fillna(
            self.median_
        )
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]