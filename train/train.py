# 📦 Imports
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import os
print(os.getcwd())

# =========================
# 🔧 Custom Transformer
# =========================
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
        X["nb_ventes_commune"] = X["nom_commune"].map(self.commune_counts_)
        X["nb_ventes_commune"] = X["nb_ventes_commune"].fillna(self.median_)
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


# =========================
# 📊 Chargement données
# =========================
df = pd.read_parquet('data/prod/df_model_appart_2020.parquet.gz', engine='pyarrow')

df["has_dependance"] = df["has_dependance"].astype(int)

FEATURES_BASE = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "has_dependance",
]

TARGET = "prix_m2"

X = df[FEATURES_BASE + ["nom_commune"]].copy()
y = df[TARGET]

# =========================
# ✂️ Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🚀 Pipeline
# =========================
pipeline = Pipeline(steps=[
    ("commune_encoder", CommuneSalesEncoder()),
    ("feature_selector", FeatureSelector(FEATURES_BASE + ["nb_ventes_commune"])),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=22,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ))
])

# =========================
# 🎯 Train
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 📈 Predict
# =========================
y_pred = pipeline.predict(X_test)

# =========================
# 📊 Evaluation
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Pipeline Random Forest")
print("RMSE :", rmse)
print("R2   :", r2)