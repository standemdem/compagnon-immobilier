import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score



df = pd.read_parquet('../data/parquet/full_2020.csv.parquet', engine='pyarrow')



# Liste des colonnes à supprimer (remplissage < 10%)
cols_to_drop_1 = [
    'ancien_code_commune', 'ancien_nom_commune', 'ancien_id_parcelle',
    'lot1_numero', 'lot1_surface_carrez', 'lot2_numero', 'lot2_surface_carrez',
    'lot3_numero', 'lot3_surface_carrez', 'lot4_numero', 'lot4_surface_carrez',
    'lot5_numero', 'lot5_surface_carrez'
]
# Liste des colonnes avec infos inutiles
cols_to_drop_2 = [
    'id_mutation', 'numero_disposition', 'adresse_numero', 'adresse_suffixe',
    'adresse_nom_voie', 'adresse_code_voie', 'nom_commune', 'id_parcelle',
    'numero_volume', 'code_nature_culture', 'nature_culture','code_nature_culture_speciale'
]
df.drop(columns=cols_to_drop_1, inplace=True)
# Vérification des colonnes restantes
print(f"Shape after dropping columns: {df.shape}")

def optimize_dataframe(df, parse_dates=None, category_thresh=0.05, verbose=True):
    """
    Optimise les types d'un DataFrame pour réduire l'utilisation mémoire :
    - convertit les objets en catégories si nombre de modalités faible
    - convertit les float64 en float32
    - convertit les int64 en int32
    - convertit les colonnes de dates

    Parameters:
    - df : DataFrame à optimiser
    - parse_dates : liste de colonnes à parser comme dates
    - category_thresh : seuil max de ratio modalité/nb lignes pour transformer en 'category'
    - verbose : affiche la mémoire gagnée

    Returns:
    - df optimisé
    """

    initial_memory = df.memory_usage(deep=True).sum() / 1024**2

    # Dates
    if parse_dates:
        for col in parse_dates:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Float → float32
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float32')

    # Int → int32
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].isnull().any():
            df[col] = df[col].astype('Int32')
        else:
            df[col] = df[col].astype('int32')

    # Object → category si peu de modalités
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique(dropna=False) / len(df) <= category_thresh:
            df[col] = df[col].astype('category')

    final_memory = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"💾 Mémoire utilisée : {initial_memory:.2f} Mo → {final_memory:.2f} Mo ({100 * (1 - final_memory/initial_memory):.1f}% gagné)")

    return df

# Application test sur ech_annonces_ventes_68.csv
optimized_df1 = optimize_dataframe(df, parse_dates=['date_mutation'], verbose=True)
optimized_df1.info()
optimized_df1.head()


# Vérification si le DataFrame est vide
if optimized_df1.empty:
    raise ValueError("Le DataFrame 'optimized_df1' est vide après suppression des valeurs manquantes.")

# Calcul de la fréquence des communes
commune_freq = optimized_df1['nom_commune'].value_counts()
optimized_df1['commune_freq'] = optimized_df1['nom_commune'].map(commune_freq)

# Création de la colonne 'coordonnees'
optimized_df1['coordonnees'] = list(zip(optimized_df1['latitude'], optimized_df1['longitude']))

# Colonnes choisies
colonnes_choisies = [
    'valeur_fonciere', 'surface_terrain', 'surface_reelle_bati',
    'code_type_local', 'nombre_pieces_principales',
    'commune_freq', 'nombre_lots'
]

# Vérification des colonnes manquantes
colonnes_manquantes = [col for col in colonnes_choisies if col not in optimized_df1.columns]
if colonnes_manquantes:
    raise ValueError(f"Les colonnes suivantes sont manquantes dans 'optimized_df1' : {colonnes_manquantes}")

# Création du DataFrame final
dfRFR = optimized_df1[colonnes_choisies].copy()
# Suppression des valeurs manquantes
dfRFR = dfRFR.dropna()
# Vérification si le DataFrame est vide
if dfRFR.empty:
    raise ValueError("Le DataFrame 'dfRFR' est vide après sélection des colonnes.")



# ---------------------------
# Début de l'app Streamlit
# ---------------------------
st.title("🌲 Random Forest Regressor - Analyse et Visualisation")
st.subheader("Aperçu du dataset :")
st.dataframe(dfRFR.head())


# ---------------------------
# Choix des variables
# ---------------------------
st.subheader("🔧 Choix des variables")
target = st.selectbox("Variable cible (y)", dfRFR.columns)

features = st.multiselect(
    "Variables explicatives (X)",
    dfRFR.columns,
    default=[col for col in dfRFR.columns if col != target]
)

if not features:
    st.info("Sélectionne au moins une variable explicative pour continuer.")
    st.stop()

X = dfRFR[features]
y = dfRFR[target]

# Vérifications basiques
if X.empty or y.empty:
    st.error("X ou y est vide après sélection. Vérifie tes colonnes.")
    st.stop()

# Ajuster test_size si nécessaire (garde ta logique)
test_size = 0.2
if len(dfRFR) < 5:
    test_size = 1 / len(dfRFR)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.markdown(f"**X_train shape :** {X_train.shape}<br>**X_test shape :** {X_test.shape}", unsafe_allow_html=True)

# ---------------------------
# Random Forest Regressor
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------
# Métriques
# ---------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Performances du modèle")
st.metric("RMSE", f"{rmse:.4f}")
st.metric("R² Score", f"{r2:.4f}")

# ---------------------------
# Importance des variables
# ---------------------------
st.subheader("📌 Importance des variables")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot.barh(ax=ax)
plt.title("Importance des variables")
st.pyplot(fig)

# ---------------------------
# Prédiction utilisateur
# ---------------------------
st.subheader("📝 Prédiction sur de nouvelles valeurs")
new_values = []
for col in X.columns:
    default_val = float(X[col].mean()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0
    val = st.number_input(f"{col}", value=default_val)
    new_values.append(val)

if st.button("Prédire"):
    # s'assurer que la forme est correcte (1, n_features)
    pred = model.predict([new_values])
    st.success(f"Prédiction : {pred[0]:.4f}")
