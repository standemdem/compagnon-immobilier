from pathlib import Path
import joblib
import streamlit as st
import pandas as pd


BASE_PATH = Path("data/models")

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_PATH / "random_forest_prix_m2.joblib")
    sales_per_commune = joblib.load(BASE_PATH / "sales_per_commune.joblib")
    median_sales = joblib.load(BASE_PATH / "median_sales.joblib")
    return model, sales_per_commune, median_sales



FEATURES_FINAL = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "latitude",
    "longitude",
    "has_dependance",
    "nb_ventes_commune",
]

def add_nb_ventes_commune(
    df: pd.DataFrame,
    sales_per_commune,
    median_sales: float,
) -> pd.DataFrame:
    df = df.copy()

    df["nom_commune"] = df["nom_commune"].str.strip()

    df["nb_ventes_commune"] = (
        df["nom_commune"]
        .map(sales_per_commune)
        .fillna(median_sales)
    )

    df["commune_inconnue"] = df["nom_commune"].map(sales_per_commune).isna()
    return df

def predict_prix_m2(input_df: pd.DataFrame) -> tuple[float, bool]:
    model, sales_per_commune, median_sales = load_artifacts()

    df = add_nb_ventes_commune(input_df, sales_per_commune, median_sales)

    pred = model.predict(df[FEATURES_FINAL])[0]
    commune_inconnue = bool(df["commune_inconnue"].iloc[0])

    return float(pred), commune_inconnue


st.set_page_config(page_title="Prédiction €/m²", page_icon="🔮", layout="wide")
st.title("🔮 Prédiction du prix au m²")

st.markdown("### 🧾 Caractéristiques du bien")

c1, c2, c3 = st.columns(3)
with c1:
    surface = st.number_input("Surface (m²)", 5.0, 300.0, 40.0)
with c2:
    pieces = st.number_input("Nombre de pièces", 1, 10, 2)
with c3:
    has_dep = st.selectbox("Dépendance", [0, 1], format_func=lambda x: "Oui" if x else "Non")

st.markdown("### 📍 Localisation")
c1, c2, c3 = st.columns(3)
with c1:
    latitude = st.number_input("Latitude", format="%.6f", value=48.8566)
with c2:
    longitude = st.number_input("Longitude", format="%.6f", value=2.3522)
with c3:
    commune = st.text_input("Nom de la commune", value="Paris")

if st.button("Estimer le prix au m²", type="primary"):
    input_df = pd.DataFrame([{
        "surface_reelle_bati": surface,
        "nombre_pieces_principales": pieces,
        "latitude": latitude,
        "longitude": longitude,
        "has_dependance": int(has_dep),
        "nom_commune": commune,
    }])

    pred, commune_inconnue = predict_prix_m2(input_df)

    st.success(f"💰 **{pred:,.0f} € / m²**".replace(",", " "))
    st.info(f"Prix total estimé : **{pred * surface:,.0f} €**".replace(",", " "))

    if commune_inconnue:
        st.warning(
            "⚠️ Commune absente du jeu d’entraînement. "
            "La prédiction utilise une valeur médiane nationale."
        )

    with st.expander("🔍 Variables envoyées au modèle"):
        st.dataframe(input_df)
