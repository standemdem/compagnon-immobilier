import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="Carte", layout="wide")

# ----------- PARAMÈTRES -----------
FILE_PATH = Path("data/parquet/optimized_2020.parquet")  # <- plutôt que chemin absolu
MAX_POINTS = 100000

# ----------- CHARGEMENT DES DONNÉES -----------
@st.cache_data
def load_data(path: str):
    cols = [
        "longitude", "latitude",
        "code_departement", "nom_commune",
        "valeur_fonciere", "surface_reelle_bati"
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df.dropna(subset=["longitude", "latitude", "valeur_fonciere"])
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    df["valeur_fonciere"] = df["valeur_fonciere"].astype(float)
    return df

df = load_data(str(FILE_PATH))

st.title("🗺️ Carte des transactions")

# ----------- SIDEBAR : FILTRES GÉOGRAPHIQUES -----------
st.sidebar.title("Filtres géographiques")

departements = df["code_departement"].sort_values().unique()
selected_dep = st.sidebar.selectbox("Choix du département", departements)

df_dep = df[df["code_departement"] == selected_dep]
communes = df_dep["nom_commune"].sort_values().unique()
selected_com = st.sidebar.selectbox("Choix de la commune", communes)

df_filtered = df_dep[df_dep["nom_commune"] == selected_com]

# ----------- SLIDER PRIX AVEC SAISIE -----------
st.sidebar.markdown("### Filtrer par valeur foncière (€)")

min_price_total = int(df_filtered["valeur_fonciere"].min())
max_price_total = int(df_filtered["valeur_fonciere"].max())

col1, col2 = st.sidebar.columns(2)
price_min_input = col1.number_input(
    "Min", min_value=min_price_total, max_value=max_price_total,
    value=min_price_total, step=1000, format="%d"
)
price_max_input = col2.number_input(
    "Max", min_value=min_price_total, max_value=max_price_total,
    value=max_price_total, step=1000, format="%d"
)

price_range = st.sidebar.slider(
    "Ajuster la plage (rapide)",
    min_value=min_price_total,
    max_value=max_price_total,
    value=(price_min_input, price_max_input),
    step=1000,
    format="€%d"
)

price_min, price_max = price_range
df_filtered = df_filtered[
    (df_filtered["valeur_fonciere"] >= price_min) &
    (df_filtered["valeur_fonciere"] <= price_max)
]

st.write(f"🔎 **Nombre de points affichés** : {len(df_filtered):,}")

# Cas 0 point (évite crash sur mean / zoom)
if df_filtered.empty:
    st.warning("Aucun résultat avec ces filtres. Élargis la plage de prix ou change de commune.")
    st.stop()

# ----------- ÉCHANTILLONNAGE -----------
df_sample = df_filtered.sample(MAX_POINTS, random_state=42) if len(df_filtered) > MAX_POINTS else df_filtered.copy()

# ----------- TOOLTIP DYNAMIQUE -----------
tooltip_cols = ["longitude", "latitude", "valeur_fonciere", "surface_reelle_bati", "nom_commune"]
existing_cols = [col for col in tooltip_cols if col in df_sample.columns]

df_sample = df_sample[existing_cols].copy().dropna(subset=["longitude", "latitude"])
df_sample["longitude"] = df_sample["longitude"].astype(float)
df_sample["latitude"] = df_sample["latitude"].astype(float)

if "surface_reelle_bati" in df_sample.columns:
    df_sample["surface_reelle_bati"] = df_sample["surface_reelle_bati"].fillna("N/A")

tooltip_html = "<b>Commune :</b> {nom_commune}<br/><b>Prix :</b> {valeur_fonciere} €"
if "surface_reelle_bati" in df_sample.columns:
    tooltip_html += "<br/><b>Surface :</b> {surface_reelle_bati} m²"

tooltip = {
    "html": tooltip_html,
    "style": {"backgroundColor": "white", "color": "black", "fontSize": "12px"}
}

# ----------- AFFICHAGE CARTE -----------
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_sample,
    get_position='[longitude, latitude]',
    get_color='[0, 100, 200, 160]',
    get_radius=30,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=float(df_sample["latitude"].mean()),
    longitude=float(df_sample["longitude"].mean()),
    zoom=12,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
))
