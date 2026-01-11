import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import os

# ----------- PARAMÈTRES -----------
FILE_PATH = Path("data/parquet/optimized_2020.parquet")  # <- plutôt que chemin absolu
MAX_POINTS = 10000


st.subheader("Données du fichier CSV")
st.write("")
with st.expander("Voir le descriptif des colonnes", expanded=False):

    data = [
        ("id_mutation", "Identifiant de mutation (non stable, sert à grouper les lignes)"),
        ("date_mutation", "Date de la mutation au format ISO-8601 (YYYY-MM-DD)"),
        ("numero_disposition", "Numéro de disposition"),
        ("nature_mutation", "Nature de la mutation"),
        ("valeur_fonciere", "Valeur foncière (séparateur décimal = point)"),
        ("adresse_numero", "Numéro de l'adresse"),
        ("adresse_suffixe", "Suffixe du numéro de l'adresse (B, T, Q)"),
        ("adresse_code_voie", "Code FANTOIR de la voie (4 caractères)"),
        ("adresse_nom_voie", "Nom de la voie de l'adresse"),
        ("code_postal", "Code postal (5 caractères)"),
        ("code_commune", "Code commune INSEE (5 caractères)"),
        ("nom_commune", "Nom de la commune (accentué)"),
        ("ancien_code_commune", "Ancien code commune INSEE (si différent lors de la mutation)"),
        ("ancien_nom_commune", "Ancien nom de la commune (si différent lors de la mutation)"),
        ("code_departement", "Code département INSEE (2 ou 3 caractères)"),
        ("id_parcelle", "Identifiant de parcelle (14 caractères)"),
        ("ancien_id_parcelle", "Ancien identifiant de parcelle (si différent lors de la mutation)"),
        ("numero_volume", "Numéro de volume"),
        ("lot_1_numero", "Numéro du lot 1"),
        ("lot_1_surface_carrez", "Surface Carrez du lot 1"),
        ("lot_2_numero", "Numéro du lot 2"),
        ("lot_2_surface_carrez", "Surface Carrez du lot 2"),
        ("lot_3_numero", "Numéro du lot 3"),
        ("lot_3_surface_carrez", "Surface Carrez du lot 3"),
        ("lot_4_numero", "Numéro du lot 4"),
        ("lot_4_surface_carrez", "Surface Carrez du lot 4"),
        ("lot_5_numero", "Numéro du lot 5"),
        ("lot_5_surface_carrez", "Surface Carrez du lot 5"),
        ("nombre_lots", "Nombre de lots"),
        ("code_type_local", "Code de type de local"),
        ("type_local", "Libellé du type de local"),
        ("surface_reelle_bati", "Surface réelle du bâti"),
        ("nombre_pieces_principales", "Nombre de pièces principales"),
        ("code_nature_culture", "Code de nature de culture"),
        ("nature_culture", "Libellé de nature de culture"),
        ("code_nature_culture_speciale", "Code de nature de culture spéciale"),
        ("nature_culture_speciale", "Libellé de nature de culture spéciale"),
        ("surface_terrain", "Surface du terrain"),
        ("longitude", "Longitude du centre de la parcelle (WGS-84)"),
        ("latitude", "Latitude du centre de la parcelle (WGS-84)")
    ]
    df = pd.DataFrame(data, columns=["Colonne", "Description"])
    st.dataframe(
        df,
        use_container_width=True,
        height=450
    )

df = pd.read_parquet(FILE_PATH, engine='pyarrow')
st.subheader("Aperçu du dataset")
st.dataframe(df.head())



st.subheader("Analyse des valeurs manquantes")
st.write("")
with st.expander("Voir l'analyse des valeurs manquantes", expanded=False):

    # --- 1. Colonnes très peu renseignées ---
    st.markdown("### Colonnes très peu renseignées (> 90 %)")
    df_high_nan = pd.DataFrame(
        [
            (
                "ancien_code_commune, lotX_surface_carrez, lotX_numero",
                "> 90 %",
                "À supprimer"
            )
        ],
        columns=["Colonne", "Taux de NaN", "Action recommandée"]
    )
    st.table(df_high_nan)

    # --- 2. Colonnes partiellement manquantes ---
    st.markdown("### Colonnes partiellement manquantes (10 % à 60 %)")
    df_mid_nan = pd.DataFrame(
        [
            (
                "surface_reelle_bati",
                "59 %",
                "Critique pour calcul prix/m² → filtrer les lignes nulles"
            ),
            (
                "nombre_pieces_principales, code_type_local",
                "43 %",
                "Supprimer ou étudier si remplaçables"
            ),
            (
                "adresse_numero, surface_terrain",
                "30–40 %",
                "Peu critiques → garder si utiles, sinon ignorer"
            )
        ],
        columns=["Colonne", "Taux de NaN", "Action recommandée"]
    )
    st.table(df_mid_nan)

    # --- 3. Colonnes faiblement incomplètes ---
    st.markdown("### Colonnes faiblement incomplètes (< 5 %)")
    df_low_nan = pd.DataFrame(
        [
            (
                "valeur_fonciere, code_postal, adresse_nom_voie",
                "< 2 %",
                "Très exploitables → conserver, filtrer les NaN à l’usage"
            ),
            (
                "latitude, longitude",
                "3.1 %",
                "Géolocalisation très utile → garder, filtrer ponctuellement"
            )
        ],
        columns=["Colonne", "Taux de NaN", "Action recommandée"]
    )
    st.table(df_low_nan)

st.subheader("Colonnes à supprimer et justification")
st.write("")
with st.expander("Voir les colonnes à supprimer et les justifications", expanded=False):

    st.markdown("### Colonnes avec très peu de données (< 10%)")
    st.markdown(
        "Ces colonnes contiennent majoritairement des valeurs manquantes et n'apportent pas d'information fiable. "
        "Nous avons choisis de les supprimer pour éviter des biais ou des erreurs lors des analyses."
    )
    cols_to_drop_1 = [
        'ancien_code_commune', 'ancien_nom_commune', 'ancien_id_parcelle',
        'lot1_numero', 'lot1_surface_carrez', 'lot2_numero', 'lot2_surface_carrez',
        'lot3_numero', 'lot3_surface_carrez', 'lot4_numero', 'lot4_surface_carrez',
        'lot5_numero', 'lot5_surface_carrez'
    ]
    st.write("Colonnes supprimées :", cols_to_drop_1)
    

    st.markdown("### Colonnes avec informations peu pertinentes ou redondantes")
    st.markdown(
        "Ces colonnes apportent peu d'information utile pour l'analyse principale. "
        "Et ne sont pas interressante pour le calcul de prila valeur foncière ou les analyses géographiques."
    )
    cols_to_drop_2 = [
        'numero_disposition', 'adresse_numero', 'adresse_suffixe',
        'adresse_nom_voie', 'adresse_code_voie', 'nature_culture_speciale',
        'numero_volume', 'code_nature_culture', 'nature_culture','code_nature_culture_speciale'
    ]
    st.write("Colonnes supprimées :", cols_to_drop_2)

st.subheader("Étude des variables catégorielles")
st.write("")
with st.expander("Voir l'étude des variables catégorielles", expanded=False):
    # Sélection des colonnes catégorielles
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ['id_mutation', 'id_parcelle']]

    st.markdown(f"**Nombre de colonnes catégorielles : {len(cat_cols)}**")
    st.write("Liste des colonnes catégorielles :", cat_cols)

    for col in cat_cols:
        st.markdown(f"---\n### {col}")
        st.write(df[col].value_counts(dropna=False).head(10))


st.subheader("Choix des variables catégorielles")
st.write("")
st.write("Nous avons décidé de nous concentrer sur la nature_mutation ainsi que type_mutation pour étudier les différentes répartitions")
st.write("Voici quelques graphiques importants qui nous ont permis d'un peu plus comprendre le dataset ainsi que de prendre certaine décision")


st.subheader("Graphiques")

# Liste des chemins vers les images locales
image_paths = [
    "streamlit/assets/images/repartion_modalite_nature_mutation.png",
    "streamlit/assets/images/repartion_modalite_type_local.png",
    "streamlit/assets/images/repartion_type_mutation.png",
    "streamlit/assets/images/prix_median_type_mutation.png",
    "streamlit/assets/images/surface_moyenne_type_mutation.png",
    "streamlit/assets/images/valeur_fonciere_par_type_mutation.png",
    "streamlit/assets/images/log_valeur_fonciere.png"
]

with st.expander("Voir les images", expanded=False):
    for path in image_paths:
        if os.path.exists(path):
            image = Image.open(path)
            st.image(image, caption=os.path.basename(path))
            st.write("")
            st.write("")
        else:
            st.warning(f"Image non trouvée : {path}")