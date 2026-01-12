import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Random Forest – Présentation du travail",
    layout="wide"
)

st.title("📊 Modèle Random Forest – Présentation du travail réalisé")

st.markdown(
    """
    Cette page présente **les analyses réalisées pour le Projet**
    avec le modèle **Random Forest**

    **Voici une documentation des résultats obtenus**.
    """
)

# ===============================
# TABS
# ===============================
tabs = st.tabs([
    "🧩 Préparation des données",
    "🌲 Choix du modèle",
    "📊 Performances",
    "🧠 Importance des variables",
    "📈 PDP",
    "🔍 Interpretation d'érreurs",
    "📐 Calibration",
    "🌲 Surrogate model"
])

# ===============================
# TAB 1 – DATA PREP
# ===============================
with tabs[0]:
    st.header("Préparation des données")

    st.markdown(
        """
        Les variables utilisées dans le modèle Random Forest sont :

        - surface_reelle_bati  
        - nombre_pieces_principales  
        - latitude  
        - longitude  
        - has_dependance  

        Les variables ajouter suite au feature engineering sont :
        - nb_ventes_commune

        La variable cible est :
        - **prix_m2**
        """
    )

    with st.expander("Traitements appliqués"):
        st.markdown(
            """
            - Encodage de `has_dependance` en variable binaire
            - Sélection volontaire d’un ensemble restreint de variables
            - Ajout de features pour enrichir le modèle 
            """
        )

# ===============================
# TAB 2 – MODELE
# ===============================
with tabs[1]:
    st.header("Modèle Random Forest")

    st.markdown(
        """
        Après avoir comparé plusieurs modèles, nous avons choisi d’utiliser un Random Forest car c’est celui qui obtenait les meilleurs résultats sur nos données. 

        Il nous a permis de mieux capturer les relations non linéaires entre les caractéristiques des biens et le prix au m². 

        Nous avons également retenu ce modèle pour les possibilités d’interprétation qu’il offre, ce qui nous a permis d’analyser et de justifier les prédictions obtenues. 

        Ce choix nous semble être un bon compromis entre performance et compréhension du modèle.
        """
    )


    st.markdown(
        """
        ## Comparaison avec le Gradient Boosting Regressor

        Nous avons dans un deuxième temps testé un Gradient Boosting Regressor. 

        Le modèle a montrait des performances plutôt décevante (RMSE ≈ 1115 et R² ≈ 0,81), les résultats obtenus restaient inférieurs à ceux du Random Forest. 

        Ce dernier permettait une meilleure précision de prédiction sur nos données, 

        ce qui nous a conduits à ne pas retenir le Gradient Boosting pour la suite du projet.

        ## Comparaison avec le LightGBM Regressor

        Nous avons également expérimenté un modèle LightGBM Regressor, qui a donné des résultats légèrement meilleurs que le Gradient Boosting 

        (RMSE ≈ 1105 et R² ≈ 0,82). Toutefois, ces performances restent en dessous de celles obtenues avec le Random Forest (RMSE ≈ 1055 et R² ≈ 0,83). 

        Pour cette raison, nous avons finalement choisi le Random Forest comme modèle principal de notre étude.
        """
    )

    



# ===============================
# TAB 3 – PERFORMANCE
# ===============================
with tabs[2]:
    st.header("Évaluation du modèle")

    st.markdown(
            """
            Pour les résultats du Random Forest on obtient : 

            #### RMSE : 1055.2417058588

            #### R2   : 0.8334907754404861

            Ce résultat est plutot moyen voir mauvais pour une prédiction sur le prix au m² ce qui nous a amené à nous demander pourquoi ?
            
            Ainsi nous avons interprété les données suivantes pour essayer de comprendre ce que l'on pouvait améliorer
            """
        )

    with st.expander("Lecture des résultats"):
        st.markdown(
            """
            Ces diagramme montre que la localisation du bien (latitude et longitude) est le facteur le plus déterminant dans la prédiction du prix au m², 
            
            ce qui confirme l’importance de l’emplacement sur le marché immobilier. Le nombre de ventes dans la commune influence fortement la prédiction

            En revanche, la surface du bâti, la présence de dépendances et le nombre de pièces ont un impact plus limité sur le prix au m². 

            Globalement, le modèle met en évidence des relations cohérentes avec les caractéristiques économiques du marché immobilier.
            """
        )


    col1, col2 = st.columns(2)

    with col1:
        st.image(
            Image.open("streamlit/assets/images/summary_plot.png"),
            caption="PDP – Surface",
            width=600
        )

    with col2:
        st.image(
            Image.open("streamlit/assets/images/feature_value.png"),
            caption="Métriques de performance du Random Forest",
            width = 600
        )


   
    

# ===============================
# TAB 4 – FEATURE IMPORTANCE
# ===============================
with tabs[3]:
    st.header("Importance des variables")

    with st.expander("Ce que montre ce graphique"):
        st.markdown(
            """
            Ce graphique illustre la décomposition de la prédiction du prix au m² pour un bien donné. 

            La prédiction finale (environ 4 550 €/m²) résulte d’une valeur de base du modèle, ajustée par l’influence des différentes variables. 

            Les variables en rouge (notamment le nombre de ventes dans la commune et la longitude) contribuent à augmenter le prix au m² prédit, 

            tandis que les variables en bleu (comme la latitude, la surface du bâti et l’absence de dépendance) ont un effet réducteur.
            """
        )

    st.image(
        Image.open("streamlit/assets/images/force_plot.png"),
        caption="Importance des variables – Random Forest",
        width=1250
    )



# ===============================
# TAB 5 – PDP
# ===============================
with tabs[4]:
    st.header("Partial Dependence Plots")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            Image.open("streamlit/assets/images/partial_dependence.png"),
            caption="PDP – Surface",
            width=500
        )

    with col2:
        st.markdown(
        """
        On observe clairement sur ce graphique que plus le **nombre de ventes**
        dans une commune est **élevé**, plus la **surface moyenne** des biens est **faible**.

        Ce phénomène peut s’expliquer par la densité de population :
        plus une métropole est dense, plus les surfaces habitables ont
        tendance à être réduites.
        """
        )

   

# ===============================
# TAB 6 – SHAP
# ===============================
with tabs[5]:
    st.header("Interprétation d'erreur")

    with st.expander("Distribution des résidus"):
        st.markdown(
            """
            Le graphique montre que le modele arrive moins à prédire les habitations avec un faible prix au m² très probablement car il y a moins de données concernant celles ci

            A l'inverse il ne semble pas avoir de difficulté quand il s'agit de prix élevé encore une fois probablement car il y a plus de données 

            car > métropole > plus de gens > plus de vente > plus de données
            """
        )

    st.image(
        Image.open("streamlit/assets/images/distribution_résidu.png"),
        width=700
    )




# ===============================
# TAB 7 – CALIBRATION
# ===============================
with tabs[6]:
    st.header("calibration du modèle")

    with st.expander("Interprétation de la calibration"):
        st.markdown(
            """
            Ce graphique de calibration montre une bonne corrélation entre le prix réel et le prix prédit au m², indiquant que le modèle capte correctement la tendance générale du marché. 

            La dispersion des points augmente pour les valeurs élevées, ce qui suggère une précision plus faible pour les biens chers. 

            On observe également une légère surévaluation des biens peu chers et une sous-évaluation des biens très chers, traduisant un effet de lissage du modèle.
            """
        )

    st.image(
        Image.open("streamlit/assets/images/calibration_du_modele.png"),
        caption="Courbe de calibration du Random Forest",
        width=700
    )


# ===============================
# TAB 8 – SURROGATE
# ===============================
with tabs[7]:
    st.header("Arbre de décision du modèle")

    st.image(
        Image.open("streamlit/assets/images/decision_tree_regressor.png"),
        caption="Modèle surrogate du Random Forest",
        width=1000
    )

    with st.expander("Pourquoi un arbre de décision ?"):
        st.markdown(
            """
            Le decision tree regressor permet d’approximer le comportement
            du Random Forest avec un modèle plus simple et lisible.
            """
        )