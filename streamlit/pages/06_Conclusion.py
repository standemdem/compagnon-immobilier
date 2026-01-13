import streamlit as st

st.set_page_config(
    page_title="Conclusion",
    layout="wide"
)

# ===============================
# TAB – CONCLUSION
# ===============================
st.header("🏁 Conclusion du projet")

conclusion_tabs = st.tabs([
    "🎯 Objectifs & démarche",
    "📊 Résultats obtenus",
    "⚠️ Limites identifiées",
    "🚀 Pistes d’amélioration",
    "🎓 Bilan personnel"
])

# ===============================
# TAB 1 – OBJECTIFS
# ===============================
with conclusion_tabs[0]:
    st.subheader("Objectifs et démarche")

    st.markdown(
        """
        L’objectif principal de ce projet était de **prédire le prix au m² des biens immobiliers**
        à partir de données issues du marché immobilier, tout en cherchant à **comprendre les facteurs
        qui influencent réellement ces prix**.

        Nous avons suivi une démarche complète de data science :
        - sélection et préparation des données  
        - feature engineering  
        - comparaison de plusieurs modèles de machine learning  
        - interprétation des résultats à l’aide d’outils explicatifs  

        L’enjeu n’était donc pas uniquement d’obtenir de bonnes performances,
        mais aussi de proposer un modèle **compréhensible et justifiable**.
        """
    )

# ===============================
# TAB 2 – RESULTATS
# ===============================
with conclusion_tabs[1]:
    st.subheader("Résultats du modèle")

    st.markdown(
        """
        Le **Random Forest Regressor** s’est imposé comme le modèle le plus performant parmi ceux testés,
        avec les résultats suivants :

        - **RMSE ≈ 1055 €**
        - **R² ≈ 0,83**

        Ces performances restent correctes compte tenu de la complexité du marché immobilier,
        même si elles ne permettent pas une prédiction parfaitement précise du prix au m².

        Les analyses d’importance des variables, de PDP et de SHAP ont montré que :
        - la **localisation** (latitude, longitude) est le facteur le plus déterminant  
        - le **nombre de ventes dans la commune** joue un rôle clé  
        - les caractéristiques propres au bien (surface, nombre de pièces, dépendances)
          ont un impact plus limité sur le prix au m²  

        Ces résultats sont cohérents avec la réalité économique du marché immobilier.
        """
    )

# ===============================
# TAB 3 – LIMITES
# ===============================
with conclusion_tabs[2]:
    st.subheader("Limites du modèle")

    st.markdown(
        """
        Malgré des résultats satisfaisants, plusieurs limites ont été identifiées :

        - Le modèle a plus de difficultés à prédire les **biens à faible prix au m²**,
          probablement en raison d’un manque de données dans ces segments.
        - On observe un **effet de lissage** :
            - légère surévaluation des biens peu chers  
            - sous-évaluation des biens très chers  
        - Le Random Forest capte bien les tendances globales,
          mais peine à modéliser les situations atypiques ou très locales.

        Ces limites montrent que la performance du modèle dépend fortement
        de la **richesse et de la diversité des variables disponibles**.
        """
    )

# ===============================
# TAB 4 – AMELIORATIONS
# ===============================
with conclusion_tabs[3]:
    st.subheader("Pistes d’amélioration")

    st.markdown(
        """
        Plusieurs axes d’amélioration pourraient permettre d’augmenter
        la précision et la robustesse du modèle :

        **Ajout de nouvelles variables socio-démographiques :**
        - nombre d’habitants par km²  
        - revenu médian de la commune  
        - taux de chômage  
        - typologie de la zone (urbaine, périurbaine, rurale)

        **Feature engineering plus avancé :**
        - nombre d’habitants par m² habitable  
        - distance au centre-ville ou aux pôles économiques  
        - proximité des transports, écoles ou commerces  

        **Approche par segmentation :**
        - modèles distincts selon le type de bien ou la zone géographique  
        - ou intégration de ces informations via des variables catégorielles  

        Ces améliorations permettraient de mieux capter
        les dynamiques locales du marché immobilier.
        """
    )

# ===============================
# TAB 5 – BILAN
# ===============================
with conclusion_tabs[4]:
    st.subheader("Bilan du projet")

    st.markdown(
        """
        Ce projet nous a permis de mobiliser l’ensemble des compétences
        abordées au cours de la formation, aussi bien techniques que méthodologiques.

        Au-delà des performances chiffrées, nous avons surtout appris à :
        - analyser la pertinence d’un modèle  
        - interpréter ses prédictions  
        - identifier ses limites  

        Le Random Forest s’est révélé être un outil pertinent pour ce type de problématique,
        tout en laissant entrevoir de nombreuses possibilités d’amélioration.

        En conclusion, ce travail constitue une **base solide** pour aller plus loin
        dans la modélisation du marché immobilier et dans l’application
        de méthodes de data science à des problématiques réelles.
        """
    )
