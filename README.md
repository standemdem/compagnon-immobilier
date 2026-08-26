# Compagnon Immobilier 🏡

Projet Data Science collaboratif – Aide à l'achat immobilier en France.

## Objectifs 🎯
- Comparer les territoires à partir de données socio-économiques, transports, éducation, criminalité, etc.
- Estimer le prix au m² d’un logement donné
- Prédire l’évolution du marché immobilier

## Organisation du repo 🗂️
- `data/` : données brutes et traitées
- `notebooks/` : notebooks d'exploration, modélisation, visualisation
- `src/` : scripts Python (nettoyage, entraînement, affichage)
- `outputs/` : graphiques, cartes, résultats
- `scripts/` : automatisation de certaines tâches (dl datasets)

## Membres de l'équipe 👥
- [@standemdem](https://github.com/standemdem)
- [@NajehRhaiem](https://github.com/NajehRhaiem)
- [@YlanF](https://github.com/YlanF)

## Lancement du projet 🚀
1. 🔁 Cloner le dépôt
```bash
git clone git@github.com:TON-UTILISATEUR/compagnon-immobilier.git
cd compagnon-immobilier
```
Remplace TON-UTILISATEUR par ton nom d'utilisateur ou organisation GitHub.

2. 🧪 Créer et activer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```
Sur Windows (PowerShell) :

```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\Activate.ps1
```
3. 📦 Installer les dépendances
```bash
pip install -r requirements.txt
```
4. 📥 Télécharger, extraire et convertir les fichiers de données en .parquet

```bash
python3 scripts/dl_csvs.py
```








## note pour executer le train du model il faut se placer à la racine du projet et executer
```
python -m train.train
```


## pour faire une prédiction
```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "x-api-key: test_compagnon_immo" \
  -d '{
    "surface_reelle_bati": 85.5,
    "nombre_pieces_principales": 4,
    "latitude": 48.8985,
    "longitude": 2.0932,
    "has_dependance": 1,
    "nom_commune": "Saint-Germain-en-Laye"
  }'
```