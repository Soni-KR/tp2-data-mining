# Détection de fraude explicable

Application ENSI : explorer les transactions, détecter les alertes et comprendre les résultats du modèle XGBoost.

## Lancer

```powershell
docker compose up --build
```

- Interface : http://localhost:8501
- API : http://localhost:8000/docs

## Utiliser

**Transaction Monitoring → Try one transaction → Score sample** pour essayer une transaction.
Le chargement CSV, les seuils, les budgets et les explications SHAP sont disponibles dans les autres sections.

## Fichiers

| Fichier | Rôle |
|---|---|
| `dashboard.py` | Interface Streamlit |
| `api.py` | Prédiction FastAPI |
| `fraud_core.py`, `client.py` | Validation, décisions et appels API |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Installation et lancement |
| `.streamlit/config.toml` | Thème sombre |
| `preprocessor.pkl`, `xgboost_model.json`, `config.json` | Modèle entraîné et variables attendues |
| CSV et JSON de résultats | Exemples, graphiques et évaluation |
| [ONE_PAGER.pdf](ONE_PAGER.pdf) | Présentation du projet en une page |

## Résultats

IEEE-CIS : 590 540 transactions, environ 3,50 % de fraudes. Test final : **AP 0,4380 · ROC-AUC 0,8861**.
Les comparaisons des quatre modèles utilisent les mêmes 20 000 lignes de validation.

Les entrées doivent contenir les 68 variables préparées, dont l'historique. Le score n'est pas une probabilité calibrée ; les coûts sont illustratifs. Aucun réentraînement au lancement.

Les fichiers de travail, tests locaux et rapports intermédiaires sont exclus de Git. Les modèles et données nécessaires restent inclus pour que le projet fonctionne après clonage.

Structure simple inspirée de [tp1-data-mining](https://github.com/Soni-KR/tp1-data-mining), avec Docker conservé pour ce projet.
