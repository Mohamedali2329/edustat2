# EduStat TN — Orientation Universitaire

> Système intelligent de recommandation et de prédiction de filières universitaires tunisiennes, basé sur les données historiques d'admission 2022–2025.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange)](https://xgboost.ai)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-blue)](https://mlflow.org)

---

## Structure du Projet

```
edustat2/
├── backend/
│   ├── main.py              ← API FastAPI (5 endpoints)
│   └── requirements.txt     ← Dépendances backend
├── frontend/
│   └── index.html           ← Interface utilisateur (HTML/CSS/JS)
├── data/
│   ├── tunisie_orientation_complete.csv   ← Dataset (2880 lignes)
│   ├── best_orientation_model.joblib      ← Modèle XGBoost entraîné
│   ├── label_encoders.joblib              ← Encodeurs LabelEncoder
│   ├── model_metadata.json                ← Métriques du modèle
│   ├── classification_report.txt          ← Rapport détaillé
│   ├── demo_recommendations.json          ← Recommandations test
│   └── eda_0[1-6]_*.png                   ← 6 graphiques EDA
├── modeling.py              ← Pipeline ML complet
├── requirements.txt         ← Dépendances globales
└── README.md
```

---

## Démarrage Rapide

### 1. Installer les dépendances

```bash
cd edustat2

# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Installer les dépendances
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2. Entraîner le modèle (si pas encore fait)

```bash
python modeling.py
```

> Génère `data/best_orientation_model.joblib` et les graphiques EDA.

### 3. Lancer le backend API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API disponible sur : `http://127.0.0.1:8000`
Documentation Swagger : `http://127.0.0.1:8000/docs`

### 4. Ouvrir le frontend

Ouvrir directement dans le navigateur :

```
frontend/index.html
```

Ou servir avec Python :

```bash
python -m http.server 3000 --directory frontend
# → http://localhost:3000
```

### 5. Visualiser les runs MLflow (optionnel)

```bash
mlflow ui --backend-store-uri file:///CHEMIN_ABSOLU/edustat2/data/mlruns --port 5000
# → http://localhost:5000
```

---

## API Endpoints

| Méthode | Route            | Description                        |
| ------- | ---------------- | ---------------------------------- |
| `GET`   | `/`              | Health check + métriques modèle    |
| `GET`   | `/api/sections`  | Liste des sections bac disponibles |
| `GET`   | `/api/domaines`  | Liste des domaines avec comptage   |
| `GET`   | `/api/stats`     | Statistiques globales du dataset   |
| `POST`  | `/api/recommend` | Recommandations de filières        |
| `POST`  | `/api/predict`   | Prédiction ML du domaine           |

### Exemple — Recommandation

```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"section_bac": "M", "score_etudiant": 155.0, "top_n": 5}'
```

### Exemple — Prédiction ML

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"section_bac": "S", "score_2025": 130.0}'
```

---

## Pipeline ML

| Étape                | Détail                                                         |
| -------------------- | -------------------------------------------------------------- |
| **Dataset**          | 2880 enregistrements, 149 filières, 12 universités, 7 sections |
| **Features**         | Scores 4 années, moyenne, tendance, stabilité, section encodée |
| **Cible**            | Domaine (11 classes métier)                                    |
| **Rééquilibrage**    | SMOTE (k_neighbors adaptatif)                                  |
| **Optimisation**     | GridSearchCV + StratifiedKFold(5)                              |
| **Modèles comparés** | Random Forest vs XGBoost                                       |
| **Meilleur modèle**  | **XGBoost** (F1=0.60, Accuracy=0.60)                           |
| **Tracking**         | MLflow (params + métriques + artefacts)                        |

---

## Résultats

| Modèle                | Accuracy   | F1 (weighted) | Précision  | Rappel     |
| --------------------- | ---------- | ------------- | ---------- | ---------- |
| Random Forest + SMOTE | 0.5538     | 0.5499        | 0.5604     | 0.5538     |
| **XGBoost + SMOTE**   | **0.6024** | **0.6036**    | **0.6221** | **0.6024** |

---

## Dataset

- **Source** : Données officielles d'orientation universitaire tunisienne
- **Période** : 2022 – 2025
- **Champs** : Code Filière, Université, Établissement, Section Bac, Scores annuels

---

## Dépendances Principales

```
fastapi        # API REST
uvicorn        # Serveur ASGI
xgboost        # Modèle ML principal
scikit-learn   # Preprocessing, métriques, GridSearchCV
imbalanced-learn  # SMOTE
mlflow         # Tracking expériences
pandas / numpy # Manipulation données
```

---

_EduStat TN — Tunisie 2025_
