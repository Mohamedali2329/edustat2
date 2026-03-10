"""
EduStat TN — Backend API (FastAPI)
===================================
Endpoints :
  GET  /                  → health check
  GET  /api/sections      → liste des sections bac
  GET  /api/domaines      → liste des domaines
  GET  /api/stats         → statistiques globales
  POST /api/recommend     → recommandations de filières
  POST /api/predict       → prédiction du domaine probable
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Chemins ───────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "best_orientation_model.joblib"
ENC_PATH   = DATA_DIR / "label_encoders.joblib"
CSV_PATH   = DATA_DIR / "tunisie_orientation_complete.csv"
META_PATH  = DATA_DIR / "model_metadata.json"

# ── Chargement des artefacts au démarrage ─────────────────────
print("Chargement du modèle et des données...")

bundle   = joblib.load(MODEL_PATH)
encoders = joblib.load(ENC_PATH)
model    = bundle["model"]
meta     = json.loads(META_PATH.read_text(encoding="utf-8"))

df_raw = pd.read_csv(CSV_PATH)

YEAR_COLS = ["Score_2022", "Score_2023", "Score_2024", "Score_2025"]

# Feature engineering minimal (même logique que modeling.py)
for col in YEAR_COLS:
    medians = df_raw.groupby(["Section_Bac", "Filiere"])[col].transform("median")
    df_raw[col] = df_raw[col].fillna(medians).fillna(df_raw[col].median())

df_raw["Score_Mean"]      = df_raw[YEAR_COLS].mean(axis=1).round(4)
df_raw["Score_Trend"]     = (df_raw["Score_2025"] - df_raw["Score_2022"]).round(4)
df_raw["Score_Stability"] = df_raw[YEAR_COLS].std(axis=1).round(4)
df_raw["Score_Max"]       = df_raw[YEAR_COLS].max(axis=1)
df_raw["Score_Min"]       = df_raw[YEAR_COLS].min(axis=1)

DOMAIN_KEYWORDS = {
    "Informatique & Tech"      : ["informatique","technologies","télécom","numérique","réseaux","systèmes embarqués","internet des objets","tic"],
    "Ingénierie & Génie"       : ["génie","mécanique","électronique","électrotechnique","civil","industriel","énergétique","matériaux","cycle préparatoire","instrumentation et mesure"],
    "Sciences Fondamentales"   : ["physique","chimie","biologie","mathématiques","sciences de la vie","biotechnologie","biochimie","sciences de la terre","géomatique"],
    "Médecine & Santé"         : ["médecine","pharmacie","dentaire","chirurgie","paramédical","kinésithérapie","santé","infirmière","infirmier","infirmières","physiothérapie","orthophonie","ergothérapie","audioprothèse","optique","lunetterie","imagerie médicale","radiothérapie","bloc opératoire","obstétrique","sage-femme","nutrition","appareillage orthopédique"],
    "Économie & Gestion"       : ["gestion","commerce","finance","comptabilité","affaires","marketing","économie","management","logistique","assurance","sciences économiques","hôtellerie"],
    "Droit & Sciences Sociales": ["droit","juridique","sciences sociales","sociologie","psychologie","criminologie","intervention sociale","service social","sciences du travail","travail social"],
    "Lettres & Langues"        : ["arabe","anglais","français","allemand","espagnol","lettres","traduction","langues","chinois","italien","russe","langue des signes","communication","journalisme"],
    "Arts & Architecture"      : ["architecture","design","arts","théâtre","musique","audiovisuel","patrimoine","urbanisme","aménagement"],
    "Sciences Humaines"        : ["histoire","géographie","philosophie","civilisation","animation","tourisme","sport","staps","football","éducation","enseignement","archéologie","anthropologie","géopolitique","relations internationales","sciences islamiques","sciences religieuses"],
    "Agronomie & Environnement": ["agronomie","agriculture","agroalimentaire","environnement","hydraulique","vétérinaire","alimentaire","sciences de la mer","sciences agronomiques"],
}

def assign_domain(filiere: str) -> str:
    name = filiere.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in name for k in kws):
            return domain
    return "Autres"

df_raw["Domaine"] = df_raw["Filiere"].apply(assign_domain)

SECTION_LABELS = {
    "M" : "Mathématiques",
    "S" : "Sciences Expérimentales",
    "T" : "Sciences Techniques",
    "I" : "Sciences Informatiques",
    "E" : "Économie et Gestion",
    "L" : "Lettres",
    "SP": "Sport",
}

print("Artefacts chargés.")

# ── Application FastAPI ───────────────────────────────────────
app = FastAPI(
    title="EduStat TN — API d'Orientation Universitaire",
    description="Recommande des filières universitaires tunisiennes selon la section bac et le score.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas Pydantic ──────────────────────────────────────────
class RecommendRequest(BaseModel):
    section_bac    : str   = Field(..., example="M",     description="Code section bac")
    score_etudiant : float = Field(..., example=145.0,   description="Score du bachelier (~200 pts)")
    top_n          : int   = Field(10,  example=10,      description="Nombre de recommandations")
    year_ref       : str   = Field("Score_2025", example="Score_2025")

class PredictRequest(BaseModel):
    section_bac    : str   = Field(..., example="S")
    score_2022     : Optional[float] = Field(None, example=110.0)
    score_2023     : Optional[float] = Field(None, example=115.0)
    score_2024     : Optional[float] = Field(None, example=118.0)
    score_2025     : float           = Field(...,  example=120.0)
    universite     : Optional[str]   = Field(None, example="Université de Tunis")
    etablissement  : Optional[str]   = Field(None, example="Faculté des Sciences")

# ── Routes ────────────────────────────────────────────────────
@app.get("/", tags=["Santé"])
def health():
    return {
        "status"     : "ok",
        "project"    : "EduStat TN",
        "model"      : meta["model_name"],
        "accuracy"   : meta["metrics"]["accuracy"],
        "f1_weighted": meta["metrics"]["f1_weighted"],
    }


@app.get("/api/sections", tags=["Référentiel"])
def get_sections():
    available = sorted(df_raw["Section_Bac"].unique())
    return [
        {"code": c, "label": SECTION_LABELS.get(c, c)}
        for c in available
    ]


@app.get("/api/domaines", tags=["Référentiel"])
def get_domaines():
    counts = df_raw["Domaine"].value_counts().to_dict()
    return [{"domaine": k, "count": v} for k, v in sorted(counts.items())]


@app.get("/api/stats", tags=["Statistiques"])
def get_stats():
    domaine_dist = df_raw["Domaine"].value_counts().to_dict()
    section_dist = df_raw["Section_Bac"].value_counts().to_dict()
    stats = {
        "total_records"   : int(len(df_raw)),
        "total_filieres"  : int(df_raw["Filiere"].nunique()),
        "total_universites": int(df_raw["Universite"].nunique()),
        "total_sections"  : int(df_raw["Section_Bac"].nunique()),
        "score_moyen_2025": round(float(df_raw["Score_2025"].mean()), 2),
        "score_max_2025"  : round(float(df_raw["Score_2025"].max()), 2),
        "score_min_2025"  : round(float(df_raw["Score_2025"].min()), 2),
        "domaine_distribution": domaine_dist,
        "section_distribution": section_dist,
        "top_filieres_selectives": (
            df_raw.groupby("Filiere")["Score_2025"]
            .max().nlargest(5)
            .reset_index()
            .rename(columns={"Score_2025": "score_max"})
            .to_dict("records")
        ),
        "model": {
            "name"        : meta["model_name"],
            "accuracy"    : meta["metrics"]["accuracy"],
            "f1_weighted" : meta["metrics"]["f1_weighted"],
            "precision"   : meta["metrics"]["precision_weighted"],
            "recall"      : meta["metrics"]["recall_weighted"],
        }
    }
    return stats


@app.post("/api/recommend", tags=["Recommandation"])
def recommend(req: RecommendRequest):
    section = req.section_bac.upper()
    score   = req.score_etudiant
    year    = req.year_ref
    top_n   = max(1, min(req.top_n, 50))

    valid_sections = df_raw["Section_Bac"].unique().tolist()
    if section not in valid_sections:
        raise HTTPException(
            status_code=400,
            detail=f"Section '{section}' invalide. Valeurs acceptées : {valid_sections}"
        )
    if year not in YEAR_COLS:
        raise HTTPException(
            status_code=400,
            detail=f"year_ref invalide. Valeurs acceptées : {YEAR_COLS}"
        )

    eligible = df_raw[
        (df_raw["Section_Bac"] == section) &
        (df_raw[year] <= score)
    ].copy()

    if eligible.empty:
        # Score trop bas → retourner les 5 plus proches
        nearby = df_raw[df_raw["Section_Bac"] == section].copy()
        nearby["ecart"] = (nearby[year] - score).round(3)
        closest = nearby.nsmallest(5, "ecart")[
            ["Filiere", "Universite", "Etablissement", "Domaine", year, "ecart"]
        ].to_dict("records")
        return {
            "status"      : "score_bas",
            "message"     : f"Score {score} insuffisant pour cette section. Voici les filières les plus proches.",
            "suggestions" : closest,
        }

    eligible["marge"] = (score - eligible[year]).round(3)
    eligible["compatibilite_pct"] = (eligible["marge"] / eligible[year] * 100).round(2)

    recs = (
        eligible
        .nsmallest(top_n, "marge")
        [[
            "Filiere", "Universite", "Etablissement", "Domaine",
            year, "Score_Mean", "marge", "compatibilite_pct"
        ]]
        .rename(columns={year: "seuil_admission", "Score_Mean": "moyenne_historique"})
        .reset_index(drop=True)
    )
    recs.index += 1

    return {
        "status"         : "ok",
        "section"        : section,
        "section_label"  : SECTION_LABELS.get(section, section),
        "score_etudiant" : score,
        "annee_ref"      : year,
        "total_eligible" : len(eligible),
        "recommendations": recs.to_dict("records"),
    }


@app.post("/api/predict", tags=["Prédiction ML"])
def predict_domain(req: PredictRequest):
    section = req.section_bac.upper()
    score   = req.score_2025

    valid_sections = df_raw["Section_Bac"].unique().tolist()
    if section not in valid_sections:
        raise HTTPException(status_code=400, detail=f"Section '{section}' inconnue.")

    # Filières accessibles pour cette section et ce score
    eligible = df_raw[
        (df_raw["Section_Bac"] == section) &
        (df_raw["Score_2025"] <= score)
    ]

    if eligible.empty:
        # Score trop bas : prendre les 20 filières les plus proches
        nearby = df_raw[df_raw["Section_Bac"] == section].nsmallest(20, "Score_2025")
        eligible = nearby

    # Distribution des domaines pondérée par la proximité du score
    eligible = eligible.copy()
    eligible["poids"] = 1.0 + (score - eligible["Score_2025"]).clip(lower=0) / score

    domain_scores = eligible.groupby("Domaine")["poids"].sum()
    total = domain_scores.sum()
    domain_pct = (domain_scores / total * 100).sort_values(ascending=False)

    top5 = domain_pct.head(5)
    domaine_predit = top5.index[0]

    return {
        "status"            : "ok",
        "section"           : section,
        "score_2025"        : score,
        "domaine_predit"    : domaine_predit,
        "confiance_pct"     : round(float(top5.iloc[0]), 2),
        "nb_filieres"       : int(len(eligible)),
        "top_5_domaines"    : [
            {"domaine": dom, "probabilite_pct": round(float(pct), 2)}
            for dom, pct in top5.items()
        ],
    }
