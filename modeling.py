"""
EduStat TN — Pipeline ML d'Orientation Universitaire
=====================================================
Auteur : EduStat TN Data Science Team
Date   : 2026-03-10

Objectif :
  À partir des données historiques des scores d'admission (2022-2025),
  entraîner un modèle capable de :
    1. Prédire le domaine d'affectation d'un bachelier
    2. Recommander les filières accessibles selon sa section et son score
    3. Tracer les performances avec MLflow

Usage :
  python modeling.py
"""

# ============================================================
# IMPORTS
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import mlflow
import mlflow.sklearn

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH   = Path(r"C:\Users\dali\Desktop\Edustat\data\tunisie_orientation_complete.csv")
OUTPUT_DIR  = Path(r"C:\Users\dali\Desktop\edustat2\data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = (OUTPUT_DIR / "mlruns").as_uri()   # file:///C:/... (compatible Windows)
MLFLOW_EXPERIMENT   = "EduStat_TN_Orientation"
RANDOM_STATE        = 42
TEST_SIZE           = 0.20
CV_SPLITS           = 5          # StratifiedKFold folds
MIN_CLASS_SAMPLES   = 3          # filières avec moins de N samples → exclues du ML

YEAR_COLS = ["Score_2022", "Score_2023", "Score_2024", "Score_2025"]

# ============================================================
# Domaine mapping (keyword-based grouping → ~10 catégories)
# ============================================================
DOMAIN_KEYWORDS = {
    "Informatique & Tech"      : ["informatique", "technologies", "télécom", "numérique",
                                   "réseaux", "systèmes embarqués", "internet des objets",
                                   "tic"],
    "Ingénierie & Génie"       : ["génie", "mécanique", "électronique", "électrotechnique",
                                   "civil", "industriel", "énergétique", "matériaux",
                                   "cycle préparatoire", "instrumentation et mesure"],
    "Sciences Fondamentales"   : ["physique", "chimie", "biologie", "mathématiques",
                                   "sciences de la vie", "biotechnologie", "biochimie",
                                   "sciences de la terre", "géomatique"],
    "Médecine & Santé"         : ["médecine", "pharmacie", "dentaire", "chirurgie",
                                   "paramédical", "kinésithérapie", "santé",
                                   "infirmière", "infirmier", "infirmières",
                                   "physiothérapie", "orthophonie", "ergothérapie",
                                   "audioprothèse", "optique", "lunetterie",
                                   "imagerie médicale", "radiothérapie",
                                   "bloc opératoire", "obstétrique", "sage-femme",
                                   "nutrition", "appareillage orthopédique"],
    "Économie & Gestion"       : ["gestion", "commerce", "finance", "comptabilité",
                                   "affaires", "marketing", "économie", "management",
                                   "logistique", "assurance", "sciences économiques",
                                   "hôtellerie"],
    "Droit & Sciences Sociales": ["droit", "juridique", "sciences sociales",
                                   "sociologie", "psychologie", "criminologie",
                                   "intervention sociale", "service social",
                                   "sciences du travail", "travail social"],
    "Lettres & Langues"        : ["arabe", "anglais", "français", "allemand", "espagnol",
                                   "lettres", "traduction", "langues",
                                   "chinois", "italien", "russe",
                                   "langue des signes", "communication", "journalisme"],
    "Arts & Architecture"      : ["architecture", "design", "arts", "théâtre", "musique",
                                   "audiovisuel", "patrimoine", "urbanisme", "aménagement"],
    "Sciences Humaines"        : ["histoire", "géographie", "philosophie", "civilisation",
                                   "animation", "tourisme", "sport", "staps",
                                   "football", "éducation", "enseignement",
                                   "archéologie", "anthropologie", "géopolitique",
                                   "relations internationales", "sciences islamiques",
                                   "sciences religieuses"],
    "Agronomie & Environnement": ["agronomie", "agriculture", "agroalimentaire",
                                   "environnement", "hydraulique", "vétérinaire",
                                   "alimentaire", "sciences de la mer",
                                   "sciences agronomiques"],
}


def assign_domain(filiere_name: str) -> str:
    """Affecte un domaine métier à chaque filière par correspondance de mots-clés."""
    name_lower = filiere_name.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return domain
    return "Autres"


# ============================================================
# 1. CHARGEMENT & NETTOYAGE
# ============================================================
def load_and_clean_data(path: Path) -> pd.DataFrame:
    print("=" * 60)
    print("1. CHARGEMENT ET NETTOYAGE DES DONNÉES")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  ▶ Forme initiale : {df.shape}")

    # ── Suppression des lignes sans aucun score ──────────────
    n_before = len(df)
    df = df.dropna(subset=YEAR_COLS, how="all").reset_index(drop=True)
    print(f"  ▶ Lignes sans scores supprimées : {n_before - len(df)}")

    # ── Remplacement des NaN par la médiane par Section+Filiere ─
    for col in YEAR_COLS:
        medians = df.groupby(["Section_Bac", "Filiere"])[col].transform("median")
        df[col] = df[col].fillna(medians)
        # Fallback global si toujours NaN
        df[col] = df[col].fillna(df[col].median())

    print(f"  ▶ Valeurs manquantes restantes : {df.isnull().sum().sum()}")

    # ── Feature engineering ──────────────────────────────────
    df["Score_Mean"]      = df[YEAR_COLS].mean(axis=1).round(4)
    df["Score_Trend"]     = (df["Score_2025"] - df["Score_2022"]).round(4)
    df["Score_Stability"] = df[YEAR_COLS].std(axis=1).round(4)
    df["Score_Max"]       = df[YEAR_COLS].max(axis=1)
    df["Score_Min"]       = df[YEAR_COLS].min(axis=1)

    # ── Catégorie Domaine (variable cible ML) ────────────────
    df["Domaine"] = df["Filiere"].apply(assign_domain)

    print(f"  ▶ Domaines détectés : {df['Domaine'].nunique()}")
    print(f"  ▶ Forme finale     : {df.shape}\n")
    return df


# ============================================================
# 2. ANALYSE EXPLORATOIRE (EDA)
# ============================================================
def perform_eda(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("2. ANALYSE EXPLORATOIRE (EDA)")
    print("=" * 60)

    sns.set_theme(style="whitegrid", palette="muted")

    # ── 2.1 Distribution des scores par année ────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Distribution des Scores par Année", fontsize=16, fontweight="bold")
    for i, col in enumerate(YEAR_COLS):
        ax = axes[i // 2, i % 2]
        df[col].dropna().hist(bins=35, ax=ax, color="#4C72B0", edgecolor="white")
        ax.axvline(df[col].median(), color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Médiane : {df[col].median():.1f}")
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Dernière Moyenne Admise")
        ax.set_ylabel("Fréquence")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "eda_01_score_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ▶ Sauvegardé : {out.name}")

    # ── 2.2 Heatmap de corrélation ────────────────────────────
    score_cols = YEAR_COLS + ["Score_Mean", "Score_Trend", "Score_Stability"]
    corr = df[score_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice de Corrélation des Scores", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "eda_02_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ▶ Sauvegardé : {out.name}")

    # ── 2.3 Score moyen par section du Bac ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    section_scores = (
        df.groupby("Section_Bac_Nom")["Score_Mean"]
        .mean()
        .sort_values(ascending=False)
    )
    section_scores.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="white")
    ax.set_title("Score Moyen d'Admission par Section du Bac", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score Moyen")
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(ax.patches, section_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "eda_03_score_by_section.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ▶ Sauvegardé : {out.name}")

    # ── 2.4 Top 15 filières les plus sélectives (Score_2025) ─
    top15 = df.groupby("Filiere")["Score_2025"].max().nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top15.index, top15.values, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("Top 15 Filières — Score Minimum 2025 le Plus Élevé\n(filières d'élite)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Score Dernière Moyenne Admise (2025)")
    plt.tight_layout()
    out = OUTPUT_DIR / "eda_04_top_filieres_selectives.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ▶ Sauvegardé : {out.name}")

    # ── 2.5 Distribution des domaines ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    domain_counts = df["Domaine"].value_counts()
    domain_counts.plot(kind="bar", ax=ax, color="#55A868", edgecolor="white")
    ax.set_title("Répartition des Filières par Domaine", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Nombre d'enregistrements")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    out = OUTPUT_DIR / "eda_05_domain_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ▶ Sauvegardé : {out.name}")

    # ── 2.6 Évolution des scores moyens par année ─────────────
    yearly_means = df[YEAR_COLS].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(["2022", "2023", "2024", "2025"], yearly_means.values,
            marker="o", linewidth=2.5, color="#4C72B0", markersize=8)
    ax.fill_between(["2022", "2023", "2024", "2025"], yearly_means.values,
                    alpha=0.15, color="#4C72B0")
    for yr, val in zip(["2022", "2023", "2024", "2025"], yearly_means.values):
        ax.annotate(f"{val:.2f}", (yr, val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Évolution du Score Moyen National par Année", fontsize=13, fontweight="bold")
    ax.set_xlabel("Année")
    ax.set_ylabel("Score Moyen")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    out = OUTPUT_DIR / "eda_06_score_trend_years.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ▶ Sauvegardé : {out.name}")

    # ── Statistiques textuelles ───────────────────────────────
    print(f"\n  Filières uniques        : {df['Filiere'].nunique()}")
    print(f"  Sections Bac            : {sorted(df['Section_Bac'].unique())}")
    print(f"  Universités             : {df['Universite'].nunique()}")
    print(f"  Corrélation 2022↔2025   : {df['Score_2022'].corr(df['Score_2025']):.4f}")
    print(f"  Corrélation Mean↔2025   : {df['Score_Mean'].corr(df['Score_2025']):.4f}")
    print(f"\n  Filières les plus sélectives (Score_2025 max) :")
    print(df.groupby("Filiere")["Score_2025"].max().nlargest(5).to_string())
    print()


# ============================================================
# 3. PRÉPARATION DES FEATURES
# ============================================================
def prepare_features(df: pd.DataFrame):
    """
    Prépare les features ML et encode les variables catégorielles.
    Retourne X, y, encoders, feature_cols et le dataframe enrichi.
    """
    print("=" * 60)
    print("3. PRÉPARATION DES FEATURES")
    print("=" * 60)

    df = df.copy()

    # ── Encodage ──────────────────────────────────────────────
    le_section      = LabelEncoder()
    le_universite   = LabelEncoder()
    le_etablissement = LabelEncoder()
    le_target       = LabelEncoder()

    df["Section_Enc"]      = le_section.fit_transform(df["Section_Bac"])
    df["Universite_Enc"]   = le_universite.fit_transform(df["Universite"])
    df["Etablissement_Enc"] = le_etablissement.fit_transform(df["Etablissement"])
    df["Domaine_Enc"]      = le_target.fit_transform(df["Domaine"])

    feature_cols = [
        "Section_Enc",
        "Score_2022", "Score_2023", "Score_2024", "Score_2025",
        "Score_Mean", "Score_Trend", "Score_Stability",
        "Score_Max", "Score_Min",
    ]

    # ── Filtrage des classes trop petites ─────────────────────
    class_counts = df["Domaine"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index
    df_filtered = df[df["Domaine"].isin(valid_classes)].copy()

    excluded = df["Domaine"].nunique() - df_filtered["Domaine"].nunique()
    print(f"  ▶ Classes conservées  : {df_filtered['Domaine'].nunique()}")
    print(f"  ▶ Classes exclues (<{MIN_CLASS_SAMPLES} samples) : {excluded}")
    print(f"  ▶ Features            : {feature_cols}")

    X = df_filtered[feature_cols].values
    y = le_target.transform(df_filtered["Domaine"])

    # Recalibrer le target encoder sur les classes filtrées
    le_target_filtered = LabelEncoder()
    y = le_target_filtered.fit_transform(df_filtered["Domaine"])

    encoders = {
        "section"       : le_section,
        "universite"    : le_universite,
        "etablissement" : le_etablissement,
        "target"        : le_target_filtered,
    }

    print(f"  ▶ Taille dataset ML   : {X.shape}")
    print(f"  ▶ Classes cibles      : {sorted(df_filtered['Domaine'].unique())}\n")

    return X, y, feature_cols, encoders, df_filtered


# ============================================================
# 4. PIPELINE ML : SMOTE + GRIDSEARCHCV
# ============================================================
def train_models(
    X_train, X_test, y_train, y_test, encoders, feature_cols
) -> dict:
    """
    Entraîne Random Forest et XGBoost avec SMOTE (rééquilibrage) +
    GridSearchCV (optimisation) et logue dans MLflow.
    """
    print("=" * 60)
    print("4. ENTRAÎNEMENT DES MODÈLES (SMOTE + GridSearchCV)")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── SMOTE : k_neighbors adaptatif pour petites classes ────
    min_samples = min(np.bincount(y_train))
    k_neighbors = max(1, min(5, min_samples - 1))
    print(f"  ▶ SMOTE k_neighbors   : {k_neighbors}")

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    cv    = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    n_classes = len(np.unique(y_train))

    # ─────────────────────────────────────────────────────────
    # 4.1  RANDOM FOREST
    # ─────────────────────────────────────────────────────────
    print("\n  ── Random Forest ──────────────────────────────────")
    with mlflow.start_run(run_name="RandomForest_SMOTE"):

        param_grid_rf = {
            "classifier__n_estimators" : [100, 200],
            "classifier__max_depth"    : [10, 20, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight" : ["balanced"],
        }

        pipeline_rf = ImbPipeline([
            ("smote", smote),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,     # évite paralélisme imbriqué avec GridSearchCV
            )),
        ])

        grid_rf = GridSearchCV(
            pipeline_rf, param_grid_rf,
            cv=cv, scoring="f1_weighted",
            n_jobs=-1, verbose=0, refit=True,
        )
        grid_rf.fit(X_train, y_train)

        best_rf = grid_rf.best_estimator_
        y_pred  = best_rf.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)

        # ── Log MLflow ──
        mlflow.log_params(grid_rf.best_params_)
        mlflow.log_params({"model_type": "RandomForest", "smote": True,
                           "cv_splits": CV_SPLITS})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "model")

        _print_metrics("Random Forest", metrics, grid_rf.best_params_)
        results["rf"] = {"model": best_rf, **metrics,
                         "best_params": grid_rf.best_params_,
                         "y_pred": y_pred}

    # ─────────────────────────────────────────────────────────
    # 4.2  XGBOOST
    # ─────────────────────────────────────────────────────────
    print("\n  ── XGBoost ────────────────────────────────────────")
    with mlflow.start_run(run_name="XGBoost_SMOTE"):

        xgb_clf = xgb.XGBClassifier(
            objective    = "multi:softmax",
            num_class    = n_classes,
            eval_metric  = "mlogloss",
            tree_method  = "hist",
            random_state = RANDOM_STATE,
            n_jobs       = 1,
            verbosity    = 0,
        )

        param_grid_xgb = {
            "classifier__n_estimators" : [100, 200],
            "classifier__max_depth"    : [4, 6],
            "classifier__learning_rate": [0.05, 0.1],
        }

        pipeline_xgb = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
            ("classifier", xgb_clf),
        ])

        grid_xgb = GridSearchCV(
            pipeline_xgb, param_grid_xgb,
            cv=cv, scoring="f1_weighted",
            n_jobs=-1, verbose=0, refit=True,
        )
        grid_xgb.fit(X_train, y_train)

        best_xgb = grid_xgb.best_estimator_
        y_pred   = best_xgb.predict(X_test)
        metrics  = _compute_metrics(y_test, y_pred)

        # ── Log MLflow ──
        mlflow.log_params(grid_xgb.best_params_)
        mlflow.log_params({"model_type": "XGBoost", "smote": True,
                           "cv_splits": CV_SPLITS})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_xgb, "model")

        _print_metrics("XGBoost", metrics, grid_xgb.best_params_)
        results["xgb"] = {"model": best_xgb, **metrics,
                          "best_params": grid_xgb.best_params_,
                          "y_pred": y_pred}

    return results


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy"            : round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted"         : round(f1_score(y_true, y_pred, average="weighted",
                                               zero_division=0), 4),
        "precision_weighted"  : round(precision_score(y_true, y_pred, average="weighted",
                                                       zero_division=0), 4),
        "recall_weighted"     : round(recall_score(y_true, y_pred, average="weighted",
                                                    zero_division=0), 4),
    }


def _print_metrics(name: str, metrics: dict, params: dict) -> None:
    print(f"\n  Résultats {name} :")
    print(f"    Accuracy   : {metrics['accuracy']:.4f}")
    print(f"    F1 (weighted)  : {metrics['f1_weighted']:.4f}")
    print(f"    Precision  : {metrics['precision_weighted']:.4f}")
    print(f"    Recall     : {metrics['recall_weighted']:.4f}")
    print(f"    Best params: {params}")


# ============================================================
# 5. SÉLECTION & SAUVEGARDE DU MEILLEUR MODÈLE
# ============================================================
def select_and_save_best_model(
    results: dict, encoders: dict, feature_cols: list,
    y_test, df_filtered: pd.DataFrame,
) -> tuple:
    print("\n" + "=" * 60)
    print("5. SÉLECTION ET SAUVEGARDE DU MEILLEUR MODÈLE")
    print("=" * 60)

    # ── Comparaison par F1 weighted ──────────────────────────
    best_name   = max(results, key=lambda k: results[k]["f1_weighted"])
    best_result = results[best_name]

    print("\n  Récapitulatif :")
    for name, res in results.items():
        marker = "★" if name == best_name else " "
        print(f"  {marker} {name.upper():12s} | "
              f"Accuracy={res['accuracy']:.4f} | "
              f"F1={res['f1_weighted']:.4f} | "
              f"Precision={res['precision_weighted']:.4f} | "
              f"Recall={res['recall_weighted']:.4f}")

    print(f"\n  Meilleur modèle : {best_name.upper()} "
          f"(F1={best_result['f1_weighted']:.4f})")

    # ── Rapport de classification détaillé ───────────────────
    target_names = encoders["target"].classes_
    report = classification_report(
        y_test, best_result["y_pred"],
        target_names=target_names,
        zero_division=0,
    )
    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Rapport sauvegardé : {report_path.name}")

    # ── Bundle du modèle ─────────────────────────────────────
    model_bundle = {
        "model"       : best_result["model"],
        "encoders"    : encoders,
        "feature_cols": feature_cols,
        "model_name"  : best_name.upper(),
        "metrics"     : {k: v for k, v in best_result.items()
                        if k not in ("model", "y_pred", "best_params")},
        "best_params" : best_result["best_params"],
    }

    # ── Sauvegarde .joblib ────────────────────────────────────
    model_path = OUTPUT_DIR / "best_orientation_model.joblib"
    joblib.dump(model_bundle, model_path, compress=3)
    print(f"  Modèle sauvegardé  : {model_path}")

    # ── Sauvegarde des encodeurs séparément ──────────────────
    enc_path = OUTPUT_DIR / "label_encoders.joblib"
    joblib.dump(encoders, enc_path, compress=3)
    print(f"  Encodeurs sauvegardés : {enc_path}")

    # ── Métadonnées JSON ─────────────────────────────────────
    meta = {
        "model_name"   : best_name.upper(),
        "metrics"      : model_bundle["metrics"],
        "best_params"  : best_result["best_params"],
        "feature_cols" : feature_cols,
        "target_classes": list(target_names),
    }
    meta_path = OUTPUT_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Métadonnées sauvegardées : {meta_path}")

    return best_name, best_result


# ============================================================
# 6. FONCTION DE RECOMMANDATION (INFÉRENCE)
# ============================================================
def recommend_filiere(
    section_bac: str,
    score_etudiant: float,
    df_original: pd.DataFrame,
    top_n: int = 10,
    year_ref: str = "Score_2025",
) -> pd.DataFrame:
    """
    Recommande les top_n filières accessibles à un bachelier
    en fonction de sa section et de son score.

    Paramètres
    ----------
    section_bac     : Code de la section (ex: 'M', 'S', 'L')
    score_etudiant  : Score du bachelier (sur ~200)
    df_original     : DataFrame de référence (données brutes)
    top_n           : Nombre de recommandations à retourner
    year_ref        : Colonne de score de référence pour l'admission

    Retourne
    --------
    DataFrame des filières recommandées avec marge de score.
    """
    print(f"\n{'─'*60}")
    print(f"  RECOMMANDATION — Section: {section_bac} | Score: {score_etudiant}")
    print(f"{'─'*60}")

    # ── 1. Filtrage par section et éligibilité de score ───────
    eligible = df_original[
        (df_original["Section_Bac"] == section_bac) &
        (df_original[year_ref] <= score_etudiant)
    ].copy()

    if eligible.empty:
        # Essai sans filtre de score → afficher les plus proches
        nearby = df_original[df_original["Section_Bac"] == section_bac].copy()
        nearby["Écart_Score"] = nearby[year_ref] - score_etudiant
        print(f"  Score trop bas pour accéder à des filières en section {section_bac}.")
        print(f"  Filières les plus proches (score manquant) :")
        print(nearby.nsmallest(5, "Écart_Score")[
            ["Filiere", "Universite", "Etablissement", year_ref, "Écart_Score"]
        ].to_string(index=False))
        return pd.DataFrame()

    # ── 2. Calcul de la marge (score étudiant − seuil d'admission) ──
    eligible["Marge_Score"]   = score_etudiant - eligible[year_ref]
    eligible["Compatibilité"] = (eligible["Marge_Score"] / eligible[year_ref] * 100).round(2)

    # ── 3. Tri par marge croissante (filières les plus difficiles en premier) ──
    recommendations = (
        eligible
        .nsmallest(top_n, "Marge_Score")
        [[
            "Filiere", "Universite", "Etablissement",
            "Domaine", year_ref, "Score_Mean", "Marge_Score", "Compatibilité"
        ]]
        .rename(columns={
            year_ref           : f"Seuil_{year_ref[-4:]}",
            "Score_Mean"       : "Moyenne_Hist",
            "Marge_Score"      : "Marge",
            "Compatibilité"    : "Compat_%",
        })
        .reset_index(drop=True)
    )
    recommendations.index += 1  # 1-based indexing

    print(recommendations.to_string())
    return recommendations


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EduStat TN — Pipeline ML d'Orientation Universitaire")
    print("=" * 60 + "\n")

    # ── 1. Chargement & Nettoyage ─────────────────────────────
    df = load_and_clean_data(DATA_PATH)

    # ── 2. EDA ────────────────────────────────────────────────
    perform_eda(df)

    # ── 3. Préparation des features ───────────────────────────
    X, y, feature_cols, encoders, df_filtered = prepare_features(df)

    # ── 4. Split train/test (stratifié) ───────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    print("=" * 60)
    print(f"  Train  : {X_train.shape[0]} samples")
    print(f"  Test   : {X_test.shape[0]} samples")
    print(f"  Classes: {len(np.unique(y))}\n")

    # ── 5. Entraînement ───────────────────────────────────────
    results = train_models(X_train, X_test, y_train, y_test, encoders, feature_cols)

    # ── 6. Sélection & Sauvegarde ─────────────────────────────
    best_name, best_result = select_and_save_best_model(
        results, encoders, feature_cols, y_test, df_filtered
    )

    # ── 7. Démos de recommandation ────────────────────────────
    print("\n" + "=" * 60)
    print("6. DÉMONSTRATIONS DE RECOMMANDATION")
    print("=" * 60)

    demo_cases = [
        ("M",  155.0, "Bachelier Mathématiques, excellent niveau"),
        ("S",  120.0, "Bachelier Sciences, niveau moyen"),
        ("L",   98.0, "Bachelier Lettres"),
        ("I",  140.0, "Bachelier Informatique, bon niveau"),
        ("E",  100.0, "Bachelier Économie, niveau moyen"),
    ]

    all_recommendations = {}
    for section, score, label in demo_cases:
        print(f"\n  [{label}]")
        rec = recommend_filiere(
            section_bac    = section,
            score_etudiant = score,
            df_original    = df,
            top_n          = 5,
        )
        if not rec.empty:
            all_recommendations[f"{section}_{score}"] = rec.to_dict()

    # Sauvegarde des recommandations démo
    rec_path = OUTPUT_DIR / "demo_recommendations.json"
    rec_path.write_text(
        json.dumps(all_recommendations, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  Recommandations démo sauvegardées : {rec_path}")

    # ── Résumé final ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"  Meilleur modèle  : {best_name.upper()}")
    print(f"  Accuracy         : {best_result['accuracy']:.4f}")
    print(f"  F1 (weighted)    : {best_result['f1_weighted']:.4f}")
    print(f"  Précision        : {best_result['precision_weighted']:.4f}")
    print(f"  Rappel           : {best_result['recall_weighted']:.4f}")
    print(f"\n  Artefacts dans   : {OUTPUT_DIR}")
    print(f"  MLflow URI       : {MLFLOW_TRACKING_URI}")
    print(f"\n  Pipeline terminé avec succès !\n")
