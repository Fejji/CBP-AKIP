# main.py (Version corrigée pour le déploiement sur Render)

import sys
import joblib
import json
import logging
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# NOUVELLE IMPORTATION
from fastapi.responses import FileResponse 
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# --- Importations et configuration initiales ---
from custom_objects import RobustSelectFromModel
# Astuce pour permettre à joblib de trouver la classe personnalisée lors du chargement
sys.modules['__main__'].RobustSelectFromModel = RobustSelectFromModel

MODEL_PATH = "aki_hybrid_model_final.joblib"
METADATA_PATH = "aki_hybrid_metadata_final.json"
SHAP_BACKGROUND_PATH = "aki_hybrid_shap_background_final.joblib"

# --- Variables Globales ---
model_pipeline = None
metadata = {}
shap_explainer = None
feature_names_in_order = []

logging.basicConfig(level=logging.INFO)

# --- Création de l'application FastAPI ---
app = FastAPI(
    title="CPB-AKIP Score API",
    description="API pour prédire le risque d'Insuffisance Rénale Aiguë post-opératoire.",
    version="1.0.0"
)

# --- Middleware CORS ---
# Permet à votre frontend (même s'il est servi par une autre origine) de communiquer avec l'API.
origins = ["*"]  # Ou spécifiez les domaines autorisés
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Événement de démarrage ---
# Charge les modèles et métadonnées une seule fois au lancement de l'application.
@app.on_event("startup")
def load_resources():
    global model_pipeline, metadata, shap_explainer, feature_names_in_order
    try:
        logging.info("Chargement du pipeline de modèle...")
        model_pipeline = joblib.load(MODEL_PATH)
        logging.info("Chargement des métadonnées...")
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        feature_names_in_order = metadata.get('selected_features_final', [])
        if not feature_names_in_order:
            raise ValueError("La liste des caractéristiques ('selected_features_final') est vide ou non trouvée dans les métadonnées.")

        logging.info("Chargement des données d'arrière-plan pour SHAP...")
        background_data_joblib = joblib.load(SHAP_BACKGROUND_PATH)
        background_df = pd.DataFrame(background_data_joblib, columns=feature_names_in_order)
        
        logging.info("Initialisation de l'explainer SHAP...")
        # On utilise la fonction de prédiction de probabilité du modèle
        predict_fn = lambda x: model_pipeline.predict_proba(x)[:, 1]
        shap_explainer = shap.KernelExplainer(predict_fn, background_df)
        
        logging.info("Ressources chargées avec succès.")

    except FileNotFoundError as e:
        logging.error(f"Fichier non trouvé : {e}. Assurez-vous que les fichiers de modèle, métadonnées et SHAP sont présents.")
        raise
    except Exception as e:
        logging.error(f"Erreur lors du chargement des ressources : {e}")
        raise

# --- Modèle de données Pydantic pour l'entrée ---
class PatientInput(BaseModel):
    Age: Optional[float] = None; Sexe: Optional[float] = None; IMC: Optional[float] = None
    Diabete: Optional[float] = Field(None, alias='Diabète'); HTA: Optional[float] = None
    IRC: Optional[float] = None; FEVG_pre: Optional[float] = Field(None, alias='FEVG_pré')
    NYHA: Optional[float] = None; ASA: Optional[float] = None; Euroscore: Optional[float] = None
    Creat_pre: Optional[float] = Field(None, alias='Créat_pré'); Clairance_pre: Optional[float] = Field(None, alias='Clairance_pré')
    Hb_pre: Optional[float] = Field(None, alias='Hb_pré'); Chir: Optional[float] = None
    Urg: Optional[float] = None; Redux: Optional[float] = None
    Duree_CEC: Optional[float] = Field(None, alias='Durée_CEC'); Duree_clamp: Optional[float] = Field(None, alias='Durée_clamp')
    PAM_CEC: Optional[float] = None; CGR_per: Optional[float] = None; Lac_fin: Optional[float] = None
    NAD_fin: Optional[float] = None; Dobu_fin: Optional[float] = None; PFC_per: Optional[float] = None
    CPS_per: Optional[float] = None; SCA_pre: Optional[float] = Field(None, alias='SCA_pré')
    AVC_pre: Optional[float] = Field(None, alias='AVC_pré'); BPCO: Optional[float] = None
    IH: Optional[float] = None; Tabagisme: Optional[float] = None; pH_0: Optional[float] = None
    Ht_0: Optional[float] = None; PAPS_pre: Optional[float] = Field(None, alias='PAPS_pré'); Ht_pre: Optional[float] = Field(None, alias='Ht_pré')

    class Config:
        anystr_strip_whitespace = True
        
# =================================================================
# NOUVELLE ROUTE POUR SERVIR LA PAGE D'ACCUEIL (index.html)
# =================================================================
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse('index.html')
# =================================================================


# --- Endpoints de l'API ---

@app.get("/health", summary="Vérifier l'état de santé de l'API")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}
    
@app.get("/metadata", summary="Obtenir les métadonnées (valeurs de formulaire, etc.)")
def get_metadata():
    return {
        "feature_value_maps": metadata.get("feature_value_maps"),
        "numerical_ranges": metadata.get("numerical_ranges")
    }

@app.post("/predict", summary="Prédire le risque d'IRA pour un patient")
def predict_risk(patient_data: PatientInput) -> Dict[str, Any]:
    if not model_pipeline: raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    try:
        patient_df = pd.DataFrame([patient_data.dict(by_alias=True)], columns=feature_names_in_order)
        patient_df.replace({None: np.nan}, inplace=True)
        probability = model_pipeline.predict_proba(patient_df)[0][1]
        threshold = metadata.get('optimal_threshold_youden', 0.2715)
        is_at_risk = bool(probability >= threshold)
        return {"probability_of_ira": float(probability), "is_at_risk": is_at_risk, "optimal_threshold_used": threshold}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain", summary="Expliquer une prédiction unique (locale) avec valeur de base")
def explain_prediction(patient_data: PatientInput) -> Dict[str, Any]:
    if not shap_explainer: raise HTTPException(status_code=503, detail="L'explainer SHAP n'est pas chargé.")
    try:
        patient_df = pd.DataFrame([patient_data.dict(by_alias=True)], columns=feature_names_in_order)
        patient_df.replace({None: np.nan}, inplace=True)
        
        shap_values = shap_explainer.shap_values(patient_df).flatten()
        
        # On retourne aussi la valeur de base de l'explainer
        base_value = shap_explainer.expected_value
        
        explanation = {feature: round(value, 4) for feature, value in zip(feature_names_in_order, shap_values)}
        
        return {"base_value": round(base_value, 4), "shap_values": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
