# main.py (Version finale pour graphique en cascade)

import sys
import joblib
import json
import logging
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# --- Importations et configuration initiales ---
from custom_objects import RobustSelectFromModel
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

class PatientInput(BaseModel):
    Age: Optional[float] = None; Sexe: Optional[float] = None; IMC: Optional[float] = None
    Diabete: Optional[float] = Field(None, alias='Diabète'); HTA: Optional[float] = None
    IRC: Optional[float] = None; FEVG_pre: Optional[float] = Field(None, alias='FEVG_pré')
    NYHA: Optional[float] = None; ASA: Optional[float] = None; Euroscore: Optional[float] = None
    Creat_pre: Optional[float] = Field(None, alias='Créat_pré')
    Clairance_pre: Optional[float] = Field(None, alias='Clairance_pré')
    Hb_pre: Optional[float] = Field(None, alias='Hb_pré'); Chir: Optional[float] = None
    Urg: Optional[float] = None; Redux: Optional[float] = None
    Duree_CEC: Optional[float] = Field(None, alias='Durée_CEC')
    Duree_clamp: Optional[float] = Field(None, alias='Durée_clamp'); PAM_CEC: Optional[float] = None
    CGR_per: Optional[float] = None; Lac_fin: Optional[float] = None
    NAD_fin: Optional[float] = None; Dobu_fin: Optional[float] = None
    PFC_per: Optional[float] = None; CPS_per: Optional[float] = None
    SCA_pre: Optional[float] = Field(None, alias='SCA_pré')
    AVC_pre: Optional[float] = Field(None, alias='AVC_pré'); BPCO: Optional[float] = None
    IH: Optional[float] = None; Tabagisme: Optional[float] = None; pH_0: Optional[float] = None
    Ht_0: Optional[float] = None; PAPS_pre: Optional[float] = Field(None, alias='PAPS_pré')
    Ht_pre: Optional[float] = Field(None, alias='Ht_pré')

app = FastAPI(title="API de Prédiction du Risque d'IRA", version="14.0.0 (Waterfall Plot)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def load_artifacts():
    global model_pipeline, metadata, shap_explainer, feature_names_in_order
    logging.info("--- Démarrage de l'API : Chargement des artefacts... ---")
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        feature_names_in_order = metadata.get('selected_features_initial', [])
        
        model_pipeline = joblib.load(MODEL_PATH)
        logging.info("Modèle de prédiction principal chargé.")
        
        stacker = model_pipeline.estimator
        rf_pipeline_for_shap = stacker.estimators_[1]
        
        background_data_np = joblib.load(SHAP_BACKGROUND_PATH)

        def rf_predict_proba_for_shap(data):
            df = pd.DataFrame(data, columns=feature_names_in_order)
            return rf_pipeline_for_shap.predict_proba(df)[:, 1]

        shap_explainer = shap.KernelExplainer(rf_predict_proba_for_shap, background_data_np)
        logging.info("Explainer SHAP (local) configuré avec succès.")

    except Exception as e:
        logging.error(f"--- ERREUR CRITIQUE AU DÉMARRAGE --- : {e}", exc_info=True)
        raise e

# --- Endpoints ---

@app.post("/predict", summary="Prédire le risque d'IRA pour un patient")
def predict_aki(patient_data: PatientInput):
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
        
        # MISE À JOUR : On retourne aussi la valeur de base de l'explainer
        base_value = shap_explainer.expected_value
        
        explanation = {feature: round(value, 4) for feature, value in zip(feature_names_in_order, shap_values)}
        
        return {
            "shap_values": explanation,
            "base_value": float(base_value)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))