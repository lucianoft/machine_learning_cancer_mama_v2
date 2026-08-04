from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Pipeline otimizada pelo algoritmo genetico (ga/run_ga_experiments.py) e servida por
# padrao; se ainda nao foi gerada (ex.: ambiente novo, antes do primeiro
# `python -m ga.run_ga_experiments`), cai pra pipeline original da Fase 1 — a API
# nunca fica fora do ar por causa disso.
GA_PIPELINE_PATH = MODELS_DIR / "pipeline_breast_cancer_ga_optimized.pkl"
GA_METADATA_PATH = MODELS_DIR / "pipeline_metadata_ga_optimized.pkl"
FALLBACK_PIPELINE_PATH = MODELS_DIR / "pipeline_breast_cancer.pkl"
FALLBACK_METADATA_PATH = MODELS_DIR / "pipeline_metadata.pkl"


class ModelService:
    """Carrega e executa a pipeline treinada (otimizada via GA, com fallback pra
    original da Fase 1)."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.metadata: dict | None = None

    def load(self) -> None:
        if GA_PIPELINE_PATH.exists() and GA_METADATA_PATH.exists():
            pipeline_path, metadata_path = GA_PIPELINE_PATH, GA_METADATA_PATH
        elif FALLBACK_PIPELINE_PATH.exists() and FALLBACK_METADATA_PATH.exists():
            pipeline_path, metadata_path = FALLBACK_PIPELINE_PATH, FALLBACK_METADATA_PATH
        else:
            raise FileNotFoundError(
                f"Nenhuma pipeline encontrada em {MODELS_DIR} "
                "(nem otimizada pelo GA, nem a original da Fase 1)."
            )

        self.pipeline = joblib.load(pipeline_path)
        self.metadata = joblib.load(metadata_path)

    @property
    def feature_columns(self) -> list[str]:
        if not self.metadata:
            raise RuntimeError("Modelo não carregado.")
        return self.metadata["feature_columns"]

    def predict(self, patients: list[dict]) -> list[dict]:
        if not self.pipeline or not self.metadata:
            raise RuntimeError("Modelo não carregado.")

        df = pd.DataFrame(patients)
        missing = set(self.feature_columns) - set(df.columns)
        extra = set(df.columns) - set(self.feature_columns)
        if missing:
            raise ValueError(f"Colunas ausentes: {sorted(missing)}")
        if extra:
            raise ValueError(f"Colunas não esperadas: {sorted(extra)}")

        x = df[self.feature_columns]
        predictions = self.pipeline.predict(x)
        probabilities = self.pipeline.predict_proba(x)
        labels = self.metadata["target_labels"]

        results = []
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            results.append({
                "predicao": pred_int,
                "label": labels[pred_int],
                "probabilidade_benigno": float(probabilities[i][0]),
                "probabilidade_maligno": float(probabilities[i][1]),
            })
        return results


model_service = ModelService()
