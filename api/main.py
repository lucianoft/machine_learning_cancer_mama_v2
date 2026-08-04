from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_service import model_service
from api.observability import configure_logging, instrument, log_prediction
from api.schemas import (
    ExplainResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)
from llm.explainer import LLMUnavailableError, gerar_explicacao
from llm.prompts import DISCLAIMER

configure_logging()

API_DESCRIPTION = f"""
API de apoio à classificação **benigno vs maligno** em massas mamárias, com base em
características de núcleos celulares extraídas de imagens de **PAAF** (Punção Aspirativa
por Agulha Fina).

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI / Kaggle.

A pipeline de classificação (`/predict`) é otimizada via **algoritmo genético** (ver
`ga/`). O endpoint `/predict/explain` usa uma **LLM** para traduzir o resultado em
linguagem natural para o profissional de saúde.

> {DISCLAIMER}
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Tech Challenge — Câncer de Mama (Breast Cancer Wisconsin)",
    description=API_DESCRIPTION,
    version="2.0.0",
    lifespan=lifespan,
)
instrument(app)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health() -> HealthResponse:
    """Verifica se a API e o modelo estão carregados."""
    return HealthResponse(
        status="ok",
        model_name=model_service.metadata.get("model_name") if model_service.metadata else None,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["Sistema"])
def metadata() -> MetadataResponse:
    """Retorna metadados do modelo treinado para câncer de mama."""
    meta = model_service.metadata or {}
    return MetadataResponse(
        model_name=meta["model_name"],
        target_column=meta["target_column"],
        target_labels=meta["target_labels"],
        feature_columns=meta["feature_columns"],
        dataset=meta["dataset"],
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predição — câncer de mama"],
    summary="Classificação benigno/maligno",
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Estima se a amostra é **benigna** ou **maligna** com base nas 30 medidas
    de núcleos celulares (mean, error e worst).
    """
    with log_prediction("predict") as ctx:
        try:
            results = model_service.predict([request.patient.to_feature_dict()])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        resultado = results[0]
        ctx["label"] = resultado["label"]
        ctx["confianca"] = max(resultado["probabilidade_benigno"], resultado["probabilidade_maligno"])
        return PredictResponse(**resultado)


@app.post(
    "/predict/explain",
    response_model=ExplainResponse,
    tags=["Predição — câncer de mama"],
    summary="Classificação benigno/maligno + explicação em linguagem natural (LLM)",
)
def predict_explain(request: PredictRequest) -> ExplainResponse:
    """
    Mesma classificação do `/predict`, mais uma explicação em linguagem natural gerada
    por uma LLM — pensada pra apoiar a leitura do resultado por profissionais de saúde.
    Requer `OPENAI_API_KEY` configurada no ambiente.
    """
    with log_prediction("predict_explain") as ctx:
        patient_features = request.patient.to_feature_dict()
        try:
            results = model_service.predict([patient_features])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        resultado = results[0]
        ctx["label"] = resultado["label"]
        ctx["confianca"] = max(resultado["probabilidade_benigno"], resultado["probabilidade_maligno"])

        try:
            explicacao = gerar_explicacao(
                resultado, patient_features, model_service.pipeline, model_service.feature_columns
            )
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return ExplainResponse(**resultado, explicacao=explicacao)
