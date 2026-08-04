"""Integracao com LLM pra traduzir o resultado do modelo em explicacao pro
profissional de saude. Endpoint que usa isso (POST /predict/explain) fica separado do
/predict original, que continua rapido e sem dependencia externa.

Provedor configuravel via LLM_PROVIDER=openai|groq (default: groq). Groq expoe uma API
compativel com a da OpenAI (mesmo formato de request/response, servindo Llama 3 —
um dos modelos sugeridos no PDF do desafio), entao reusamos o mesmo SDK `openai`, so
trocando `base_url` e a chave — sem dependencia nova.
"""

import os

import numpy as np
from openai import OpenAI, OpenAIError
from sklearn.pipeline import Pipeline

from llm.prompts import SYSTEM_PROMPT, build_user_prompt


class LLMUnavailableError(RuntimeError):
    """Levantado quando a chamada a LLM falha (sem API key, provedor invalido, rede etc.)."""


PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
    },
}


def _resolve_provider() -> dict:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in PROVIDERS:
        raise LLMUnavailableError(
            f"LLM_PROVIDER={provider!r} invalido — use 'openai' ou 'groq'."
        )

    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise LLMUnavailableError(
            f"{cfg['api_key_env']} nao configurada (LLM_PROVIDER={provider})."
        )

    model = os.getenv(cfg["model_env"], cfg["default_model"])
    return {"provider": provider, "api_key": api_key, "base_url": cfg["base_url"], "model": model}


def select_relevant_features(
    pipeline: Pipeline,
    patient_features: dict[str, float],
    feature_cols: list[str],
    top_n: int = 5,
) -> dict[str, float]:
    """Usa o StandardScaler ja treinado na pipeline pra achar as features que mais se
    desviam da media da populacao de treino (|z-score|) — sem precisar de um artefato
    de estatisticas separado."""
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    values = np.array([patient_features[col] for col in feature_cols])
    z_scores = (values - scaler.mean_) / scaler.scale_

    ranked = sorted(zip(feature_cols, z_scores), key=lambda item: abs(item[1]), reverse=True)
    top_cols = [col for col, _ in ranked[:top_n]]
    return {col: patient_features[col] for col in top_cols}


def gerar_explicacao(
    prediction: dict,
    patient_features: dict[str, float],
    pipeline: Pipeline,
    feature_cols: list[str],
) -> str:
    relevant = select_relevant_features(pipeline, patient_features, feature_cols)
    user_prompt = build_user_prompt(prediction, relevant)

    config = _resolve_provider()

    try:
        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
    except OpenAIError as exc:
        raise LLMUnavailableError(
            f"Falha ao chamar a LLM (provedor={config['provider']}): {exc}"
        ) from exc

    return response.choices[0].message.content
