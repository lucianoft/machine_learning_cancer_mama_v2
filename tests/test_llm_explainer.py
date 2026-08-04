from types import SimpleNamespace

import pytest

from api.model_service import ModelService
from llm.explainer import LLMUnavailableError, gerar_explicacao, select_relevant_features
from llm.prompts import DISCLAIMER


@pytest.fixture
def pipeline_e_features(sample_patient_payload):
    service = ModelService()
    service.load()
    return service.pipeline, service.feature_columns, sample_patient_payload["patient"]


@pytest.fixture(autouse=True)
def default_provider_env(monkeypatch):
    """Provedor default (groq) com uma chave falsa — os testes mockam o client OpenAI,
    entao a chave nunca e usada de verdade, so precisa existir pra passar da validacao."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")


def _fake_openai_response(texto: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto))])


def test_select_relevant_features_devolve_subconjunto(pipeline_e_features):
    pipeline, feature_cols, patient = pipeline_e_features
    relevantes = select_relevant_features(pipeline, patient, feature_cols, top_n=5)

    assert len(relevantes) == 5
    assert set(relevantes.keys()).issubset(set(feature_cols))


def test_gerar_explicacao_inclui_disclaimer_no_prompt_de_sistema(pipeline_e_features, mocker):
    pipeline, feature_cols, patient = pipeline_e_features
    prediction = {
        "predicao": 0,
        "label": "Benigno",
        "probabilidade_benigno": 0.91,
        "probabilidade_maligno": 0.09,
    }

    texto_esperado = f"Explicacao gerada. {DISCLAIMER}"
    mock_openai = mocker.patch("llm.explainer.OpenAI")
    mock_openai.return_value.chat.completions.create.return_value = _fake_openai_response(texto_esperado)

    resultado = gerar_explicacao(prediction, patient, pipeline, feature_cols)

    assert resultado == texto_esperado
    _, kwargs = mock_openai.return_value.chat.completions.create.call_args
    system_message = kwargs["messages"][0]["content"]
    assert DISCLAIMER in system_message


def test_gerar_explicacao_usa_groq_por_padrao(pipeline_e_features, mocker):
    pipeline, feature_cols, patient = pipeline_e_features
    prediction = {"predicao": 0, "label": "Benigno", "probabilidade_benigno": 0.9, "probabilidade_maligno": 0.1}

    mock_openai = mocker.patch("llm.explainer.OpenAI")
    mock_openai.return_value.chat.completions.create.return_value = _fake_openai_response("ok")

    gerar_explicacao(prediction, patient, pipeline, feature_cols)

    _, kwargs = mock_openai.call_args
    assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert kwargs["api_key"] == "test-groq-key"


def test_gerar_explicacao_troca_pra_openai_via_env(pipeline_e_features, mocker, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    pipeline, feature_cols, patient = pipeline_e_features
    prediction = {"predicao": 1, "label": "Maligno", "probabilidade_benigno": 0.1, "probabilidade_maligno": 0.9}

    mock_openai = mocker.patch("llm.explainer.OpenAI")
    mock_openai.return_value.chat.completions.create.return_value = _fake_openai_response("ok")

    gerar_explicacao(prediction, patient, pipeline, feature_cols)

    _, kwargs = mock_openai.call_args
    assert kwargs["base_url"] is None
    assert kwargs["api_key"] == "test-openai-key"


def test_gerar_explicacao_falha_clara_sem_api_key(pipeline_e_features, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    pipeline, feature_cols, patient = pipeline_e_features
    prediction = {"predicao": 0, "label": "Benigno", "probabilidade_benigno": 0.9, "probabilidade_maligno": 0.1}

    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        gerar_explicacao(prediction, patient, pipeline, feature_cols)


def test_gerar_explicacao_propaga_erro_de_rede_como_llm_unavailable(pipeline_e_features, mocker):
    from openai import OpenAIError

    pipeline, feature_cols, patient = pipeline_e_features
    prediction = {
        "predicao": 1,
        "label": "Maligno",
        "probabilidade_benigno": 0.1,
        "probabilidade_maligno": 0.9,
    }

    mock_openai = mocker.patch("llm.explainer.OpenAI")
    mock_openai.return_value.chat.completions.create.side_effect = OpenAIError("falha de rede")

    with pytest.raises(LLMUnavailableError):
        gerar_explicacao(prediction, patient, pipeline, feature_cols)
