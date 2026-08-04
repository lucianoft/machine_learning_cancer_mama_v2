def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"]


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert len(body["feature_columns"]) == 30
    assert body["target_labels"] == {"0": "Benigno", "1": "Maligno"}


def test_predict_com_payload_valido(client, sample_patient_payload):
    response = client.post("/predict", json=sample_patient_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicao"] in (0, 1)
    assert body["label"] in ("Benigno", "Maligno")


def test_predict_com_payload_invalido(client, sample_patient_payload):
    payload = {"patient": dict(sample_patient_payload["patient"])}
    del payload["patient"]["mean radius"]

    response = client.post("/predict", json=payload)
    # campo obrigatorio ausente -> Pydantic rejeita com 422 antes mesmo de chegar
    # no ModelService
    assert response.status_code == 422


def test_predict_explain_com_llm_mockada(client, sample_patient_payload, mocker):
    mocker.patch(
        "api.main.gerar_explicacao",
        return_value="Explicacao de teste. Ferramenta de apoio a decisao. "
                     "Nao substitui avaliacao medica, mamografia, ultrassonografia "
                     "nem biopsia. O profissional de saude tem a palavra final.",
    )

    response = client.post("/predict/explain", json=sample_patient_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["explicacao"].startswith("Explicacao de teste")
    assert body["predicao"] in (0, 1)


def test_predict_explain_retorna_503_quando_llm_indisponivel(client, sample_patient_payload, mocker):
    from llm.explainer import LLMUnavailableError

    mocker.patch(
        "api.main.gerar_explicacao",
        side_effect=LLMUnavailableError("sem OPENAI_API_KEY"),
    )

    response = client.post("/predict/explain", json=sample_patient_payload)
    assert response.status_code == 503


def test_metrics_expostas(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests" in response.content or b"# HELP" in response.content
