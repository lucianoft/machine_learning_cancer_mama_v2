from api.model_service import ModelService


def test_load_e_predict_com_pipeline_real(sample_patient_payload):
    service = ModelService()
    service.load()

    patient = sample_patient_payload["patient"]
    resultados = service.predict([patient])

    assert len(resultados) == 1
    resultado = resultados[0]
    assert resultado["predicao"] in (0, 1)
    assert resultado["label"] in ("Benigno", "Maligno")
    assert 0.0 <= resultado["probabilidade_benigno"] <= 1.0
    assert 0.0 <= resultado["probabilidade_maligno"] <= 1.0
    assert abs(resultado["probabilidade_benigno"] + resultado["probabilidade_maligno"] - 1.0) < 1e-6


def test_predict_reclama_de_coluna_ausente(sample_patient_payload):
    service = ModelService()
    service.load()

    patient = dict(sample_patient_payload["patient"])
    del patient["mean radius"]

    try:
        service.predict([patient])
        assert False, "deveria ter lancado ValueError"
    except ValueError as exc:
        assert "mean radius" in str(exc)


def test_feature_columns_tem_30_features():
    service = ModelService()
    service.load()
    assert len(service.feature_columns) == 30
