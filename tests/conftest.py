import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_patient_payload() -> dict:
    with open(BASE_DIR / "api" / "example_request.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as test_client:
        yield test_client
