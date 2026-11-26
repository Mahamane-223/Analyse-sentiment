"""Tests de l'API d'analyse de sentiments (avec le vrai modèle du projet)."""

import pytest
from fastapi.testclient import TestClient

import api.main as main
from src.config import LABELS


@pytest.fixture(scope="module")
def client():
    # Le bloc "with" déclenche le lifespan : le modèle est chargé une seule fois.
    with TestClient(main.app) as test_client:
        yield test_client


def predict(client, payload):
    return client.post("/predict", json=payload)


# --- Fonctionnement normal ---------------------------------------------------


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("I loved the ending.", "positif"),
        ("Great acting and a wonderful story.", "positif"),
        ("The movie was boring.", "négatif"),
        ("Terrible acting, a waste of time.", "négatif"),
    ],
)
def test_predict_returns_expected_label(client, text, expected_label):
    response = predict(client, {"text": text})

    assert response.status_code == 200
    assert response.json()["label"] == expected_label


def test_predict_response_matches_streamlit_contract(client):
    response = predict(client, {"text": "I loved this movie"})
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"label", "proba"}
    assert body["label"] in LABELS.values()
    assert isinstance(body["proba"], float)
    assert 0.0 <= body["proba"] <= 1.0


def test_predict_proba_is_probability_of_predicted_class(client):
    text = "The movie was boring."
    probas = client.app.state.model.predict_proba([text])[0]

    body = predict(client, {"text": text}).json()

    assert body["proba"] == pytest.approx(probas.max(), abs=1e-4)
    assert body["proba"] >= 0.5


# --- Validation des entrées --------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"text": ""}, id="texte-vide"),
        pytest.param({"text": "   \n\t "}, id="espaces-seuls"),
        pytest.param({}, id="champ-text-manquant"),
        pytest.param({"text": "Great movie", "extra": 1}, id="champ-supplementaire"),
        pytest.param({"text": 123}, id="type-nombre"),
        pytest.param({"text": None}, id="type-null"),
        pytest.param({"text": ["Great movie"]}, id="type-liste"),
        pytest.param({"text": "a" * (main.MAX_TEXT_LENGTH + 1)}, id="texte-trop-long"),
    ],
)
def test_predict_rejects_invalid_input_with_422(client, payload):
    response = predict(client, payload)

    assert response.status_code == 422
    assert response.json()["detail"]


def test_predict_rejects_malformed_json_with_422(client):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_predict_accepts_text_at_max_length(client):
    response = predict(client, {"text": "a" * main.MAX_TEXT_LENGTH})

    assert response.status_code == 200


def test_validation_error_does_not_echo_user_input(client):
    response = predict(client, {"text": "a" * (main.MAX_TEXT_LENGTH + 1)})

    assert "aaaa" not in response.text


# --- Gestion du modèle -------------------------------------------------------


def test_model_is_loaded_at_startup(client):
    model = client.app.state.model

    assert hasattr(model, "predict_proba")
    assert {int(c) for c in model.classes_} == set(LABELS)


def test_startup_fails_with_clear_error_when_model_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "absent.joblib")

    with pytest.raises(RuntimeError, match="Model file is missing"):
        with TestClient(main.app):
            pass


class FailingModel:
    classes_ = [0, 1]

    def predict_proba(self, texts):
        raise ValueError("internal detail: C:/secret/path/model.joblib")


@pytest.fixture
def failing_model(client):
    # Sauvegarde/restauration explicite : monkeypatch.setattr sur app.state
    # (objet State de Starlette) supprime l'attribut au lieu de le restaurer.
    real_model = client.app.state.model
    client.app.state.model = FailingModel()
    yield
    client.app.state.model = real_model


@pytest.mark.usefixtures("failing_model")
def test_predict_returns_generic_500_when_model_fails(client):
    response = predict(client, {"text": "I loved this movie"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert "secret" not in response.text
