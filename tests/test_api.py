from statistics import median
import time

from fastapi.testclient import TestClient
import pytest

from mlops.api import main as api_main


class DummyModel:
    def predict(self, features_df):
        amount = float(features_df["Amount"].iloc[0])
        return [1 if amount >= 15 else 0]

    def predict_proba(self, features_df):
        amount = float(features_df["Amount"].iloc[0])
        probability = 0.85 if amount >= 15 else 0.15
        return [[1.0 - probability, probability]]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api_main, "load_model", lambda: DummyModel())
    return TestClient(api_main.app)


def make_payload(*, feature_vector: list[float] | None = None) -> dict[str, float | list[float]]:
    payload: dict[str, float | list[float]] = {
        column_name: float(index) for index, column_name in enumerate(api_main.FEATURE_COLUMNS)
    }
    if feature_vector is not None:
        payload["feature_vector"] = feature_vector
    return payload


def test_predict_happy_path_returns_probability(client: TestClient) -> None:
    response = client.post("/predict", json=make_payload())

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["fraud_probability"], float)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in {0, 1}


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (lambda: {key: value for key, value in make_payload().items() if key != "Amount"}, 422),
        (lambda: {**make_payload(), "Amount": "abc"}, 422),
        (lambda: make_payload(feature_vector=[0.0] * 29), 422),
    ],
)
def test_predict_validation_errors(
    client: TestClient,
    payload,
    expected_status: int,
) -> None:
    response = client.post("/predict", json=payload())

    assert response.status_code == expected_status


def test_predict_latency_median_under_threshold(client: TestClient) -> None:
    timings: list[float] = []

    for _ in range(100):
        started_at = time.perf_counter()
        response = client.post("/predict", json=make_payload())
        timings.append(time.perf_counter() - started_at)
        assert response.status_code == 200

    assert median(timings) <= 0.2