from flask.testing import FlaskClient

from flask_app import create_app


def test_health_endpoint():
    app = create_app()
    client: FlaskClient = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200


def test_predict_bad_payload():
    app = create_app()
    client: FlaskClient = app.test_client()
    resp = client.post("/predict", json={"texts": "not-a-list"})
    assert resp.status_code == 400

