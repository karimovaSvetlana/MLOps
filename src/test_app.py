from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_train_model():
    model_name = "linear_regression"
    hyperparameters = {
        "fit_intercept": True,
        "copy_X": True
    }
    training_data = {
        "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        "labels": [2.5, 3.5, 4.5]
    }
    response = client.post(f"/train_model/{model_name}",
                           json={"hyperparameters": hyperparameters, "training_data": training_data})
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"][:17] == model_name


def test_list_models():
    response = client.get("/list_models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["models"], list)


def test_predict():
    model_name = "linear_regression_1"
    prediction_data = {
        "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    }
    response = client.post(f"/predict/{model_name}", json=prediction_data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["prediction"], list)


def test_retrain_model():
    model_name = "linear_regression_1"
    hyperparameters = {
        "fit_intercept": True,
        "copy_X": True
    }
    training_data = {
        "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        "labels": [2.5, 3.5, 4.5]
    }
    response = client.put(f"/retrain_model/{model_name}",
                          json={"hyperparameters": hyperparameters, "training_data": training_data})
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == model_name


def test_delete_model():
    model_name = "linear_regression_1"
    response = client.delete(f"/delete_model/{model_name}")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == model_name
