from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException

from helpers.create_models import create_model
from helpers.typing_models import (
    LinRegHyperparameters,
    DecisionTreeHyperparameters,
    RandomForestHyperparameters,
    TrainingData,
    ModelInfo,
    ModelList,
    PredictionData
)
from helpers.save_model_minio import FileSave

app = FastAPI()
file_saver = FileSave("127.0.0.1:9000", "minioadmin", "minioadmin")
# шобы не падало надо поднять сервер: minio server /Users/isupport/Desktop/code/MLOps. Если сервер не стоит - это плохо

# JSON с моделью и гиперпараметрами по pydantic - СЕЙЧАС ПОЛОМАНО так как удалила хранение в оперативке
# также jsonф с примерами создать для версионирования датасетов
# собрать все в docker (и запустить в docker-compose), как запушить в docker-hub?


@app.post("/train_model/{model_name}", response_model=ModelInfo)
def train_model(
    model_name: str,
    hyperparameters: Union[
        LinRegHyperparameters, DecisionTreeHyperparameters, RandomForestHyperparameters
    ],
    training_data: TrainingData,
):
    """
    <pre>
    Args:
        model_name: name of the model as string, available models
            * linear_regression
            * decision_tree
            * random_forest
        hyperparameters: dict of hyperparameters names as keys and its values as values
        training_data:
            json like {
                "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "labels": [2.5, 3.5, 4.5]
            }
    Returns:
        dict with name of the trained model and it"s hyperparameters
    </pre>
    """
    model = create_model(model_name, hyperparameters)
    model.fit(training_data.features, training_data.labels)

    model_name = f"{model_name}_{len([i for i in file_saver.list_of_models_minio() if model_name in i]) + 1}"
    file_saver.save_model_to_minio(model, model_name, hyperparameters)

    return {"model_name": model_name, "hyperparameters": hyperparameters}


@app.get("/list_models", response_model=ModelList)
def list_models():
    models_list = file_saver.list_of_models_minio()
    return {"models": models_list}


@app.post("/predict/{model_name}", response_model=Dict[str, List[float]])
def predict(model_name: str, prediction_data: PredictionData):
    """
    <pre>
    Args:
        model_name: name of the model as string, see available models in list_models
            * linear_regression
            * decision_tree
            * random_forest
        prediction_data:
            json like {
                "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
            }

    Returns:
        dict with predictions
    </pre>
    """
    model, hyperparameters = file_saver.load_model_from_minio(model_name)
    prediction = model.predict(prediction_data.features)
    return {"prediction": prediction}


# Endpoint to retrain a specific model
@app.put("/retrain_model/{model_name}", response_model=ModelInfo)
def retrain_model(
    model_name: str,
    hyperparameters: Union[
        LinRegHyperparameters, DecisionTreeHyperparameters, RandomForestHyperparameters
    ],
    training_data: TrainingData
):
    """
    <pre>
    Args:
        model_name: name of the model as string, see available models in list_models
            * linear_regression
            * decision_tree
            * random_forest
        hyperparameters: dict of hyperparameters names as keys and its values as values
        training_data:
            json like {
                "features": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "labels": [2.5, 3.5, 4.5]
            }
    Returns:
        dict with name of the trained model and it"s hyperparameters
    </pre>
    """
    if model_name not in file_saver.list_of_models_minio():
        raise HTTPException(status_code=404, detail="Model not found")

    file_saver.delete_model_from_minio(model_name)

    model = create_model("_".join(model_name.split('_')[:2]), hyperparameters)
    model.fit(training_data.features, training_data.labels)

    file_saver.save_model_to_minio(model, model_name, hyperparameters)
    return {"model_name": model_name, "hyperparameters": hyperparameters}


# Endpoint to delete a specific model
@app.delete("/delete_model/{model_name}", response_model=ModelInfo)
def delete_model(model_name: str):
    """
    <pre>
    Args:
        model_name: name of the model as string, see available models in list_models
            * linear_regression
            * decision_tree
            * random_forest
    Returns:
        dict of model name and its hyperparameters
    </pre>
    """
    models_list = file_saver.list_of_models_minio()
    if model_name not in models_list:
        raise HTTPException(status_code=404, detail="Model not found")

    model, hyperparameters = file_saver.load_model_from_minio(model_name)
    file_saver.delete_model_from_minio(model_name})

    return {
        "model_name": model_name,
        "hyperparameters": hyperparameters
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
