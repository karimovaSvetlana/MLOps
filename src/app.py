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
from helpers.save_minio import FileSave

app = FastAPI()
file_saver = FileSave('127.0.0.1:9000', 'minioadmin', 'minioadmin')

trained_models = {}


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
                'features': [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                'labels': [2.5, 3.5, 4.5]
            }
    Returns:
        dict with name of the trained model and it's hyperparameters
    </pre>
    """
    model = create_model(model_name, hyperparameters)
    model.fit(training_data.features, training_data.labels)

    model_name = (
        f"{model_name}_{len([i for i in trained_models.keys() if model_name in i]) + 1}"
    )
    trained_models[model_name] = {"model": model, "hyperparameters": hyperparameters}

    file_saver.save_model_to_minio(model, model_name)

    return {"model_name": model_name, "hyperparameters": hyperparameters}


# Endpoint to list available models
@app.get("/list_models", response_model=ModelList)
def list_models():
    return {"models": list(trained_models.keys())}


# Endpoint to make predictions using a specific model
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
                'features': [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
            }

    Returns:
        dict with predictions
    </pre>
    """
    file_saver.load_model_from_minio(model_name)

    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = trained_models[model_name]["model"]
    prediction = model.predict(prediction_data.features)
    return {"prediction": prediction}


# Endpoint to retrain a specific model
@app.put("/retrain_model/{model_name}", response_model=ModelInfo)
def retrain_model(
    model_name: str,
    hyperparameters: Union[
        LinRegHyperparameters, DecisionTreeHyperparameters, RandomForestHyperparameters
    ],
    training_data: TrainingData,
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
                'features': [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                'labels': [2.5, 3.5, 4.5]
            }
    Returns:
        dict with name of the trained model and it's hyperparameters
    </pre>
    """
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = trained_models[model_name]["model"]
    model.fit(training_data.features, training_data.labels)
    trained_models[model_name]["hyperparameters"] = hyperparameters
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
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    deleted_model_info = trained_models.pop(model_name)
    file_saver.delete_model_from_minio(model_name)
    return {
        "model_name": model_name,
        "hyperparameters": deleted_model_info["hyperparameters"],
    }


# Implement Swagger documentation
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
