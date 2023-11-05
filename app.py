from fastapi import FastAPI, HTTPException, Query, Path, Depends
from pydantic import BaseModel
from typing import List, Dict, Union
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# A dictionary to store trained models
trained_models = {}


# Pydantic models for request and response data
class Hyperparameters(BaseModel):
    parameter1: float
    parameter2: float


class TrainingData(BaseModel):
    features: List[List[float]]
    labels: List[float]


class ModelInfo(BaseModel):
    model_name: str
    hyperparameters: Hyperparameters


class ModelList(BaseModel):
    models: List[str]


class PredictionData(BaseModel):
    features: List[List[float]]


# Function to create a model instance based on model_name
def create_model(model_name: str) -> object:
    if model_name == "linear_regression":
        return LinearRegression()
    elif model_name == "decision_tree":
        return DecisionTreeRegressor()
    elif model_name == "random_forest":
        return RandomForestRegressor()
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")


# Endpoint to train a model
@app.post("/train_model/{model_name}", response_model=ModelInfo)
def train_model(
        model_name: str,
        hyperparameters: Hyperparameters,
        training_data: TrainingData
):
    """
    <pre>
    Args:
        model_name: name of the model as string, available models
            * linear_regression
            * decision_tree
            * random_forest
        hyperparameters: TODO
        training_data:
            json like {
                'features': [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                'labels': [2.5, 3.5, 4.5]
            }
    Returns:
        dict with name of the trained model and it's hyperparameters
    </pre>
    """
    model = create_model(model_name)
    model.fit(training_data.features, training_data.labels)
    trained_models[model_name] = {"model": model, "hyperparameters": hyperparameters}
    return {"model_name": model_name, "hyperparameters": hyperparameters}


# Endpoint to list available models
@app.get("/list_models", response_model=ModelList)
def list_models():
    return {"models": list(trained_models.keys())}


# Endpoint to make predictions using a specific model
@app.post("/predict/{model_name}", response_model=Dict[str, float])
def predict(
        model_name: str,
        prediction_data: PredictionData
):
    """
    <pre>
    Args:
        model_name: name of the model as string, available models
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
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = trained_models[model_name]["model"]
    prediction = model.predict(prediction_data.features)
    return {"prediction": prediction.tolist()}


# Endpoint to retrain a specific model
@app.put("/retrain_model/{model_name}", response_model=ModelInfo)
def retrain_model(
        model_name: str,
        hyperparameters: Hyperparameters,
        training_data: TrainingData
):
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = trained_models[model_name]["model"]
    model.fit(training_data.features, training_data.labels)
    trained_models[model_name]["hyperparameters"] = hyperparameters
    return {"model_name": model_name, "hyperparameters": hyperparameters}


# Endpoint to delete a specific model
@app.delete("/delete_model/{model_name}", response_model=ModelInfo)
def delete_model(
        model_name: str
):
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    deleted_model_info = trained_models.pop(model_name)
    return {"model_name": model_name, "hyperparameters": deleted_model_info["hyperparameters"]}


# Implement Swagger documentation
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
