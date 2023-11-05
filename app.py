from typing import List, Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

app = FastAPI()

# A dictionary to store trained models
trained_models = {}


# Pydantic models for request and response data
class LinRegHyperparameters(BaseModel):
    fit_intercept: bool = Field(
        description="Whether to calculate the intercept for this model",
        example=False
    )
    copy_X: bool = Field(
        description="If True, X will be copied; else, it may be overwritten",
        example=False
    )


class DecisionTreeHyperparameters(BaseModel):
    max_depth: int = Field(
        description="The maximum depth of the tree",
        example=50
    )
    min_samples_split: int = Field(
        description="The minimum number of samples required to split an internal node",
        example=10
    )
    random_state: int = Field(
        description="Controls the randomness of the estimator",
        example=52
    )


class RandomForestHyperparameters(BaseModel):
    n_estimators: int = Field(
        description="The number of trees in the forest",
        example=100
    )
    max_depth: int = Field(
        description="The maximum depth of the tree",
        example=50
    )
    random_state: int = Field(
        description="Controls the randomness of the estimator",
        example=52
    )


class TrainingData(BaseModel):
    features: List[List[float]] = Field(
        description="Whether to calculate the intercept for this model",
        example=[[1.2, 2.0], [2.9, 3.3], [3.1, 4.0]]
    )
    labels: List[float] = Field(
        description="Whether to calculate the intercept for this model",
        example=[2.5, 3.5, 4.5]
    )


class ModelInfo(BaseModel):
    model_name: str
    hyperparameters: Union[
        LinRegHyperparameters,
        DecisionTreeHyperparameters,
        RandomForestHyperparameters
    ]


class ModelList(BaseModel):
    models: List[str]


class PredictionData(BaseModel):
    features: List[List[float]] = Field(
        description="Whether to calculate the intercept for this model",
        example=[[1, 2], [9.0, 3.3], [3.5, 4.0]]
    )


# Function to create a model instance based on model_name
def create_model(
        model_name: str,
        hyperparameters: Union[
            LinRegHyperparameters,
            DecisionTreeHyperparameters,
            RandomForestHyperparameters
        ]
) -> object:
    """

    Args:
        model_name: name of the model as string, available models
            * linear_regression
            * decision_tree
            * random_forest
        hyperparameters: dict of hyperparameters names as keys and its values as values

    Returns:
        model: object of sklearn model
    """
    if model_name == "linear_regression":
        # if not isinstance(hyperparameters, LinRegHyperparameters):
        #     raise HTTPException(status_code=400, detail="Invalid parameters type")
        model = LinearRegression()
        model.fit_intercept = hyperparameters.fit_intercept
        model.copy_X = hyperparameters.copy_X

    elif model_name == "decision_tree":
        # if not isinstance(hyperparameters, DecisionTreeHyperparameters):
        #     raise HTTPException(status_code=400, detail="Invalid parameters type")
        model = DecisionTreeRegressor()
        model.max_depth = hyperparameters.max_depth
        model.min_samples_split = hyperparameters.min_samples_split
        model.random_state = hyperparameters.random_state

    elif model_name == "random_forest":
        # if not isinstance(hyperparameters, RandomForestHyperparameters):
        #     raise HTTPException(status_code=400, detail="Invalid parameters type")
        model = RandomForestRegressor()
        model.n_estimators = hyperparameters.n_estimators
        model.max_depth = hyperparameters.max_depth

    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return model


# Endpoint to train a model
@app.post("/train_model/{model_name}", response_model=ModelInfo)
def train_model(
        model_name: str,
        hyperparameters: Union[
            LinRegHyperparameters,
            DecisionTreeHyperparameters,
            RandomForestHyperparameters
        ],
        training_data: TrainingData
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

    model_name = f"{model_name}_{len([i for i in trained_models.keys() if model_name in i]) + 1}"
    trained_models[model_name] = {"model": model, "hyperparameters": hyperparameters}

    return {"model_name": model_name, "hyperparameters": hyperparameters}


# Endpoint to list available models
@app.get("/list_models", response_model=ModelList)
def list_models():
    return {"models": list(trained_models.keys())}


# Endpoint to make predictions using a specific model
@app.post("/predict/{model_name}", response_model=Dict[str, List[float]])
def predict(
        model_name: str,
        prediction_data: PredictionData
):
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
            LinRegHyperparameters,
            DecisionTreeHyperparameters,
            RandomForestHyperparameters
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
def delete_model(
        model_name: str
):
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
    return {"model_name": model_name, "hyperparameters": deleted_model_info["hyperparameters"]}


# Implement Swagger documentation
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
