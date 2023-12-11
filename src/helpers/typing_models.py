from pydantic import BaseModel, Field
from typing import List, Union


class LinRegHyperparameters(BaseModel):
    fit_intercept: bool = Field(
        description="Whether to calculate the intercept for this model", example=False
    )
    copy_X: bool = Field(
        description="If True, X will be copied; else, it may be overwritten",
        example=False,
    )


class DecisionTreeHyperparameters(BaseModel):
    max_depth: int = Field(description="The maximum depth of the tree", example=50)
    min_samples_split: int = Field(
        description="The minimum number of samples required to split an internal node",
        example=10,
    )
    random_state: int = Field(
        description="Controls the randomness of the estimator", example=52
    )


class RandomForestHyperparameters(BaseModel):
    n_estimators: int = Field(
        description="The number of trees in the forest", example=100
    )
    max_depth: int = Field(description="The maximum depth of the tree", example=50)
    random_state: int = Field(
        description="Controls the randomness of the estimator", example=52
    )


class TrainingData(BaseModel):
    features: List[List[float]] = Field(
        description="Whether to calculate the intercept for this model",
        example=[[1.2, 2.0], [2.9, 3.3], [3.1, 4.0]],
    )
    labels: List[float] = Field(
        description="Whether to calculate the intercept for this model",
        example=[2.5, 3.5, 4.5],
    )


class ModelInfo(BaseModel):
    model_name: str
    hyperparameters: Union[
        LinRegHyperparameters, DecisionTreeHyperparameters, RandomForestHyperparameters
    ]


class ModelList(BaseModel):
    models: List[str]


class PredictionData(BaseModel):
    features: List[List[float]] = Field(
        description="Whether to calculate the intercept for this model",
        example=[[1, 2], [9.0, 3.3], [3.5, 4.0]],
    )
