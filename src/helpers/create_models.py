from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from typing import Union

from typing_models import LinRegHyperparameters, DecisionTreeHyperparameters, RandomForestHyperparameters


def create_model(
    model_name: str,
    hyperparameters: Union[
        LinRegHyperparameters, DecisionTreeHyperparameters, RandomForestHyperparameters
    ],
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
        model = LinearRegression(
            fit_intercept=hyperparameters.fit_intercept, copy_X=hyperparameters.copy_X
        )

    elif model_name == "decision_tree":
        model = DecisionTreeRegressor(
            max_depth=hyperparameters.max_depth,
            min_samples_split=hyperparameters.min_samples_split,
            random_state=hyperparameters.random_state,
        )

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=hyperparameters.n_estimators,
            max_depth=hyperparameters.max_depth,
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return model
