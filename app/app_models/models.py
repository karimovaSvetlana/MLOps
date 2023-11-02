import lightgbm as lgb
import pandas as pd

from catboost import CatBoostClassifier
from catboost import Pool

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def prep_data_for_train(path_to_data: str, target_name: str = 'target', test_size: float = 0.2) -> tuple:
    """

    Args:
        path_to_data: path to data from current directory. File of data should be csv
        target_name: name of the target field in train data
        test_size: share of data to take as a test_data

    Returns:
        splitted_datasets: tuple of (x_train, x_test, y_train, y_test)
    """
    df = pd.read_csv(path_to_data)
    x = df.drop(target_name, axis=1)
    y = df[target_name]
    splitted_datasets = train_test_split(x, y, test_size=test_size, random_state=52)

    return splitted_datasets


def train_lgb(path_to_data: str, target_name: str = 'target', params: dict = None):
    """Trains LGBM model and returns trained model and metric

    Args:
        path_to_data: path to data from current directory. File of data should be csv
        target_name: name of the target field in train data
        params: parameters of LightGBM model

    Returns:
        results: dict with keys 'model' and 'metrics'
    """
    x_train, x_test, y_train, y_test = prep_data_for_train(path_to_data, target_name)

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

    if params is None:
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

    num_round = 100
    model = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

    accuracy = accuracy_score(y_test, y_pred_binary)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    results = {
        'model': model,
        'metrics': accuracy
    }
    return results


def train_catboost(path_to_data: str, target_name: str = 'target', params: dict = None):
    """Trains CatBoost model and returns trained model and metric

    Args:
        path_to_data: path to data from current directory. File of data should be csv
        target_name: name of the target field in train data
        params: parameters of CatBoost model

    Returns:
        results: dict with keys 'model' and 'metrics'
    """
    x_train, x_test, y_train, y_test = prep_data_for_train(path_to_data, target_name)

    train_pool = Pool(x_train, y_train)
    
    if params is None:
        params = {
            'iterations': 100, 
            'depth': 5,
            'learning_rate': 0.1,
            'loss_function': 'Logloss'
        }

    model = CatBoostClassifier(**params)
    model.fit(train_pool)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    results = {
        'model': model,
        'metrics': accuracy
    }
    return results


def train_logreg(path_to_data: str, target_name: str = 'target', params: dict = None):
    """Trains LogReg model and returns trained model and metric

    Args:
        path_to_data: path to data from current directory. File of data should be csv
        target_name: name of the target field in train data
        params: parameters of LightGBM model

    Returns:
        results: dict with keys 'model' and 'metrics'
    """
    x_train, x_test, y_train, y_test = prep_data_for_train(path_to_data, target_name)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    results = {
        'model': model,
        'metrics': accuracy
    }
    return results



