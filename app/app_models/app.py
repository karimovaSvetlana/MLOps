from flask import Flask, request, jsonify
from app.app_models.models import train_lgb, train_catboost, train_logreg

app = Flask("ultra_dumb")

# Dictionary to store trained models
trained_models = {}


@app.route('/train_model/<model_name>', methods=['POST'])
def train_model(model_name):
    try:
        data = request.get_json()

        if model_name == 'LightGBM':
            model = train_lgb()
        elif model_name == 'CatBoost':
            model = train_catboost()
        elif model_name == 'LogReg':
            model = train_logreg()
        else:
            return jsonify({"error": "Invalid model name"}), 400

        features = data.get('features', [])
        labels = data.get('labels', [])
        model.fit(features, labels)

        # Storing the trained model
        trained_models[model_name] = model

        return jsonify({"message": f"Model {model_name} trained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/list_models', methods=['GET'])
def list_models():
    return jsonify({"models": list(trained_models.keys())})


@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in trained_models:
        return jsonify({"error": "Model not found"}), 404

    try:
        data = request.get_json()
        features = data.get('features', [])
        prediction = trained_models[model_name].predict(features).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/retrain_model/<model_name>', methods=['POST'])
def retrain_model(model_name):
    if model_name not in trained_models:
        return jsonify({"error": "Model not found"}), 404

    try:
        data = request.get_json()
        features = data.get('features', [])
        labels = data.get('labels', [])
        trained_models[model_name].fit(features, labels)
        return jsonify({"message": f"Model {model_name} retrained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete_model/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    if model_name in trained_models:
        del trained_models[model_name]
        return jsonify({"message": f"Model {model_name} deleted"})
    return jsonify({"error": "Model not found"}), 404


# # Training a Model with Hyperparameters
# @app.route('/train_model/<model_name>', methods=['POST'])
# def train_model(model_name):
#     # Extract data from the request's JSON payload
#     data = request.get_json()
#     hyperparameters = data.get('hyperparameters')
#     training_data = data.get('training_data')
#
#     # Perform model training and return a response
#     # ...
#
#
# # Listing Available Models
# @app.route('/list_models', methods=['GET'])
# def list_models():
#
#
# # Return a list of available model names
# # ...
#
# # Making Predictions with a Specific Model
# @app.route('/predict/<model_name>', methods=['POST'])
# def predict(model_name):
#     # Extract input data from the request's JSON payload
#     input_data = request.get_json().get('input_data')
#
#     # Make predictions using the specified model
#     # ...
#
#
# # Retraining a Model
# @app.route('/retrain_model/<model_name>', methods=['POST'])
# def retrain_model(model_name):
#
#
# # Extract updated hyperparameters and training data
# # ...
#
# # Deleting a Model
# @app.route('/delete_model/<model_name>', methods=['DELETE'])
# def delete_model(model_name):
#
#
# # Delete the specified model
# # ...
#
# if __name__ == '__main__':
#     app.run(debug=True)
