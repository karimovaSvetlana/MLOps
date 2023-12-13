from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error
import pickle
import io


class FileSave:
    def __init__(self, minio_host, access_key, secret_key, secure=False):
        self.minio_client = Minio(minio_host, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = "models"

    def save_model_to_minio(self, model, model_name, hyperparameters):
        model_info = {'model': model, 'hyperparameters': hyperparameters}
        model_bytes = pickle.dumps(model_info)
        object_name = f"{model_name}.pkl"

        try:
            model_data = io.BytesIO(model_bytes)
            self.minio_client.put_object(self.bucket_name, object_name, model_data, len(model_bytes))
            print(f"Model '{model_name}' uploaded to Minio bucket '{self.bucket_name}'")
        except S3Error as e:
            print(f"Error uploading model: {e}")

    def load_model_from_minio(self, model_name):
        object_name = f"{model_name}.pkl"

        try:
            model_data = self.minio_client.get_object(self.bucket_name, object_name)
            loaded_model_info = pickle.loads(model_data.read())
            print(f"Model '{model_name}' downloaded from Minio")
        except S3Error as e:
            print(f"Error downloading model: {e}")
            return None

        return loaded_model_info['model'], loaded_model_info['hyperparameters']

    def list_of_models_minio(self):
        try:
            objects = self.minio_client.list_objects(self.bucket_name, recursive=True)
            model_names = [obj.object_name.replace('.pkl', '') for obj in objects if isinstance(obj.object_name, str) and obj.object_name]
            return model_names

        except S3Error as e:
            print(f"Error getting list of models: {e}")

    def delete_model_from_minio(self, model_name):
        object_name = f"{model_name}.pkl"

        try:
            self.minio_client.remove_object(self.bucket_name, object_name)
            print(f"Model '{object_name}' deleted from Minio bucket '{self.bucket_name}'")
        except S3Error as e:
            print(f"Error deleting model: {e}")
