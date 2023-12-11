from minio import Minio
from minio.error import S3Error
import pickle
import io


class FileSave:
    def __init__(self, minio_host, access_key, secret_key, secure=False):
        self.minio_client = Minio(minio_host, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = "models"

    def save_model_to_minio(self, model, model_name):
        model_bytes = pickle.dumps(model)
        object_name = f"{model_name}.pkl"

        try:
            model_data = io.BytesIO(model_bytes)
            self.minio_client.put_object(self.bucket_name, object_name, model_data, len(model_bytes))
            print(f"Model '{model_name}' uploaded to Minio bucket '{self.bucket_name}'")
        except S3Error as e:
            print(f"Error uploading model: {e}")

    def load_model_from_minio(self, model_name):
        object_name = f"{model_name}.pkl"
        local_model_path = f"models/local_{model_name}.pkl"

        try:
            self.minio_client.fget_object(self.bucket_name, object_name, local_model_path)
            print(f"Model '{model_name}' downloaded from Minio to '{local_model_path}'")
        except S3Error as e:
            print(f"Error downloading model: {e}")
            return None

        with open(local_model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        return loaded_model

    def delete_model_from_minio(self, model_name):
        object_name = f"{model_name}.pkl"

        try:
            self.minio_client.remove_object(self.bucket_name, object_name)
            print(f"Model '{model_name}' deleted from Minio bucket '{self.bucket_name}'")
        except S3Error as e:
            print(f"Error deleting model: {e}")
