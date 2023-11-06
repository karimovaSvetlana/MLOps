import grpc
from concurrent import futures
import model_service_pb2
import model_service_pb2_grpc

# Import the logic for model training
from app import train_model  # Assuming you have a train_model function


# Implement the gRPC service
class ModelService(model_service_pb2_grpc.ModelServiceServicer):
    def TrainModel(self, request, context):
        # Implement your model training logic here
        model_name = request.model_name
        hyperparameters = request.hyperparameters
        training_data = request.training_data

        # Call the train_model function from your models module
        result = train_model(model_name, hyperparameters, training_data)

        # Create and return a response
        response = model_service_pb2.TrainModelResponse(
            model_name=model_name,
            status=result
        )
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port('[::]:50051')  # Specify the gRPC server port
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
