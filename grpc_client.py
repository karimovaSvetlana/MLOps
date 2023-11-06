import grpc
import model_service_pb2
import model_service_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')  # Replace with the actual server address and port
stub = model_service_pb2_grpc.ModelServiceStub(channel)

request = model_service_pb2.TrainModelRequest(
    model_name="linear_regression",
    parameter1=True,
    parameter2=False,
    features=[1.0, 2.0, 3.0],
    labels=[2.5, 3.5, 4.5]
)

response = stub.TrainModel(request)

print(f"Model Name: {response.model_name}")
print(f"Status: {response.status}")
