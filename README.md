# App
Project within MLOps subject in HSE

! All the code is checked with `flake8` and formatted with `black`

### Files
1. `app.py` - the main realisation of FastAPI
2. `test_app.py` - tests with pytest
3. `poetry.lock` - dependencies
4. `pyprogect.toml` - result of venv creation
5. <details>
    <summary>grpc archive</summary>
    gRPC files:
    - `model_service.proto` - main file that creates configuration for the app
    - `model_service_pb2.py` - file created with compilation of .proto file
    - `model_service_pb2_grpc.py` - file created with compilation of .proto file
    - `grpc_server.py` - file to raise a server
    - `grpc_client.py` - file to test if everything works
</details>


### Usage

To start using app - firstly pull images from dockerhub

To interact with app use the command ```docker-compose up``` and than have a good time with Swagger)))

<details>
    <summary>archive</summary>
    To start using app - firstly install all libs
    ```
    poetry init  # creates your personal venv
    poetry install  # installing all dependencies from poetry.lock file
    ```
    To interact with app use the command below
    ```
    poetry run python app.py
    ```
    To check tests use
    ```
    poetry run pytest test_app.py
    ```
</details>

Done!<br>
<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExYW40bHNpbHo0bW1qeHEzNGhieWVtNjI3eWF5dmptOTNnbnRndTBuZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/a7aOFe45g61JbiCl80/giphy.gif)https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExYW40bHNpbHo0bW1qeHEzNGhieWVtNjI3eWF5dmptOTNnbnRndTBuZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/a7aOFe45g61JbiCl80/giphy.gif" height="200" />

