# App
Project within MLOps subject in HSE

! All the code is checked with `flake8` and formatted with `black`

### Files
1. `app.py` - the main realisation of FastAPI
2. `test_app.py` - tests with pytest
3. `poetry.lock` - dependencies
4. `pyprogect.toml` - result of venv creation

### Usage
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

Done!<br>
<img src="https://media.giphy.com/media/KzM1lAfJjCWNq/giphy.gif" width="240" height="200" />

