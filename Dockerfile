FROM python:3.9-slim

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip
RUN pip install poetry
RUN pip install uvicorn
RUN poetry install

EXPOSE 8000

CMD poetry run python src/app.py
