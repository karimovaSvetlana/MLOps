FROM python:3.9-slim

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-root

COPY . /app

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
