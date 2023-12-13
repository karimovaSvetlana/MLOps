FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-root --no-dev

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]