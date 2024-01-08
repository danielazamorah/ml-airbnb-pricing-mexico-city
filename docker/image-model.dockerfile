FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["model.keras", "predict.py", "./"]

EXPOSE 8080
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8080", "predict:app"]