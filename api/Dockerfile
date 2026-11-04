FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache Hugging Face (monté en volume par docker compose : le modèle n'est téléchargé qu'une fois)
    HF_HOME=/models

WORKDIR /app

# PyTorch en version CPU : environ 5 fois plus léger que la version GPU par défaut
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Code ET tests : on lance pytest dans le conteneur (docker compose exec api pytest -v)
COPY . .

# Dossiers montés en volume par docker compose
RUN mkdir -p /models /app/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
