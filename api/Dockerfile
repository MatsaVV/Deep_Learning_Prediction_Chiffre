# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY main.py .
COPY model/ model/
COPY data/ /app/data/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'API
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
