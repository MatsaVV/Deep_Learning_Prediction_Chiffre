from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

# Initialiser l'API FastAPI
app = FastAPI()

# Charger le modèle CNN avec un chemin absolu
model_path = os.path.join(os.path.dirname(__file__), "model", "model.h5")
model = load_model(model_path)

# Fonction de prétraitement des images
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convertir en niveaux de gris
    image = image.resize((28, 28))  # Redimensionner à 28x28
    image = np.array(image) / 255.0  # Normalisation
    image = image.reshape(1, 28, 28, 1)  # Format pour le modèle
    return image

# Endpoint pour vérifier le statut de l’API
@app.get("/health")
def health_check():
    return {"status": "API en ligne"}

# Endpoint pour la prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire et prétraiter l'image
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # Prédiction avec le modèle CNN
        prediction = model.predict(image)
        predicted_label = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {"prediction": predicted_label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

# Lancer l'API si on exécute directement le fichier
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
