from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io

# Initialiser l'API FastAPI
app = FastAPI()

# Charger le modèle CNN
model = load_model("model/model.h5")

# Fonction de prétraitement des images
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convertir en niveaux de gris
    image = image.resize((28, 28))  # Redimensionner à 28x28 (comme MNIST)
    image = np.array(image)
    image = image / 255.0  # Normalisation
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
        # Lire l'image
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # Prédiction avec le modèle CNN
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction)

        return {"prediction": int(predicted_label), "confidence": float(np.max(prediction))}
    except Exception as e:
        return {"error": str(e)}

# Lancer l'API si on exécute directement le fichier
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
