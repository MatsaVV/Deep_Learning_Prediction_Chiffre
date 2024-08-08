import pytest
import numpy as np
from tensorflow.keras.models import load_model
from app import predict_image

model = load_model('model/model.h5')

@pytest.fixture
def sample_image():
    image = np.random.rand(28, 28).astype(np.float32)
    return image

def test_predict_image(sample_image):
    predicted_label = predict_image(sample_image)
    assert isinstance(predicted_label, int), "La prédiction devrait être un entier"
    assert 0 <= predicted_label < 10, "La prédiction devrait être entre 0 et 9"
