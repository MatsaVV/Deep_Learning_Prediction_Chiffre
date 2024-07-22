import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Chargement du modèle et des données
model = load_model('model.h5')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

st.title('Reconnaissance de chiffres manuscrits')

# Canvas pour dessiner un chiffre
st.subheader("Dessinez un chiffre:")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage
    stroke_width=10,
    stroke_color="#ffffff",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Bouton pour prédire le chiffre dessiné
if st.button('Prédire le chiffre dessiné'):
    if canvas_result.image_data is not None:
        # Traitement de l'image pour la prédiction
        img = canvas_result.image_data
        img = Image.fromarray((img[:, :, 0]).astype('uint8'))  # Convertir en image PIL, utiliser un seul canal
        img = img.resize((28, 28), Image.ANTIALIAS)  # Redimensionner l'image comme les données d'entraînement
        img = np.array(img) / 255.0  # Normalisation
        img = img.reshape(1, 28, 28, 1)  # Reshape pour le modèle

        # Prédiction
        pred = model.predict(img)
        predicted_label = np.argmax(pred)
        st.write(f'Prédiction du chiffre dessiné : {predicted_label}')

# Fonction pour charger une image aléatoire et prédire
def load_random_image_and_predict():
    if 'index' not in st.session_state or st.session_state.update_image:
        st.session_state.index = np.random.randint(0, X_test.shape[0])
        st.session_state.update_image = False

    index = st.session_state.index
    image = X_test[index].reshape(28, 28)
    label = y_test[index]

    st.image(image, caption='Image aléatoire du dataset', width=150)
    if st.button('Prédire'):
        st.session_state.predicted = True
        image = image.reshape(1, 28, 28, 1)  # Reshape pour le modèle
        pred = model.predict(image)
        predicted_label = np.argmax(pred)
        st.session_state.predicted_label = predicted_label
        st.write(f'Prédiction : {predicted_label}')
    elif 'predicted' in st.session_state and st.session_state.predicted:
        st.write(f'Prédiction précédente : {st.session_state.predicted_label}')

    return label, st.session_state.get('predicted_label', None)

# Initialiser les valeurs de session
if 'update_image' not in st.session_state:
    st.session_state.update_image = True
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

true_label, predicted_label = load_random_image_and_predict()

# Boutons pour validation de la prédiction
if predicted_label is not None:
    correct = st.button('Correct')
    incorrect = st.button('Incorrect')
    if correct or incorrect:
        if correct:
            st.success('Merci pour votre confirmation!')
        elif incorrect:
            st.error(f'Oups! Le bon chiffre était {true_label}.')
        st.session_state.update_image = True  # Marque pour charger une nouvelle image
        st.session_state.predicted = False  # Réinitialise l'état de prédiction

# Bouton pour charger une nouvelle image
if st.button('Nouvelle image'):
    st.session_state.update_image = True
    st.session_state.predicted = False
