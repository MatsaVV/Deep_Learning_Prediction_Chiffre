import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests

# Définir l’URL de l’API REST
API_URL = "http://localhost:8000/predict"

# Fonction pour envoyer une image à l’API REST et recevoir une prédiction
def predict_image(image):
    image = image.reshape(28, 28)  # Assurer la bonne dimension
    image = (image * 255).astype('uint8')  # Convertir en uint8 pour l’API
    files = {"file": ("image.png", image.tobytes(), "image/png")}

    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            return "Erreur"
    except requests.exceptions.RequestException as e:
        return f"Erreur de connexion à l’API : {e}"

# Charger les données de test
X_test_new = np.load('data/X_test_new.npy')

# Initialisation de l'état de l'application
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.correct_predictions = 0
    st.session_state.incorrect_predictions = 0
    st.session_state.predictions = []

# Titre principal de l'application
st.title('Reconnaissance de chiffres manuscrits')

# Barre de menu
menu = st.sidebar.selectbox(
    'Menu',
    ['Image aléatoire', 'Dessin', 'Jeux']
)

# Fonction pour afficher le tableau des prédictions
def display_prediction_table():
    st.subheader('Tableau des prédictions')
    st.write(f"Prédictions correctes: {st.session_state.correct_predictions}")
    st.write(f"Prédictions incorrectes: {st.session_state.incorrect_predictions}")

# Fonction pour valider la prédiction
def validate_prediction(true_label):
    correct = st.button('Correct')
    incorrect = st.button('Incorrect')
    if correct or incorrect:
        if correct:
            st.success('Merci pour votre confirmation!')
            st.session_state.correct_predictions += 1
        elif incorrect:
            st.error(f'Oups! Mauvaise prédiction!')
            st.session_state.incorrect_predictions += 1
        del st.session_state.predicted_label

# Affichage du contenu en fonction de la sélection du menu
if menu == 'Image aléatoire':
    st.header('Image aléatoire')
    if 'index' not in st.session_state or st.session_state.update_image:
        st.session_state.index = np.random.randint(0, X_test_new.shape[0])
        st.session_state.update_image = False

    index = st.session_state.index
    image = X_test_new[index].reshape(28, 28)
    st.image(image, caption='Image aléatoire du dataset', width=150)

    if st.button('Prédire'):
        st.session_state.predicted = True
        predicted_label = predict_image(image)
        st.write(f'Prédiction : {predicted_label}')
    elif 'predicted' in st.session_state and st.session_state.predicted:
        st.write(f'Prédiction précédente : {st.session_state.predicted_label}')

    if 'predicted_label' in st.session_state:
        validate_prediction(true_label=None)

    if st.button('Nouvelle image'):
        st.session_state.update_image = True
        st.session_state.predicted = False

    display_prediction_table()

elif menu == 'Dessin':
    st.header('Dessinez un chiffre')
    st.subheader("Dessinez un chiffre:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button('Prédire le chiffre dessiné'):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = Image.fromarray((img[:, :, 0]).astype('uint8'))
            img = img.resize((28, 28), Image.ANTIALIAS)
            img = np.array(img) / 255.0
            img = img.reshape(1, 28, 28, 1)
            predicted_label = predict_image(img)
            st.write(f'Prédiction du chiffre dessiné : {predicted_label}')

    if 'predicted_label' in st.session_state:
        true_label = st.number_input('Entrez le vrai chiffre dessiné :', min_value=0, max_value=9, step=1)
        validate_prediction(true_label)

    display_prediction_table()

elif menu == 'Jeux':
    st.header('Jeux')
    st.subheader("Dessinez un chiffre:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas_game"
    )

    if st.button('Soumettre le dessin'):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = Image.fromarray((img[:, :, 0]).astype('uint8'))
            img = img.resize((28, 28), Image.ANTIALIAS)
            img = np.array(img) / 255.0
            img = img.reshape(1, 28, 28, 1)

            predicted_label = predict_image(img)
            st.write(f'Prédiction du chiffre dessiné : {predicted_label}')
            st.session_state.predicted_label = predicted_label

    if 'predicted_label' in st.session_state:
        true_label = st.number_input('Entrez le vrai chiffre dessiné :', min_value=0, max_value=9, step=1)
        validate_prediction(true_label)

    display_prediction_table()
