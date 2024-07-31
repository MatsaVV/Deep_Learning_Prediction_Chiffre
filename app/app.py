import os
os.system('sh setup.sh')

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Chargement du modèle et des nouvelles données de test
model = load_model('model/model.h5')
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

# Fonction pour charger une image aléatoire et prédire
def load_random_image_and_predict():
    if 'index' not in st.session_state or st.session_state.update_image:
        st.session_state.index = np.random.randint(0, X_test_new.shape[0])
        st.session_state.update_image = False

    index = st.session_state.index
    image = X_test_new[index].reshape(28, 28)

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

    return None, st.session_state.get('predicted_label', None)

# Initialiser les valeurs de session pour l'image aléatoire
if 'update_image' not in st.session_state:
    st.session_state.update_image = True
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Affichage du contenu en fonction de la sélection du menu
if menu == 'Image aléatoire':
    st.header('Image aléatoire')
    true_label, predicted_label = load_random_image_and_predict()

    # Boutons pour validation de la prédiction
    if predicted_label is not None:
        correct = st.button('Correct')
        incorrect = st.button('Incorrect')
        if correct or incorrect:
            if correct:
                st.success('Merci pour votre confirmation!')
                st.session_state.correct_predictions += 1
            elif incorrect:
                st.error(f'Oups! Le bon chiffre était {true_label}.')
                st.session_state.incorrect_predictions += 1
            st.session_state.update_image = True  # Marque pour charger une nouvelle image
            st.session_state.predicted = False  # Réinitialise l'état de prédiction

    # Bouton pour charger une nouvelle image
    if st.button('Nouvelle image'):
        st.session_state.update_image = True
        st.session_state.predicted = False

    # Affichage du tableau des prédictions
    st.subheader('Tableau des prédictions')
    st.write(f"Prédictions correctes: {st.session_state.correct_predictions}")
    st.write(f"Prédictions incorrectes: {st.session_state.incorrect_predictions}")

elif menu == 'Dessin':
    st.header('Dessinez un chiffre')
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
            st.session_state.predicted_label = predicted_label  # Stocker la prédiction dans la session

    # Boutons pour validation de la prédiction pour la page "Dessin"
    if 'predicted_label' in st.session_state:
        true_label = st.number_input('Entrez le vrai chiffre dessiné :', min_value=0, max_value=9, step=1)
        correct = st.button('Correct')
        incorrect = st.button('Incorrect')
        if correct or incorrect:
            if correct:
                st.success('Merci pour votre confirmation!')
                st.session_state.correct_predictions += 1
            elif incorrect:
                st.error(f'Oups! Le bon chiffre était {true_label}.')
                st.session_state.incorrect_predictions += 1
            del st.session_state.predicted_label  # Réinitialiser la prédiction

    # Affichage du tableau des prédictions
    st.subheader('Tableau des prédictions')
    st.write(f"Prédictions correctes: {st.session_state.correct_predictions}")
    st.write(f"Prédictions incorrectes: {st.session_state.incorrect_predictions}")

elif menu == 'Jeux':
    st.header('Jeux')
    st.write('Travail à réaliser')
