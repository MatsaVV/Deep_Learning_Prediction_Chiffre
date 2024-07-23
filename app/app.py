import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Chargement du modèle et des données
model = load_model('model/model.h5')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Initialisation de l'état de l'application
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
    st.session_state.correct_predictions = 0
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
            elif incorrect:
                st.error(f'Oups! Le bon chiffre était {true_label}.')
            st.session_state.update_image = True  # Marque pour charger une nouvelle image
            st.session_state.predicted = False  # Réinitialise l'état de prédiction

    # Bouton pour charger une nouvelle image
    if st.button('Nouvelle image'):
        st.session_state.update_image = True
        st.session_state.predicted = False

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

elif menu == 'Jeux':
    st.header('Jeux')
    st.subheader("Dessinez un chiffre et voyez combien de fois le modèle peut le prédire correctement en 10 essais.")

    # Afficher le canevas pour dessiner
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas_game"
    )

    # Bouton pour soumettre le dessin
    if st.button('Soumettre le dessin'):
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

            # Mettre à jour les statistiques
            st.session_state.attempts += 1
            st.session_state.predictions.append(predicted_label)
            st.write(f'Prédiction {st.session_state.attempts}: {predicted_label}')

            if st.session_state.attempts >= 10:
                correct_predictions = sum([1 for p in st.session_state.predictions if p == predicted_label])
                st.write(f"Nombre de prédictions correctes : {correct_predictions} sur 10")
                st.write(f"Précision : {correct_predictions * 10}%")

                # Réinitialiser les statistiques
                st.session_state.attempts = 0
                st.session_state.correct_predictions = 0
                st.session_state.predictions = []
