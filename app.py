import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model, Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

# Fonction pour prédire une image
def predict_image(image):
    image = image.reshape(1, 28, 28, 1)
    pred = model.predict(image)
    predicted_label = np.argmax(pred)
    st.session_state.predicted_label = predicted_label
    return predicted_label

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
            st.error(f'Oups!')
            st.session_state.incorrect_predictions += 1
        del st.session_state.predicted_label

# Fonction pour afficher les activations des couches
def plot_layer_activations(model, img):
    layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name or 'max_pooling2d' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    for layer_name, layer_activation in zip([layer.name for layer in model.layers if 'conv2d' in layer.name or 'max_pooling2d' in layer.name], activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // 16
        display_grid = np.zeros((size * n_cols, size * 16))

        for col in range(n_cols):
            for row in range(16):
                channel_image = layer_activation[0, :, :, col * 16 + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        fig, ax = plt.subplots(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        ax.set_title(layer_name)
        ax.grid(False)
        ax.imshow(display_grid, aspect='auto', cmap='viridis')
        st.pyplot(fig)

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

            st.subheader("Activations des Couches")
            plot_layer_activations(model, img)

    if 'predicted_label' in st.session_state:
        true_label = st.number_input('Entrez le vrai chiffre dessiné :', min_value=0, max_value=9, step=1)
        validate_prediction(true_label)

    display_prediction_table()
