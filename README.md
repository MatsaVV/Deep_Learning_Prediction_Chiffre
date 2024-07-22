# Reconnaissance de Chiffres Manuscrits

## Description du Projet

Ce projet est une application de reconnaissance de chiffres manuscrits développée avec TensorFlow et Streamlit. Elle permet aux utilisateurs de tester la reconnaissance de chiffres soit en utilisant des images aléatoires du dataset MNIST soit en dessinant directement dans l'application.

## Structure

digit_recognition/
│
├── app/ # Dossier pour l'application Streamlit
│ └── app.py # Script principal de l'application Streamlit
│
├── models/ # Dossier pour les modèles entraînés et les scripts d'entraînement
│ ├── train_model.py # Script pour entraîner le modèle
│ └── model.h5 # Modèle entraîné sauvegardé
│
├── data/ # Dossier pour les données utilisées par le modèle
│ ├── X_test.npy # Données de test 
│ └── y_test.npy # Labels de test
│
├── .gitignore # Fichier pour ignorer les fichiers/dossiers non nécessaires
└── README.md # Fichier README pour le projet


## Fonctionnalités

- Prédiction de chiffres manuscrits à partir d'images du dataset MNIST.
- Interface pour dessiner un chiffre et le soumettre pour prédiction.

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

