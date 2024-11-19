import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Répertoires des données
data_dir = 'processed_images'
target_size = (224, 224)  # Doit correspondre au prétraitement
batch_size = 32

# Création d'un générateur d'images
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisation des pixels
    validation_split=0.2  # Séparer les données en 80% train / 20% validation
)

# Chargement des données d'entraînement
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',  # Pour un problème binaire : "mine" ou "non-mine"
    subset='training'
)

# Chargement des données de validation
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)
# Vérifier que les données ont été correctement chargées
print(f"Nombre d'images d'entraînement : {train_data.samples}")
print(f"Nombre d'images de validation : {val_data.samples}")
