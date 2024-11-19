import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chemin vers le dossier racine contenant les sous-dossiers
base_dir = "images"  # Change si nécessaire
batch_size = 32
image_size = (224, 224)  # Taille des images redimensionnées (comme prétraitées)

# Création d'un générateur d'images avec augmentation des données
datagen = ImageDataGenerator(
    rescale=1./255,       # Normalisation des pixels entre 0 et 1
    validation_split=0.2  # Séparer en 80% entraînement / 20% validation
)

# Chargement des données d'entraînement
train_data = datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Classes binaires : Mine (1) ou PasMine (0)
    subset='training'     # Ensemble d'entraînement
)

# Chargement des données de validation
val_data = datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'   # Ensemble de validation
)

# Affichage des classes associées
print(f"Classes trouvées : {train_data.class_indices}")
