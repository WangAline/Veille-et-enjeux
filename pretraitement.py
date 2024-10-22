import os
from PIL import Image
import numpy as np

# Dossier contenant les images
image_folder = 'imagesMine'
processed_folder = 'processed_images'

# Taille cible des images
target_size = (224, 224)

# Fonction pour prétraiter les images
def preprocess_images(image_folder, processed_folder, target_size):
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Parcourir chaque fichier dans le dossier d'images
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            # Chemin complet de l'image
            img_path = os.path.join(image_folder, filename)
            
            # Charger l'image
            img = Image.open(img_path)
            
            # Redimensionner l'image
            img = img.resize(target_size)
            
            # Convertir l'image en tableau NumPy
            img_array = np.array(img)
            
            # Normaliser les valeurs des pixels (0 à 255) en les ramenant entre 0 et 1
            img_array = img_array / 255.0
            
            # Sauvegarder l'image redimensionnée et normalisée
            processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
            processed_img.save(os.path.join(processed_folder, filename))
            
            print(f"Processed {filename}")
    
    print("All images have been processed!")

# Appeler la fonction pour prétraiter les images
preprocess_images(image_folder, processed_folder, target_size)
