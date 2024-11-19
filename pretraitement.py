import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



# Taille cible des images
target_size = (224, 224)

# Fonction pour prétraiter les images
def preprocess_images(image_folder, processed_folder, target_size):
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Parcourir chaque fichier dans le dossier d'images
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp')):
            img_path = os.path.join(image_folder, filename)
            try:
                # Charger l'image
                img = Image.open(img_path).convert('RGB')

                # Redimensionner l'image
                img = img.resize(target_size)

                # Convertir l'image en tableau NumPy et normaliser
                img_array = np.array(img) / 255.0

                # Sauvegarder l'image redimensionnée et normalisée
                processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
                processed_img.save(os.path.join(processed_folder, filename))
                print(f"Processed {filename}")

                # Sauvegarder dans un log
                with open('preprocessing_log.txt', 'a') as log_file:
                    log_file.write(f"Processed {filename}\n")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    print("All images have been processed!")

# Afficher une image prétraitée pour vérification
def display_sample_image(processed_folder):
    sample_files = os.listdir(processed_folder)
    if sample_files:
        sample_path = os.path.join(processed_folder, sample_files[0])
        sample_img = Image.open(sample_path)
        plt.imshow(sample_img)
        plt.axis('off')
        plt.show()
    else:
        print("No processed images to display.")

# Dossier contenant les images
image_folder_mines = 'imagesMine'
processed_folder_mines = 'processed_mines'
# Dossier contenant les images
image_folder_pasmines = 'imagesPasMine'
processed_folder_pasmines = 'processed_pasmines'

# Appeler la fonction pour prétraiter les images
preprocess_images(image_folder_mines, processed_folder_mines, target_size)
preprocess_images(image_folder_pasmines, processed_folder_pasmines, target_size)

# Afficher un échantillon
display_sample_image(processed_folder_mines)
