from PIL import Image
import os

# Chemin du dossier contenant les images
folder_path = r'C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\intermediaire'
output_folder = r'C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\predicted_images'

# Assurez-vous que le dossier de sortie existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Coordonnées de la sous-image "predicted image" (à ajuster selon vos images)
x1, y1, x2, y2 = 656, 394, 1080, 815  # Exemple de coordonnées

# Traitement des images
for i in range(11, 101):
    file_path = os.path.join(folder_path, f'epoch_{i}.png')
    if os.path.exists(file_path):
        with Image.open(file_path) as img:
            # Découper la sous-image
            predicted_image = img.crop((x1, y1, x2, y2))
            # Enregistrer la sous-image
            predicted_image.save(os.path.join(output_folder, f'epoch_{i}.png'))
            print(f'Image {i} traitée.')
    else:
        print(f'Le fichier {file_path} n\'existe pas.')

print("Extraction terminée.")
