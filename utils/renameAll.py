import os
import re

# Chemin du dossier contenant les images
folder_path = r'C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\intermédiaire'

# Parcourir tous les fichiers dans le dossier
for filename in os.listdir(folder_path):
    print(f"Traitement du fichier : {filename}")  # Ajout d'une instruction print pour le débogage
    # Vérifier si le fichier correspond au modèle désiré (nombre suivi de .jpg)
    if re.match(r'^\d+\.png$', filename):
        # Extraire le numéro de l'image
        number = int(re.findall(r'\d+', filename)[0])
        # Construire le nouveau nom de fichier
        new_filename = f'epoch_{number}.png'
        # Renommer le fichier
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    else:
        print(f"Le fichier {filename} ne correspond pas au modèle attendu.")

print("Renommage terminé.")
