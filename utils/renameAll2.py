import os

# Chemin du dossier contenant les images
folder_path = r'C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\intermediaire'

# Renommer les fichiers de epoch_149.png à epoch_100.png
for i in range(149, 99, -1):  # Commence à 149 et va jusqu'à 100
    old_filename = os.path.join(folder_path, f'epoch_{i}.png')
    new_filename = os.path.join(folder_path, f'epoch_{i + 1}.png')

    # Vérifier si le fichier ancien existe avant de le renommer
    if os.path.exists(old_filename):
        print(f'Renommage de {old_filename} en {new_filename}')
        os.rename(old_filename, new_filename)
    else:
        print(f'Le fichier {old_filename} n\'existe pas et ne sera pas renommé.')

print("Renommage terminé.")
