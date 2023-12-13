import matplotlib.pyplot as plt
from PIL import Image

# Chemin vers l'image
image_path = r'C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\intermediaire\epoch_11.png'

# Fonction pour gérer l'événement de clic sur l'image
def onclick(event):
    ix, iy = event.xdata, event.ydata
    print(f'Coordonnées du point cliqué : x = {ix}, y = {iy}')

# Ouvrir l'image
img = Image.open(image_path)

# Afficher l'image
fig, ax = plt.subplots()
ax.imshow(img)

# Connecter l'événement de clic
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
