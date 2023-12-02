import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import numpy as np
import os

# Chemin vers le dossier où les images sont stockées
intermediate_dir = 'dataset/intermediaire'
checkpoint_path = './checkpoints/train'

# Charger les checkpoints
ckpt = tf.train.Checkpoint(generator_g=pix2pix.unet_generator(3, norm_type='instancenorm'),
                           generator_f=pix2pix.unet_generator(3, norm_type='instancenorm'),
                           discriminator_x=pix2pix.discriminator(norm_type='instancenorm', target=False),
                           discriminator_y=pix2pix.discriminator(norm_type='instancenorm', target=False))
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restaurer le dernier checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# Fonction pour charger et préparer les images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

# Initialiser les listes pour stocker les scores
disc_x_scores = []
disc_y_scores = []

# Tester les discriminateurs sur les images intermédiaires
for epoch in range(1, 201):  # suppose qu'il y a 200 époques
    image_path = os.path.join(intermediate_dir, f'epoch_{epoch}.png')
    test_image = preprocess_image(image_path)

    # Ajouter une dimension de batch
    test_image = tf.expand_dims(test_image, 0)

    # Obtenir les prédictions des discriminateurs
    disc_x_pred = ckpt.discriminator_x(test_image, training=False)
    disc_y_pred = ckpt.discriminator_y(test_image, training=False)

    # Convertir les prédictions en scores scalaires
    disc_x_score = tf.reduce_mean(disc_x_pred)
    disc_y_score = tf.reduce_mean(disc_y_pred)

    # Ajouter les scores aux listes
    disc_x_scores.append(disc_x_score.numpy())
    disc_y_scores.append(disc_y_score.numpy())

# Plot des scores des discriminateurs
epochs_range = range(1, 201)
plt.figure(figsize=(15, 5))

# Plot pour le discriminateur X
plt.subplot(1, 2, 1)
plt.plot(epochs_range, disc_x_scores, label='Discriminateur X Score')
plt.title('Scores du Discriminateur X par Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

# Plot pour le discriminateur Y
plt.subplot(1, 2, 2)
plt.plot(epochs_range, disc_y_scores, label='Discriminateur Y Score')
plt.title('Scores du Discriminateur Y par Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.show()
