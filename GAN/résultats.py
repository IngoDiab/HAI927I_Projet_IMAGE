import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import numpy as np
import os

intermediate_dir = 'dataset/intermediaire'
checkpoint_path = './checkpoints/train'

ckpt = tf.train.Checkpoint(generator_g=pix2pix.unet_generator(3, norm_type='instancenorm'),
                           generator_f=pix2pix.unet_generator(3, norm_type='instancenorm'),
                           discriminator_x=pix2pix.discriminator(norm_type='instancenorm', target=False),
                           discriminator_y=pix2pix.discriminator(norm_type='instancenorm', target=False))
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

disc_x_scores = []
disc_y_scores = []

for epoch in range(1, 201):
    image_path = os.path.join(intermediate_dir, f'epoch_{epoch}.png')
    test_image = preprocess_image(image_path)

    test_image = tf.expand_dims(test_image, 0)

    disc_x_pred = ckpt.discriminator_x(test_image, training=False)
    disc_y_pred = ckpt.discriminator_y(test_image, training=False)

    disc_x_score = tf.reduce_mean(disc_x_pred)
    disc_y_score = tf.reduce_mean(disc_y_pred)

    disc_x_scores.append(disc_x_score.numpy())
    disc_y_scores.append(disc_y_score.numpy())

epochs_range = range(1, 201)
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, disc_x_scores, label='Discriminateur X Score')
plt.title('Scores du Discriminateur X par Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, disc_y_scores, label='Discriminateur Y Score')
plt.title('Scores du Discriminateur Y par Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.show()
