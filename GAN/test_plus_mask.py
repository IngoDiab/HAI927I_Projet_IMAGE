import warnings
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

warnings.filterwarnings("ignore", category=FutureWarning)

AUTOTUNE = tf.data.AUTOTUNE

# Chemins des dossiers de données
path_test_hommes = 'dataset/testhommes'
path_test_femmes = 'dataset/testfemmes'


# Chargement et prétraitement des données
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [128, 128])
    image = (image / 127.5) - 1
    return image


# Création du masque
def create_face_mask(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mask = np.ones_like(image) * 255  # Fond blanc
    for (x, y, w, h) in faces:
        mask[y:y + h, x:x + w] = 0  # Visage en noir
    return mask


# Fusion des images
def combine_images_with_mask(generated, original, mask):
    combined = np.where(mask == 0, generated, original)
    return combined


# Création du générateur et du discriminateur
OUTPUT_CHANNELS = 3
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 10
def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# Fonction pour générer les images
def generate_images(model, input_tensor, filename):
    # Générer l'image
    prediction = model(input_tensor, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

    # Charger l'image originale et créer le masque
    original_image = cv2.imread(filename)
    original_image = cv2.resize(original_image, (256, 256))
    mask = create_face_mask(filename)

    # Combiner l'image générée avec l'originale
    combined_image = combine_images_with_mask(prediction, original_image, mask)

    # Afficher les images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(combined_image)
    plt.title("Combined Image")
    plt.axis("off")

    plt.show()


# Parcourir les images et les traiter
for i, file_path in enumerate(tf.io.gfile.glob(path_test_hommes + '/*.jpg')[:5]):
    test_input = load_and_preprocess_image(file_path)
    generate_images(generator_g, test_input, file_path)

for i, file_path in enumerate(tf.io.gfile.glob(path_test_femmes + '/*.jpg')[:5]):
    test_input = load_and_preprocess_image(file_path)
    generate_images(generator_f, test_input, file_path)
