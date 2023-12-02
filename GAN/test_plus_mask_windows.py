import warnings
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.utils import save_img

warnings.filterwarnings("ignore", category=FutureWarning)

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10

# Chemins des dossiers de données
path_test_hommes = 'dataset/testhommes'
path_test_femmes = 'dataset/testfemmes'

test_hommes = tf.data.Dataset.list_files(path_test_hommes + '/*.jpg')
test_femmes = tf.data.Dataset.list_files(path_test_femmes + '/*.jpg')


# Chargement et prétraitement des données
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image_test(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = normalize(image)
    return image, path


test_hommes = test_hommes.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_femmes = test_femmes.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)


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

total = 0

import os


# Fonction pour générer les images
def generate_images(model, input_tensor):
    # Récupération de l'image et du chemin
    image_tensor, path_tensor = input_tensor
    image_tensor = tf.image.resize(image_tensor, [256, 256])

    prediction = model(image_tensor, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    prediction = cv2.resize(prediction, (128, 128))

    # Récupération du chemin d'origine et création du masque
    path = path_tensor.numpy()[0].decode('utf-8')
    original_image = cv2.imread(path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (128, 128))
    mask = create_face_mask(path)

    # Combiner l'image générée avec l'originale
    combined_image = combine_images_with_mask(prediction, original_image, mask)

    # Construction du chemin de sortie
    output_dir = 'dataset/testresults'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(path))

    # Sauvegarder l'image combinée
    save_img(output_path, combined_image)


# Parcourir les images et les traiter
for image, path in test_hommes.take(5):
    generate_images(generator_g, (image, path))

for image, path in test_femmes.take(5):
    generate_images(generator_f, (image, path))

