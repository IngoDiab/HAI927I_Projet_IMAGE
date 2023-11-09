# -*- coding: utf-8 -*-
"""test_gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vF2LqdBFSf5dSq88hgrfOGES8ds0xyFr
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# librairies générales
import pickle # pour charger le modèle
import pandas as pd
import string
import unicodedata
import wordcloud
from random import randint
import re
from tabulate import tabulate
import time
import numpy as np
import base64
import sys
# librairie affichage
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
# librairies scikit learn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

#Ploting
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

"""Début CycleGan ici"""

#!pip install git+https://github.com/tensorflow/examples.git

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE

"""Pipeline d'entrée"""

path_hommes = 'dataset/hommes'
path_femmes = 'dataset/femmes'
path_test_hommes = 'dataset/testhommes'
path_test_femmes = 'dataset/testfemmes'

train_hommes = tf.data.Dataset.list_files(path_hommes + '/*.jpg').map(tf.io.read_file).map(tf.image.decode_jpeg)
train_femmes = tf.data.Dataset.list_files(path_femmes + '/*.jpg').map(tf.io.read_file).map(tf.image.decode_jpeg)
test_hommes = tf.data.Dataset.list_files(path_test_hommes + '/*.jpg').map(tf.io.read_file).map(tf.image.decode_jpeg)
test_femmes = tf.data.Dataset.list_files(path_test_femmes + '/*.jpg').map(tf.io.read_file).map(tf.image.decode_jpeg)

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = normalize(image)
  return image

train_hommes = train_hommes.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_femmes = train_femmes.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_hommes = test_hommes.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_femmes = test_femmes.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_homme = next(iter(train_hommes))
sample_femme = next(iter(train_femmes))
sample_homme = tf.image.resize(sample_homme, [256, 256])
sample_femme = tf.image.resize(sample_femme, [256, 256])

plt.subplot(121)
plt.title('Homme')
plt.imshow(sample_homme[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Homme with random jitter')
plt.imshow(random_jitter(sample_homme[0]) * 0.5 + 0.5)

plt.subplot(121)
plt.title('Femme')
plt.imshow(sample_femme[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Femme with random jitter')
plt.imshow(random_jitter(sample_femme[0]) * 0.5 + 0.5)

"""Importez et réutilisez les modèles Pix2Pix

Importez le générateur et le discriminateur utilisés dans Pix2Pix via le package tensorflow_examples installé.

L'architecture du modèle utilisée dans ce tutoriel est très similaire à celle utilisée dans pix2pix . Certaines des différences sont :

    

* Cyclegan utilise la normalisation  d'instance au lieu de la normalisation par lots .
* L' article CycleGAN utilise un générateur basé sur resnet modifié. Ce tutoriel utilise un générateur unet modifié pour plus de simplicité.


Il y a 2 générateurs (G et F) et 2 discriminateurs (X et Y) en cours de formation ici.

* Le générateur G apprend à transformer l'image X en image Y .
* Le générateur F apprend à transformer l'image Y en image X .

Le discriminateur D_X apprend à faire la différence entre l'image X et l'image générée X ( F(Y) ).
Le discriminateur D_Y apprend à différencier l'image Y de l'image générée Y ( G(X) ).
"""

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
print("Shape before concatenation:", sample_homme.shape, sample_femme.shape)

# Redimensionner les images à la taille attendue par le générateur
#sample_homme_resized = tf.image.resize(sample_homme, [256, 256]) #taille attendue -> 256
#sample_femme_resized = tf.image.resize(sample_femme, [256, 256])

#to_femme = generator_g(sample_homme_resized)
#to_homme = generator_f(sample_femme_resized)

to_femme = generator_g(sample_homme)
to_homme = generator_f(sample_femme)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_homme, to_femme, sample_femme, to_homme]
title = ['Homme', 'To Femme', 'Femme', 'To Homme']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real femme?')
plt.imshow(discriminator_y(sample_femme)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real homme?')
plt.imshow(discriminator_x(sample_homme)[0, ..., -1], cmap='RdBu_r')

plt.show()

"""Fonctions de perte"""

LAMBDA = 10
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
  print ('Latest checkpoint restored!!')

"""Entrainement"""

EPOCHS = 50
def generate_images(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_hommes, train_femmes)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_homme) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_homme)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

for inp in test_hommes.take(5):
    inp = tf.image.resize(inp, [256, 256])
    generate_images(generator_g, inp)

for inp in test_femmes.take(5):
    inp = tf.image.resize(inp, [256, 256])
    generate_images(generator_f, inp)