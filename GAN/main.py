from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import cv2

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(generator_g=pix2pix.unet_generator(3, norm_type='instancenorm'),
                           generator_f=pix2pix.unet_generator(3, norm_type='instancenorm'))
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

def create_face_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    mask = np.ones_like(image) * 255
    for (x, y, w, h) in faces:
        mask[y:y + h, x:x + w] = 0
    return mask

def combine_images_with_mask(generated, original, mask):
    combined = np.where(mask == 0, generated, original)
    return combined

def load_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = (image / 127.5) - 1
    return image.astype(np.float32)

def generate_images(model, input_image):
    # Generate the prediction
    prediction = model(tf.expand_dims(input_image, 0), training=False)[0].numpy()
    prediction = prediction * 0.5 + 0.5  # Rescale the image
    prediction = np.clip(prediction, 0, 1)
    prediction_image = (prediction * 255).astype(np.uint8)

    original_image = (input_image * 0.5 + 0.5) * 255
    mask = create_face_mask(original_image.astype(np.uint8))
    combined_image = combine_images_with_mask(prediction_image, original_image.astype(np.uint8), mask)

    return combined_image

def update_canvas(canvas, image):
    image = Image.fromarray(image)
    photo_image = ImageTk.PhotoImage(image)
    canvas.photo_image = photo_image  # Keep reference to avoid garbage collection
    canvas.create_image(0, 0, image=photo_image, anchor=NW)

def run_model():
    if not image_path.get():
        return
    input_image = load_image(image_path.get())
    generated_image_g = generate_images(ckpt.generator_g, input_image)
    generated_image_f = generate_images(ckpt.generator_f, input_image)
    update_canvas(colorTransfertCanvas, generated_image_g)
    update_canvas(styleTransfertCanvas, generated_image_f)


def select_image():
    file_path = askopenfilename()
    if file_path:
        image_path.set(file_path)
        original_image = cv2.imread(file_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (256, 256))
        update_canvas(previewCanvas, original_image)


def save_generated_image(canvas):
    file_path = asksaveasfilename(defaultextension=".png")
    if file_path and canvas.photo_image:
        canvas.photo_image._PhotoImage__photo.write(file_path)

gui = Tk()
gui.geometry("900x312")
image_path = StringVar()

menubar = Menu(gui)
file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label="Importer l'image", command=select_image)
menubar.add_cascade(label="Fichier", menu=file_menu)

run_menu = Menu(menubar, tearoff=0)
run_menu.add_command(label="DeepLearning", command=run_model)
menubar.add_cascade(label="Run", menu=run_menu)

# Canvas setup
label_preview = Label(gui, text="Preview :")
label_preview.pack()
label_preview.place(x=20, y=0)

previewCanvas = Canvas(gui, width=256, height=256, bg="white")
previewCanvas.place(x=20, y=20)

label_TColor_txt = Label(gui, text="Image femme")
label_TColor_txt.pack()
label_TColor_txt.place(x=300, y=0)

colorTransfertCanvas = Canvas(gui, width=256, height=256, bg="white")
colorTransfertCanvas.place(x=300, y=20)

label_TStyle_txt = Label(gui, text="Image homme")
label_TStyle_txt.pack()
label_TStyle_txt.place(x=600, y=0)

styleTransfertCanvas = Canvas(gui, width=256, height=256, bg="white")
styleTransfertCanvas.place(x=600, y=20)

gui.config(menu=menubar)
gui.mainloop()
