import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import shutil


# Define the main application class
class PhotoSorterApp:
    def __init__(self, root, image_folder, men_folder, women_folder):
        self.root = root
        self.image_folder = image_folder
        self.men_folder = men_folder
        self.women_folder = women_folder
        self.images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_index = 0

        self.setup_ui()
        self.display_image()

    def setup_ui(self):
        # Set up the main window
        self.root.title('Celebrity Photo Sorter')
        self.root.geometry('800x600')

        # Set up the Image label
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        # Set up buttons
        men_button = tk.Button(self.root, text="Homme", command=self.sort_to_men)
        men_button.pack(side=tk.LEFT, padx=60)

        women_button = tk.Button(self.root, text="Femme", command=self.sort_to_women)
        women_button.pack(side=tk.RIGHT, padx=60)

    def display_image(self):
        # Display the current image
        if self.image_index < len(self.images):
            image_path = os.path.join(self.image_folder, self.images[self.image_index])
            img = Image.open(image_path)
            img.thumbnail((750, 500))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img  # Keep a reference.
        else:
            self.image_label.config(text="No more images.")

    def sort_to_men(self):
        # Move the image to men's folder
        current_image_path = os.path.join(self.image_folder, self.images[self.image_index])
        shutil.move(current_image_path, self.men_folder)
        self.next_image()

    def sort_to_women(self):
        # Move the image to women's folder
        current_image_path = os.path.join(self.image_folder, self.images[self.image_index])
        shutil.move(current_image_path, self.women_folder)
        self.next_image()

    def next_image(self):
        # Go to the next image
        self.image_index += 1
        self.display_image()


# Define the main execution
def main():
    root = tk.Tk()
    app = PhotoSorterApp(
        root,
        image_folder=r"C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\100k",
        men_folder=r"C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\testhommes",
        women_folder=r"C:\Users\docto\PycharmProjects\HAI927I_Projet_IMAGE\GAN\dataset\testfemmes"
    )
    root.mainloop()


# Execute the program
if __name__ == "__main__":
    main()
