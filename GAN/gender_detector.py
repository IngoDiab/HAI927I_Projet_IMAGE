import cv2
import numpy as np
import os

# Model and configuration paths
GENDER_MODEL = 'deploy_gender.prototxt'
GENDER_PROTO = 'gender_net.caffemodel'
FACE_PROTO = "deploy.prototxt.txt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

# Load models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Initialize frame size
frame_width = 1280
frame_height = 720

def get_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            faces.append((max(start_x, 0), max(start_y, 0), max(end_x, 0), max(end_y, 0)))
    return faces

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation = inter)

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1

def process_images_in_directory(input_dir):
    output_dir = "detection"
    os.makedirs(output_dir, exist_ok=True)

    male_count, female_count = 0, 0

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)
            frame = img.copy()
            if frame.shape[1] > frame_width:
                frame = image_resize(frame, width=frame_width)
            faces = get_faces(frame)
            for (start_x, start_y, end_x, end_y) in faces:
                face_img = frame[start_y: end_y, start_x: end_x]
                blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender_index = gender_preds[0].argmax()  # Modification ici
                gender = GENDER_LIST[gender_index]
                gender_confidence_score = gender_preds[0][gender_index]  # Correction de la variable
                label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
                box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
                cv2.putText(frame, label, (start_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)
                if gender == "Male":
                    male_count += 1
                else:
                    female_count += 1
            output_file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_file_path, frame)

    print(f"Total hommes detectes: {male_count}")
    print(f"Total femmes detectees: {female_count}")

if __name__ == '__main__':
    directory_path = input("Path vers les images: ")
    process_images_in_directory(directory_path)

