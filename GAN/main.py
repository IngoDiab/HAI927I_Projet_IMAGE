from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import cv2

### TRADI ###
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from enum import Enum
import dlib
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from matplotlib.path import Path

########################## ENUM ##########################

class Features(Enum):
    LEFT_EYE = 1
    RIGHT_EYE = 2
    EDGE_NOSE = 3
    LOWER_NOSE = 4
    JAW = 5
    MOUTH = 6
    LEFT_EYEBROW = 7
    RIGHT_EYEBROW = 8
    NOSE = 9
    BEARD = 10

########################## FUNCTIONS ##########################

def ConvertFeatureToIndices(feature):
    if(feature == Features.LEFT_EYE):
        return list(range(36, 42))

    elif(feature == Features.RIGHT_EYE):
        return list(range(42, 48))

    elif(feature == Features.EDGE_NOSE):
        return list(range(27, 31))

    elif(feature == Features.LOWER_NOSE):
        return list(range(31, 36))

    elif(feature == Features.NOSE):
        return list(range(27, 36))

    elif(feature == Features.JAW):
        return list(range(0, 17))

    elif(feature == Features.MOUTH):
        return list(range(48, 68))

    elif(feature == Features.LEFT_EYEBROW):
        return list(range(17, 22))

    elif(feature == Features.RIGHT_EYEBROW):
        return list(range(22, 27))

    elif(feature == Features.BEARD):
        return [49] + list(list(range(3, 13)) + list(range(54, 59)))

    else:
        return list(range(68))

def length(vector):
  return math.sqrt(vector[0]**2 + vector[1]**2)

def normalize_vector(vector):
  magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
  normalized_vector = (vector[0] / magnitude, vector[1] / magnitude)
  return normalized_vector


def LoadImage(pathImage):
    img = cv2.imread(pathImage, cv2.IMREAD_ANYCOLOR)
    return img

def imPlot(img):
    _convertedImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(_convertedImg)
    plt.show()
    return img

def match_colors(source_image, reference_image):
  _source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2Lab)
  _reference_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2Lab)

  _mean_source, _std_source = cv2.meanStdDev(_source_lab)
  _mean_reference, _std_reference = cv2.meanStdDev(_reference_lab)

  _result_lab = _source_lab.copy()
  _result_lab[:, :, 0] = np.clip(_result_lab[:, :, 0] * (_std_reference[0] / _std_source[0]), 0, 255)
  _result_lab[:, :, 1] = _result_lab[:, :, 1] + (_mean_reference[1] - _mean_source[1])
  _result_lab[:, :, 2] = _result_lab[:, :, 2] + (_mean_reference[2] - _mean_source[2])

  result_image = cv2.cvtColor(_result_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

  return result_image

def ImageToFeatureImage(position, minBBY, minBBX):
  _featurePosX = position[0]-minBBX
  _featurePosY = position[1]-minBBY
  return (_featurePosX, _featurePosY)

#Get features with dlib
def DetectionFace(image):
    _grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor("Dlib/shape_predictor_68_face_landmarks.dat")
    _allFacesRecognized = _detector(_grayImg)

    if _allFacesRecognized:
        _portrait = _allFacesRecognized[0]
        _handles = _predictor(_grayImg, _portrait)
        return _handles

def MarkHandlesRecognition(image):
    _handles = DetectionFace(image)

    if _handles:
        _imageWithHandles = image.copy()

        for i in range(68):
            _xPos, _yPos = _handles.part(i).x, _handles.part(i).y
            cv2.circle(_imageWithHandles, (_xPos, _yPos), 2, (0, 0, 255), -1)

        return _imageWithHandles

def Extract(image, feature, convexShape, thresholdX = 0, thresholdY = 0):
    _handles = DetectionFace(image)
    if(convexShape):
      return ExtractConvexFromHandles(image, feature, _handles, thresholdX, thresholdY)
    else:
      return ExtractFromHandles(image, feature, _handles, thresholdX, thresholdY)

def ExtractFromHandles(image, feature, handles, thresholdX = 0, thresholdY = 0):
    if(handles):
        _indices = ConvertFeatureToIndices(feature)
        _featureHandles = np.array([(handles.part(i).x, handles.part(i).y) for i in _indices], dtype=np.float32)
        _leftCornerX, _leftCornerY, _width, _height = cv2.boundingRect(_featureHandles)
        _leftCornerX -= math.floor(thresholdX/2)
        _leftCornerY -= math.floor(thresholdY/2)
        _width += thresholdX
        _height += thresholdY
        _imageFeature = image[_leftCornerY : _leftCornerY + _height, _leftCornerX : _leftCornerX + _width]
        return _imageFeature, _leftCornerX, _leftCornerY, _width, _height

def ExtractConvexFromHandles(image, feature, handles, thresholdX = 0, thresholdY = 0, debug = False):
    if(handles):

      #Get handles
      _indices = ConvertFeatureToIndices(feature)
      _imageFeature, _leftCornerX, _leftCornerY, _width, _height = ExtractFromHandles(image, feature, handles, thresholdX, thresholdY)
      _featureHandles = np.array([(handles.part(i).x, handles.part(i).y) for i in _indices], dtype=np.float32)

      #Get convex
      _hull = ConvexHull(_featureHandles)
      _pathConvex = Path(_featureHandles[_hull.vertices])

      #Get centroid
      _centroidX = 0
      _centroidY = 0
      for i in range(0,len(_pathConvex)):
        _centroidX += _pathConvex.vertices[i][0]
        _centroidY += _pathConvex.vertices[i][1]
      _centroidX /= len(_pathConvex)
      _centroidY /= len(_pathConvex)

      #Resize convex
      _color = (0, 0, 255)
      _thickness = 1
      if(debug):
        cv2.circle(image, (math.floor(_centroidX),math.floor(_centroidY)), 1, (0, 255, 0), _thickness)
      for i in range(0,len(_pathConvex)):
        directionCentroidPoint = _pathConvex.vertices[i] - (_centroidX, _centroidY)
        directionCentroidPointNormalized = normalize_vector(directionCentroidPoint)
        directionCentroidPointResized = (directionCentroidPointNormalized[0] * (thresholdX/2), directionCentroidPointNormalized[1] *  (thresholdY/2))
        _pathConvex.vertices[i] = _pathConvex.vertices[i] + directionCentroidPointResized

      #Draw convex
      if(debug):
        for i in range(0,len(_pathConvex)-1):
          cv2.line(image, (int(_pathConvex.vertices[i][0]), int(_pathConvex.vertices[i][1])), (int(_pathConvex.vertices[i+1][0]), int(_pathConvex.vertices[i+1][1])), _color, _thickness)
        cv2.line(image, (int(_pathConvex.vertices[len(_pathConvex)-1][0]), int(_pathConvex.vertices[len(_pathConvex)-1][1])), (int(_pathConvex.vertices[0][0]), int(_pathConvex.vertices[0][1])), _color, _thickness)

      #Image referentiel -> Feature referentiel
      for i in range(0,len(_pathConvex)):
        _pathConvex.vertices[i][0] -= _leftCornerX
        _pathConvex.vertices[i][1] -= _leftCornerY

      #CreateMask
      _mask = np.zeros((_height, _width, 1), np.uint8)
      for row in range(0, _height):
        for column in range(0, _width):
          if(_pathConvex.contains_point((column, row))):
            _mask[row,column] = 255
          else:
            _mask[row,column] = 0
      return _imageFeature, _leftCornerX, _leftCornerY, _width, _height, _mask

def Triangulation(image, handles = None):
  if(not handles):
    handles = DetectionFace(image)
  _featureHandles = np.array([(handles.part(i).x, handles.part(i).y) for i in range(68)], dtype=np.uint32)
  _triangles = Delaunay(_featureHandles)
  return _triangles, _featureHandles

def DrawDelaunayImage(image, triangleValues, _color = (0, 0, 255)):
  _thickness = 1
  imageDrawnOn = image.copy()
  for triangle in triangleValues:
    cv2.line(imageDrawnOn, tuple(triangle[0].astype(int)), tuple(triangle[1].astype(int)), _color, _thickness)
    cv2.line(imageDrawnOn, tuple(triangle[1].astype(int)), tuple(triangle[2].astype(int)), _color, _thickness)
    cv2.line(imageDrawnOn, tuple(triangle[0].astype(int)), tuple(triangle[2].astype(int)), _color, _thickness)
  return imageDrawnOn

def ConnexionDelaunay(imageSource, imageDesti, feature = None):
  handlesSources = DetectionFace(imageSource)
  handlesDesti = DetectionFace(imageDesti)

  trianglesSources, _ = Triangulation(imageSource, handlesSources)
  triangleValuesFeatureSources = []
  _indices = ConvertFeatureToIndices(feature)
  for i, triangle in enumerate(trianglesSources.simplices):
    trianglesIndexOK = 0
    if triangle[0] in _indices :
      trianglesIndexOK+=1
    if triangle[1] in _indices :
      trianglesIndexOK+=1
    if triangle[2] in _indices :
      trianglesIndexOK+=1
    if trianglesIndexOK >=2:
      triangleValuesFeatureSources.append((triangle[0], triangle[1], triangle[2]))

  triangleValuesFeatureSourcesArray = np.array(triangleValuesFeatureSources)

  _nbSimplices = triangleValuesFeatureSourcesArray.shape[0]

  triangleValuesSources = []
  triangleValuesDesti = []

  for i, triangle in enumerate(triangleValuesFeatureSourcesArray):
    trianglePosS = ((handlesSources.part(triangle[0]).x, handlesSources.part(triangle[0]).y), (handlesSources.part(triangle[1]).x, handlesSources.part(triangle[1]).y), (handlesSources.part(triangle[2]).x, handlesSources.part(triangle[2]).y))
    triangleValuesSources.append(trianglePosS)

    trianglePosD = ((handlesDesti.part(triangle[0]).x, handlesDesti.part(triangle[0]).y), (handlesDesti.part(triangle[1]).x, handlesDesti.part(triangle[1]).y), (handlesDesti.part(triangle[2]).x, handlesDesti.part(triangle[2]).y))
    triangleValuesDesti.append(trianglePosD)

  return np.array(triangleValuesSources, dtype=np.float32), np.array(triangleValuesDesti, dtype=np.float32)

def Swap(imageModel, imageCopyFrom, imagePasteOn, feature, useConvex = True, thresholdXM = 0, thresholdYM = 0, thresholdXC = 0, thresholdYC = 0):
    if useConvex:
      _feature, _leftCornerXP, _leftCornerYP, _widthP, _heightP, _ = Extract(imageModel, feature, True, thresholdXM, thresholdYM)
      _imageCopyFromFeature, _, _, _widthC, _heightC, _mask = Extract(imageCopyFrom, feature, True, thresholdXC, thresholdYC)
    else:
      _feature, _leftCornerXP, _leftCornerYP, _widthP, _heightP = Extract(imageModel, feature, False, thresholdXM, thresholdYM)
      _imageCopyFromFeature, _, _, _widthC, _heightC = Extract(imageCopyFrom, feature, False, thresholdXC, thresholdYC)

    _scaleOnX = _widthP/_widthC
    _scaleOnY = _heightP/_heightC
    _imageCopyFromFeatureResized = cv2.resize(_imageCopyFromFeature, (0, 0), fx=_scaleOnX, fy=_scaleOnY)

    _imageCopyFromFeatureResizedSpecified = match_colors(_imageCopyFromFeatureResized, _feature)

    if useConvex:
      _maskResized = cv2.resize(_mask, (0, 0), fx=_scaleOnX, fy=_scaleOnY)
      _newHeight, _newWidth, _ = _imageCopyFromFeatureResizedSpecified.shape

    if useConvex:
      for row in range(0, _newHeight):
        for column in range(0, _newWidth):
          if _maskResized[row,column] == 255:
            imagePasteOn[_leftCornerYP + row, _leftCornerXP + column] = _imageCopyFromFeatureResizedSpecified[row,column]
    else:
      imagePasteOn[_leftCornerYP : _leftCornerYP + _newHeight, _leftCornerXP : _leftCornerXP + _widthP] = _imageCopyFromFeatureResizedSpecified

def DetectBeard(imageSource):
  handlesSources = DetectionFace(imageSource)
  trianglesSources, _ = Triangulation(imageSource, handlesSources)
  triangleValuesSources = []

  _indices = ConvertFeatureToIndices(Features.__members__["BEARD"])

  for i, triangle in enumerate(trianglesSources.simplices):
    trianglesIndexOK = 0
    if triangle[0] in _indices :
      trianglesIndexOK+=1
    if triangle[1] in _indices :
      trianglesIndexOK+=1
    if triangle[2] in _indices :
      trianglesIndexOK+=1
    if trianglesIndexOK >=2:
      triangleValuesSources.append(((handlesSources.part(triangle[0]).x, handlesSources.part(triangle[0]).y), (handlesSources.part(triangle[1]).x, handlesSources.part(triangle[1]).y), (handlesSources.part(triangle[2]).x, handlesSources.part(triangle[2]).y)))

  triangleValuesSourcesArray = np.array(triangleValuesSources, dtype=np.float32)
  return triangleValuesSourcesArray

def SwapBeard(imageSource, imageDest, debug = False):
  _test = DetectBeard(imageSource)
  _beardTriangleSource, _beardTriangleDest = ConnexionDelaunay(imageSource, imageDest, Features.__members__["BEARD"])
  imagePasteOn = imageDest.copy()

  if(debug):
    imagePasteOn = DrawDelaunayImage(imagePasteOn, _beardTriangleDest)

  for i in range(0, _beardTriangleSource.shape[0]):
    triangleSource = _beardTriangleSource[i]
    triangleDest = _beardTriangleDest[i]

    _leftCornerXS, _leftCornerYS, _widthS, _heightS = cv2.boundingRect(triangleSource)
    _leftCornerXD, _leftCornerYD, _widthD, _heightD = cv2.boundingRect(triangleDest)

     #Convex
    if (triangleSource == 0).all() or (triangleDest == 0).all():
      continue

    _hullSource = ConvexHull(triangleSource)
    _pathConvexSource = Path(triangleSource[_hullSource.vertices])

    _hullDest = ConvexHull(triangleDest)
    _pathConvexDest = Path(triangleDest[_hullDest.vertices])

     #Image referentiel -> Feature referentiel
    for i in range(0,len(_pathConvexSource)):
      _pathConvexSource.vertices[i][0] -= _leftCornerXS
      _pathConvexSource.vertices[i][1] -= _leftCornerYS
      _pathConvexDest.vertices[i][0] -= _leftCornerXD
      _pathConvexDest.vertices[i][1] -= _leftCornerYD

    #Mask
    _maskS = np.zeros((_heightS, _widthS, 1), np.uint8)
    for row in range(0, _heightS):
      for column in range(0, _widthS):
        if(_pathConvexSource.contains_point((column, row))):
          _maskS[row,column] = 255
        else:
          _maskS[row,column] = 0

    _maskD = np.zeros((_heightD, _widthD, 1), np.uint8)
    for row in range(0, _heightD):
      for column in range(0, _widthD):
        if(_pathConvexDest.contains_point((column, row))):
          _maskD[row,column] = 255
        else:
          _maskD[row,column] = 0

    _imageTriangleSource = imageSource[_leftCornerYS : _leftCornerYS + _heightS, _leftCornerXS : _leftCornerXS + _widthS].copy()
    _imageTriangleDest = imageDest[_leftCornerYD : _leftCornerYD + _heightD, _leftCornerXD : _leftCornerXD + _widthD].copy()

    for row in range(0, _heightS):
      for column in range(0, _widthS):
        if _maskS[row,column] == 0:
          _imageTriangleSource[row, column] = (0, 0, 0)

    for row in range(0, _heightD):
      for column in range(0, _widthD):
        if _maskD[row,column] == 0:
          _imageTriangleDest[row, column] = (0, 0, 0)

    triangleSourceF32 = np.float32(triangleSource)
    triangleDestF32 = np.float32(triangleDest)
    M = cv2.getAffineTransform(triangleSourceF32, triangleDestF32)
    M[0, 2] = 0 # Normaliser tx par la largeur de l'image source
    M[1, 2]  = 0  # Normaliser ty par la hauteur de l'image source
    warped_triangle = cv2.warpAffine(_imageTriangleSource, M, (_widthD, _heightD))
    _maskSW = cv2.warpAffine(_maskS, M, (_widthD, _heightD))

    warped_triangle = match_colors(warped_triangle, _imageTriangleDest)

    for row in range(0, _heightD):
      for column in range(0, _widthD):
        if not _maskD[row,column] == 0 and not _maskSW[row,column] == 0 and not warped_triangle[row,column].all() == 0:
          imagePasteOn[_leftCornerYD + row, _leftCornerXD + column] = warped_triangle[row,column]

  return imagePasteOn
### TRADI ###

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


def select_imageDL():
    file_path = askopenfilename()
    if file_path:
        image_path.set(file_path)
        original_image = cv2.imread(file_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (256, 256))
        update_canvas(previewCanvas, original_image)

def select_imageT():
    file_path = askopenfilename()
    if file_path:
        image_path_Tradi.set(file_path)
        original_image = cv2.imread(file_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (256, 256))
        update_canvas(previewCanvasT, original_image)

def save_generated_image(canvas):
    file_path = asksaveasfilename(defaultextension=".png")
    if file_path and canvas.photo_image:
        canvas.photo_image._PhotoImage__photo.write(file_path)

def run_tradiVH():
    if not image_path_Tradi.get():
        return
    input_image = cv2.imread(image_path_Tradi.get(), cv2.IMREAD_ANYCOLOR)
    imgMan = LoadImage("Images_NoDL/man.jpg")
    imgSwapped = SwapBeard(imgMan, input_image)
    Swap(input_image, imgMan, imgSwapped, Features.__members__["MOUTH"], thresholdXM = 2, thresholdYM = 2, thresholdXC = 2, thresholdYC = 2)
    Swap(input_image, imgMan, imgSwapped, Features.__members__["LEFT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
    Swap(input_image, imgMan, imgSwapped, Features.__members__["RIGHT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
    imgSwapped = cv2.cvtColor(imgSwapped, cv2.COLOR_BGR2RGB)
    imgSwapped = cv2.resize(imgSwapped, (256, 256))
    update_canvas(othergenderTCanvas, imgSwapped)

def run_tradiVF():
    if not image_path_Tradi.get():
        return
    input_image = cv2.imread(image_path_Tradi.get(), cv2.IMREAD_ANYCOLOR)
    imgWoman = LoadImage("Images_NoDL/woman.jpg")
    imgSwapped = SwapBeard(imgWoman, input_image)
    Swap(input_image, imgWoman, imgSwapped, Features.__members__["MOUTH"], thresholdXM = 2, thresholdYM = 2, thresholdXC = 2, thresholdYC = 2)
    Swap(input_image, imgWoman, imgSwapped, Features.__members__["LEFT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
    Swap(input_image, imgWoman, imgSwapped, Features.__members__["RIGHT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
    imgSwapped = cv2.cvtColor(imgSwapped, cv2.COLOR_BGR2RGB)
    imgSwapped = cv2.resize(imgSwapped, (256, 256))
    update_canvas(othergenderTCanvas, imgSwapped)

gui = Tk()
gui.geometry("900x600")
image_path = StringVar()
image_path_Tradi = StringVar() 

menubar = Menu(gui)
file_menu = Menu(menubar, tearoff=0)
choice_menu = Menu(menubar, tearoff=0)
choice_menu.add_command(label="pour Deep Learning", command=select_imageDL)
choice_menu.add_command(label="pour Traditionnelle", command=select_imageT)
file_menu.add_cascade(label="Importer l'image", menu=choice_menu)
menubar.add_cascade(label="Fichier", menu=file_menu)

run_menu = Menu(menubar, tearoff=0)
run_menu.add_command(label="DeepLearning", command=run_model)

run_tradi_menu = Menu(menubar, tearoff=0)
run_tradi_menu.add_command(label="Vers homme", command=run_tradiVH)
run_tradi_menu.add_command(label="Vers femme", command=run_tradiVF)
run_menu.add_cascade(label="Traditionnelle", menu=run_tradi_menu)

menubar.add_cascade(label="Run", menu=run_menu)

# Canvas setup
#DeepL
label_preview = Label(gui, text="Preview :")
label_preview.pack()
label_preview.place(x=20, y=0)

previewCanvas = Canvas(gui, width=256, height=256, bg="white")
previewCanvas.place(x=20, y=20)

label_TColor_txt = Label(gui, text="Image femme :")
label_TColor_txt.pack()
label_TColor_txt.place(x=300, y=0)

colorTransfertCanvas = Canvas(gui, width=256, height=256, bg="white")
colorTransfertCanvas.place(x=300, y=20)

label_TStyle_txt = Label(gui, text="Image homme :")
label_TStyle_txt.pack()
label_TStyle_txt.place(x=600, y=0)

styleTransfertCanvas = Canvas(gui, width=256, height=256, bg="white")
styleTransfertCanvas.place(x=600, y=20)

#Tradi
label_previewT = Label(gui, text="Preview :")
label_previewT.pack()
label_previewT.place(x=20, y=300)

previewCanvasT = Canvas(gui, width=256, height=256, bg="white")
previewCanvasT.place(x=20, y=320)

label_othergenderT = Label(gui, text="Image du genre oppose :")
label_othergenderT.pack()
label_othergenderT.place(x=300, y=300)

othergenderTCanvas = Canvas(gui, width=256, height=256, bg="white")
othergenderTCanvas.place(x=300, y=320)

gui.config(menu=menubar)
gui.mainloop()
