import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib
import math
from enum import Enum

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
    NOSE = 8

########################## FUNCTIONS ##########################

def ConvertFeatureToIndices(feature):
    match feature:

        case Features.LEFT_EYE:
            return list(range(36, 42))

        case Features.RIGHT_EYE:
            return list(range(42, 48))

        case Features.EDGE_NOSE:
            return list(range(27, 31))

        case Features.LOWER_NOSE:
            return list(range(31, 36))

        case Features.NOSE:
            return list(range(27, 36))

        case Features.JAW:
            return list(range(0, 17))

        case Features.MOUTH:
            return list(range(48, 68))

        case Features.LEFT_EYEBROW:
            return list(range(17, 22))

        case Features.RIGHT_EYEBROW:
            return list(range(22, 27))

        case _:
            return list()

def LoadImage(pathImage):
    img = cv2.imread(pathImage, cv2.IMREAD_ANYCOLOR)
    return img

def imPlot(img):
    _convertedImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(_convertedImg)
    plt.show()
    return img

def imPlotMultiple(images):
    _convertedImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(_convertedImg)
    plt.show()
    return img

#Get features with dlib
def DetectionFace(image):
    _grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
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

def Extract(image, feature, thresholdX = 0, thresholdY = 0):
    _handles = DetectionFace(image)
    return ExtractFromHandles(image, feature, _handles, thresholdX, thresholdY)   

def ExtractFromHandles(image, feature, handles, thresholdX = 0, thresholdY = 0):
    if(handles):
        _indices = ConvertFeatureToIndices(feature)
        _featureHandles = np.array([(handles.part(i).x, handles.part(i).y) for i in _indices], dtype=np.float32)
        _leftCornerX, _leftCornerY, _width, _height = cv2.boundingRect(_featureHandles)
        _imageFeature = image[_leftCornerY - math.floor(thresholdY/2):_leftCornerY + _height + math.ceil(thresholdY/2), _leftCornerX - math.floor(thresholdX/2) :_leftCornerX + _width + math.ceil(thresholdX/2)]
        return _imageFeature, _leftCornerX, _leftCornerY, _width, _height

def Swap(imageModel, imageCopyFrom, imagePasteOn, feature, thresholdXM = 0, thresholdYM = 0, thresholdXC = 0, thresholdYC = 0):
    _imageModelFeature, _leftCornerXP, _leftCornerYP, _widthP, _heightP = Extract(imageModel, feature, thresholdXM, thresholdYM)
    _imageCopyFromFeature, _leftCornerXC, _leftCornerYC, _widthC, _heightC = Extract(imageCopyFrom, feature, thresholdXC, thresholdYC)

    _scaleOnX = (_widthP+thresholdYM)/(_widthC+thresholdXC)
    _scaleOnY = (_heightP+thresholdYM)/(_heightC+thresholdYC)
    _imageCopyFromFeatureResized = cv2.resize(_imageCopyFromFeature, (0, 0), fx=_scaleOnX, fy=_scaleOnY)

    imagePasteOn[_leftCornerYP - math.floor(thresholdYM/2):_leftCornerYP + _heightP + math.floor(thresholdYM/2), _leftCornerXP - math.floor(thresholdXM/2):_leftCornerXP + _widthP + math.floor(thresholdXM/2)] = _imageCopyFromFeatureResized

########################## MAIN ##########################

imgMan = LoadImage("man.jpg")
imgWoman = LoadImage("woman.jpg")

imgSwapped = imgMan.copy()

# imgWithHandles = MarkHandlesRecognition(imgWoman)
# imPlot(imgWithHandles)

# extracted, _, _, _, _ = Extract(imgWoman, Features.__members__["LEFT_EYE"], 10,10)
# imPlot(extracted)

# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["NOSE"])
Swap(imgMan, imgWoman, imgSwapped, Features.__members__["MOUTH"])
Swap(imgMan, imgWoman, imgSwapped, Features.__members__["LEFT_EYE"],10,10, 10,10)
Swap(imgMan, imgWoman, imgSwapped, Features.__members__["RIGHT_EYE"],10,10, 10,10)
# #Swap(imgMan, imgWoman, imgSwapped, Features.__members__["LOWER_NOSE"])
imPlot(imgSwapped)