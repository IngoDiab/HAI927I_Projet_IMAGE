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

        case Features.BEARD:
            return [49] + list(list(range(3, 13)) + list(range(54, 59)))

        case _:
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
  print("ConnexionDelaunay")
  print(triangleValuesFeatureSourcesArray)

  _nbSimplices = triangleValuesFeatureSourcesArray.shape[0]

  triangleValuesSources = [] #np.zeros(( _nbSimplices, 3, 2), dtype=int)
  triangleValuesDesti = [] #np.zeros(( _nbSimplices, 3, 2), dtype=int)

  for i, triangle in enumerate(triangleValuesFeatureSourcesArray):
    trianglePosS = ((handlesSources.part(triangle[0]).x, handlesSources.part(triangle[0]).y), (handlesSources.part(triangle[1]).x, handlesSources.part(triangle[1]).y), (handlesSources.part(triangle[2]).x, handlesSources.part(triangle[2]).y))
    triangleValuesSources.append(trianglePosS)
    # triangleValuesSources[i][1] = (handlesSources.part(triangle[1]).x, handlesSources.part(triangle[1]).y)
    # triangleValuesSources[i][2] = (handlesSources.part(triangle[2]).x, handlesSources.part(triangle[2]).y)

    trianglePosD = ((handlesDesti.part(triangle[0]).x, handlesDesti.part(triangle[0]).y), (handlesDesti.part(triangle[1]).x, handlesDesti.part(triangle[1]).y), (handlesDesti.part(triangle[2]).x, handlesDesti.part(triangle[2]).y))
    triangleValuesDesti.append(trianglePosD)
    # triangleValuesDesti[i][1] = (handlesDesti.part(triangle[1]).x, handlesDesti.part(triangle[1]).y)
    # triangleValuesDesti[i][2] = (handlesDesti.part(triangle[2]).x, handlesDesti.part(triangle[2]).y)

  print("ConnexionDelaunay2")
  print(np.array(triangleValuesSources, dtype=np.float32))

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

    # _imageCopyFromFeatureResized = np.uint8(_imageCopyFromFeatureResized)
    # _feature = np.uint8(_feature)
    # print(_imageCopyFromFeatureResized.shape)
    # print(cv2.cvtColor(_imageCopyFromFeatureResized, cv2.COLOR_BGR2RGB).shape)
    _imageCopyFromFeatureResizedSpecified = match_colors(_imageCopyFromFeatureResized, _feature)
    #_imageCopyFromFeatureResizedSpecified = cv2.cvtColor(_imageCopyFromFeatureResizedSpecified, cv2.COLOR_RGB2BGR)

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
      # triangleValuesSources[i][1] = (handlesSources.part(triangle[1]).x, handlesSources.part(triangle[1]).y)
      # triangleValuesSources[i][2] = (handlesSources.part(triangle[2]).x, handlesSources.part(triangle[2]).y)

  triangleValuesSourcesArray = np.array(triangleValuesSources, dtype=np.float32)
  print("DetectBeard")
  print(triangleValuesSourcesArray.shape[0])
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
    print(triangleDest)

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

    # imPlot(_imageTriangleSource)
    # imPlot(_maskS)
    # imPlot(_imageTriangleDest)
    # imPlot(_maskD)

    print(M)
    imPlot(warped_triangle)

    imPlot(imagePasteOn)
  return imagePasteOn

########################## MAIN ##########################

imgMan = LoadImage("man.jpg")
imgWoman = LoadImage("woman.jpg")

#imgSwapped = imgMan.copy()

#SWAP
# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["NOSE"])
# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["NOSE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["MOUTH"], thresholdXM = 2, thresholdYM = 2, thresholdXC = 2, thresholdYC = 2)
# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["LEFT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["RIGHT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
# Swap(imgMan, imgWoman, imgSwapped, Features.__members__["LOWER_NOSE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
# imPlot(imgSwapped)

#HANDLES
# imgWithHandles = MarkHandlesRecognition(imgMan)
# imPlot(imgWithHandles)

#EXTRACTION NO CONVEX
#extracted, _, _, _, _ = Extract(imgMan, Features.__members__["NOSE"], False, 12,0)
#imPlot(extracted)

#TRIANGULATION
# triangleValuesMan, triangleValuesWoman = ConnexionDelaunay(imgMan, imgWoman)
# imgManDelaunay = DrawDelaunayImage(imgMan, triangleValuesMan)
# imgWomanDelaunay = DrawDelaunayImage(imgWoman, triangleValuesWoman)
# imPlot(imgManDelaunay)
# imPlot(imgWomanDelaunay)

#EXTRACTION CONVEX
#_imageFeature, mask = Extract(imgMan, Features.__members__["MOUTH"], True, 10,10)
# _imageFeature, mask = Extract(imgMan, Features.__members__["LEFT_EYE"], True, 10,10)
# imPlot(imgMan)
# imPlot(_imageFeature)
# imPlot(mask)

#MATCH COLORS
# extractedM, _, _, _widthM, _heightM = Extract(imgMan, Features.__members__["MOUTH"], False, 10,10)
# extractedW, _, _, _widthW, _heightW = Extract(imgWoman, Features.__members__["MOUTH"], False, 10,10)
# imPlot(extractedM)
# imPlot(extractedW)
# _scaleOnX = _widthM/_widthW
# _scaleOnY = _heightM/_heightW
# extractedWResized = cv2.resize(extractedW, (0, 0), fx=_scaleOnX, fy=_scaleOnY)
# imgMatch = match_colors(imgMan, imgWoman)
# imPlot(imgMatch)

#DETECT BEARD
# triangleValuesMan = DetectBeard(imgWoman)
# imgManDelaunay = DrawDelaunayImage(imgWoman, triangleValuesMan)
# imPlot(imgManDelaunay)

#SWAP BEARD
imgSwapped = SwapBeard(imgMan, imgWoman)
Swap(imgWoman, imgMan, imgSwapped, Features.__members__["MOUTH"], thresholdXM = 2, thresholdYM = 2, thresholdXC = 2, thresholdYC = 2)
Swap(imgWoman, imgMan, imgSwapped, Features.__members__["LEFT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
Swap(imgWoman, imgMan, imgSwapped, Features.__members__["RIGHT_EYE"], thresholdXM = 10, thresholdYM = 10, thresholdXC = 10, thresholdYC = 10)
imPlot(imgSwapped)