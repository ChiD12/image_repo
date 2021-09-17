from py import app
from py.repo import rGetAllImages, rGetMaxId, rGetImageWithId ,rPostImage, rGetLikesByUser, rAddLike, rGetAllLikes, rGetAllLikesWithId
from py.model import compute_similar_images, nomalizeImgShape
import secrets
import os
import urllib.request
import io
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import potrace

def sGetAllImages():
    recipes = rGetAllImages()
    dicts = []
    dicts = parseImages(recipes, dicts)
    return dicts

def sGetAllLikes():
    likes = rGetAllLikes()
    dicts = []
    dicts = parseLikes(likes, dicts)
    return dicts

def sGetAllLikesWithId(userId):
    likes = rGetAllLikesWithId(userId)
    likedIds = []
    for like in likes:
        likedIds.append(like.imgId)
    likedIds.sort()
    return likedIds

    
def sGetImageWithId(id):
    imageClass = rGetImageWithId(id)
    return imageClass.name


def sGetSimilarImages(imgName):
    IMAGESTOSEND = 9
    folderName = './256/'
    img = Image.open(folderName + imgName).convert("RGB")
    similarImageIds = compute_similar_images(img, IMAGESTOSEND+1)

    
    firstName = rGetImageWithId(similarImageIds[0]).name
    FirstImage = Image.open(folderName + firstName).convert("RGB")

    npImg = np.asarray(img)
    npFirstImage = np.asarray(FirstImage)

    #check if the most similar image is identical to given image, if so remove it
    if np.array_equal(npImg, npFirstImage):
        similarImageIds.pop(0)
    else:
        similarImageIds.pop(IMAGESTOSEND)
    
    imageList = []
    for id in similarImageIds:
        imageClass = rGetImageWithId(id)
        imageList.append({"id": imageClass.id, "name": imageClass.name, "likes": imageClass.likes})
    
    return imageList


def sPostImage(img, isURL, imageType = None):
    lastID = rGetMaxId()
    if lastID == None:
        lastID = 0

    print("in postimage")
    #create name for file with its id prefixed
    prefix = "{}{}".format(lastID+1, chr(secrets.randbelow(26) + 97))
    random = secrets.token_hex(8)

    if isURL:
        if isinstance(img, (bytes, bytearray)):
            decoded = img.decode()
        URL = decoded

        _, ext = os.path.splitext(img)

        if '%' in ext or '?' in ext:
            ext = ext.split('?')[0]
            ext = ext.split('%')[0]

        fileName = prefix + random + ext

        absolutePath = os.path.join(app.root_path, '../images', fileName)
        urllib.request.urlretrieve(URL, absolutePath)
        image = Image.open(absolutePath).convert("RGB")
        
        
    else:
        #TODO add functionality to download if image is passed
        # _, ext = os.path.splitext(img)

        print("in else")
        fileName = prefix + random + '.' + imageType
        absolutePath = os.path.join(app.root_path, '../images', fileName)


        image = Image.open(io.BytesIO(img)).convert("RGB")
        image.save(absolutePath)

    tensorImg = transforms.ToTensor()(image)
    tensorImg = tensorImg.unsqueeze(0)

    smallImg = nomalizeImgShape(tensorImg[0])
    
    smallAbsolutePath = os.path.join(app.root_path, '../256', fileName)

    smallImg = transforms.ToPILImage(mode='RGB')(smallImg)
    smallImg.save(smallAbsolutePath)

    newEntry = rPostImage(fileName)
    newList = {"id": newEntry.id, "name": newEntry.name}


    return newList

def sLikeImage(userId, imgId):
    likesList = rGetLikesByUser(userId)
    if not didUserAlreadyLike(likesList, imgId):
        return rAddLike(userId, imgId)

    return -1

def didUserAlreadyLike(likesList, imgId):
    for like in likesList:
        if like.imgId == int(imgId):
            return True
    return False


def parseImages(imgs, dicts):
    for img in imgs:
        dictRecipe = {"name": img.name, "id": img.id, "likes": img.likes}
        dicts.append(dictRecipe)
    return dicts

def parseLikes(likes, dicts):
    for like in likes:
        dictRecipe = {"userId": like.userId, "imgId": like.imgId, "likeId": like.likeId}
        dicts.append(dictRecipe)
    return dicts


def sCreateGraph(imgName):
    print("in here")
    folderName = './256/'
    image = cv2.imread(folderName + imgName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    bil = cv2.bilateralFilter(image, 5, 50, 50)
    canny = cv2.Canny(bil, 67, 134, L2gradient =True)
    canny = canny[::-1]

    for i in range(len(canny)):
        canny[i][canny[i] > 1] = 1
    bmp = potrace.Bitmap(canny)
    print("here")
    path = bmp.trace(2, potrace.TURNPOLICY_MINORITY, 1.0, 1, .5)
    # path = bmp
    latex = []

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start
            if segment.is_corner:
                x1, y1 = segment.c
                x2, y2 = segment.end_point
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            else:
                x1, y1 = segment.c1
                x2, y2 = segment.c2
                x3, y3 = segment.end_point
                latex.append('((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),\
                (1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))' % \
                (x0, x1, x1, x2, x1, x2, x2, x3, y0, y1, y1, y2, y1, y2, y2, y3))
            start = segment.end_point
    return latex


    # cv2.imshow("bil",canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return
    