from py import app
from py.repo import rGetAllImages, rGetMaxId, rGetImageWithId ,rPostImage
from py.model import compute_similar_images, nomalizeImgShape
import secrets
import os
import urllib.request
import io
import numpy as np
from PIL import Image
import torchvision.transforms as transforms



# def sPostImage(img):

def sGetAllImages():
    recipes = rGetAllImages()
    dicts = []

    dicts = parseResults(recipes, dicts)

    return dicts

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
        imageList.append({"id": imageClass.id, "name": imageClass.name})
    
    return imageList


def sPostImage(img, isURL, imageType = None):
    lastID = rGetMaxId()
    if lastID == None:
        lastID = 0

    #create name for file with its id prefixed
    prefix = "{}{}".format(lastID+1, chr(secrets.randbelow(26) + 97))
    random = secrets.token_hex(8)
    

    if isURL:
        if isinstance(img, (bytes, bytearray)):
            img = img.decode()
        URL = img

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




def parseResults(imgs, dicts):
    for img in imgs:
        dictRecipe = {"name": img.name, "id": img.id}
        dicts.append(dictRecipe)

    return dicts