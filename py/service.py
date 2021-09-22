from py import app
#from py.repo import rGetAllImages, rGetMaxId, rGetImageWithId ,rPostImage, rGetLikesByUser, rAddLike, rGetAllLikes, rGetAllLikesWithId
# from py.repo import repo
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

class Service:

    def __init__(self, repo):
        self.repo= repo

    def sGetAllImages(self):
        recipes = self.repo.rGetAllImages()
        dicts = []
        dicts = self.parseImages(recipes, dicts)
        return dicts

    def sGetAllLikes(self):
        likes = self.repo.rGetAllLikes()
        dicts = []
        dicts = self.parseLikes(likes, dicts)
        return dicts

    def sGetAllLikesWithId(self, userId):
        likes = self.repo.rGetAllLikesWithId(userId)
        likedIds = []
        for like in likes:
            likedIds.append(like.imgId)
        likedIds.sort()
        return likedIds

        
    def sGetImageWithId(self, id):
        imageClass = self.repo.rGetImageWithId(id)
        return imageClass.name


    def sGetSimilarImages(self, imgName):
        IMAGESTOSEND = 9
        folderName = './256/'
        img = Image.open(folderName + imgName).convert("RGB")
        similarImageIds = compute_similar_images(img, IMAGESTOSEND+1)

        
        firstName = self.repo.rGetImageWithId(similarImageIds[0]).name
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
            imageClass = self.repo.rGetImageWithId(id)
            imageList.append({"id": imageClass.id, "name": imageClass.name, "likes": imageClass.likes})
        
        return imageList


    def sPostImage(self, img, isURL, imageType = None):
        lastID = self.repo.rGetMaxId()
        if lastID == None:
            lastID = 0

        #create name for file with its id prefixed
        prefix = "{}{}".format(lastID+1, chr(secrets.randbelow(26) + 97))
        random = secrets.token_hex(8)

        if isURL:
            if isinstance(img, (bytes, bytearray)):
                decoded = img.decode()
            URL = decoded

            _, ext = os.path.splitext(URL)

            if '%' in ext or '?' in ext:
                ext = ext.split('?')[0]
                ext = ext.split('%')[0]

            fileName = prefix + random + ext

            absolutePath = os.path.join(app.root_path, '../images', fileName)
            # urllib.request.urlretrieve(URL, absolutePath)
            self.download(False, absolutePath, URL)
            image = Image.open(absolutePath).convert("RGB")
            
        else:
            fileName = prefix + random + '.' + imageType
            absolutePath = os.path.join(app.root_path, '../images', fileName)

            image = Image.open(io.BytesIO(img)).convert("RGB")
            self.download(True, absolutePath, img=image)

        tensorImg = transforms.ToTensor()(image)
        tensorImg = tensorImg.unsqueeze(0)

        smallImg = nomalizeImgShape(tensorImg[0])
        
        smallAbsolutePath = os.path.join(app.root_path, '../256', fileName)

        smallImg = transforms.ToPILImage(mode='RGB')(smallImg)
        self.download(True, smallAbsolutePath, img=smallImg)
        # smallImg.save(smallAbsolutePath)

        newEntry = self.repo.rPostImage(fileName)
        newList = {"id": newEntry.id, "name": newEntry.name}

        return newList

    def download(self, isImage, path, URL='', img = None):
        if isImage:
            img.save(path)
        else:
            urllib.request.urlretrieve(URL, path)

    def sLikeImage(self, userId, imgId):
        likesList = self.repo.rGetLikesByUser(userId)
        if not self.didUserAlreadyLike(likesList, imgId):
            return self.repo.rAddLike(userId, imgId)

        return -1

    def didUserAlreadyLike(self, likesList, imgId):
        for like in likesList:
            if like.imgId == int(imgId):
                return True
        return False


    def parseImages(self, imgs, dicts):
        for img in imgs:
            dictRecipe = {"name": img.name, "id": img.id, "likes": img.likes}
            dicts.append(dictRecipe)
        return dicts

    def parseLikes(self, likes, dicts):
        for like in likes:
            dictRecipe = {"userId": like.userId, "imgId": like.imgId, "likeId": like.likeId}
            dicts.append(dictRecipe)
        return dicts


    def sCreateGraph(self, imgName):
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
