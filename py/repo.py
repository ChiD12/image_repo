from py.dbmodels import Image, UserLikes
#from py import db
from sqlalchemy import func 

class Repo:
    def __init__(self, db):
        self.db = db
        # self.Image = 

    def rGetAllImages(self):
        imgs = Image.query.all()
        return imgs

    def rGetAllLikes(self):
        likes = UserLikes.query.all()
        return likes

    def rGetMaxId(self):
        lastId = self.db.session.query(func.max(Image.id)).first()
        return lastId[0]

    def rGetImageWithId(self, id):
        img = Image.query.filter_by(id=id).first()
        return img

    def rGetAllLikesWithId(self, userId):
        likes = UserLikes.query.filter_by(userId=userId).all()
        return likes

    def rPostImage(self, name):
        image = Image(name=name, likes=0)
        self.db.session.add(image)
        self.db.session.commit()
        return image

    def rGetLikesByUser(self, userId):
        userLikesList = UserLikes.query.filter_by(userId = userId)
        return userLikesList

    def rAddLike(self, userId, imgId):
        like = UserLikes(userId=userId, imgId=imgId)
        self.db.session.add(like)

        img = Image.query.filter_by(id=imgId).first()
        setattr(img, 'likes', img.likes+1)

        self.db.session.commit()

        return img.likes