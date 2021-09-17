
from py.dbmodels import Image, UserLikes
from py import db
from sqlalchemy import func 

def rGetAllImages():
    imgs = Image.query.all()
    return imgs

def rGetAllLikes():
    likes = UserLikes.query.all()
    return likes

def rGetMaxId():
    lastId = db.session.query(func.max(Image.id)).first()
    return lastId[0]

def rGetImageWithId(id):
    img = Image.query.filter_by(id=id).first()
    return img

def rGetAllLikesWithId(userId):
    likes = UserLikes.query.filter_by(userId=userId).all()
    return likes

def rPostImage(name):
    image = Image(name=name, likes=0)
    db.session.add(image)
    db.session.commit()
    return image

def rGetLikesByUser(userId):
    userLikesList = UserLikes.query.filter_by(userId = userId)
    return userLikesList

def rAddLike(userId, imgId):
    like = UserLikes(userId=userId, imgId=imgId)
    db.session.add(like)

    img = Image.query.filter_by(id=imgId).first()
    setattr(img, 'likes', img.likes+1)

    db.session.commit()

    return img.likes