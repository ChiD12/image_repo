
from py.dbmodels import Image
from py import db
from sqlalchemy import func 

def rGetAllImages():
    imgs = Image.query.all()
    return imgs

def rGetMaxId():
    lastId = db.session.query(func.max(Image.id)).first()
    return lastId[0]

def rGetImageWithId(id):
    img = Image.query.filter_by(id=id).first()
    return img

def rPostImage(name):
    image = Image(name=name)
    db.session.add(image)
    db.session.commit()
    return image
