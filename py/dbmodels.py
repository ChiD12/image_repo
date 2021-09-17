from py import db

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False, unique=True)
    likes = db.Column(db.Integer)

    def __repr__(self):
        return f"Image('{self.id}', '{self.name}', {self.likes})"

class UserLikes(db.Model):
    userId = db.Column(db.Integer)
    imgId = db.Column(db.Integer, db.ForeignKey('image.id'))
    likeId = db.Column(db.Integer, primary_key=True)

    def __repr__(self):
        return f"UserLikes('{self.userId}' likes '{self.imgId}, id '{self.likeId}')"