from py import app
from flask import send_file, request, send_file
from py.service import sGetAllImages, sGetImageWithId, sGetSimilarImages, sPostImage, sCreateGraph, sLikeImage, sGetAllLikes, sGetAllLikesWithId
from py.model import trainModels
import json

@app.route('/', methods=['GET'])
def index():
    return {"res":"API"}

@app.route('/get_image/<id>')
def get_image(id):
    imageName = sGetImageWithId(id)
    return send_file("../images/{}".format(imageName))


@app.route('/all')
def all():
    items = {"items": sGetAllImages()}
    print("sent all images")
    return items, 200

@app.route('/likes/all')
def getAllLikes():
    items = {"items": sGetAllLikes()}
    return items, 200

@app.route('/likes/<userId>')
def getAllLikesWithId(userId):
    items = {"likedIds": sGetAllLikesWithId(userId)}
    return items, 200


@app.route('/getSimilarImages/<imgName>')
def getSimilarImages(imgName):
    items = {"items": sGetSimilarImages(imgName)}
    print(items)
    return items, 200


@app.route('/createImage', methods = ['POST'])
def postImage():
    img = request.data
    contentType = request.content_type.split('/')

    if contentType[0] == "image":
        isUrl = False
    else:
        isUrl = True

    try:
        result = {"items": [sPostImage(img, isUrl, contentType[1])]}
    except Exception:
        print(Exception)
        return {}, 400

    print(result)
    return result, 200


@app.route('/train/qwer')
def train():
    trainModels()
    return "trained", 200

@app.route('/graph/<imgName>')
def getGraph(imgName):
    latex = sCreateGraph(imgName)
    return {"data": latex}, 200

#curl -i "localhost:5000/api/foo?userId=546&imgId=4"  
#localhost:5000/like?userId=a1eea67-cdca-2341-3057-8dfbbdd1dda5&imgId=4
@app.route('/like', methods = ['POST'])
def putLike():
    # args = request.args.to_dict()
    decoded = request.data.decode()
    dataJson = json.loads(decoded)
    likes = sLikeImage(dataJson["userId"], dataJson["imgId"])
    return {"likes": likes}, 200

