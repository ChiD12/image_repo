from py import app
import jsonpickle
from flask import send_file, url_for, request, Response, redirect, send_from_directory, send_file
from py.repo import rGetMaxId, rPostImage, rGetAllImages
from py.service import sGetAllImages, sGetImageWithId, sGetSimilarImages, sPostImage
from py.model import trainModels

@app.route('/', methods=['GET'])
def index():
    # r = rPostImage(1,2)
    g = rGetMaxId()
    print(g)
    return {"res":g}


# @app.route('/image/<url>', methods=['GET'])
# def image(url):
#     print(url)
#     return send_file(url_for('images', filename=url), mimetype='image/gif')

@app.route('/get_image/<id>')
def get_image(id):
    #TODO redirect to image with that id
    # return send_file("images/0.jpg", mimetype='image/gif')
    imageName = sGetImageWithId(id)
    return send_file("../images/{}".format(imageName))




@app.route('/all')
def all():
    # if request.args.get('type') == '1':
    #    filename = 'ok.gif'
    # else:
    #    filename = 'error.gif'
    items = {"items": sGetAllImages()}
    return items, 200


@app.route('/getSimilarImages/<imgName>')
def getSimilarImages(imgName):
    items = {"items": sGetSimilarImages(imgName)}
    return items, 200


@app.route('/createImage', methods = ['POST'])
def postImage():
    img = request.data
    contentType = request.content_type.split('/')

    if contentType[0] == "image":
        print("image")
        isUrl = False
    else:
        print("url")
        isUrl = True

    try:
        result = {"items": [sPostImage(img, isUrl, contentType[1])]}
    except Exception:
        print(Exception)
        return {}, 400

    return result, 200


@app.route('/train/qwer')
def train():
    trainModels()
    return "trained", 200
