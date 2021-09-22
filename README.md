# Image Repo backend

[Frontend Found here](https://github.com/ChiD12/image_repo_frontend)
\
[Live Demo](http://image.danielr.tech/).

Image Repository whose features include storing images, delivering images, and uses a Convalutional Neural Network to create image encodings to find similarities between images as well as the image represented in a graphing calculator and likes

# Installation

Follow instuctions found here to install [pypotrace](https://pypi.org/project/pypotrace/)

### `pip install -r requirements.txt`
First to install dependancies run `pip install -r requirements.txt`
These are quite large as it includes pytorch

### `python app.py`
To start back end run: `python app.py`
\
App runs on port 5000, to use either make requests from the endpoints or use the front end set up.

a normal request normally travels from:
Routes(Endpoints) &#8594; Service(Business Logic) &#8594; Repo(Database Calls)

sqlite is used so databse is stores in site.db

### `python tests.py`
To run tests

### Endpoints

### `/all`
`get` Returns all images

##### Return Example

```
items: {[
            {"id": 1, "likes": 1, "name": "1p1e187a0c1e961090.jpg" },
            {"id": 2, "likes": 2, "name": "2de90db2acd3759635.jpg" },
        ]
    
}
```  

### `/likes/all`
`get` Returns all likes

##### Return Example

```
items: {[
            {"imgId": 4, "likeId": 1, "userId": "a1eea67-cdca-2341-3057-8dfbbdd1dda5" },
            {"imgId": 10, "likeId": 2, "userId": "a1eea67-cdca-2341-3057-8dfbbdd1dda5" },
        ]
    
}
```  

### `/likes/<userId>`
`get` Id of images liked by user

##### Return Example

```
"likedIds":[1,2,3,4,7,10,15]
```  

### `/getSimilarImages/<imgName>`
`get` Returns the 9 most similar images

##### Return Example

```
items: {[
            {"id": 12, "likes": 1, "name": "12gda3c006df972cc80.webp" },
            {"id": 16, "likes": 0, "name": "16f084f16c404f20fa0.jpg" },
            {"id": 21, "likes": 1, "name": "21oc49b9428cee99de3.jpg" },
            {"id": 5, "likes": 1, "name": "5jd23dca78f1f3f350.jpeg" },
            {"id": 22, "likes": 1, "name": "22zba77c48ae663542a.webp" },
            {"id": 6, "likes": 0, "name": "6m5b159fdb82004f98.jpg" },
            {"id": 27, "likes": 0, "name": "27g962cdba88ed3f2da.jpg" },
            {"id": 30, "likes": 0, "name": "30ofa2693cbe76352ec.jpg" },
            {"id": 3, "likes": 1, "name": "3fd977705e2397f102.jpg" },
        ]
}
```  

### `/createImage`
`post` Adds an image to the image repo, body can include either a url, or the image itself, if image is sent header must include `"Content-Type": "image/jpeg"` and specify the images type

##### Return Example

```
items: {[
            {"id": 12, "likes": 0, "name": "12gda3c006df972cc80.webp" },

        ]
}
```  


### `/graph/<imgName>`
`get` Returns a list of math equations that when graphed creates the image

##### Return Example

```
data: {[
           	"((1-t)((1-t)((1-t)58.613…8.841118+t247.704999)))",
            "((1-t)((1-t)((1-t)54.375…4.573221+t242.220794)))",
            "((1-t)((1-t)((1-t)52.000…7.384250+t235.651129)))",
            "((1-t)((1-t)((1-t)50.586…0.700000+t228.500000)))",
            "((1-t)((1-t)((1-t)47.937…9.775000+t214.000000)))" ...
        ]
}
```  
### `/like`

`post` Adds a like to specified image from specified userId

##### Body Example
```
{userId: a1eea67-cdca-2341-3057-8dfbbdd1dda5, imgId: "12gda3c006df972cc80.webp"}
```  

Returns integer for how many likes that image has
