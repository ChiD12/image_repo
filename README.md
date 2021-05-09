# Image Repo backend for the Shopify challenge

[Frontend Found here](https://github.com/ChiD12/image_repo_frontend)
\
[Live Demo](http://image.danielr.tech/).

Image Repository whose features include storing images, delivering images, and uses a Convalutional Neural Network to create image encodings to find similarities between images

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

#Folders
###images
Image storage folder
###256
Stores downscaled 256x256 versions of images to be used in conv neural network
###models
Stores trained models as well as encodings for the images
###py
Python files


