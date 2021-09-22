from py import app, db

import unittest
from unittest.mock import MagicMock
from py.service import Service
from py.repo import Repo
from py.routes import setService
from unittest.mock import patch
from py.dbmodels import Image, UserLikes

class FlaskTest(unittest.TestCase):

    def test_all(self):
        #setup
        test = app.test_client(self)
        expected = [
            {"id": 1, "likes": 1, "name": "1p1e187a0c1e961090.jpg" },
            {"id": 2, "likes": 2, "name": "2de90db2acd3759635.jpg" },
        ]
        
        dbReturned = [
            Image(id=1, name='1p1e187a0c1e961090.jpg', likes=1),
            Image(id=2, name='2de90db2acd3759635.jpg', likes=2)
        ]
        

        testRepo = Repo(db)
        testRepo.rGetAllImages = MagicMock(return_value=dbReturned)
        testService = Service(testRepo)
        setService(testService)

        #calls
        response = test.get("/all")
        items = (eval(response.data.decode()))['items']
        statuscode = response.status_code
        contentType = response.content_type

        #asserts
        self.assertEqual(statuscode, 200, "Status should be 200")
        self.assertEqual(contentType, 'application/json', "Content type should be application/json")
        self.assertEqual(items, expected, "Returned dict should be same as expected")

    def test_likes_all(self):
        #setup
        test = app.test_client(self)

        expected = [
            {"imgId": 4, "likeId": 1, "userId": "a1eea67-cdca-2341-3057-8dfbbdd1dda5" },
            {"imgId": 10, "likeId": 2, "userId": "a1eea67-cdca-2341-3057-8dfbbdd1dda5" },
        ]
        
        dbReturned = [
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=4, likeId=1),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=10, likeId=2)
        ]
        
        testRepo = Repo(db)
        testRepo.rGetAllLikes = MagicMock(return_value=dbReturned)
        testService = Service(testRepo)
        setService(testService)

        #calls
        response = test.get("/likes/all")
        items = (eval(response.data.decode()))['items']
        statuscode = response.status_code
        contentType = response.content_type

        #asserts
        self.assertEqual(statuscode, 200, "Status should be 200")
        self.assertEqual(contentType, 'application/json', "Content type should be application/json")
        self.assertEqual(items, expected, "Returned dict should be same as expected")

    def test_likes_userId(self):
        #setup
        test = app.test_client(self)

        expected = { "likedIds":[1,2,3,4,7,10,15]}
        
        dbReturned = [
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=1, likeId=1),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=2, likeId=2),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=3, likeId=3),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=4, likeId=4),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=7, likeId=5),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=10, likeId=6),
            UserLikes(userId="a1eea67-cdca-2341-3057-8dfbbdd1dda5", imgId=15, likeId=7)
        ]
    
        testRepo = Repo(db)
        testRepo.rGetAllLikesWithId = MagicMock(return_value=dbReturned)
        testService = Service(testRepo)
        setService(testService)

        #calls
        response = test.get("/likes/a1eea67-cdca-2341-3057-8dfbbdd1dda5")
        items = (eval(response.data.decode()))
        statuscode = response.status_code
        contentType = response.content_type

        #asserts
        self.assertEqual(statuscode, 200, "Status should be 200")
        self.assertEqual(contentType, 'application/json', "Content type should be application/json")
        self.assertEqual(items, expected, "Returned dict should be same as expected")


    def test_create_image(self):
        #setup
        test = app.test_client(self)

        expected = [{"id": 6, "name": '6p1e187a0c1e961090.jpg' }]

        dbReturned = Image(id=6, name='6p1e187a0c1e961090.jpg', likes=0)

        body = 'https://www.ukiahdailyjournal.com/wp-content/uploads/2020/12/MAILMAN3.jpeg'
        
        headers= {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "image/jpeg"
            }

        img = open("39hd0d57634e34f16cd.jpeg", 'rb').read()

        testRepo = Repo(db)
        testRepo.rGetMaxId = MagicMock(return_value=5)
        testRepo.rPostImage = MagicMock(return_value=dbReturned)
        testService = Service(testRepo)
        testService.download = MagicMock()
        setService(testService)

        #calls
        response = test.post("/createImage", data= img, headers=headers)
        items = (eval(response.data.decode()))["items"]
        statuscode = response.status_code
        contentType = response.content_type

        #asserts
        self.assertEqual(statuscode, 200, "Status should be 200")
        self.assertEqual(contentType, 'application/json', "Content type should be application/json")
        self.assertEqual(items, expected, "Returned dict should be same as expected")

if __name__ == "__main__":
    unittest.main()