from os import environ
from waitress import serve
from py import app
import socket

# print(socket.gethostbyname(socket.gethostname()))
# print(socket.gethostname())
port = int(environ.get('PORT', 5000))

if __name__ == "__main__":
    #app.run(environ.get('PORT'))
    serve(app, host = '0.0.0.0', port=port)