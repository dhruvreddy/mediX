from flask import Flask
from imports import *

def app():
    app = Flask(__name__)
    from .diabetes import diabetes
    app.register_blueprint(diabetes, url_prefix='/')
    from .lungCancer import lungCancer
    app.register_blueprint(lungCancer, url_prefix='/' )
    return app
