from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .config import Config


app = Flask (__name__)
app.config.from_object (Config)
app.secret_key = '48dfa6608110b22ecb622e0300ab9ed4958b92309503bb07'

# pip install Flask Flask-SQLAlchemy mysqlclient
db = SQLAlchemy(app)
from app import routes
