from . import db

class User(db.Model): 
    __tablename__ = 'users'
    id = db.Column('user_id', db.Integer, primary_key = True, autoincrement= True)
    username  = db.Column('username', db.String (100), unique = True)
    role_id = db.Column('role_id', db.Integer)
    password = db.Column('password', db.String (100))

    def __repr__(self):
        return f"<User {self.username}>"

