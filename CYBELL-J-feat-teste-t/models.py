from database import db
from flask_login import UserMixin
import time

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    
    # NOVOS CAMPOS DE SEGURANÇA (Anti-Brute Force no Servidor)
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    lockout_until = db.Column(db.Float, default=0.0, nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"