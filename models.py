from database import db

from flask_login import UserMixin

# Model responsável por representar os usuários cadastrados no sistema
class User(UserMixin, db.Model):
    __tablename__ = "users"

    # Identificador único do usuário
    id = db.Column(db.Integer, primary_key=True)

    # Nome de usuário utilizado no login
    # Não permite valores duplicados
    username = db.Column( db.String(80), unique=True, nullable=False)

    # Senha criptografada do usuário
    password = db.Column(db.String(255), nullable=False)

    # Representação textual utilizada principalmente para debug
    def __repr__(self):
        return f"<User {self.username}>"