import re

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from werkzeug.security import generate_password_hash, check_password_hash

from database import db
from models import User

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configurações principais da aplicação
app.config["SECRET_KEY"] = "cybell_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Inicializa a conexão com o banco de dados
db.init_app(app)

# Configuração do Flask-Login responsável pelo gerenciamento de sessões dos usuários
login_manager = LoginManager()

login_manager.init_app(app)

# Define a rota padrão caso o usuário tente acessar uma página protegida
login_manager.login_view = "login"

# Remove a mensagem automática em inglês exibida pelo Flask-Login
login_manager.login_message = None

# Função utilizada pelo Flask-Login para recuperar o usuário da sessão
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Cria as tabelas automaticamente caso ainda não existam
with app.app_context():
    db.create_all()

# Redireciona a rota principal para a página de login
@app.route("/")
def home():
    return redirect(url_for("login"))


# Página responsável pelo login e autenticação do usuário
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()

        # Verifica se os campos foram preenchidos
        if not username or not password:
            flash("Preencha usuário e senha.", "error")

            return redirect(url_for("login"))

        # Verifica se o usuário informado existe no banco de dados
        user = User.query.filter_by(username=username).first()

        if not user:
            flash("Usuário não encontrado.", "error")

            return redirect(url_for("login"))

        # Compara a senha digitada com o hash salvo no banco
        password_correct = check_password_hash(user.password, password)

        if not password_correct:
            flash("Senha incorreta.", "error")

            return redirect(url_for("login"))

        # Realiza o login do usuário e cria a sessão automaticamente
        login_user(user)

        return redirect(url_for("dashboard"))

    return render_template("login.html")

# Página responsável pelo cadastro de novos usuários
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()

        # Verifica se todos os campos foram preenchidos
        if not username or not password:
            flash("Preencha todos os campos.", "error")

            return redirect(url_for("register"))

        # Define tamanho mínimo permitido para o username
        if len(username) < 3:
            flash("Usuário deve ter pelo menos 3 caracteres.", "error")

            return redirect(url_for("register"))

        # Define tamanho máximo permitido para o username
        if len(username) > 20:
            flash("Usuário pode ter no máximo 20 caracteres.", "error")

            return redirect(url_for("register"))

        # Permite apenas letras, números e underline
        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            flash("Usuário deve conter apenas letras, números e _", "error")

            return redirect(url_for("register"))

        # Define tamanho mínimo permitido para a senha
        if len(password) < 6:
            flash("Senha deve ter pelo menos 6 caracteres.", "error")

            return redirect(url_for("register"))

        # Verifica se já existe um usuário com o mesmo nome
        username_exists = User.query.filter_by(username=username).first()

        if username_exists:
            flash("Esse usuário já existe.", "error")

            return redirect(url_for("register"))

        # Criptografa a senha antes de salvar no banco
        hashed_password = generate_password_hash(password)

        # Cria um novo usuário
        new_user = User(username=username, password=hashed_password)

        # Salva no banco de dados
        db.session.add(new_user)

        db.session.commit()

        # Realiza login automático após o cadastro
        login_user(new_user)

        return redirect(url_for("dashboard"))

    return render_template("register.html")


# Dashboard principal do sistema
# Apenas usuários autenticados podem acessar
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)


# Realiza logout do usuário e encerra a sessão atual
@app.route("/logout")
@login_required
def logout():
    logout_user()

    return redirect(url_for("login"))


# Executa a aplicação em modo de desenvolvimento
if __name__ == "__main__":
    app.run(debug=True)