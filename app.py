import re
import os

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify  
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from werkzeug.security import generate_password_hash, check_password_hash

from database import db
from models import User
from api import api_bp

from flasgger import Swagger

from flask_wtf import CSRFProtect

from dotenv import load_dotenv

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import time

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

# Ativa documentação Swagger da API
ENABLE_SWAGGER = os.getenv("ENABLE_SWAGGER", "false").lower() == "true"

if ENABLE_SWAGGER:
    Swagger(app)

# Configurações principais da aplicação
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

if not app.config["SECRET_KEY"]:
    raise ValueError("A variável de ambiente SECRET_KEY não foi encontrada. Verifique o arquivo .env")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

limiter = Limiter(get_remote_address, app=app, default_limits=[])

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # mude para True quando tiver HTTPS em produção

csrf = CSRFProtect(app)

# Inicializa a conexão com o banco de dados
db.init_app(app)

# Registra as rotas da API separadas no arquivo api.py
app.register_blueprint(api_bp)

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
    """
    Redirecionamento inicial
    ---
    tags:
      - Navegação
    responses:
      302:
        description: Redireciona o usuário para a página de login
    """
    return redirect(url_for("login"))


# Página responsável pelo login e autenticação do usuário
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Preencha usuário e senha.", "error")
            return redirect(url_for("login"))

        user = User.query.filter_by(username=username).first()

        if not user:
            flash("Usuário ou senha inválidos.", "error")
            return redirect(url_for("login"))

        # VERIFICAÇÃO DE SEGURANÇA: Bloqueio por Força Bruta no Servidor
        current_time = time.time()
        if user.lockout_until > current_time:
            remaining_seconds = int(user.lockout_until - current_time)
            flash(f"Conta temporariamente bloqueada por segurança. Tente novamente em {remaining_seconds} segundos.", "error")
            return redirect(url_for("login"))

        # Compara a senha
        if not check_password_hash(user.password, password):
            user.failed_login_attempts += 1
            
            # Se errar 5 vezes, bloqueia por 30 segundos no servidor
            if user.failed_login_attempts >= 5:
                user.lockout_until = current_time + 30
                db.session.commit()
                flash("Muitas tentativas falhas. Conta bloqueada por 30 segundos.", "error")
            else:
                db.session.commit()
                flash("Usuário ou senha inválidos.", "error")
                
            return redirect(url_for("login"))

        # Sucesso: Reseta os contadores de falha
        user.failed_login_attempts = 0
        user.lockout_until = 0.0
        db.session.commit()

        login_user(user)
        return redirect(url_for("dashboard"))

    return render_template("login.html")

# Página responsável pelo cadastro de novos usuários
@app.route("/register", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def register():
    """
    Cadastro de usuário
    ---
    tags:
      - Autenticação
    parameters:
      - name: username
        in: formData
        type: string
        required: true
      - name: password
        in: formData
        type: string
        required: true
    responses:
      200:
        description: Página de cadastro carregada com sucesso
      302:
        description: Usuário cadastrado e redirecionado para dashboard
    """
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
    """
    Dashboard principal
    ---
    tags:
      - Dashboard
    responses:
      200:
        description: Dashboard exibido com sucesso
      401:
        description: Usuário não autenticado
    """
    return render_template("dashboard.html", username=current_user.username)


# Realiza logout do usuário e encerra a sessão atual
@app.route("/logout")
@login_required
def logout():
    """
    Logout do usuário
    ---
    tags:
      - Autenticação
    responses:
      302:
        description: Usuário desconectado e redirecionado para login
    """
    logout_user()

    return redirect(url_for("login"))


# Executa a aplicação em modo de desenvolvimento
if __name__ == "__main__":
    app.run(debug=True)