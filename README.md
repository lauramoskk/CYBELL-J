# Guia simples para testar o sistema CYBELL-J

## 1. Instalar as dependências
No terminal, execute:
```bash
pip install -r requirements.txt
```
Caso ainda não exista um arquivo `requirements.txt`, instale manualmente:
```bash
pip install flask flask-sqlalchemy flask-login werkzeug
```

---

# 2. Executar o sistema
No terminal, execute:
```bash
python app.py
```
Se tudo estiver funcionando corretamente, aparecerá algo parecido com:
```bash
Running on http://127.0.0.1:5000
```
Abra o navegador e acesse:
```text
http://127.0.0.1:5000
```

---

# 3. Testar o registro de usuário
1. Clique em "Registrar"
2. Digite um usuário
3. Digite uma senha
4. Clique em "Registrar"

Validações implementadas:
* Não permite campos vazios
* Não permite usuários duplicados
* Username deve ter entre 3 e 20 caracteres
* Username aceita apenas letras, números e _
* Senha deve possuir pelo menos 6 caracteres
* Senha é salva criptografada no banco

Após registrar:
* o usuário é autenticado automaticamente
* o sistema redireciona para o dashboard

---

# 4. Testar login
1. Faça logout
2. Entre novamente usando:

* usuário cadastrado
* senha cadastrada

Validações implementadas:
* Verifica campos vazios
* Verifica se o usuário existe
* Verifica se a senha está correta
* Mantém sessão autenticada

---

# 5. Testar logout
1. Clique no botão "Sair"
2. O sistema encerrará a sessão
3. O usuário será redirecionado para login

---

# 6. Testar captura comportamental
Após logar:
1. Vá para o dashboard
2. Digite no campo de texto
3. Mova o mouse pela tela
4. Clique em diferentes áreas

O sistema captura:
* teclas pressionadas
* tempo de pressionamento
* tempo entre teclas
* movimentos do mouse
* cliques do mouse
* coordenadas X/Y
* timestamps

---

# 7. Visualizar dados no console

Abra o console do navegador:
* Chrome/Edge:
  * F12
  * Aba "Console"

Execute:
```javascript
exportBehaviorData()
```

Isso exibirá todos os dados coletados.

---

# 8. Baixar JSON com os dados
No console do navegador execute:
```javascript
downloadBehaviorData()
```

O sistema fará download automático de um arquivo JSON contendo os eventos capturados.
