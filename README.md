# Guia de Instalação e Testes do Sistema CYBELL-J

## Pré-requisitos e Configuração de Ambiente

Para o funcionamento da captura contínua e persistência de dados em nuvem, este projeto utiliza o **MongoDB Atlas** e variáveis de ambiente.

1. Crie um arquivo chamado **`.env`** na raiz do projeto (na mesma pasta do `app.py`).
2. Adicione sua URI do MongoDB Atlas no arquivo `.env`:
   ```env
   MONGO_URI=sua_uri_do_mongodb_atlas_aqui
   SECRET_KEY=uma_chave_aleatoria_grande_aqui
   ENABLE_SWAGGER=true
   ```
   
   Para gerar uma `SECRET_KEY` segura, rode no terminal:
   ```
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
   
**Importante:** sem a `SECRET_KEY` definida, a aplicação não inicia (ela valida isso na subida).
      
`ENABLE_SWAGGER=true` habilita a documentação em `/apidocs`.
   
_(Nota: O arquivo `.env` já está listado no `.gitignore` para proteger suas credenciais de segurança)._

## 1. Instalar as dependências

No terminal, execute o comando para instalar todas as bibliotecas necessárias (incluindo o suporte a variáveis de ambiente e banco NoSQL):
```
pip install -r requirements.txt
```

## 2. Executar o sistema

No terminal, execute:


```
python app.py
```

Se tudo estiver funcionando corretamente, aparecerá a mensagem:



```
Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
```

Abra o navegador e acesse:

```
[http://127.0.0.1:5000](http://127.0.0.1:5000)
```

## 3. Testar o registro de usuário

1.  Clique em "Registrar".
    
2.  Digite um usuário (de 3 a 20 caracteres, apenas letras, números e `_`).
    
3.  Digite uma senha (mínimo de 6 caracteres).
    
4.  Clique em "Registrar".
    

**Segurança implementada:**

-   Validação robusta de campos e formato no servidor.
    
-   As senhas são criptografadas com _hash_ seguro (`werkzeug.security`) antes de serem salvas no SQLite.
    
-   Após o cadastro, o usuário é autenticado automaticamente e redirecionado para o dashboard.
    

## 4. Testar login e Proteção Anti-Força Bruta

1.  Faça logout e tente entrar novamente com suas credenciais.
    
2.  **Nova Proteção no Servidor:** O sistema agora possui bloqueio ativo contra ataques de força bruta (_Brute Force_). Se houver 5 tentativas falhas consecutivas, a conta do usuário é temporariamente bloqueada por segurança no servidor.
    

## 5. Testar o Dashboard e Simulador Corporativo

Após logar, você será direcionado ao Dashboard, que conta com:

-   **Simulador de Trabalho:** Campos de preenchimento para simular rotinas corporativas.
    
-   **Captura Invisível de Comportamento:** Monitoramento contínuo em segundo plano (digitação, ritmo, posições de mouse e atalhos de edição como `Ctrl+C`, `Ctrl+V`, `Ctrl+X` e `Ctrl+A` com mensuração de _hold time_).
    
-   **Confiança Degradada:** Sistema de segurança por ociosidade que exibe um modal de reautenticação caso o usuário fique inativo por vários ciclos.
    

## 6. Verificar persistência no MongoDB Atlas (Nuvem)

O sistema envia automaticamente os blocos de eventos biométricos coletados para a API a cada 10 segundos:

1.  Interaja com o dashboard digitando e movimentando o mouse.
    
2.  Abra o aplicativo **MongoDB Compass** e conecte-se utilizando a sua string de conexão do **MongoDB Atlas**.
    
3.  Acesse o banco de dados `cybell_db`.
    
4.  Verifique as coleções **`keystrokes`** e **`mouse_events`**. Os dados estarão particionados e salvos de forma segura por usuário.
    

## 7. Testar a rota de verificação da IA

A API possui uma rota de simulação (`/api/verify`) integrada com documentação Swagger (`/apidocs`). Para testá-la via terminal (PowerShell), execute:

```
Invoke-RestMethod -Uri [http://127.0.0.1:5000/api/verify](http://127.0.0.1:5000/api/verify) -Method Post -Body '{}' -ContentType 'application/json'
```

## Proteção contra CSRF
 
Todos os formulários (login e cadastro) e as chamadas via `fetch` (`/api/behavior` e `/api/reauth`) agora exigem um token CSRF válido, gerado pelo Flask-WTF. Isso é transparente para quem usa a interface normalmente, o token é injetado automaticamente nas páginas. Se você modificar os templates ou o `behavior.js`, garanta que o token continue sendo enviado, senão as requisições passam a falhar com erro 400.
