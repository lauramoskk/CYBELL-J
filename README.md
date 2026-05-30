# Guia simples para testar o sistema CYBELL-J

## Pré-requisitos

Para o funcionamento da captura contínua de dados, certifique-se de ter o **MongoDB Community Server** instalado e rodando localmente em sua máquina (porta 27017), bem como o **MongoDB Compass** para a visualização do banco de dados.

## 1. Instalar as dependências

No terminal, execute:

```
pip install -r requirements.txt
```

Caso ainda não exista um arquivo `requirements.txt`, instale manualmente:


```
pip install flask flask-sqlalchemy flask-login werkzeug pymongo
```

# 2. Executar o sistema

No terminal, execute:

```
python app.py
```

Se tudo estiver funcionando corretamente, aparecerá algo parecido com:

```
Running on http://127.0.0.1:5000
```

Abra o navegador e acesse:

```
http://127.0.0.1:5000
```

# 3. Testar o registro de usuário

1.  Clique em "Registrar"
    
2.  Digite um usuário
    
3.  Digite uma senha
    
4.  Clique em "Registrar"
    

Validações implementadas:

-   Não permite campos vazios
    
-   Não permite usuários duplicados
    
-   Username deve ter entre 3 e 20 caracteres
    
-   Username aceita apenas letras, números e _
    
-   Senha deve possuir pelo menos 6 caracteres
    
-   Senha é salva criptografada no banco
    

Após registrar:

-   o usuário é autenticado automaticamente
    
-   o sistema redireciona para o dashboard
    

# 4. Testar login

1.  Faça logout
    
2.  Entre novamente usando:
    

-   usuário cadastrado
    
-   senha cadastrada
    

Validações implementadas:

-   Verifica campos vazios
    
-   Verifica se o usuário existe
    
-   Verifica se a senha está correta
    
-   Mantém sessão autenticada
    

# 5. Testar logout

1.  Clique no botão "Sair"
    
2.  O sistema encerrará a sessão
    
3.  O usuário será redirecionado para login
    

# 6. Testar captura comportamental

Após logar:

1.  Vá para o dashboard
    
2.  Digite no campo de texto
    
3.  Mova o mouse pela tela
    
4.  Clique em diferentes áreas
    

O sistema captura:

-   teclas pressionadas
    
-   tempo de pressionamento
    
-   tempo entre teclas
    
-   movimentos do mouse
    
-   cliques do mouse
    
-   coordenadas X/Y
    
-   timestamps
    

# 7. Visualizar dados no console

Abra o console do navegador:

-   Chrome/Edge:
    
    -   F12
        
    -   Aba "Console"
        

Execute:

```
exportBehaviorData()
```

Isso exibirá todos os dados coletados.

# 8. Baixar JSON com os dados

No console do navegador execute:

```
downloadBehaviorData()

```
O sistema fará download automático de um arquivo JSON contendo os eventos capturados.

# 9. Testar envio automático para o MongoDB (Banco de Dados NoSQL)

O sistema foi atualizado para enviar os dados comportamentais capturados diretamente para a API a cada 10 segundos.

1.  No navegador, com o painel "Console" (F12) aberto, continue interagindo com a tela.
    
2.  A cada 10 segundos, você deverá ver a confirmação: `Status do MongoDB: Dados biométricos salvos no MongoDB`.
    
3.  Para validar fisicamente, abra o aplicativo **MongoDB Compass** e conecte-se à URI padrão (`mongodb://localhost:27017/`).
    
4.  Acesse o banco de dados `cybell_db`.
    
5.  Verifique as coleções `keystrokes` e `mouse_events`. Os dados gerados pelos seus testes estarão registrados nestas coleções em formato JSON.
    

# 10. Testar a rota de verificação da IA

Foi disponibilizada uma rota de simulação (`/api/verify`) que retornará o status futuro da Inteligência Artificial. Para testá-la, abra um terminal (como o PowerShell) e execute:

```
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/verify -Method Post -Body '{}' -ContentType 'application/json'
```