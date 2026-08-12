// Alterna a visibilidade da senha quando o usuário clica no ícone de olho ao lado do input
const togglePassword =
    document.getElementById("togglePassword");

if (togglePassword) {
    togglePassword.addEventListener("click", () => {
        const passwordInput =
            document.getElementById("password");

        // Verifica se a senha está oculta
        if (passwordInput.type === "password") {
            // Mostra a senha digitada
            passwordInput.type = "text";

            // Troca o ícone para olho cortado
            togglePassword.classList.remove("fa-eye");

            togglePassword.classList.add("fa-eye-slash");

        } else {
            // Oculta novamente a senha
            passwordInput.type = "password";

            // Retorna o ícone original
            togglePassword.classList.remove("fa-eye-slash");

            togglePassword.classList.add("fa-eye");
        }
    });
}

// Seleciona todas as mensagens de alerta exibidas na tela para removê-las automaticamente após alguns segundos
const flashMessages =
    document.querySelectorAll(".flash-message");

setTimeout(() => {
    flashMessages.forEach((message) => {
        // Aplica efeito de desaparecimento suave
        message.style.opacity = "0";

        setTimeout(() => {
            // Remove o elemento do HTML
            message.remove();
        }, 300);
    });
}, 3000);

// Cria e exibe uma mensagem de flash dinamicamente, reaproveitando o mesmo
// estilo (.flash-message) usado no restante do sistema, e some sozinha
// depois de alguns segundos, igual as mensagens renderizadas pelo servidor
function showFlashMessage(container, text, type = "success") {
    if (!container) return;

    const message = document.createElement("div");
    message.className = `flash-message ${type}`;
    message.textContent = text;

    container.appendChild(message);

    setTimeout(() => {
        message.style.opacity = "0";

        setTimeout(() => {
            message.remove();
        }, 300);
    }, 3000);
}

//Botão "Abrir Chamado" do dashboard 
const saveRecordBtn = document.getElementById("saveRecordBtn");
const dashboardFlashContainer = document.getElementById("dashboardFlashContainer");

// Proteção contra Brute Force no Client-Side (Login)
const loginForm = document.getElementById("loginForm");

if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
        let failedAttempts = parseInt(localStorage.getItem("login_failures") || "0");
        let lockoutUntil = parseInt(localStorage.getItem("lockout_until") || "0");
        const now = Date.now();

        // Verifica se o usuário está bloqueado
        if (now < lockoutUntil) {
            e.preventDefault();
            const remainingSeconds = Math.ceil((lockoutUntil - now) / 1000);
            alert(`Muitas tentativas falhas. Aguarde ${remainingSeconds} segundos para tentar novamente.`);
            return;
        }

        // Se o login falhar, o Flask vai recarregar a página com erro. 
        // Podemos incrementar a falha aqui ou capturar o erro da rota.
        // Uma forma simples para o protótipo: toda vez que submete, incrementa se não houver sucesso imediato,
        // ou controlamos via contador de erros na tela de login.
    });
}