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