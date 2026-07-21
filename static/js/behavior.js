let keyboardData = [];
let mouseData = [];
let shortcutsData = [];
let ctrlPressTime = null;
let idleCycles = 0; // Contador de inatividade

// 1. CAPTURA DE TECLADO E ATALHOS
document.addEventListener('keydown', (event) => {
    if (!event.isTrusted) return; // Anti-Bot

    const timestamp = Date.now();

    if (event.key === 'Control') {
        ctrlPressTime = timestamp;
    }

    // Captura Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+A
    if (event.ctrlKey && ['c', 'v', 'x', 'a'].includes(event.key.toLowerCase())) {
        const holdTime = ctrlPressTime ? (timestamp - ctrlPressTime) : 0;
        shortcutsData.push({
            key: `Ctrl+${event.key.toUpperCase()}`,
            event_type: 'shortcut',
            hold_time: holdTime,
            timestamp: timestamp
        });
    } else if (event.key !== 'Control') {
        keyboardData.push({
            key: event.key,
            event_type: 'keydown',
            timestamp: timestamp
        });
    }
});

document.addEventListener('keyup', (event) => {
    if (!event.isTrusted) return;
    if (event.key === 'Control') {
        ctrlPressTime = null;
    }
});

// 2. CAPTURA DE MOUSE (Movimento e Clique)
document.addEventListener('mousemove', (event) => {
    if (!event.isTrusted) return;
    mouseData.push({
        event_type: 'mouse_move',
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

document.addEventListener('click', (event) => {
    if (!event.isTrusted) return;
    mouseData.push({
        event_type: 'mouse_click',
        button: event.button, // 0 = Esquerdo, 1 = Meio, 2 = Direito
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// 3. ENVIO DE DADOS PARA A API
function sendDataToBackend() {
    const payload = {
        keyboard: keyboardData,
        mouse: mouseData,
        shortcuts: shortcutsData
    };

    fetch('/api/behavior', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        // GATILHO: Verifica se a janela veio vazia (Confiança Degradada)
        if (data.status === 'empty_window') {
            idleCycles++;
            if (idleCycles >= 3) { // 3 ciclos de 10s (30 segundos)
                document.getElementById('degradedTrustModal').style.display = 'flex';
            }
        } else if (data.status === 'bot_detected') {
            alert("Atividade suspeita detectada pelo sistema de segurança.");
        } else {
            idleCycles = 0; // Zera o contador se houve atividade real
        }
    })
    .catch(error => console.error('Erro na comunicação com a API:', error));

    // Limpa os arrays para o próximo ciclo
    keyboardData = [];
    mouseData = [];
    shortcutsData = [];
}

// 4. FUNÇÃO DO MODAL DE DESBLOQUEIO
function verifyReauth() {
    const pass = document.getElementById('reauthPassword').value;
    if (pass.length > 0) {
        document.getElementById('degradedTrustModal').style.display = 'none';
        document.getElementById('reauthPassword').value = '';
        idleCycles = 0; // Reinicia a contagem
    }
}

// 5. INICIA O LOOP (10 segundos)
setInterval(sendDataToBackend, 10000);