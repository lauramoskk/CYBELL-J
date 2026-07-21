let keyboardData = [];
let mouseData = [];
let shortcutsData = [];
let ctrlPressTime = null;
let idleCycles = 0;

// Captura de Teclado e Atalhos (Ctrl+C, Ctrl+V, etc.)
document.addEventListener('keydown', (event) => {
    if (!event.isTrusted) return; // Anti-bot básico por evento nativo

    const timestamp = Date.now();

    if (event.key === 'Control') {
        ctrlPressTime = timestamp;
    }

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

// Captura de Mouse (Movimento e Cliques)
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
        button: event.button,
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// Envio periódico para a API (A cada 10 segundos)
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
        if (data.status === 'empty_window') {
            idleCycles++;
            if (idleCycles >= 3) { // 3 ciclos de 10s = 30 segundos ocioso
                const modal = document.getElementById('degradedTrustModal');
                if (modal) modal.style.display = 'flex';
            }
        } else {
            idleCycles = 0;
        }
    })
    .catch(error => console.error('Erro ao enviar dados comportamentais:', error));

    keyboardData = [];
    mouseData = [];
    shortcutsData = [];
}

function verifyReauth() {
    const pass = document.getElementById('reauthPassword').value;
    if (pass.length > 0) {
        const modal = document.getElementById('degradedTrustModal');
        if (modal) modal.style.display = 'none';
        document.getElementById('reauthPassword').value = '';
        idleCycles = 0;
    }
}

setInterval(sendDataToBackend, 10000);