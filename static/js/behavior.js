// ========================================
// IDENTIFICAÇÃO DO USUÁRIO
// ========================================
// Obtém o username enviado pelo backend através do body
function getUsername() {
    return document.body.dataset.user;
}

// Identificador único da sessão atual
const sessionId = crypto.randomUUID();

// ========================================
// ARRAYS DE DADOS
// ========================================
// Armazena eventos de teclado
const keyboardData = [];

// Armazena eventos de mouse
const mouseData = [];

// ========================================
// CONFIGURAÇÕES
// ========================================
// Limite máximo de eventos armazenados em memória
const MAX_EVENTS = 1000;

// Controle de captura de mouse
let lastMouseCapture = 0;

// Intervalo mínimo entre capturas
const mouseThrottleTime = 100;

// ========================================
// TECLADO
// ========================================
// Armazena teclas pressionadas
const pressedKeys = {};

// Armazena tempo de soltura da última tecla
let lastKeyReleaseTime = null;

// Evento ao pressionar tecla
document.addEventListener("keydown", (event) => {
    // Ignora campos de senha
    if (event.target.type === "password") {
        return;
    }

    // Evita duplicação caso a tecla permaneça pressionada
    if (!pressedKeys[event.key]) {
        pressedKeys[event.key] = Date.now();
    }
});

// Evento ao soltar tecla
document.addEventListener("keyup", (event) => {
    // Ignora senha
    if (event.target.type === "password") {
        return;
    }

    const pressTime = pressedKeys[event.key];

    if (!pressTime) {
        return;
    }

    const releaseTime = Date.now();

    // Tempo que a tecla permaneceu pressionada
    const holdTime = releaseTime - pressTime;

    // Tempo entre uma tecla e a próxima
    let flightTime = null;

    if (lastKeyReleaseTime !== null) {
        flightTime = pressTime - lastKeyReleaseTime;
    }

    // Atualiza última tecla solta
    lastKeyReleaseTime = releaseTime;

    // Estrutura JSON do evento
    const keystrokeEvent = {
        user_id: getUsername(),
        session_id: sessionId,
        event_type: "keystroke",
        data: {
            key: event.key,
            press_time: pressTime,
            release_time: releaseTime,
            hold_time: holdTime,
            flight_time: flightTime
        }
    };

    // Salva evento
    keyboardData.push(
        keystrokeEvent
    );

    // Limita tamanho do array
    if (keyboardData.length > MAX_EVENTS) {
        keyboardData.shift();
    }

    // Remove tecla armazenada
    delete pressedKeys[event.key];
});

// ========================================
// MOVIMENTO DO MOUSE
// ========================================
// Evento de movimentação
document.addEventListener("mousemove", (event) => {
    const now = Date.now();

    // Throttle para evitar excesso de eventos
    if (now - lastMouseCapture < mouseThrottleTime) {
        return;
    }

    lastMouseCapture = now;

    const mouseEvent = {
        user_id: getUsername(),
        session_id: sessionId,
        event_type: "mouse_move",
        data: {
            timestamp: now,
            x: event.clientX,
            y: event.clientY,
            event: "move"
        }
    };

    // Salva evento
    mouseData.push(
        mouseEvent
    );

    // Limita tamanho do array
    if (mouseData.length > MAX_EVENTS) {
        mouseData.shift();
    }
});

// ========================================
// CLICK DO MOUSE
// ========================================
// Captura cliques do mouse
document.addEventListener("click", (event) => {
    let button = "unknown";

    if (event.button === 0) {
        button = "left";
    }

    if (event.button === 1) {
        button = "middle";
    }

    if (event.button === 2) {
        button = "right";
    }

    const clickEvent = {
        user_id: getUsername(),
        session_id: sessionId,
        event_type: "mouse_click",
        data: {
            timestamp: Date.now(),
            x: event.clientX,
            y: event.clientY,
            button: button,
            event: "click"
        }
    };

    mouseData.push(
        clickEvent
    );

    // Limita tamanho do array
    if (mouseData.length > MAX_EVENTS) {
        mouseData.shift();
    }
});

// ========================================
// EXPORTAÇÃO DOS DADOS
// ========================================
// Retorna todos os dados coletados
function exportBehaviorData() {
    return {
        keyboard: keyboardData,
        mouse: mouseData
    };
}

// Exporta arquivo JSON
function downloadBehaviorData() {
    const data = exportBehaviorData();

    const json = JSON.stringify(data, null, 2);

    const blob =
        new Blob(
            [json],
            {
                type: "application/json"
            }
        );

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;

    link.download = "behavior_data.json";

    link.click();

    URL.revokeObjectURL(url);
}

// ========================================
// ESTATÍSTICAS
// ========================================
// Atualiza informações na tela
function updateStats() {
    const keyboardCount = document.getElementById("keyboard-count");

    const mouseCount = document.getElementById("mouse-count");

    const totalCount = document.getElementById("total-count");

    if (!keyboardCount ||
        !mouseCount ||
        !totalCount) {

        return;
    }

    keyboardCount.textContent = keyboardData.length;

    mouseCount.textContent = mouseData.length;

    totalCount.textContent = keyboardData.length + mouseData.length;
}

// Atualiza estatísticas constantemente
setInterval(() => {
    updateStats();
}, 500);

// ========================================
// DEBUG
// ========================================
// Atualiza console automaticamente
setInterval(() => {
    console.clear();

    console.log("Keyboard Data:", keyboardData);

    console.log("Mouse Data:", mouseData);

}, 3000);

// ========================================
// FUNÇÕES GLOBAIS
// ========================================
// Disponibiliza funções no console do navegador

window.exportBehaviorData = exportBehaviorData;

window.downloadBehaviorData = downloadBehaviorData;