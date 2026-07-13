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
// DETECÇÃO DE DISPOSITIVO (MOUSE VS TRACKPAD)
// ========================================
// Variável de estado do dispositivo apontador (Padrão inicial: mouse)
let currentPointingDevice = "mouse";

// Monitora o comportamento da rolagem para diferenciar Rodinha de Mouse vs Trackpad
document.addEventListener("wheel", (event) => {
    // Trackpads costumam gerar valores decimais contínuos e pequenos (ex: 3.14, 1.05)
    // Mouses tradicionais (rodinha) geram saltos inteiros padronizados (ex: 100, -100, 120)
    const isFractional = !Number.isInteger(event.deltaY) && event.deltaY !== 0;
    const isSmallStep = Math.abs(event.deltaY) > 0 && Math.abs(event.deltaY) < 20;

    if (isFractional || isSmallStep) {
        currentPointingDevice = "trackpad";
    } else if (Math.abs(event.deltaY) >= 50) {
        currentPointingDevice = "mouse";
    }
}, { passive: true });

// ========================================
// ARRAYS DE DADOS
// ========================================
// Armazena eventos de teclado
const keyboardData = [];

// Armazena eventos de mouse/trackpad
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

// Função interna para anonimizar a tecla (Garante RNF04 - Privacidade)
function getKeyCategory(event) {
    // Teclas de controle essenciais para analisar o comportamento
    const controlKeys = [
        'Backspace', 'Enter', 'Space', 'Tab', 
        'ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
        'AltLeft', 'AltRight', 'CapsLock'
    ];
    
    // Se for controle, retorna o nome físico do botão
    if (controlKeys.includes(event.code)) {
        return event.code;
    }
    // Se for letra ou número, substitui pelo rótulo genérico (não grava o caractere)
    if (event.code.startsWith('Key')) return 'CHARACTER_KEY';
    if (event.code.startsWith('Digit')) return 'DIGIT_KEY';
    
    return 'OTHER_KEY';
}

// Evento ao pressionar tecla
document.addEventListener("keydown", (event) => {
    // Ignora campos de senha
    if (event.target.type === "password") {
        return;
    }

    // Usa a categoria no lugar da letra exata
    const keyLabel = getKeyCategory(event);

    // Evita duplicação caso a tecla permaneça pressionada
    if (!pressedKeys[keyLabel]) {
        pressedKeys[keyLabel] = Date.now();
    }
});

// Evento ao soltar tecla
document.addEventListener("keyup", (event) => {
    // Ignora senha
    if (event.target.type === "password") {
        return;
    }

    // Usa a categoria no lugar da letra exata
    const keyLabel = getKeyCategory(event);

    const pressTime = pressedKeys[keyLabel];

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
            key: keyLabel,
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
    delete pressedKeys[keyLabel];
});

// ========================================
// MOVIMENTO DO MOUSE / TRACKPAD
// ========================================
// Evento de movimentação
document.addEventListener("mousemove", (event) => {
    const now = Date.now();

    // Throttle para evitar excesso de eventos
    if (now - lastMouseCapture < mouseThrottleTime) {
        return;
    }

    lastMouseCapture = now;

    // Trackpads costumam disparar sub-coordenadas flutuantes ao deslizar suavemente
    if (!Number.isInteger(event.clientX) || !Number.isInteger(event.clientY)) {
        currentPointingDevice = "trackpad";
    }

    const mouseEvent = {
        user_id: getUsername(),
        session_id: sessionId,
        event_type: "mouse_move",
        data: {
            timestamp: now,
            x: event.clientX,
            y: event.clientY,
            event: "move",
            device_type: currentPointingDevice // <- AQUI ESTÁ A IDENTIFICAÇÃO ENVIADA PARA A IA
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
// CLICK DO MOUSE / TRACKPAD
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
            event: "click",
            device_type: currentPointingDevice // <- AQUI TAMBÉM INCLUI A DEFINIÇÃO DO DISPOSITIVO
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

// ========================================
// ENVIO PARA A API
// ========================================
// Envia o JSON com os dados capturados para o backend via POST
function sendDataToBackend() {
    const data = exportBehaviorData();
    
    // Só envia se houver dados novos para não sobrecarregar o servidor
    if (data.keyboard.length === 0 && data.mouse.length === 0) {
        return;
    }

    fetch('/api/behavior', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log("Status do MongoDB:", result.message);
        
        // Opcional: Limpar os arrays depois de enviar com sucesso
        // keyboardData.length = 0;
        // mouseData.length = 0;
    })
    .catch(error => {
        console.error("Erro ao enviar dados para a API:", error);
    });
}

// Dispara o envio dos dados a cada 10 segundos
setInterval(sendDataToBackend, 10000);