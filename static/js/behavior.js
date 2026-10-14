'use strict';

// ============================================
// CYBELL-J — Captura e verificação automática
// ============================================

let keyboardData = [];
let mouseData = [];
let shortcutsData = [];

let ctrlPressTime = null;
let idleCycles = 0;
let isSendingData = false;

let automaticVerificationInProgress = false;
let consecutiveSuspiciousResults = 0;
let reauthCooldownUntil = 0;
let reauthInProgress = false;

// Token CSRF
const csrfTokenMeta = document.querySelector(
    'meta[name="csrf-token"]'
);

const csrfToken = csrfTokenMeta
    ? csrfTokenMeta.getAttribute('content')
    : '';

let currentDeviceType = 'mouse';

// ============================================
// Captura de teclado e atalhos
// ============================================

document.addEventListener('keydown', (event) => {
    if (!event.isTrusted) return;

    const target = event.target;

    if (target) {
        const isPasswordField =
            target.tagName === 'INPUT' &&
            target.type === 'password';

        const isIgnored =
            typeof target.hasAttribute === 'function' &&
            target.hasAttribute('data-behavior-ignore');

        if (isPasswordField || isIgnored) return;
    }

    const timestamp = Date.now();

    if (event.key === 'Control') {
        ctrlPressTime = timestamp;
    }

    if (
        event.ctrlKey &&
        ['c', 'v', 'x', 'a'].includes(event.key.toLowerCase())
    ) {
        const holdTime = ctrlPressTime
            ? timestamp - ctrlPressTime
            : 0;

        shortcutsData.push({
            key: `Ctrl+${event.key.toUpperCase()}`,
            event_type: 'shortcut',
            hold_time: holdTime,
            timestamp
        });
    } else if (event.key !== 'Control') {
        keyboardData.push({
            key: event.code,
            event_type: 'keydown',
            timestamp
        });
    }
});

document.addEventListener('keyup', (event) => {
    if (!event.isTrusted) return;

    if (event.key === 'Control') {
        ctrlPressTime = null;
    }
});

// ============================================
// Captura de rolagem
// ============================================

document.addEventListener('wheel', (event) => {
    if (!event.isTrusted) return;

    if (
        Math.abs(event.deltaY) < 50 ||
        event.deltaX !== 0 ||
        !Number.isInteger(event.deltaY)
    ) {
        currentDeviceType = 'trackpad';
    } else {
        currentDeviceType = 'mouse';
    }

    mouseData.push({
        event_type: 'mouse_wheel',
        device_type: currentDeviceType,
        delta_x: event.deltaX,
        delta_y: event.deltaY,
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// ============================================
// Captura de movimento do mouse
// ============================================

document.addEventListener('mousemove', (event) => {
    if (!event.isTrusted) return;

    mouseData.push({
        event_type: 'mouse_move',
        device_type: currentDeviceType,
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// ============================================
// Captura de cliques
// ============================================

document.addEventListener('click', (event) => {
    if (!event.isTrusted) return;

    mouseData.push({
        event_type: 'mouse_click',
        device_type: currentDeviceType,
        button: event.button,
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// ============================================
// Captura de pressionamento do mouse
// ============================================

document.addEventListener('mousedown', (event) => {
    if (!event.isTrusted) return;

    mouseData.push({
        event_type: 'mouse_down',
        device_type: currentDeviceType,
        button: event.button,
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// ============================================
// Captura de soltura do mouse
// ============================================

document.addEventListener('mouseup', (event) => {
    if (!event.isTrusted) return;

    mouseData.push({
        event_type: 'mouse_up',
        device_type: currentDeviceType,
        button: event.button,
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now()
    });
});

// ============================================
// Divide eventos em lotes
// ============================================

function createBatches(events, batchSize) {
    const batches = [];

    for (let i = 0; i < events.length; i += batchSize) {
        batches.push(events.slice(i, i + batchSize));
    }

    return batches;
}

// ============================================
// Envio de eventos ao backend
// ============================================

async function sendDataToBackend() {
    if (isSendingData) return;

    if (
        keyboardData.length === 0 &&
        mouseData.length === 0 &&
        shortcutsData.length === 0
    ) {
        return;
    }

    isSendingData = true;

    const keyboardSnapshot = keyboardData;
    const mouseSnapshot = mouseData;
    const shortcutsSnapshot = shortcutsData;

    keyboardData = [];
    mouseData = [];
    shortcutsData = [];

    const keyboardBatches = createBatches(
        keyboardSnapshot,
        200
    );

    const mouseBatches = createBatches(
        mouseSnapshot,
        400
    );

    const totalBatches = Math.max(
        keyboardBatches.length,
        mouseBatches.length,
        shortcutsSnapshot.length > 0 ? 1 : 0,
        1
    );

    let completedBatches = 0;

    try {
        for (let i = 0; i < totalBatches; i++) {
            const payloadBatch = {
                keyboard: keyboardBatches[i] || [],
                mouse: mouseBatches[i] || [],
                shortcuts: i === 0 ? shortcutsSnapshot : []
            };

            const response = await fetch('/api/behavior', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify(payloadBatch)
            });

            let data;

            try {
                data = await response.json();
            } catch {
                throw new Error(
                    `Resposta inválida do servidor (HTTP ${response.status}).`
                );
            }

            if (!response.ok) {
                throw new Error(
                    data.message ||
                    data.mensagem ||
                    `Erro HTTP ${response.status} ao enviar eventos.`
                );
            }

            completedBatches = i + 1;

            // Janela de reautenticação NÃO é aberta por inatividade.
            // A solicitação de senha é feita pela análise da IA.
            if (data.status === 'empty_window') {
                idleCycles++;
            } else {
                idleCycles = 0;
            }
        }
    } catch (error) {
        console.error(
            'Erro ao enviar dados comportamentais:',
            error
        );

        const keyboardSent = Math.min(
            completedBatches * 200,
            keyboardSnapshot.length
        );

        const mouseSent = Math.min(
            completedBatches * 400,
            mouseSnapshot.length
        );

        const shortcutsSent = completedBatches > 0;

        keyboardData = [
            ...keyboardSnapshot.slice(keyboardSent),
            ...keyboardData
        ];

        mouseData = [
            ...mouseSnapshot.slice(mouseSent),
            ...mouseData
        ];

        if (!shortcutsSent) {
            shortcutsData = [
                ...shortcutsSnapshot,
                ...shortcutsData
            ];
        }
    } finally {
        isSendingData = false;
    }
}

// ============================================
// Elementos da janela de reautenticação
// ============================================

const reauthModal = document.getElementById(
    'degradedTrustModal'
);

const reauthForm = document.getElementById('reauthForm');

const reauthPassword = document.getElementById(
    'reauthPassword'
);

const reauthSubmitBtn = document.getElementById(
    'reauthSubmitBtn'
);

const reauthMessage = document.getElementById(
    'reauthMessage'
);

function showReauthModal() {
    if (!reauthModal) return;

    reauthModal.style.display = 'flex';

    if (reauthMessage) {
        reauthMessage.textContent = '';
    }

    if (reauthPassword) {
        reauthPassword.value = '';
        reauthPassword.focus();
    }
}

// ============================================
// Reautenticação por senha
// ============================================

async function verifyReauth() {
    if (reauthInProgress) return;

    if (!reauthPassword || !reauthSubmitBtn) return;

    const password = reauthPassword.value;

    if (!password) {
        if (reauthMessage) {
            reauthMessage.textContent = 'Digite sua senha.';
        }

        reauthPassword.focus();
        return;
    }

    reauthInProgress = true;
    reauthSubmitBtn.disabled = true;

    if (reauthMessage) {
        reauthMessage.style.color = '#64748b';
        reauthMessage.textContent = 'Verificando senha...';
    }

    try {
        const response = await fetch('/api/reauth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                password
            })
        });

        let data;

        try {
            data = await response.json();
        } catch {
            data = {};
        }

        if (response.ok && data.status === 'success') {
            reauthPassword.value = '';

            if (reauthModal) {
                reauthModal.style.display = 'none';
            }

            if (reauthMessage) {
                reauthMessage.textContent = '';
            }

            idleCycles = 0;
            consecutiveSuspiciousResults = 0;

            // Intervalo antes de uma nova solicitação de senha.
            reauthCooldownUntil = Date.now() + 5 * 60 * 1000;

        } else {
            // Mantém a janela aberta se a senha estiver incorreta.
            reauthPassword.value = '';

            if (reauthMessage) {
                reauthMessage.style.color = '#b91c1c';
                reauthMessage.textContent =
                    data.message ||
                    data.mensagem ||
                    'Senha incorreta. Tente novamente.';
            }

            reauthPassword.focus();
        }
    } catch (error) {
        console.error('Erro ao reautenticar:', error);

        if (reauthMessage) {
            reauthMessage.style.color = '#b91c1c';
            reauthMessage.textContent =
                'Não foi possível verificar a senha. Tente novamente.';
        }
    } finally {
        reauthInProgress = false;
        reauthSubmitBtn.disabled = false;
    }
}

// O formulário também pode ser enviado com Enter.
if (reauthForm) {
    reauthForm.addEventListener('submit', (event) => {
        event.preventDefault();
        verifyReauth();
    });
}

// ============================================
// Verificação automática de identidade pela IA
// ============================================

async function verifyBehaviorAutomatically() {
    if (automaticVerificationInProgress) return;

    if (Date.now() < reauthCooldownUntil) return;

    // Não solicita outra senha enquanto a janela já está aberta.
    if (
        reauthModal &&
        reauthModal.style.display === 'flex'
    ) {
        return;
    }

    automaticVerificationInProgress = true;

    try {
        const response = await fetch('/api/verify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({})
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                `Resposta inválida do servidor (HTTP ${response.status}).`
            );
        }

        // Ainda não há dados suficientes para a análise.
        if (data.status === 'insufficient_data') {
            consecutiveSuspiciousResults = 0;
            return;
        }

        if (!response.ok) {
            console.error(
                'Erro na verificação automática:',
                data.message ||
                data.mensagem ||
                `HTTP ${response.status}`
            );

            return;
        }

        if (data.resultado === 'suspeito') {
            consecutiveSuspiciousResults++;

            console.warn(
                'Resultado suspeito consecutivo:',
                consecutiveSuspiciousResults
            );

            // Duas respostas suspeitas consecutivas são necessárias.
            if (consecutiveSuspiciousResults >= 2) {
                consecutiveSuspiciousResults = 0;
                showReauthModal();
            }
        } else {
            // Inconclusivo não é considerado suspeito.
            consecutiveSuspiciousResults = 0;
        }

    } catch (error) {
        console.error(
            'Erro na verificação automática de identidade:',
            error
        );
    } finally {
        automaticVerificationInProgress = false;
    }
}

// ============================================
// Inicialização
// ============================================

// Envia os eventos a cada 10 segundos.
setInterval(sendDataToBackend, 10000);

// Verifica a identidade periodicamente.
// A primeira verificação ocorre após 30 segundos, para dar tempo
// de acumular eventos comportamentais.
setTimeout(() => {
    verifyBehaviorAutomatically();

    setInterval(
        verifyBehaviorAutomatically,
        60000
    );
}, 30000);