// Widgets do dashboard
// Cada widget aqui existe para gerar um tipo diferente de sinal comportamental (dwell time,
// trajetória de mouse, drag, hold time, scroll)

// Estado dos chamados: liga "Abrir Chamado" ao Kanban e ao seletor de encerramento
let nextTicketId = 1004; // os cartões de exemplo do HTML já usam 1001–1003
const FIRST_USER_TICKET_ID = 1004; // tudo com id >= isso foi criado pelo usuário no formulário
const MAX_USER_TICKETS = 4; // limite de quantos chamados a PESSOA pode criar (não conta os de exemplo)

function getOpenTickets() {
    if (!kanbanBoard) return [];
    return Array.from(kanbanBoard.querySelectorAll(".kanban-card")).map((card) => ({
        id: card.dataset.id,
        subject: card.querySelector(".kanban-card-subject")?.textContent.trim() || "",
    }));
}

function countUserCreatedOpenTickets() {
    return getOpenTickets().filter((t) => parseInt(t.id, 10) >= FIRST_USER_TICKET_ID).length;
}

function updateTicketLimitUI() {
    const warning = document.getElementById("ticketLimitWarning");
    const atLimit = countUserCreatedOpenTickets() >= MAX_USER_TICKETS;

    if (typeof saveRecordBtn !== "undefined" && saveRecordBtn) {
        saveRecordBtn.disabled = atLimit;
    }
    if (warning) {
        warning.style.display = atLimit ? "flex" : "none";
    }
}

function refreshCloseTicketSelect() {
    const select = document.getElementById("closeTicketSelect");
    if (!select) return;

    const tickets = getOpenTickets();
    const previousValue = select.value;

    if (tickets.length === 0) {
        select.innerHTML = `<option value="">Nenhum chamado aberto</option>`;
        select.disabled = true;
    } else {
        select.disabled = false;
        select.innerHTML = tickets
            .map((t) => `<option value="${t.id}">#${t.id} — ${t.subject}</option>`)
            .join("");
        if (tickets.some((t) => t.id === previousValue)) {
            select.value = previousValue;
        }
    }

    if (closeTicketBtn) {
        closeTicketBtn.disabled = tickets.length === 0;
    }

    updateTicketLimitUI();
}

function createKanbanCard(subject) {
    if (!kanbanBoard) return;
    const openDropzone = kanbanBoard.querySelector('.kanban-column[data-column="aberto"] .kanban-dropzone');
    if (!openDropzone) return;

    const id = nextTicketId++;
    const card = document.createElement("div");
    card.className = "kanban-card";
    card.dataset.id = String(id);
    card.innerHTML = `<strong>#${id}</strong> — <span class="kanban-card-subject">${escapeHtml(subject)}</span>`;
    openDropzone.appendChild(card);
    refreshCloseTicketSelect();
}

function removeKanbanCard(ticketId) {
    if (!kanbanBoard || !ticketId) return;
    const card = kanbanBoard.querySelector(`.kanban-card[data-id="${ticketId}"]`);
    if (card) card.remove();
    refreshCloseTicketSelect();
}

// Depois de encerrar um chamado, limpa avaliação e assinatura
function resetCloseForm() {
    selectedRating = 0;
    hasSignature = false;

    document.querySelectorAll("#starRating i").forEach((star) => {
        star.classList.remove("fa-solid");
        star.classList.add("fa-regular");
    });

    const canvas = document.getElementById("signatureCanvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// "Abrir Chamado": valida o formulário, cria um novo cartão em "Aberto" e limpa os campos depois 
if (typeof saveRecordBtn !== "undefined" && saveRecordBtn) {
    saveRecordBtn.addEventListener("click", () => {
        if (countUserCreatedOpenTickets() >= MAX_USER_TICKETS) {
            updateTicketLimitUI();
            return; // limite atingido, o botão já fica desabilitado e o aviso visível
        }

        const nameInput = document.getElementById("clientNameInput");
        const phoneInputEl = document.getElementById("phoneInput");
        const emailInputEl = document.getElementById("emailInput");
        const summaryInput = document.getElementById("ticketSummaryInput");
        const descriptionInput = document.getElementById("ticketDescriptionInput");

        const fields = [nameInput, phoneInputEl, emailInputEl, summaryInput, descriptionInput];
        const allFilled = fields.every((el) => el && el.value.trim().length > 0);

        if (!allFilled) {
            const flashContainer = document.getElementById("dashboardFlashContainer");
            if (typeof showFlashMessage === "function" && flashContainer) {
                showFlashMessage(flashContainer, "Preencha todos os campos antes de abrir o chamado.", "error");
            }
            return;
        }

        let subject = summaryInput.value.trim();
        if (subject.length > 42) subject = subject.slice(0, 39) + "...";

        createKanbanCard(subject);

        // Limpa o formulário depois de registrar o chamado
        fields.forEach((el) => { if (el) el.value = ""; });
        if (priorityRange) priorityRange.value = "5";
        if (priorityValue) priorityValue.textContent = "5";

        const flashContainer = document.getElementById("dashboardFlashContainer");
        if (typeof showFlashMessage === "function" && flashContainer) {
            showFlashMessage(
                flashContainer,
                "Chamado aberto com sucesso!",
                "success"
            );
        }
    });
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Máscara de telefone
const phoneInput = document.getElementById("phoneInput");
if (phoneInput) {
    phoneInput.addEventListener("input", () => {
        let digits = phoneInput.value.replace(/\D/g, "").slice(0, 11);
        if (digits.length > 6) {
            phoneInput.value = `(${digits.slice(0, 2)}) ${digits.slice(2, 7)}-${digits.slice(7)}`;
        } else if (digits.length > 2) {
            phoneInput.value = `(${digits.slice(0, 2)}) ${digits.slice(2)}`;
        } else {
            phoneInput.value = digits;
        }
    });
}

// Slider de prioridade
const priorityRange = document.getElementById("priorityRange");
const priorityValue = document.getElementById("priorityValue");
if (priorityRange && priorityValue) {
    priorityRange.addEventListener("input", () => {
        priorityValue.textContent = priorityRange.value;
    });
}

// Avaliação por estrelas
let selectedRating = 0; 
const starRating = document.getElementById("starRating");
if (starRating) {
    const stars = Array.from(starRating.querySelectorAll("i"));

    function paintStars(value) {
        stars.forEach((star) => {
            const active = parseInt(star.dataset.value, 10) <= value;
            star.classList.toggle("fa-solid", active);
            star.classList.toggle("fa-regular", !active);
        });
    }

    stars.forEach((star) => {
        star.addEventListener("mouseenter", () => paintStars(parseInt(star.dataset.value, 10)));
        star.addEventListener("click", () => {
            selectedRating = parseInt(star.dataset.value, 10);
            paintStars(selectedRating);
        });
    });
    starRating.addEventListener("mouseleave", () => paintStars(selectedRating));
}

// Quadro Kanban com arraste customizado (mouse-based)
// Implementado com mousedown/mousemove/mouseup em vez de HTML5 Drag and Drop nativo de propósito: o DnD nativo 
// do navegador NÃO dispara "mousemove" durante o arraste, o que apagaria justamente o sinal que queremos capturar
const kanbanBoard = document.getElementById("kanbanBoard");
if (kanbanBoard) {
    let draggedCard = null;
    let offsetX = 0;
    let offsetY = 0;

    kanbanBoard.addEventListener("mousedown", (e) => {
        const card = e.target.closest(".kanban-card");
        if (!card) return;

        draggedCard = card;
        const rect = card.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;

        card.classList.add("dragging");
        card.style.width = `${rect.width}px`;
        document.body.appendChild(card);
        card.style.position = "fixed";
        card.style.left = `${rect.left}px`;
        card.style.top = `${rect.top}px`;
        card.style.zIndex = "1000";
    });

    document.addEventListener("mousemove", (e) => {
        if (!draggedCard) return;
        draggedCard.style.left = `${e.clientX - offsetX}px`;
        draggedCard.style.top = `${e.clientY - offsetY}px`;
    });

    document.addEventListener("mouseup", (e) => {
        if (!draggedCard) return;

        draggedCard.classList.remove("dragging");
        draggedCard.style.position = "";
        draggedCard.style.left = "";
        draggedCard.style.top = "";
        draggedCard.style.width = "";
        draggedCard.style.zIndex = "";

        const dropzone = document
            .elementsFromPoint(e.clientX, e.clientY)
            .find((el) => el.classList.contains("kanban-dropzone"));

        if (dropzone) {
            dropzone.appendChild(draggedCard);
        } else {
            // Se soltou fora de uma coluna, devolve pra coluna original
            const originalColumn = kanbanBoard.querySelector(".kanban-dropzone");
            if (originalColumn) originalColumn.appendChild(draggedCard);
        }

        draggedCard = null;
    });
}

// Assinatura em canvas
let hasSignature = false; 
const signatureCanvas = document.getElementById("signatureCanvas");
if (signatureCanvas) {
    const ctx = signatureCanvas.getContext("2d");
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    let signing = false;

    function getCanvasPos(e) {
        const rect = signatureCanvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    signatureCanvas.addEventListener("mousedown", (e) => {
        signing = true;
        hasSignature = true;
        const { x, y } = getCanvasPos(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
    });

    signatureCanvas.addEventListener("mousemove", (e) => {
        if (!signing) return;
        const { x, y } = getCanvasPos(e);
        ctx.lineTo(x, y);
        ctx.stroke();
    });

    document.addEventListener("mouseup", () => {
        signing = false;
    });

    const clearBtn = document.getElementById("clearSignatureBtn");
    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            ctx.clearRect(0, 0, signatureCanvas.width, signatureCanvas.height);
            hasSignature = false;
        });
    }
}

// Botão "segure para confirmar" (hold time) 
const closeTicketBtn = document.getElementById("closeTicketBtn");
if (closeTicketBtn) {
    const HOLD_DURATION_MS = 1200;
    let holdTimer = null;
    let holdStart = null;

    function startHold() {
        const select = document.getElementById("closeTicketSelect");
        if (!select || !select.value) return; // nada selecionado pra encerrar

        // Só deixa encerrar depois que a pessoa avaliou (estrelas) e assinou
        if (selectedRating === 0 || !hasSignature) {
            const flashContainer = document.getElementById("closeFlashContainer");
            if (typeof showFlashMessage === "function" && flashContainer) {
                showFlashMessage(flashContainer, "Avalie o atendimento e assine antes de encerrar.", "error");
            }
            return;
        }

        holdStart = Date.now();
        closeTicketBtn.classList.add("holding");
        holdTimer = setTimeout(() => {
            closeTicketBtn.classList.remove("holding");
            closeTicketBtn.classList.add("confirmed");

            removeKanbanCard(select.value);
            resetCloseForm();

            const flashContainer = document.getElementById("closeFlashContainer");
            if (typeof showFlashMessage === "function" && flashContainer) {
                showFlashMessage(flashContainer, "Atendimento encerrado com sucesso!", "success");
            }
            setTimeout(() => closeTicketBtn.classList.remove("confirmed"), 1500);
        }, HOLD_DURATION_MS);
    }

    function cancelHold() {
        clearTimeout(holdTimer);
        closeTicketBtn.classList.remove("holding");
    }

    closeTicketBtn.addEventListener("mousedown", startHold);
    closeTicketBtn.addEventListener("mouseup", cancelHold);
    closeTicketBtn.addEventListener("mouseleave", cancelHold);
}

// Popula o seletor de encerramento com os chamados que já vêm no quadro
refreshCloseTicketSelect();
