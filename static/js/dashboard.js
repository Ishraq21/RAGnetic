// ============================================
// RAGnetic Dashboard — Agent Management Logic
// ============================================

let allAgents = [];
let confirmCallback = null;
let refreshInterval = null;
let isActionInProgress = false;  // Prevent race conditions during API calls

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    loadAgents();
    // Auto-refresh every 15 seconds
    refreshInterval = setInterval(() => {
        // Skip auto-refresh if modal is open or an action is in progress
        const modalOpen = document.getElementById('create-modal')?.classList.contains('open');
        const confirmOpen = document.getElementById('confirm-dialog')?.classList.contains('open');
        if (!modalOpen && !confirmOpen && !isActionInProgress) {
            loadAgents();
        }
    }, 15000);

    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeCreateModal();
            closeConfirmDialog();
        }
    });

    // Backdrop click handlers for modals
    document.getElementById('create-modal')?.addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeCreateModal();
    });
    document.getElementById('confirm-dialog')?.addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeConfirmDialog();
    });
});

// --- API Helpers ---
function apiHeaders() {
    const token = localStorage.getItem('ragnetic_user_token');
    if (!token) {
        window.location.href = '/login';
        return {};
    }
    return {
        'Content-Type': 'application/json',
        'X-API-Key': token
    };
}

function showToast(message, isError = false) {
    const toast = document.getElementById('dashboard-toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = 'dashboard-toast ' + (isError ? 'error' : 'success');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
        toast.className = 'dashboard-toast';
    }, 4000);
}

// --- Load & Render Agents ---
async function loadAgents() {
    const container = document.getElementById('agents-container');
    if (!container) return;

    try {
        const resp = await fetch(`${API_BASE_URL}/agents/status`, {
            headers: apiHeaders()
        });

        if (resp.status === 401 || resp.status === 403) {
            window.location.href = '/login';
            return;
        }

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        allAgents = await resp.json();
        renderAgents(allAgents);
    } catch (err) {
        console.error('Failed to load agents:', err);
        // Only show loading error on first load (when the loading spinner is present)
        if (container.querySelector('.agents-loading')) {
            container.innerHTML = `
                <div class="agents-empty">
                    <div class="empty-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
                    </div>
                    <h2>Could not load agents</h2>
                    <p>Check that the server is running and try again.</p>
                    <button class="btn-create-agent" onclick="loadAgents()">Retry</button>
                </div>
            `;
        }
    }
}

function renderAgents(agents) {
    const container = document.getElementById('agents-container');
    const countEl = document.getElementById('agent-count');
    if (!container || !countEl) return;

    countEl.textContent = agents.length > 0 ? `(${agents.length})` : '';

    if (agents.length === 0) {
        container.innerHTML = `
            <div class="agents-empty">
                <div class="empty-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                </div>
                <h2>No agents yet</h2>
                <p>Create your first AI agent to get started. Agents can chat, retrieve documents, search the web, and more.</p>
                <button class="btn-create-agent" onclick="openCreateModal()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                    Create Agent
                </button>
            </div>
        `;
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'agents-grid';

    agents.forEach(agent => {
        const status = resolveStatus(agent.status);
        const card = document.createElement('div');
        card.className = 'agent-card';
        card.setAttribute('data-agent-name', agent.name || '');

        // Build meta chips safely
        const modelChip = `
            <span class="meta-chip">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/></svg>
                ${escapeHTML(agent.model_name || 'unknown')}
            </span>`;

        const costChip = (agent.total_cost != null && agent.total_cost > 0) ? `
            <span class="meta-chip">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v12M8 10h8M9 14h6"/></svg>
                $${Number(agent.total_cost).toFixed(4)}
            </span>` : '';

        const timeChip = agent.last_run ? `
            <span class="meta-chip">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12,6 12,12 16,14"/></svg>
                ${formatTimeAgo(agent.last_run)}
            </span>` : '';

        // Use data attributes instead of inline onclick with string interpolation (safer than escapeAttr)
        card.innerHTML = `
            <div class="agent-card-header">
                <div>
                    <h3 class="agent-card-title">${escapeHTML(agent.display_name || agent.name)}</h3>
                    ${agent.display_name ? `<div class="agent-card-display-name">${escapeHTML(agent.name)}</div>` : ''}
                </div>
                <span class="status-badge ${status.class}">
                    <span class="badge-dot"></span>
                    ${status.label}
                </span>
            </div>
            <p class="agent-card-desc">${escapeHTML(agent.description || 'No description')}</p>
            <div class="agent-card-meta">
                ${modelChip}${costChip}${timeChip}
            </div>
            <div class="agent-card-actions">
                ${status.isOnline
                    ? `<button class="card-action-btn shutdown" data-action="shutdown" data-agent="${escapeAttr(agent.name)}">
                           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
                           Shutdown
                       </button>`
                    : `<button class="card-action-btn deploy" data-action="deploy" data-agent="${escapeAttr(agent.name)}">
                           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5,3 19,12 5,21"/></svg>
                           Deploy
                       </button>`
                }
                <button class="card-action-btn" data-action="chat">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    Chat
                </button>
                <button class="card-action-btn delete" data-action="delete" data-agent="${escapeAttr(agent.name)}">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3,6 5,6 21,6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        `;

        // Attach event listeners via delegation (prevents XSS from inline handlers)
        card.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = btn.getAttribute('data-action');
                const agentName = btn.getAttribute('data-agent');
                if (action === 'deploy')   deployAgent(agentName);
                if (action === 'shutdown') shutdownAgent(agentName);
                if (action === 'delete')   confirmDeleteAgent(agentName);
                if (action === 'chat')     window.location.href = '/';
            });
        });

        grid.appendChild(card);
    });

    container.innerHTML = '';
    container.appendChild(grid);
}

function resolveStatus(status) {
    const s = (status || '').toLowerCase();
    switch (s) {
        case 'deployed':
        case 'active':
        case 'online':
            return { class: 'online', label: 'Online', isOnline: true };
        case 'deploying':
        case 'ingesting':
            return { class: 'deploying', label: 'Deploying', isOnline: false };
        case 'created':
            return { class: 'created', label: 'Created', isOnline: false };
        case 'stopped':
        case 'offline':
        case 'error':
        case 'failed':
            return { class: 'stopped', label: s === 'error' || s === 'failed' ? 'Error' : 'Offline', isOnline: false };
        default:
            return { class: 'created', label: status || 'Unknown', isOnline: false };
    }
}

// --- Agent Actions ---
async function deployAgent(name) {
    if (isActionInProgress) return;
    isActionInProgress = true;
    try {
        showToast(`Deploying ${name}…`);
        const resp = await fetch(`${API_BASE_URL}/agents/${encodeURIComponent(name)}/deploy`, {
            method: 'POST',
            headers: apiHeaders()
        });
        if (resp.status === 401 || resp.status === 403) {
            window.location.href = '/login';
            return;
        }
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }
        showToast(`${name} deployment started.`);
        await loadAgents();
    } catch (err) {
        console.error('Deploy failed:', err);
        showToast(`Failed to deploy ${name}: ${err.message}`, true);
    } finally {
        isActionInProgress = false;
    }
}

async function shutdownAgent(name) {
    if (isActionInProgress) return;
    isActionInProgress = true;
    try {
        showToast(`Shutting down ${name}…`);
        const resp = await fetch(`${API_BASE_URL}/agents/${encodeURIComponent(name)}/shutdown`, {
            method: 'POST',
            headers: apiHeaders()
        });
        if (resp.status === 401 || resp.status === 403) {
            window.location.href = '/login';
            return;
        }
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }
        showToast(`${name} is now offline.`);
        await loadAgents();
    } catch (err) {
        console.error('Shutdown failed:', err);
        showToast(`Failed to shut down ${name}: ${err.message}`, true);
    } finally {
        isActionInProgress = false;
    }
}

function confirmDeleteAgent(name) {
    const confirmText = document.getElementById('confirm-text');
    if (confirmText) {
        confirmText.textContent = `Delete agent "${name}"? This cannot be undone.`;
    }
    confirmCallback = () => deleteAgent(name);
    document.getElementById('confirm-dialog')?.classList.add('open');
}

async function deleteAgent(name) {
    closeConfirmDialog();
    if (isActionInProgress) return;
    isActionInProgress = true;
    try {
        showToast(`Deleting ${name}…`);
        const resp = await fetch(`${API_BASE_URL}/agents/${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: apiHeaders()
        });
        if (resp.status === 401 || resp.status === 403) {
            window.location.href = '/login';
            return;
        }
        // 204 No Content is success for DELETE
        if (!resp.ok && resp.status !== 204) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }
        showToast(`${name} deleted.`);
        await loadAgents();
    } catch (err) {
        console.error('Delete failed:', err);
        showToast(`Failed to delete ${name}: ${err.message}`, true);
    } finally {
        isActionInProgress = false;
    }
}

function closeConfirmDialog() {
    document.getElementById('confirm-dialog')?.classList.remove('open');
    confirmCallback = null;
}

function executeConfirmedAction() {
    if (confirmCallback) confirmCallback();
}

// --- Create Agent Modal ---
function openCreateModal() {
    document.getElementById('create-modal')?.classList.add('open');
    // Use setTimeout to ensure the DOM has fully rendered the modal before focusing
    setTimeout(() => {
        document.getElementById('agent-name')?.focus();
    }, 100);
}

function closeCreateModal() {
    document.getElementById('create-modal')?.classList.remove('open');
    const form = document.getElementById('create-agent-form');
    if (form) form.reset();
    const sourcesList = document.getElementById('sources-list');
    if (sourcesList) sourcesList.innerHTML = '';
    // Reset submit button state in case it was left disabled
    const submitBtn = document.getElementById('create-submit-btn');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Agent';
    }
}

async function handleCreateAgent(event) {
    event.preventDefault();
    const submitBtn = document.getElementById('create-submit-btn');
    if (!submitBtn || submitBtn.disabled) return;  // Prevent double-submit
    submitBtn.disabled = true;
    submitBtn.textContent = 'Creating…';

    const name = (document.getElementById('agent-name')?.value || '').trim();
    const displayName = (document.getElementById('agent-display-name')?.value || '').trim();
    const description = (document.getElementById('agent-description')?.value || '').trim();
    const model = document.getElementById('agent-model')?.value || 'gpt-4o-mini';
    const embedding = document.getElementById('agent-embedding')?.value || 'text-embedding-3-small';
    const persona = (document.getElementById('agent-persona')?.value || '').trim();

    // Client-side validation
    if (!name) {
        showToast('Agent name is required.', true);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Agent';
        return;
    }

    if (!/^[a-zA-Z0-9_-]{3,50}$/.test(name)) {
        showToast('Agent name must be 3-50 characters: letters, numbers, _, -', true);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Agent';
        return;
    }

    // Gather tools
    const tools = [];
    document.querySelectorAll('#create-agent-form input[type="checkbox"]:checked').forEach(cb => {
        tools.push(cb.value);
    });

    // Gather sources
    const sources = [];
    document.querySelectorAll('.source-entry').forEach(entry => {
        const typeSelect = entry.querySelector('select');
        const valueInput = entry.querySelector('input');
        if (!typeSelect || !valueInput) return;

        const type = typeSelect.value;
        const value = valueInput.value.trim();
        if (value) {
            const source = { type };
            if (['url', 'web_crawler', 'api'].includes(type)) {
                source.url = value;
            } else if (type === 'db') {
                source.db_connection = value;
            } else {
                source.path = value;
            }
            sources.push(source);
        }
    });

    const payload = {
        name,
        display_name: displayName || null,
        description: description || null,
        llm_model: model,
        embedding_model: embedding,
        persona_prompt: persona || 'You are a helpful assistant.',
        tools: tools.length > 0 ? tools : [],
        sources: sources.length > 0 ? sources : []
    };

    try {
        isActionInProgress = true;
        const resp = await fetch(`${API_BASE_URL}/agents`, {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify(payload)
        });

        if (resp.status === 401 || resp.status === 403) {
            window.location.href = '/login';
            return;
        }

        if (resp.status === 409) {
            showToast(`Agent "${name}" already exists.`, true);
            return;
        }

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }

        showToast(`Agent "${name}" created!`);
        closeCreateModal();
        await loadAgents();
    } catch (err) {
        console.error('Create failed:', err);
        showToast(`Failed to create agent: ${err.message}`, true);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Create Agent';
        isActionInProgress = false;
    }
}

// --- Source Entries ---
function addSourceEntry() {
    const list = document.getElementById('sources-list');
    if (!list) return;
    const entry = document.createElement('div');
    entry.className = 'source-entry';
    entry.innerHTML = `
        <select>
            <option value="local">Local File</option>
            <option value="url">URL</option>
            <option value="pdf">PDF</option>
            <option value="csv">CSV</option>
            <option value="txt">Text</option>
            <option value="docx">DOCX</option>
            <option value="code_repository">Code Repo</option>
            <option value="web_crawler">Web Crawler</option>
            <option value="api">API</option>
            <option value="db">Database</option>
        </select>
        <input type="text" placeholder="Path or URL…" />
    `;

    // Create remove button with event listener (instead of inline onclick)
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'remove-source-btn';
    removeBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
    removeBtn.addEventListener('click', () => entry.remove());

    entry.appendChild(removeBtn);
    list.appendChild(entry);
}

// --- Logout ---
function handleLogout() {
    // Clear all localStorage keys to match chat interface behavior
    localStorage.removeItem('ragnetic_user_token');
    localStorage.removeItem('ragnetic_db_user_id');
    localStorage.removeItem('ragnetic_user_id');
    localStorage.removeItem('ragnetic_agent');
    localStorage.removeItem('ragnetic_thread_id');
    // Stop auto-refresh
    if (refreshInterval) clearInterval(refreshInterval);
    window.location.href = '/login';
}

// --- Utilities ---
function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function escapeAttr(str) {
    // Full attribute escaping — prevents XSS through HTML attribute injection
    return (str || '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function formatTimeAgo(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return '';  // Guard against invalid dates
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);
    if (diff < 0) return 'just now';  // Future dates
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 2592000) return `${Math.floor(diff / 86400)}d ago`;
    return date.toLocaleDateString();
}
