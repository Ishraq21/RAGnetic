// ============================================
// RAGnetic Dashboard — Full Feature Dashboard
// ============================================

let allAgents = [];
let confirmCallback = null;
let refreshInterval = null;
let isActionInProgress = false;
let currentTab = 'agents';
let panelLoaded = { agents: true };
let currentUserData = null;  // Stores /me response for role checks

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Tab navigation
    document.querySelectorAll('.sidebar-nav-item[data-tab]').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(item.getAttribute('data-tab'));
        });
    });

    // Load current user and apply role-based sidebar visibility
    applyRoleBasedVisibility();

    // Load initial tab
    loadAgents();

    // Auto-refresh every 15 seconds (only active panel)
    refreshInterval = setInterval(() => {
        const modalOpen = document.querySelector('.modal-overlay.open');
        const confirmOpen = document.getElementById('confirm-dialog')?.classList.contains('open');
        if (!modalOpen && !confirmOpen && !isActionInProgress) {
            refreshActivePanel();
        }
    }, 15000);

    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeCreateModal();
            closeConfirmDialog();
            document.querySelectorAll('.modal-overlay.open').forEach(m => m.classList.remove('open'));
        }
    });

    // Backdrop click handlers for all modals
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === e.currentTarget) overlay.classList.remove('open');
        });
    });
    document.getElementById('confirm-dialog')?.addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeConfirmDialog();
    });
});

// --- Tab Navigation ---
function switchTab(tabName) {
    currentTab = tabName;

    // Update sidebar active state
    document.querySelectorAll('.sidebar-nav-item[data-tab]').forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-tab') === tabName);
    });

    // Show/hide panels
    document.querySelectorAll('.panel-section').forEach(section => {
        section.classList.toggle('active', section.id === `panel-${tabName}`);
    });

    // Load panel data on first visit
    if (!panelLoaded[tabName]) {
        panelLoaded[tabName] = true;
        refreshActivePanel();
    }
}

function refreshActivePanel() {
    switch (currentTab) {
        case 'agents': loadAgents(); break;
        case 'analytics': loadAnalytics(); break;
        case 'benchmarks': loadBenchmarks(); break;
        case 'monitoring': loadMonitoring(); break;
        case 'users': loadUsers(); break;
    }
}

// --- Role-based Sidebar Visibility ---
async function applyRoleBasedVisibility() {
    try {
        const resp = await apiFetch(`${API_BASE_URL}/security/me`);
        if (!resp || !resp.ok) return;
        currentUserData = await resp.json();

        const isAdmin = currentUserData.is_superuser ||
            (currentUserData.roles || []).some(r => r.name === 'admin');

        if (!isAdmin) {
            document.querySelectorAll('.sidebar-nav-item[data-admin-only]').forEach(el => {
                el.style.display = 'none';
            });
        }
    } catch (e) {
        console.error('Failed to load user role info:', e);
    }
}

// --- API Helpers ---
function apiHeaders() {
    const token = localStorage.getItem('ragnetic_user_token');
    if (!token) { window.location.href = '/login'; return {}; }
    return { 'Content-Type': 'application/json', 'X-API-Key': token };
}

function apiHeadersMultipart() {
    const token = localStorage.getItem('ragnetic_user_token');
    if (!token) { window.location.href = '/login'; return {}; }
    return { 'X-API-Key': token };
}

function showToast(message, isError = false) {
    const toast = document.getElementById('dashboard-toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = 'dashboard-toast ' + (isError ? 'error' : 'success');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { toast.className = 'dashboard-toast'; }, 4000);
}

async function apiFetch(url, options = {}) {
    // Merge headers properly — options.headers should not override auth headers
    const mergedHeaders = { ...apiHeaders(), ...(options.headers || {}) };
    const mergedOptions = { ...options, headers: mergedHeaders };
    const resp = await fetch(url, mergedOptions);
    if (resp.status === 401 || resp.status === 403) { window.location.href = '/login'; return null; }
    return resp;
}

function closeModal(id) {
    document.getElementById(id)?.classList.remove('open');
}

// ============================================
// AGENTS PANEL
// ============================================
async function loadAgents() {
    const container = document.getElementById('agents-container');
    if (!container) return;
    try {
        const resp = await apiFetch(`${API_BASE_URL}/agents/status`);
        if (!resp || !resp.ok) throw new Error(`HTTP ${resp?.status}`);
        allAgents = await resp.json();
        renderAgents(allAgents);
    } catch (err) {
        console.error('Failed to load agents:', err);
        if (container.querySelector('.agents-loading')) {
            container.innerHTML = `<div class="agents-empty"><div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg></div><h2>Could not load agents</h2><p>Check that the server is running and try again.</p><button class="btn-create-agent" onclick="loadAgents()">Retry</button></div>`;
        }
    }
}

function renderAgents(agents) {
    const container = document.getElementById('agents-container');
    const countEl = document.getElementById('agent-count');
    if (!container || !countEl) return;
    countEl.textContent = agents.length > 0 ? `(${agents.length})` : '';

    if (agents.length === 0) {
        container.innerHTML = `<div class="agents-empty"><div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg></div><h2>No agents yet</h2><p>Create your first AI agent to get started.</p><button class="btn-create-agent" onclick="openCreateModal()"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>Create Agent</button></div>`;
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'agents-grid';
    agents.forEach(agent => {
        const status = resolveStatus(agent.status);
        const card = document.createElement('div');
        card.className = 'agent-card';
        const modelChip = `<span class="meta-chip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/></svg>${escapeHTML(agent.model_name || 'unknown')}</span>`;
        const costChip = (agent.total_cost != null && agent.total_cost > 0) ? `<span class="meta-chip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v12M8 10h8M9 14h6"/></svg>$${Number(agent.total_cost).toFixed(4)}</span>` : '';
        const timeChip = agent.last_run ? `<span class="meta-chip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12,6 12,12 16,14"/></svg>${formatTimeAgo(agent.last_run)}</span>` : '';

        card.innerHTML = `
            <div class="agent-card-header"><div><h3 class="agent-card-title">${escapeHTML(agent.display_name || agent.name)}</h3>${agent.display_name ? `<div class="agent-card-display-name">${escapeHTML(agent.name)}</div>` : ''}</div><span class="status-badge ${status.class}"><span class="badge-dot"></span>${status.label}</span></div>
            <p class="agent-card-desc">${escapeHTML(agent.description || 'No description')}</p>
            <div class="agent-card-meta">${modelChip}${costChip}${timeChip}</div>
            <div class="agent-card-actions">
                ${status.isOnline
                    ? `<button class="card-action-btn shutdown" data-action="shutdown" data-agent="${escapeAttr(agent.name)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>Shutdown</button>`
                    : `<button class="card-action-btn deploy" data-action="deploy" data-agent="${escapeAttr(agent.name)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5,3 19,12 5,21"/></svg>Deploy</button>`}
                <button class="card-action-btn" data-action="chat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>Chat</button>
                <button class="card-action-btn delete" data-action="delete" data-agent="${escapeAttr(agent.name)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3,6 5,6 21,6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button>
            </div>`;

        card.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = btn.getAttribute('data-action');
                const agentName = btn.getAttribute('data-agent');
                if (action === 'deploy') deployAgent(agentName);
                if (action === 'shutdown') shutdownAgent(agentName);
                if (action === 'delete') confirmDeleteAgent(agentName);
                if (action === 'chat') window.location.href = '/';
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
        case 'deployed': case 'active': case 'online': return { class: 'online', label: 'Online', isOnline: true };
        case 'deploying': case 'ingesting': return { class: 'deploying', label: 'Deploying', isOnline: false };
        case 'created': return { class: 'created', label: 'Created', isOnline: false };
        case 'stopped': case 'offline': case 'error': case 'failed': return { class: 'stopped', label: s === 'error' || s === 'failed' ? 'Error' : 'Offline', isOnline: false };
        default: return { class: 'created', label: status || 'Unknown', isOnline: false };
    }
}

async function deployAgent(name) {
    if (isActionInProgress) return;
    isActionInProgress = true;
    try {
        showToast(`Deploying ${name}…`);
        const resp = await apiFetch(`${API_BASE_URL}/agents/${encodeURIComponent(name)}/deploy`, { method: 'POST' });
        if (!resp || !resp.ok) { const err = await resp?.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp?.status}`); }
        showToast(`${name} deployment started.`);
        await loadAgents();
    } catch (err) { showToast(`Failed to deploy ${name}: ${err.message}`, true); }
    finally { isActionInProgress = false; }
}

async function shutdownAgent(name) {
    if (isActionInProgress) return;
    isActionInProgress = true;
    try {
        showToast(`Shutting down ${name}…`);
        const resp = await apiFetch(`${API_BASE_URL}/agents/${encodeURIComponent(name)}/shutdown`, { method: 'POST' });
        if (!resp || !resp.ok) { const err = await resp?.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp?.status}`); }
        showToast(`${name} is now offline.`);
        await loadAgents();
    } catch (err) { showToast(`Failed to shut down ${name}: ${err.message}`, true); }
    finally { isActionInProgress = false; }
}

function confirmDeleteAgent(name) {
    document.getElementById('confirm-text').textContent = `Delete agent "${name}"? This cannot be undone.`;
    confirmCallback = () => deleteAgent(name);
    document.getElementById('confirm-dialog')?.classList.add('open');
}

async function deleteAgent(name) {
    closeConfirmDialog();
    if (isActionInProgress) return;
    isActionInProgress = true;
    try {
        showToast(`Deleting ${name}…`);
        const resp = await apiFetch(`${API_BASE_URL}/agents/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (resp && !resp.ok && resp.status !== 204) { const err = await resp.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp.status}`); }
        showToast(`${name} deleted.`);
        await loadAgents();
    } catch (err) { showToast(`Failed to delete ${name}: ${err.message}`, true); }
    finally { isActionInProgress = false; }
}

function closeConfirmDialog() { document.getElementById('confirm-dialog')?.classList.remove('open'); confirmCallback = null; }
function executeConfirmedAction() { if (confirmCallback) confirmCallback(); }

// --- Create Agent Modal ---
function openCreateModal() { document.getElementById('create-modal')?.classList.add('open'); setTimeout(() => document.getElementById('agent-name')?.focus(), 100); }
function closeCreateModal() {
    document.getElementById('create-modal')?.classList.remove('open');
    document.getElementById('create-agent-form')?.reset();
    const sl = document.getElementById('sources-list'); if (sl) sl.innerHTML = '';
    const btn = document.getElementById('create-submit-btn'); if (btn) { btn.disabled = false; btn.textContent = 'Create Agent'; }
}

async function handleCreateAgent(event) {
    event.preventDefault();
    const submitBtn = document.getElementById('create-submit-btn');
    if (!submitBtn || submitBtn.disabled) return;
    submitBtn.disabled = true; submitBtn.textContent = 'Creating…';

    const name = (document.getElementById('agent-name')?.value || '').trim();
    const displayName = (document.getElementById('agent-display-name')?.value || '').trim();
    const description = (document.getElementById('agent-description')?.value || '').trim();
    const model = document.getElementById('agent-model')?.value || 'gpt-4o-mini';
    const embedding = document.getElementById('agent-embedding')?.value || 'text-embedding-3-small';
    const persona = (document.getElementById('agent-persona')?.value || '').trim();

    if (!name || !/^[a-zA-Z0-9_-]{3,50}$/.test(name)) { showToast('Invalid agent name.', true); submitBtn.disabled = false; submitBtn.textContent = 'Create Agent'; return; }

    const tools = []; document.querySelectorAll('#create-agent-form input[type="checkbox"]:checked').forEach(cb => tools.push(cb.value));
    const sources = []; document.querySelectorAll('.source-entry').forEach(entry => {
        const t = entry.querySelector('select')?.value, v = entry.querySelector('input')?.value.trim();
        if (v) { const s = { type: t }; ['url','web_crawler','api'].includes(t) ? s.url = v : t === 'db' ? s.db_connection = v : s.path = v; sources.push(s); }
    });

    const payload = { name, display_name: displayName || null, description: description || null, llm_model: model, embedding_model: embedding, persona_prompt: persona || 'You are a helpful assistant.', tools, sources };
    try {
        isActionInProgress = true;
        const resp = await apiFetch(`${API_BASE_URL}/agents`, { method: 'POST', body: JSON.stringify(payload) });
        if (!resp) return;
        if (resp.status === 409) { showToast(`Agent "${name}" already exists.`, true); return; }
        if (!resp.ok) { const err = await resp.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp.status}`); }
        showToast(`Agent "${name}" created!`); closeCreateModal(); await loadAgents();
    } catch (err) { showToast(`Failed to create agent: ${err.message}`, true); }
    finally { submitBtn.disabled = false; submitBtn.textContent = 'Create Agent'; isActionInProgress = false; }
}

function addSourceEntry() {
    const list = document.getElementById('sources-list'); if (!list) return;
    const entry = document.createElement('div'); entry.className = 'source-entry';
    entry.innerHTML = `<select><option value="local">Local File</option><option value="url">URL</option><option value="pdf">PDF</option><option value="csv">CSV</option><option value="txt">Text</option><option value="docx">DOCX</option><option value="code_repository">Code Repo</option><option value="web_crawler">Web Crawler</option><option value="api">API</option><option value="db">Database</option></select><input type="text" placeholder="Path or URL…" />`;
    const removeBtn = document.createElement('button'); removeBtn.type = 'button'; removeBtn.className = 'remove-source-btn';
    removeBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
    removeBtn.addEventListener('click', () => entry.remove());
    entry.appendChild(removeBtn); list.appendChild(entry);
}

// ============================================
// ANALYTICS PANEL
// ============================================
async function loadAnalytics() {
    let latResp, usageResp, runsResp, stepsResp;
    try {
        [latResp, usageResp, runsResp, stepsResp] = await Promise.all([
            apiFetch(`${API_BASE_URL}/analytics/latency`).catch(() => null),
            apiFetch(`${API_BASE_URL}/analytics/usage-summary?limit=20`).catch(() => null),
            apiFetch(`${API_BASE_URL}/analytics/agent-runs?limit=20`).catch(() => null),
            apiFetch(`${API_BASE_URL}/analytics/agent-steps?limit=20`).catch(() => null),
        ]);
    } catch (e) { console.error('Analytics fetch error:', e); return; }

    // Latency cards
    try {
        if (latResp && latResp.ok) {
            const d = await latResp.json();
            setText('lat-p50', d.p50_latency_s != null ? `${d.p50_latency_s.toFixed(2)}s` : '—');
            setText('lat-p95', d.p95_latency_s != null ? `${d.p95_latency_s.toFixed(2)}s` : '—');
            setText('lat-p99', d.p99_latency_s != null ? `${d.p99_latency_s.toFixed(2)}s` : '—');
            setText('lat-avg', d.avg_latency_s != null ? `${d.avg_latency_s.toFixed(2)}s` : '—');
        }
    } catch (e) { console.error('Latency load error:', e); }

    // Usage table — null-safe field access
    try {
        if (usageResp && usageResp.ok) {
            const data = await usageResp.json();
            renderTable('usage-table', 'usage-empty', data, row => `
                <td>${escapeHTML(row.agent_name || '—')}</td>
                <td><span class="meta-chip">${escapeHTML(row.llm_model || '—')}</span></td>
                <td>${escapeHTML(row.user_id || '—')}</td>
                <td>${(row.total_requests ?? 0).toLocaleString()}</td>
                <td>${(row.total_tokens ?? 0).toLocaleString()}</td>
                <td>$${(row.total_estimated_cost_usd ?? 0).toFixed(4)}</td>
                <td>${(row.avg_generation_time_s ?? 0).toFixed(2)}s</td>`);
        }
    } catch (e) { console.error('Usage load error:', e); }

    // Runs table — null-safe
    try {
        if (runsResp && runsResp.ok) {
            const data = await runsResp.json();
            renderTable('runs-table', 'runs-empty', data, row => `
                <td>${escapeHTML(row.agent_name || '—')}</td>
                <td>${row.total_runs ?? 0}</td>
                <td>${((row.success_rate ?? 0) * 100).toFixed(1)}%</td>
                <td>${row.failed_runs ?? 0}</td>
                <td>${(row.avg_duration_s ?? 0).toFixed(2)}s</td>`);
        }
    } catch (e) { console.error('Runs load error:', e); }

    // Steps table — null-safe
    try {
        if (stepsResp && stepsResp.ok) {
            const data = await stepsResp.json();
            renderTable('steps-table', 'steps-empty', data, row => `
                <td>${escapeHTML(row.agent_name || '—')}</td>
                <td><span class="meta-chip">${escapeHTML(row.node_name || '—')}</span></td>
                <td>${row.total_calls ?? 0}</td>
                <td>${((row.success_rate ?? 0) * 100).toFixed(1)}%</td>
                <td>${row.failed_calls ?? 0}</td>
                <td>${(row.avg_duration_s ?? 0).toFixed(2)}s</td>`);
        }
    } catch (e) { console.error('Steps load error:', e); }
}

// ============================================
// TRAINING PANEL
// ============================================
async function loadTraining() {
    let statsResp, jobsResp, datasetsResp, modelsResp;
    try {
        [statsResp, jobsResp, datasetsResp, modelsResp] = await Promise.all([
            apiFetch(`${API_BASE_URL}/training/stats`).catch(() => null),
            apiFetch(`${API_BASE_URL}/training/jobs`).catch(() => null),
            apiFetch(`${API_BASE_URL}/training/uploaded-datasets`).catch(() => null),
            apiFetch(`${API_BASE_URL}/training/models`).catch(() => null),
        ]);
    } catch (e) { console.error('Training fetch error:', e); return; }

    // Stats cards
    try {
        if (statsResp && statsResp.ok) {
            const d = await statsResp.json();
            setText('train-running', d.running_jobs ?? 0);
            setText('train-models', d.completed_models ?? 0);
            setText('train-datasets', d.total_datasets ?? 0);
            setText('train-configs', d.total_configs ?? 0);
        }
    } catch (e) { console.error('Training stats error:', e); }

    // Jobs table
    try {
        if (jobsResp && jobsResp.ok) {
            const data = await jobsResp.json();
            const tbody = document.querySelector('#training-jobs-table tbody');
            const empty = document.getElementById('training-jobs-empty');
            if (data.length === 0) { tbody.innerHTML = ''; empty?.classList.add('visible'); }
            else {
                empty?.classList.remove('visible');
                tbody.innerHTML = data.map(job => `<tr>
                    <td>${escapeHTML(job.job_name || job.adapter_id)}</td>
                    <td><span class="meta-chip">${escapeHTML(job.base_model_name)}</span></td>
                    <td><span class="badge ${job.training_status}">${job.training_status}</span></td>
                    <td>${formatTimeAgo(job.created_at)}</td>
                    <td>${job.training_status === 'running' || job.training_status === 'pending' ? `<button class="table-action-btn danger" onclick="cancelTrainingJob('${escapeAttr(job.adapter_id)}')">Cancel</button>` : '—'}</td>
                </tr>`).join('');
            }
        }
    } catch (e) { console.error('Training jobs error:', e); }

    // Datasets
    try {
        if (datasetsResp && datasetsResp.ok) {
            const data = await datasetsResp.json();
            const tbody = document.querySelector('#datasets-table tbody');
            const empty = document.getElementById('datasets-empty');
            if (!data || data.length === 0) { if (tbody) tbody.innerHTML = ''; empty?.classList.add('visible'); }
            else {
                empty?.classList.remove('visible');
                tbody.innerHTML = data.map(ds => `<tr>
                    <td>${escapeHTML(ds.filename || ds.file_id || 'Unknown')}</td>
                    <td>${ds.size ? formatBytes(ds.size) : (ds.size_bytes ? formatBytes(ds.size_bytes) : '—')}</td>
                    <td>${ds.created_at ? formatTimeAgo(new Date(ds.created_at * 1000).toISOString()) : '—'}</td>
                    <td><button class="table-action-btn danger" onclick="deleteDataset('${escapeAttr(ds.file_id || ds.filename)}')">Delete</button></td>
                </tr>`).join('');
            }
        }
    } catch (e) { console.error('Datasets error:', e); }

    // Models
    try {
        if (modelsResp && modelsResp.ok) {
            const data = await modelsResp.json();
            renderTable('models-table', 'models-empty', data, row => `
                <td><span class="meta-chip">${escapeHTML(row.adapter_id)}</span></td>
                <td>${escapeHTML(row.base_model_name)}</td>
                <td><span class="badge ${row.training_status}">${row.training_status}</span></td>
                <td>${formatTimeAgo(row.created_at)}</td>`);
        }
    } catch (e) { console.error('Models error:', e); }
}

function openTrainingModal() { document.getElementById('training-modal')?.classList.add('open'); }

async function handleStartTraining(event) {
    event.preventDefault();
    const btn = document.getElementById('training-submit-btn');
    if (!btn || btn.disabled) return;
    btn.disabled = true; btn.textContent = 'Starting…';

    const payload = {
        job_name: document.getElementById('train-job-name')?.value.trim(),
        base_model_name: document.getElementById('train-base-model')?.value.trim(),
        dataset_path: document.getElementById('train-dataset')?.value.trim(),
        hyperparameters: {
            epochs: parseInt(document.getElementById('train-epochs')?.value) || 3,
            learning_rate: parseFloat(document.getElementById('train-lr')?.value) || 2e-4,
            batch_size: parseInt(document.getElementById('train-batch')?.value) || 4,
        },
    };

    try {
        const resp = await apiFetch(`${API_BASE_URL}/training/apply`, { method: 'POST', body: JSON.stringify(payload) });
        if (!resp || !resp.ok) { const err = await resp?.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp?.status}`); }
        showToast('Training job started!'); closeModal('training-modal');
        document.getElementById('training-form')?.reset();
        await loadTraining();
    } catch (err) { showToast(`Failed to start training: ${err.message}`, true); }
    finally { btn.disabled = false; btn.textContent = 'Start Training'; }
}

function openUploadDatasetModal() { document.getElementById('upload-dataset-modal')?.classList.add('open'); }

async function handleUploadDataset(event) {
    event.preventDefault();
    const btn = document.getElementById('upload-dataset-btn');
    if (!btn || btn.disabled) return;
    btn.disabled = true; btn.textContent = 'Uploading…';

    const fileInput = document.getElementById('dataset-file');
    const file = fileInput?.files[0];
    if (!file) { showToast('Please select a file.', true); btn.disabled = false; btn.textContent = 'Upload'; return; }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch(`${API_BASE_URL}/training/upload-dataset`, { method: 'POST', headers: apiHeadersMultipart(), body: formData });
        if (resp.status === 401 || resp.status === 403) { window.location.href = '/login'; return; }
        if (!resp.ok) { const err = await resp.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp.status}`); }
        showToast('Dataset uploaded!'); closeModal('upload-dataset-modal');
        document.getElementById('upload-dataset-form')?.reset();
        await loadTraining();
    } catch (err) { showToast(`Upload failed: ${err.message}`, true); }
    finally { btn.disabled = false; btn.textContent = 'Upload'; }
}

async function cancelTrainingJob(adapterId) {
    try {
        const resp = await apiFetch(`${API_BASE_URL}/training/jobs/${encodeURIComponent(adapterId)}/cancel`, { method: 'POST' });
        if (!resp || !resp.ok) throw new Error('Cancel failed');
        showToast('Training job cancelled.'); await loadTraining();
    } catch (err) { showToast(`Failed to cancel job: ${err.message}`, true); }
}

async function deleteDataset(fileId) {
    try {
        const resp = await apiFetch(`${API_BASE_URL}/training/uploaded-datasets/${encodeURIComponent(fileId)}`, { method: 'DELETE' });
        if (!resp || (!resp.ok && resp.status !== 204)) throw new Error('Delete failed');
        showToast('Dataset deleted.'); await loadTraining();
    } catch (err) { showToast(`Failed to delete dataset: ${err.message}`, true); }
}

// ============================================
// BENCHMARKS PANEL
// ============================================
async function loadBenchmarks() {
    let summaryResp, runsResp, testsetsResp;
    try {
        [summaryResp, runsResp, testsetsResp] = await Promise.all([
            apiFetch(`${API_BASE_URL}/analytics/benchmarks`).catch(() => null),
            apiFetch(`${API_BASE_URL}/analytics/benchmarks/runs`).catch(() => null),
            apiFetch(`${API_BASE_URL}/evaluation/test-sets`).catch(() => null),
        ]);
    } catch (e) { console.error('Benchmarks fetch error:', e); return; }

    // Summary cards
    try {
        const container = document.getElementById('benchmark-summary-cards');
        const empty = document.getElementById('benchmark-summary-empty');
        if (summaryResp && summaryResp.ok) {
            const data = await summaryResp.json();
            if (data.length === 0) { container.innerHTML = ''; empty?.classList.add('visible'); }
            else {
                empty?.classList.remove('visible');
                container.innerHTML = data.map(b => {
                    // Handle both Pydantic alias keys and Python field name keys
                    const recall = b.avg_key_fact_recall ?? b['Avg Key Fact Recall'] ?? 0;
                    const faithfulness = b.avg_faithfulness ?? b['Avg Faithfulness'] ?? 0;
                    const relevance = b.avg_answer_relevance ?? b['Avg Answer Relevance'] ?? 0;
                    const f1 = b.avg_retrieval_f1 ?? b['Avg Retrieval F1'] ?? 0;
                    const testCases = b.total_test_cases_evaluated ?? b['Total Test Cases Evaluated'] ?? 0;
                    const cost = b.total_estimated_cost_usd ?? b['Total Estimated Cost (USD)'] ?? 0;
                    const model = b.agent_llm_model ?? b['Agent LLM Model (Sample)'] ?? '—';

                    return `
                    <div class="benchmark-card">
                        <h4>${escapeHTML(b.agent_name)}</h4>
                        <div class="benchmark-scores">
                            <div class="score-item"><span class="score-label">Recall</span><span class="score-value">${(recall * 100).toFixed(1)}%</span></div>
                            <div class="score-item"><span class="score-label">Faithfulness</span><span class="score-value">${(faithfulness * 100).toFixed(1)}%</span></div>
                            <div class="score-item"><span class="score-label">Relevance</span><span class="score-value">${(relevance * 100).toFixed(1)}%</span></div>
                            <div class="score-item"><span class="score-label">Retrieval F1</span><span class="score-value">${(f1 * 100).toFixed(1)}%</span></div>
                        </div>
                        <div style="margin-top:1rem;display:flex;gap:0.75rem;flex-wrap:wrap">
                            <span class="meta-chip">${testCases} tests</span>
                            <span class="meta-chip">$${Number(cost).toFixed(4)}</span>
                            <span class="meta-chip">${escapeHTML(model)}</span>
                        </div>
                    </div>`;
                }).join('');
            }
        }
    } catch (e) { console.error('Benchmark summary error:', e); }

    // Runs table
    try {
        if (runsResp && runsResp.ok) {
            const data = await runsResp.json();
            renderTable('benchmark-runs-table', 'benchmark-runs-empty', data, row => {
                // Extract overall score from summary_metrics dict if available
                let scoreDisplay = '—';
                if (row.summary_metrics) {
                    const sm = row.summary_metrics;
                    const avgScore = sm.avg_overall_score ?? sm.overall_score ?? sm.avg_faithfulness;
                    if (avgScore != null) scoreDisplay = (avgScore * 100).toFixed(1) + '%';
                }
                return `
                <td><span class="meta-chip">${escapeHTML(String(row.run_id || row.id || '—').slice(0, 8))}</span></td>
                <td>${escapeHTML(row.agent_name || '—')}</td>
                <td><span class="badge ${row.status}">${escapeHTML(row.status || '—')}</span></td>
                <td>${scoreDisplay}</td>
                <td>${row.total_items ?? '—'}</td>
                <td>${formatTimeAgo(row.started_at || row.created_at)}</td>`;
            });
        }
    } catch (e) { console.error('Benchmark runs error:', e); }

    // Test sets
    try {
        if (testsetsResp && testsetsResp.ok) {
            const rawData = await testsetsResp.json();
            // API returns { test_sets: [...] } wrapper — unwrap it
            const data = rawData.test_sets || rawData;
            renderTable('testsets-table', 'testsets-empty', data, row => `
                <td>${escapeHTML(row.filename || row)}</td>
                <td>${row.num_questions ?? row.num_items ?? '—'}</td>
                <td><button class="table-action-btn danger" onclick="deleteTestSet('${escapeAttr(row.filename || row)}')">Delete</button></td>`);
        }
    } catch (e) { console.error('Test sets error:', e); }
}

function openBenchmarkModal() {
    // Populate agent select — guard against empty agent list
    const sel = document.getElementById('bench-agent');
    if (sel) {
        if (allAgents.length === 0) {
            sel.innerHTML = '<option value="" disabled>No agents available</option>';
        } else {
            sel.innerHTML = allAgents.map(a => `<option value="${escapeAttr(a.name)}">${escapeHTML(a.name)}</option>`).join('');
        }
    }
    // Populate test set select (load from API)
    const tsel = document.getElementById('bench-testset');
    if (tsel) tsel.innerHTML = '<option value="" disabled>Loading…</option>';
    apiFetch(`${API_BASE_URL}/evaluation/test-sets`).then(async r => {
        if (r && r.ok) {
            const data = await r.json();
            if (tsel) {
                if (!data || data.length === 0) {
                    tsel.innerHTML = '<option value="" disabled>No test sets available</option>';
                } else {
                    tsel.innerHTML = data.map(ts => `<option value="${escapeAttr(ts.filename || ts)}">${escapeHTML(ts.filename || ts)}</option>`).join('');
                }
            }
        }
    }).catch(() => { if (tsel) tsel.innerHTML = '<option value="" disabled>Failed to load</option>'; });
    document.getElementById('benchmark-modal')?.classList.add('open');
}

function openTestSetModal() {
    const sel = document.getElementById('testset-agent');
    if (sel) {
        if (allAgents.length === 0) {
            sel.innerHTML = '<option value="" disabled>No agents available</option>';
        } else {
            sel.innerHTML = allAgents.map(a => `<option value="${escapeAttr(a.name)}">${escapeHTML(a.name)}</option>`).join('');
        }
    }
    document.getElementById('testset-modal')?.classList.add('open');
}

async function handleRunBenchmark(event) {
    event.preventDefault();
    const btn = document.getElementById('bench-submit-btn');
    if (!btn || btn.disabled) return;
    btn.disabled = true; btn.textContent = 'Running…';
    try {
        const agentName = document.getElementById('bench-agent')?.value;
        const testSetFilename = document.getElementById('bench-testset')?.value;
        if (!agentName || !testSetFilename) { showToast('Please select an agent and test set.', true); btn.disabled = false; btn.textContent = 'Run'; return; }

        // Fetch the actual test set data from the API
        const tsResp = await apiFetch(`${API_BASE_URL}/evaluation/test-sets/${encodeURIComponent(testSetFilename)}`);
        if (!tsResp || !tsResp.ok) throw new Error('Failed to load test set data');
        const tsData = await tsResp.json();
        const testSetArray = tsData.data || tsData;
        if (!Array.isArray(testSetArray) || testSetArray.length === 0) throw new Error('Test set is empty or invalid');

        const payload = { agent_name: agentName, test_set: testSetArray, test_set_file: testSetFilename };
        const resp = await apiFetch(`${API_BASE_URL}/evaluation/benchmark`, { method: 'POST', body: JSON.stringify(payload) });
        if (!resp || !resp.ok) { const err = await resp?.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp?.status}`); }
        showToast('Benchmark started!'); closeModal('benchmark-modal'); await loadBenchmarks();
    } catch (err) { showToast(`Failed to start benchmark: ${err.message}`, true); }
    finally { btn.disabled = false; btn.textContent = 'Run'; }
}

async function handleGenerateTestSet(event) {
    event.preventDefault();
    const btn = document.getElementById('testset-submit-btn');
    if (!btn || btn.disabled) return;
    btn.disabled = true; btn.textContent = 'Generating…';
    try {
        const payload = { agent_name: document.getElementById('testset-agent')?.value, num_questions: parseInt(document.getElementById('testset-count')?.value) || 10 };
        const resp = await apiFetch(`${API_BASE_URL}/evaluation/test-set`, { method: 'POST', body: JSON.stringify(payload) });
        if (!resp || !resp.ok) { const err = await resp?.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp?.status}`); }
        showToast('Test set generation started!'); closeModal('testset-modal'); await loadBenchmarks();
    } catch (err) { showToast(`Failed to generate test set: ${err.message}`, true); }
    finally { btn.disabled = false; btn.textContent = 'Generate'; }
}

async function deleteTestSet(filename) {
    try {
        const resp = await apiFetch(`${API_BASE_URL}/evaluation/test-sets/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        if (!resp || (!resp.ok && resp.status !== 204)) throw new Error('Delete failed');
        showToast('Test set deleted.'); await loadBenchmarks();
    } catch (err) { showToast(`Failed to delete test set: ${err.message}`, true); }
}

// ============================================
// MONITORING PANEL
// ============================================
async function loadMonitoring() {
    let healthResp, secResp, resResp, trainResp, pipeResp;
    try {
        [healthResp, secResp, resResp, trainResp, pipeResp] = await Promise.all([
            fetch(`${API_BASE_URL.replace('/api/v1', '')}/health`).catch(() => null),
            apiFetch(`${API_BASE_URL}/monitoring/security`).catch(() => null),
            apiFetch(`${API_BASE_URL}/monitoring/resources`).catch(() => null),
            apiFetch(`${API_BASE_URL}/monitoring/training-overview`).catch(() => null),
            apiFetch(`${API_BASE_URL}/monitoring/data-pipeline`).catch(() => null),
        ]);
    } catch (e) { console.error('Monitoring fetch error:', e); return; }

    // Health
    try {
        const el = document.getElementById('health-status');
        if (healthResp && healthResp.ok) {
            el.textContent = '● Healthy';
            el.classList.add('success');
            el.classList.remove('danger');
        } else {
            el.textContent = '● Down';
            el.classList.add('danger');
            el.classList.remove('success');
        }
    } catch (e) { console.error('Health check error:', e); }

    // Security metrics
    try {
        if (secResp && secResp.ok) {
            const d = await secResp.json();
            setText('mon-sessions', d.active_sessions ?? 0);
            setText('mon-failed-auth', d.failed_auth_24h ?? 0);
            setText('mon-rate-limited', d.rate_limited_24h ?? 0);
        }
    } catch (e) { console.error('Security metrics error:', e); }

    // Resources
    try {
        if (resResp && resResp.ok) {
            const d = await resResp.json();
            setProgressBar('cpu-bar', 'cpu-val', d.cpu_percent, `${d.cpu_percent}%`);
            setProgressBar('mem-bar', 'mem-val', d.memory_percent, `${d.memory_percent}%`);
            setProgressBar('disk-bar', 'disk-val', d.disk_percent, `${d.disk_percent.toFixed(1)}%`);
        }
    } catch (e) { console.error('Resource metrics error:', e); }

    // Training overview
    try {
        if (trainResp && trainResp.ok) {
            const d = await trainResp.json();
            setText('mon-train-running', d.running_jobs ?? 0);
            setText('mon-train-models', d.total_models ?? 0);
            setText('mon-gpu-hours', `${d.gpu_hours_24h?.toFixed(1) ?? 0}h`);
            setText('mon-train-cost', `$${d.training_cost_24h?.toFixed(2) ?? '0.00'}`);
        }
    } catch (e) { console.error('Training overview error:', e); }

    // Data pipeline
    try {
        if (pipeResp && pipeResp.ok) {
            const d = await pipeResp.json();
            setText('mon-total-docs', d.total_documents?.toLocaleString() ?? 0);
            setText('mon-embeddings', d.embeddings_24h ?? 0);
            setText('mon-vs-size', `${d.vector_store_size_mb?.toFixed(1) ?? 0} MB`);
            setText('mon-failed-ingests', d.failed_ingests_24h ?? 0);
        }
    } catch (e) { console.error('Pipeline metrics error:', e); }
}

function setProgressBar(barId, valId, percent, label) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    if (bar) {
        bar.style.width = `${Math.min(percent, 100)}%`;
        bar.className = 'progress-fill' + (percent > 85 ? ' critical' : percent > 65 ? ' warn' : '');
    }
    if (val) val.textContent = label;
}

// ============================================
// USERS PANEL
// ============================================
async function loadUsers() {
    try {
        const resp = await apiFetch(`${API_BASE_URL}/security/users`);
        if (resp && resp.ok) {
            const data = await resp.json();
            renderTable('users-table', 'users-empty', data, row => {
                // roles is an array of {id, name, ...} objects — extract the first role name
                const roleNames = (row.roles || []).map(r => r.name || r).filter(Boolean);
                const primaryRole = roleNames[0] || 'viewer';
                const isAdmin = roleNames.includes('admin') || row.is_superuser;
                return `
                <td>${escapeHTML(row.username)}</td>
                <td><span class="badge ${isAdmin ? 'running' : 'completed'}">${escapeHTML(primaryRole)}</span></td>
                <td>${formatTimeAgo(row.created_at)}</td>
                <td><span class="badge ${row.is_active !== false ? 'completed' : 'failed'}">${row.is_active !== false ? 'Active' : 'Disabled'}</span></td>`;
            });
        }
    } catch (e) { console.error('Users load error:', e); }
}

function openCreateUserModal() { document.getElementById('create-user-modal')?.classList.add('open'); }

async function handleCreateUser(event) {
    event.preventDefault();
    const btn = document.getElementById('create-user-btn');
    if (!btn || btn.disabled) return;
    btn.disabled = true; btn.textContent = 'Creating…';
    try {
        const payload = {
            username: document.getElementById('new-username')?.value.trim(),
            password: document.getElementById('new-password')?.value,
            role: document.getElementById('new-role')?.value || 'viewer',
        };
        const resp = await apiFetch(`${API_BASE_URL}/security/users`, { method: 'POST', body: JSON.stringify(payload) });
        if (!resp || !resp.ok) { const err = await resp?.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${resp?.status}`); }
        showToast('User created!'); closeModal('create-user-modal');
        document.getElementById('create-user-form')?.reset();
        await loadUsers();
    } catch (err) { showToast(`Failed to create user: ${err.message}`, true); }
    finally { btn.disabled = false; btn.textContent = 'Create User'; }
}

// ============================================
// UTILITIES
// ============================================
function escapeHTML(str) { const d = document.createElement('div'); d.textContent = str || ''; return d.innerHTML; }
function escapeAttr(str) { return (str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function formatTimeAgo(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return '';
    const diff = Math.floor((new Date() - date) / 1000);
    if (diff < 0) return 'just now';
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 2592000) return `${Math.floor(diff / 86400)}d ago`;
    return date.toLocaleDateString();
}

function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function renderTable(tableId, emptyId, data, rowRenderer) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    const empty = document.getElementById(emptyId);
    if (!tbody) return;

    if (!data || data.length === 0) {
        tbody.innerHTML = '';
        empty?.classList.add('visible');
    } else {
        empty?.classList.remove('visible');
        tbody.innerHTML = data.map(row => `<tr>${rowRenderer(row)}</tr>`).join('');
    }
}

// --- Logout ---
function handleLogout() {
    localStorage.removeItem('ragnetic_user_token');
    localStorage.removeItem('ragnetic_db_user_id');
    localStorage.removeItem('ragnetic_user_id');
    localStorage.removeItem('ragnetic_agent');
    localStorage.removeItem('ragnetic_thread_id');
    if (refreshInterval) clearInterval(refreshInterval);
    window.location.href = '/login';
}
