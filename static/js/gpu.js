// GPU Control JavaScript
console.log('GPU.js loaded successfully');

class GPUControl {
    constructor() {
        this.providers = [];
        this.instances = [];
        this.refreshInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadServiceStatus();
    }

    setupEventListeners() {
        // Provision GPU form
        const provisionForm = document.getElementById('provision-gpu-form');
        if (provisionForm) {
            provisionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.provisionGPU();
            });
        }
    }

    async loadServiceStatus() {
        try {
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            if (!token) return;

            const response = await fetch(`${API_BASE_URL}/gpu/service-status`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const status = await response.json();
                this.updateServiceStatus(status);
            }
        } catch (e) {
            console.warn('Failed to load GPU service status:', e);
        }
    }

    updateServiceStatus(status) {
        const badgeEl = document.getElementById('gpu-service-badge');
        const messageEl = document.getElementById('gpu-service-message');
        
        if (badgeEl) {
            badgeEl.className = `status-badge status-${status.status_color}`;
            badgeEl.textContent = status.service_name;
        }
        
        if (messageEl) {
            messageEl.textContent = status.status_message;
        }

        // Show warning for mock mode
        if (status.service_type === 'mock') {
            this.showMockModeWarning(status);
        } else {
            this.hideMockModeWarning();
        }
    }

    showMockModeWarning(status) {
        const gpuControlContent = document.querySelector('#gpu-control-view .gpu-control-content');
        if (!gpuControlContent) return;

        // Remove existing warning
        this.hideMockModeWarning();

        const warningEl = document.createElement('div');
        warningEl.className = 'gpu-service-warning';
        warningEl.innerHTML = `
            <div class="warning-icon">🧪</div>
            <div class="warning-content">
                <div class="warning-title">Mock GPU Service Active</div>
                <div class="warning-text">
                    ${status.status_message}. GPU instances will be simulated and no real resources will be provisioned.
                    ${status.environment === 'production' ? 'Set RUNPOD_API_KEY to enable real GPU services.' : ''}
                </div>
            </div>
        `;

        gpuControlContent.insertBefore(warningEl, gpuControlContent.firstChild);
    }

    hideMockModeWarning() {
        const existingWarning = document.querySelector('.gpu-service-warning');
        if (existingWarning) {
            existingWarning.remove();
        }
    }

    async loadProviders() {
        try {
            // Ensure token is available
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            
            if (!token) {
                console.error('No authentication token available for GPU providers request');
                return;
            }
            
            const response = await fetch(`${API_BASE_URL}/gpu/providers`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.providers = Array.isArray(data) ? data : [];
            this.renderProviders();
        } catch (error) {
            console.error('Error loading GPU providers:', error);
            this.showError('Failed to load GPU providers');
        }
    }

    async loadInstances() {
        try {
            // Ensure token is available
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            
            if (!token) {
                console.error('No authentication token available for GPU instances request');
                return;
            }
            
            const response = await fetch(`${API_BASE_URL}/gpu/instances`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.instances = Array.isArray(data) ? data : [];
            this.renderInstances();
            
            // Start auto-refresh for running instances
            this.startAutoRefresh();
        } catch (error) {
            console.error('Error loading GPU instances:', error);
            this.showError('Failed to load GPU instances');
        }
    }

    renderProviders() {
        const container = document.querySelector('.gpu-control-content');
        if (!container) return;

        const providersHtml = (this.providers || []).map(provider => `
            <div class="provider-card">
                <div class="provider-header">
                    <h3>${this.escapeHtml(provider.name)}</h3>
                    <span class="provider-status ${provider.availability ? 'available' : 'unavailable'}">
                        ${provider.availability ? 'Available' : 'Unavailable'}
                    </span>
                </div>
                <div class="provider-gpus">
                    ${(provider.gpu_types || []).map(gpu => `
                        <div class="gpu-type">
                            <span class="gpu-name">${this.escapeHtml(gpu.gpu_type)}</span>
                            <span class="gpu-price">$${gpu.cost_per_hour}/hour</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');

        container.innerHTML = `
            <div class="gpu-control-header">
                <div class="section-actions">
                    <button class="btn-primary" onclick="gpuControl.showProvisionModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Provision GPU
                    </button>
                    <button class="btn-secondary" onclick="gpuControl.loadInstances()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                            <path d="M21 3v5h-5"></path>
                            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                            <path d="M3 21v-5h5"></path>
                        </svg>
                        Refresh
                    </button>
                </div>
            </div>
            
            <div class="gpu-sections">
                <div class="gpu-section">
                    <h3>Available Providers</h3>
                    <div class="providers-grid">
                        ${providersHtml}
                    </div>
                </div>
                
                <div class="gpu-section">
                    <h3>Your GPU Instances</h3>
                    <div class="instances-container" id="instances-container">
                        <div class="loading-state">
                            <div class="loading-spinner"></div>
                            <p>Loading instances...</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderInstances() {
        const container = document.getElementById('instances-container');
        if (!container) return;

        if (this.instances.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                            <line x1="8" y1="21" x2="16" y2="21"></line>
                            <line x1="12" y1="17" x2="12" y2="21"></line>
                        </svg>
                    </div>
                    <h3>No GPU Instances</h3>
                    <p>Provision your first GPU instance to start training models.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="instances-grid">
                ${(this.instances || []).map(instance => this.renderInstanceCard(instance)).join('')}
            </div>
        `;
    }

    renderInstanceCard(instance) {
        const createdDate = new Date(instance.created_at).toLocaleDateString();
        const uptime = instance.started_at ? this.calculateUptime(instance.started_at) : 'Not started';
        const cost = instance.total_cost || 0;
        
        return `
            <div class="instance-card">
                <div class="instance-header">
                    <div class="instance-title">
                        <h4>${this.escapeHtml(instance.gpu_type)}</h4>
                        <span class="instance-status status-${instance.status}">${instance.status}</span>
                    </div>
                    <div class="instance-actions">
                        ${instance.status === 'running' ? `
                            <button class="btn-icon btn-warning" onclick="gpuControl.stopInstance(${instance.id})" title="Stop Instance">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="6" y="4" width="4" height="16"></rect>
                                    <rect x="14" y="4" width="4" height="16"></rect>
                                </svg>
                            </button>
                        ` : ''}
                        <button class="btn-icon" onclick="gpuControl.viewLogs(${instance.id})" title="View Logs">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14,2 14,8 20,8"></polyline>
                                <line x1="16" y1="13" x2="8" y2="13"></line>
                                <line x1="16" y1="17" x2="8" y2="17"></line>
                                <polyline points="10,9 9,9 8,9"></polyline>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="instance-content">
                    <div class="instance-details">
                        <div class="detail">
                            <span class="detail-label">Provider</span>
                            <span class="detail-value">${this.escapeHtml(instance.provider)}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Cost/Hour</span>
                            <span class="detail-value">$${instance.cost_per_hour}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Total Cost</span>
                            <span class="detail-value">$${cost.toFixed(2)}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Uptime</span>
                            <span class="detail-value">${uptime}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Created</span>
                            <span class="detail-value">${createdDate}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showProvisionModal() {
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Provision GPU Instance</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <form id="provision-gpu-form">
                    <div class="form-group">
                        <label for="gpu-type">GPU Type</label>
                        <select id="gpu-type" name="gpu_type" required>
                            <option value="">Select GPU type...</option>
                            ${(this.providers || []).flatMap(provider => 
                                (provider.gpu_types || []).map(gpu => `
                                    <option value="${gpu.gpu_type}" data-provider="${provider.name}" data-cost="${gpu.cost_per_hour}">
                                        ${gpu.gpu_type} (${provider.name}) - $${gpu.cost_per_hour}/hour
                                    </option>
                                `)
                            ).join('')}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="max-hours">Maximum Hours</label>
                        <input type="number" id="max-hours" name="max_hours" min="0.1" max="24" step="0.1" value="1" required>
                        <small>Maximum time to run the instance (0.1-24 hours)</small>
                    </div>
                    <div class="form-group">
                        <label for="purpose">Purpose</label>
                        <select id="purpose" name="purpose" required>
                            <option value="training">Training</option>
                            <option value="inference">Inference</option>
                            <option value="development">Development</option>
                        </select>
                    </div>
                    <div class="cost-estimate" id="cost-estimate" style="display: none;">
                        <div class="estimate-card">
                            <h4>Cost Estimate</h4>
                            <div class="estimate-details">
                                <div class="estimate-item">
                                    <span>GPU Cost:</span>
                                    <span id="gpu-cost">$0.00</span>
                                </div>
                                <div class="estimate-item">
                                    <span>Total Estimated:</span>
                                    <span id="total-cost">$0.00</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                        <button type="submit" class="btn-primary">Provision GPU</button>
                    </div>
                </form>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Setup form event listeners
        const form = modal.querySelector('#provision-gpu-form');
        const gpuTypeSelect = modal.querySelector('#gpu-type');
        const maxHoursInput = modal.querySelector('#max-hours');
        const costEstimate = modal.querySelector('#cost-estimate');
        
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.provisionGPU();
        });
        
        // Update cost estimate when values change
        const updateCostEstimate = () => {
            const selectedOption = gpuTypeSelect.options[gpuTypeSelect.selectedIndex];
            if (selectedOption && selectedOption.value) {
                const costPerHour = parseFloat(selectedOption.dataset.cost);
                const maxHours = parseFloat(maxHoursInput.value) || 0;
                const totalCost = costPerHour * maxHours;
                
                modal.querySelector('#gpu-cost').textContent = `$${costPerHour.toFixed(2)}/hour`;
                modal.querySelector('#total-cost').textContent = `$${totalCost.toFixed(2)}`;
                costEstimate.style.display = 'block';
            } else {
                costEstimate.style.display = 'none';
            }
        };
        
        gpuTypeSelect.addEventListener('change', updateCostEstimate);
        maxHoursInput.addEventListener('input', updateCostEstimate);
    }

    async provisionGPU() {
        const form = document.getElementById('provision-gpu-form');
        if (!form) return;

        const formData = new FormData(form);
        const provisionData = {
            gpu_type: formData.get('gpu_type'),
            max_hours: parseFloat(formData.get('max_hours')),
            purpose: formData.get('purpose')
        };

        try {
            const response = await fetch(`${API_BASE_URL}/gpu/provision`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(provisionData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showSuccess('GPU instance provisioned successfully');
            document.querySelector('.modal').remove();
            this.loadInstances();
        } catch (error) {
            console.error('Error provisioning GPU:', error);
            this.showError(`Failed to provision GPU: ${error.message}`);
        }
    }

    async stopInstance(instanceId) {
        if (!confirm('Are you sure you want to stop this GPU instance? This will terminate any running processes.')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/gpu/instances/${instanceId}/stop`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.showSuccess('GPU instance stopped successfully');
            this.loadInstances();
        } catch (error) {
            console.error('Error stopping GPU instance:', error);
            this.showError('Failed to stop GPU instance');
        }
    }

    async viewLogs(instanceId) {
        try {
            const response = await fetch(`${API_BASE_URL}/gpu/instances/${instanceId}/logs`, {
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.showLogsModal(instanceId, data.logs);
        } catch (error) {
            console.error('Error fetching logs:', error);
            this.showError('Failed to fetch instance logs');
        }
    }

    showLogsModal(instanceId, logs) {
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content large">
                <div class="modal-header">
                    <h2>GPU Instance Logs</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="logs-container">
                        <pre class="logs-content">${this.escapeHtml(logs || 'No logs available')}</pre>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Close</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    startAutoRefresh() {
        // Clear existing interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }

        // Only auto-refresh if there are running instances
        const hasRunningInstances = this.instances.some(instance => instance.status === 'running');
        if (hasRunningInstances) {
            this.refreshInterval = setInterval(() => {
                this.loadInstances();
            }, 20000); // Refresh every 20 seconds
        }
    }

    calculateUptime(startTime) {
        const start = new Date(startTime);
        const now = new Date();
        const diffMs = now - start;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
        
        if (diffHours > 0) {
            return `${diffHours}h ${diffMinutes}m`;
        } else {
            return `${diffMinutes}m`;
        }
    }

    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showInfo(message) {
        this.showToast(message, 'info');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        const container = document.getElementById('toast-notification') || document.body;
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
}

// Initialize GPU control when DOM is loaded
let gpuControl;
document.addEventListener('DOMContentLoaded', () => {
    gpuControl = new GPUControl();
});
