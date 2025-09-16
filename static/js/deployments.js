// Deployments Management JavaScript
console.log('Deployments.js loaded successfully');

class DeploymentsManager {
    constructor() {
        this.deployments = [];
        this.agents = [];
        this.projects = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Create deployment form
        const createDeploymentForm = document.getElementById('create-deployment-form');
        if (createDeploymentForm) {
            createDeploymentForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.createApiDeployment();
            });
        }
    }

    async loadDeployments() {
        try {
            // Ensure token is available
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            
            if (!token) {
                console.error('No authentication token available for deployments request');
                return;
            }
            
            const response = await fetch(`${API_BASE_URL}/deployments`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.deployments = data;
            this.renderDeployments();
        } catch (error) {
            console.error('Error loading deployments:', error);
            this.showError('Failed to load deployments');
        }
    }

    async loadAgents() {
        try {
            const response = await fetch(`${API_BASE_URL}/agents`, {
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.agents = data;
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }

    async loadProjects() {
        try {
            const response = await fetch(`${API_BASE_URL}/projects`, {
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.projects = data;
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }

    renderDeployments() {
        const container = document.querySelector('.deployments-content');
        if (!container) return;

        if (this.deployments.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                            <polyline points="3.27,6.96 12,12.01 20.73,6.96"></polyline>
                            <line x1="12" y1="22.08" x2="12" y2="12"></line>
                        </svg>
                    </div>
                    <h3>No Deployments Yet</h3>
                    <p>Create your first API deployment to make your agents accessible via API.</p>
                    <button class="btn-primary" onclick="deploymentsManager.showCreateDeploymentModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Create Deployment
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="deployments-header">
                <div class="section-actions">
                    <button class="btn-primary" onclick="deploymentsManager.showCreateDeploymentModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Create Deployment
                    </button>
                </div>
            </div>
            <div class="deployments-grid">
                ${this.deployments.map(deployment => this.renderDeploymentCard(deployment)).join('')}
            </div>
        `;
    }

    renderDeploymentCard(deployment) {
        const createdDate = new Date(deployment.created_at).toLocaleDateString();
        const statusClass = deployment.status === 'active' ? 'active' : 'inactive';
        
        return `
            <div class="deployment-card">
                <div class="deployment-header">
                    <div class="deployment-title">
                        <h3>${this.escapeHtml(deployment.agent_name || 'Unknown Agent')}</h3>
                        <span class="deployment-status status-${statusClass}">${deployment.status}</span>
                    </div>
                    <div class="deployment-actions">
                        <button class="btn-icon" onclick="deploymentsManager.testInvoke(${deployment.id})" title="Test API">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="5,3 19,12 5,21"></polygon>
                            </svg>
                        </button>
                        <button class="btn-icon" onclick="deploymentsManager.rotateKey(${deployment.id})" title="Rotate API Key">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                                <path d="M21 3v5h-5"></path>
                                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                                <path d="M3 21v-5h5"></path>
                            </svg>
                        </button>
                        <button class="btn-icon btn-danger" onclick="deploymentsManager.revokeDeployment(${deployment.id})" title="Revoke Deployment">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3,6 5,6 21,6"></polyline>
                                <path d="M19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="deployment-content">
                    <div class="deployment-details">
                        <div class="detail">
                            <span class="detail-label">Project</span>
                            <span class="detail-value">${this.escapeHtml(deployment.project_name || 'Unknown')}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Type</span>
                            <span class="detail-value">${deployment.deployment_type}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Endpoint</span>
                            <span class="detail-value">${deployment.endpoint_path}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Created</span>
                            <span class="detail-value">${createdDate}</span>
                        </div>
                    </div>
                    <div class="deployment-api-key">
                        <div class="api-key-display">
                            <span class="api-key-label">API Key:</span>
                            <span class="api-key-value" id="api-key-${deployment.id}">${this.maskApiKey(deployment.api_key || 'Not available')}</span>
                            <button class="btn-text btn-sm" onclick="deploymentsManager.copyApiKey('${deployment.api_key || ''}')" title="Copy API Key">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showCreateDeploymentModal() {
        Promise.all([this.loadAgents(), this.loadProjects()]).then(() => {
            const modal = document.createElement('div');
            modal.className = 'modal show';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>Create API Deployment</h2>
                        <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                    </div>
                    <form id="create-deployment-form">
                        <div class="form-group">
                            <label for="project-select">Project</label>
                            <select id="project-select" name="project_id" required>
                                <option value="">Select a project...</option>
                                ${this.projects.map(project => `
                                    <option value="${project.id}">${this.escapeHtml(project.name)}</option>
                                `).join('')}
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="agent-select">Agent</label>
                            <select id="agent-select" name="agent_id" required>
                                <option value="">Select an agent...</option>
                                ${this.agents.map(agent => `
                                    <option value="${agent.id}">${this.escapeHtml(agent.name)}</option>
                                `).join('')}
                            </select>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                            <button type="submit" class="btn-primary">Create Deployment</button>
                        </div>
                    </form>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Setup form event listener
            modal.querySelector('#create-deployment-form').addEventListener('submit', (e) => {
                e.preventDefault();
                this.createApiDeployment();
            });
        });
    }

    async createApiDeployment() {
        const form = document.getElementById('create-deployment-form');
        if (!form) return;

        const formData = new FormData(form);
        const deploymentData = {
            project_id: parseInt(formData.get('project_id')),
            agent_id: parseInt(formData.get('agent_id'))
        };

        try {
            const response = await fetch(`${API_BASE_URL}/deployments/api`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(deploymentData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Show API key in a special modal
            this.showApiKeyModal(result);
            
            document.querySelector('.modal').remove();
            this.loadDeployments();
        } catch (error) {
            console.error('Error creating deployment:', error);
            this.showError(`Failed to create deployment: ${error.message}`);
        }
    }

    showApiKeyModal(result) {
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Deployment Created Successfully</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="api-key-warning">
                        <div class="warning-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                        </div>
                        <p><strong>Important:</strong> Save your API key now. You won't be able to see it again!</p>
                    </div>
                    <div class="api-key-display">
                        <label>API Key:</label>
                        <div class="api-key-container">
                            <input type="text" value="${result.api_key}" readonly class="api-key-input" id="new-api-key">
                            <button class="btn-secondary" onclick="deploymentsManager.copyApiKey('${result.api_key}')">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                                Copy
                            </button>
                        </div>
                    </div>
                    <div class="deployment-info">
                        <div class="info-item">
                            <span class="info-label">Endpoint:</span>
                            <span class="info-value">${result.endpoint_path}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Status:</span>
                            <span class="info-value status-${result.status}">${result.status}</span>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-primary" onclick="this.closest('.modal').remove()">Done</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async rotateKey(deploymentId) {
        if (!confirm('Are you sure you want to rotate the API key? The old key will be invalidated.')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/deployments/${deploymentId}/rotate-key`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showApiKeyModal(result);
            this.loadDeployments();
        } catch (error) {
            console.error('Error rotating API key:', error);
            this.showError('Failed to rotate API key');
        }
    }

    async revokeDeployment(deploymentId) {
        if (!confirm('Are you sure you want to revoke this deployment? This will permanently disable the API endpoint.')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/deployments/${deploymentId}`, {
                method: 'DELETE',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.showSuccess('Deployment revoked successfully');
            this.loadDeployments();
        } catch (error) {
            console.error('Error revoking deployment:', error);
            this.showError('Failed to revoke deployment');
        }
    }

    async testInvoke(deploymentId) {
        const deployment = this.deployments.find(d => d.id === deploymentId);
        if (!deployment) {
            this.showError('Deployment not found');
            return;
        }

        const testQuery = prompt('Enter a test query:', 'Hello, how are you?');
        if (!testQuery) return;

        try {
            const response = await fetch(`${API_BASE_URL}/invoke/${deploymentId}`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json',
                    'X-API-Key': deployment.api_key
                },
                body: JSON.stringify({ query: testQuery })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showTestResultModal(testQuery, result);
        } catch (error) {
            console.error('Error testing deployment:', error);
            this.showError(`Test failed: ${error.message}`);
        }
    }

    showTestResultModal(query, result) {
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content large">
                <div class="modal-header">
                    <h2>API Test Result</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="test-result">
                        <div class="test-query">
                            <h4>Query:</h4>
                            <p>${this.escapeHtml(query)}</p>
                        </div>
                        <div class="test-response">
                            <h4>Response:</h4>
                            <div class="response-content">${this.escapeHtml(result.response)}</div>
                        </div>
                        <div class="test-metadata">
                            <div class="metadata-item">
                                <span class="metadata-label">Cost:</span>
                                <span class="metadata-value">$${result.cost}</span>
                            </div>
                            <div class="metadata-item">
                                <span class="metadata-label">Timestamp:</span>
                                <span class="metadata-value">${new Date(result.timestamp).toLocaleString()}</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Close</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    copyApiKey(apiKey) {
        if (!apiKey) {
            this.showError('No API key to copy');
            return;
        }

        navigator.clipboard.writeText(apiKey).then(() => {
            this.showSuccess('API key copied to clipboard');
        }).catch(() => {
            this.showError('Failed to copy API key');
        });
    }

    maskApiKey(apiKey) {
        if (!apiKey || apiKey.length <= 8) {
            return '*'.repeat(apiKey?.length || 8);
        }
        return apiKey.substring(0, 4) + '*'.repeat(apiKey.length - 8) + apiKey.substring(apiKey.length - 4);
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

// Initialize deployments manager when DOM is loaded
let deploymentsManager;
document.addEventListener('DOMContentLoaded', () => {
    deploymentsManager = new DeploymentsManager();
});
