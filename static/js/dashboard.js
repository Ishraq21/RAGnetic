// Dashboard JavaScript functionality
class Dashboard {
    constructor() {
        this.currentView = 'overview';
        this.agents = [];
        this.workflows = [];
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.loadUserInfo();
        await this.loadOverviewData();
        this.setupSearch();
    }

    setupEventListeners() {
        // Tab Navigation
        document.querySelectorAll('.tab-nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const view = e.currentTarget.dataset.view;
                this.switchView(view);
                
                // Update active state
                document.querySelectorAll('.tab-nav-item').forEach(navItem => {
                    navItem.classList.remove('active');
                });
                e.currentTarget.classList.add('active');
            });
        });

        // User menu


        // Forms
        const createAgentForm = document.getElementById('create-agent-form');
        if (createAgentForm) {
            createAgentForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.createAgent();
            });
        }

        const createWorkflowForm = document.getElementById('create-workflow-form');
        if (createWorkflowForm) {
            createWorkflowForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.createWorkflow();
            });
        }

        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                this.hideModal(modal);
            });
        });

        // Close modals when clicking outside
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal(modal);
                }
            });
        });
    }

    setupSearch() {
        // Agent search
        const agentSearch = document.getElementById('agent-search');
        if (agentSearch) {
            agentSearch.addEventListener('input', (e) => {
                this.filterAgents(e.target.value);
            });
        }

        // Workflow search
        const workflowSearch = document.getElementById('workflow-search');
        if (workflowSearch) {
            workflowSearch.addEventListener('input', (e) => {
                this.filterWorkflows(e.target.value);
            });
        }
    }

    async loadUserInfo() {
        try {
            // For now, we'll use the stored user info
            // In a real app, you might want to fetch this from an API
            const username = localStorage.getItem('ragnetic_username') || 'User';
            const usernameElement = document.getElementById('username');
            if (usernameElement) {
                usernameElement.textContent = username;
            }
        } catch (error) {
            console.error('Failed to load user info:', error);
        }
    }

    async loadOverviewData() {
        try {
            await Promise.all([
                this.loadAgents(),
                this.loadWorkflows(),
                this.loadRecentActivity()
            ]);
            this.updateStats();
        } catch (error) {
            console.error('Failed to load overview data:', error);
            this.showToast('Failed to load dashboard data', 'error');
        }
    }

    async loadAgents() {
        try {
            const response = await fetch(`${API_BASE_URL}/agents`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            
            if (response.ok) {
                this.agents = await response.json();
                this.renderAgents();
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('Failed to load agents:', error);
            this.showToast('Failed to load agents', 'error');
        }
    }

    async loadWorkflows() {
        try {
            console.log('Loading workflows from:', `${API_BASE_URL}/workflows`);
            const response = await fetch(`${API_BASE_URL}/workflows`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            
            console.log('Workflows response status:', response.status);
            
            if (response.ok) {
                this.workflows = await response.json();
                console.log('Loaded workflows:', this.workflows);
                this.renderWorkflows();
            } else if (response.status === 404) {
                // No workflows endpoint yet, set empty array
                console.log('Workflows endpoint not found, setting empty array');
                this.workflows = [];
                this.renderWorkflows();
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('Failed to load workflows:', error);
            // Don't show error toast for workflows, just set empty array
            this.workflows = [];
            this.renderWorkflows();
        }
    }

    async loadRecentActivity() {
        try {
            // Load recent agent runs from audit API
            const auditUrl = `${API_BASE_URL.replace('/analytics', '/audit')}/runs?limit=5`;
            console.log('Loading recent agent runs from:', auditUrl);
            
            const agentRunsResponse = await fetch(auditUrl, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            
            console.log('Agent runs response status:', agentRunsResponse.status);
            
            if (agentRunsResponse.ok) {
                const agentRuns = await agentRunsResponse.json();
                console.log('Loaded agent runs:', agentRuns);
                this.renderRecentAgentActivity(agentRuns);
            } else {
                // If audit API fails, show no recent activity
                console.log('Audit API failed, showing no recent activity');
                this.renderRecentAgentActivity([]);
            }

            // Load recent workflow runs from analytics API
            const analyticsUrl = `${API_BASE_URL}/analytics/workflow-runs?limit=5`;
            console.log('Loading recent workflow runs from:', analyticsUrl);
            
            const workflowRunsResponse = await fetch(analyticsUrl, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            
            console.log('Workflow runs response status:', workflowRunsResponse.status);
            
            if (workflowRunsResponse.ok) {
                const workflowRuns = await workflowRunsResponse.json();
                console.log('Loaded workflow runs:', workflowRuns);
                this.renderRecentWorkflowActivity(workflowRuns);
            } else {
                // If analytics API fails, show no recent activity
                console.log('Analytics API failed, showing no recent activity');
                this.renderRecentWorkflowActivity([]);
            }
        } catch (error) {
            console.error('Failed to load recent activity:', error);
            // Show no recent activity on error
            this.renderRecentAgentActivity([]);
            this.renderRecentWorkflowActivity([]);
        }
    }

    updateStats() {
        const totalAgentsElement = document.getElementById('total-agents');
        const totalWorkflowsElement = document.getElementById('total-workflows');
        const totalRunsElement = document.getElementById('total-runs');
        const successRateElement = document.getElementById('success-rate');
        
        if (totalAgentsElement) totalAgentsElement.textContent = this.agents.length;
        if (totalWorkflowsElement) totalWorkflowsElement.textContent = this.workflows.length;
        
        // Calculate total runs and success rate (placeholder for now)
        if (totalRunsElement) totalRunsElement.textContent = '0';
        if (successRateElement) successRateElement.textContent = '0%';
    }

    renderAgents() {
        const container = document.getElementById('agents-grid');
        if (!container) return;

        if (this.agents.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                        <circle cx="9" cy="7" r="4"></circle>
                        <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                        <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                    </svg>
                    <h3>No agents yet</h3>
                    <p>Create your first AI agent to get started</p>
                    <button class="btn-primary" onclick="dashboard.showCreateAgentModal()">Create Agent</button>
                </div>
            `;
            return;
        }

        container.innerHTML = this.agents.map(agent => this.renderAgentCard(agent)).join('');
    }

    renderAgentCard(agent) {
        const status = this.getAgentStatus(agent);
        const statusClass = status === 'online' ? 'online' : status === 'deploying' ? 'deploying' : 'offline';
        
        return `
            <div class="agent-card" onclick="dashboard.showAgentDetails('${agent.name}')">
                <div class="agent-header">
                    <div class="agent-info">
                        <h3>${agent.display_name || agent.name}</h3>
                        <p>${agent.description || 'No description'}</p>
                    </div>
                    <span class="agent-status ${statusClass}">${status}</span>
                </div>
                <div class="agent-meta">
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                        </svg>
                        ${agent.llm_model}
                    </span>
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2L2 7L12 12L22 7L12 2Z"></path>
                        </svg>
                        ${agent.embedding_model}
                    </span>
                </div>
                <div class="agent-actions">
                    <button class="btn-secondary" onclick="event.stopPropagation(); dashboard.deployAgent('${agent.name}')">
                        Deploy
                    </button>
                    <button class="btn-text" onclick="event.stopPropagation(); dashboard.editAgent('${agent.name}')">
                        Edit
                    </button>
                </div>
            </div>
        `;
    }

    renderWorkflows() {
        const container = document.getElementById('workflows-list');
        if (!container) return;

        if (this.workflows.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="9 11 12 14 22 4"></polyline>
                        <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                    </svg>
                    <h3>No workflows yet</h3>
                    <p>Create your first automated workflow</p>
                    <button class="btn-primary" onclick="dashboard.showCreateWorkflowModal()">Create Workflow</button>
                </div>
            `;
            return;
        }

        container.innerHTML = this.workflows.map(workflow => this.renderWorkflowItem(workflow)).join('');
    }

    renderWorkflowItem(workflow) {
        return `
            <div class="workflow-item" onclick="dashboard.showWorkflowDetails('${workflow.name}')">
                <div class="workflow-header">
                    <div class="workflow-info">
                        <h3>${workflow.name}</h3>
                        <p>${workflow.description || 'No description'}</p>
                    </div>
                    <button class="btn-primary" onclick="event.stopPropagation(); dashboard.triggerWorkflow('${workflow.name}')">
                        Run
                    </button>
                </div>
                <div class="workflow-meta">
                    <span>Steps: ${workflow.steps?.length || 0}</span>
                    <span>Agent: ${workflow.agent_name || 'None'}</span>
                </div>
            </div>
        `;
    }

    renderRecentAgentActivity(runs) {
        const container = document.getElementById('recent-agent-activity');
        if (!container) return;

        if (!runs || runs.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent activity</p>';
            return;
        }

        container.innerHTML = runs.map(run => `
            <div class="activity-item">
                <div class="activity-icon ${run.status === 'completed' ? 'success' : run.status === 'failed' ? 'error' : 'warning'}">
                    ${run.status === 'completed' ? '✓' : run.status === 'failed' ? '✗' : '⋯'}
                </div>
                <div class="activity-content">
                    <div class="activity-title">${run.agent_name || 'Unknown Agent'}</div>
                    <div class="activity-meta">${run.status} • ${this.formatTime(run.start_time)}</div>
                </div>
            </div>
        `).join('');
    }

    renderRecentWorkflowActivity(runs) {
        const container = document.getElementById('recent-workflow-activity');
        if (!container) return;

        if (!runs || runs.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent activity</p>';
            return;
        }

        container.innerHTML = runs.map(run => `
            <div class="activity-item">
                <div class="activity-icon ${run.status === 'completed' ? 'success' : run.status === 'failed' ? 'error' : 'warning'}">
                    ${run.status === 'completed' ? '✓' : run.status === 'failed' ? '✗' : '⋯'}
                </div>
                <div class="activity-content">
                    <div class="activity-title">${run.workflow_name}</div>
                    <div class="activity-meta">${run.status} • ${this.formatTime(run.start_time)}</div>
                </div>
            </div>
        `).join('');
    }

    filterAgents(query) {
        const filtered = this.agents.filter(agent => 
            agent.name.toLowerCase().includes(query.toLowerCase()) ||
            (agent.display_name && agent.display_name.toLowerCase().includes(query.toLowerCase())) ||
            (agent.description && agent.description.toLowerCase().includes(query.toLowerCase()))
        );
        
        const container = document.getElementById('agents-grid');
        if (container) {
            container.innerHTML = filtered.map(agent => this.renderAgentCard(agent)).join('');
        }
    }

    filterWorkflows(query) {
        const filtered = this.workflows.filter(workflow => 
            workflow.name.toLowerCase().includes(query.toLowerCase()) ||
            (workflow.description && workflow.description.toLowerCase().includes(query.toLowerCase()))
        );
        
        const container = document.getElementById('workflows-list');
        if (container) {
            container.innerHTML = filtered.map(workflow => this.renderWorkflowItem(workflow)).join('');
        }
    }

    switchView(view) {
        // Update tab navigation
        document.querySelectorAll('.tab-nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.view === view);
        });

        // Hide all views
        document.querySelectorAll('.view-content').forEach(content => {
            content.classList.remove('active');
        });

        // Show selected view
        const targetView = document.getElementById(`${view}-view`);
        if (targetView) {
            targetView.classList.add('active');
        }

        this.currentView = view;

        // Load view-specific data
        if (view === 'agents') {
            this.loadAgents();
        } else if (view === 'workflows') {
            this.loadWorkflows();
        }
    }

    // Modal Management
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
        }
    }

    hideModal(modal) {
        modal.classList.remove('show');
    }

    showCreateAgentModal() {
        this.showModal('create-agent-modal');
    }

    hideCreateAgentModal() {
        this.hideModal(document.getElementById('create-agent-modal'));
    }

    showCreateWorkflowModal() {
        this.showModal('create-workflow-modal');
        this.populateWorkflowAgentSelect();
    }

    hideCreateWorkflowModal() {
        this.hideModal(document.getElementById('create-workflow-modal'));
    }

    showAgentDetails(agentName) {
        this.showModal('agent-details-modal');
        this.loadAgentDetails(agentName);
    }

    hideAgentDetailsModal() {
        this.hideModal(document.getElementById('agent-details-modal'));
    }

    showWorkflowDetails(workflowName) {
        this.showModal('workflow-details-modal');
        this.loadWorkflowDetails(workflowName);
    }

    hideWorkflowDetailsModal() {
        this.hideModal(document.getElementById('workflow-details-modal'));
    }

    // Agent Management
    async createAgent() {
        const form = document.getElementById('create-agent-form');
        const formData = new FormData(form);
        
        const agentData = {
            name: formData.get('name'),
            display_name: formData.get('display_name'),
            description: formData.get('description'),
            persona_prompt: formData.get('persona_prompt') || 'You are a helpful assistant.',
            llm_model: formData.get('llm_model'),
            embedding_model: formData.get('embedding_model'),
            tools: ['retriever'],
            sources: [],
            vector_store: {
                type: 'faiss',
                bm25_k: 5,
                semantic_k: 5,
                rerank_top_n: 5,
                hit_rate_k_value: 5,
                retrieval_strategy: 'hybrid'
            }
        };

        try {
            const response = await fetch(`${API_BASE_URL}/agents`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify(agentData)
            });

            if (response.ok) {
                this.showToast('Agent created successfully', 'success');
                this.hideCreateAgentModal();
                form.reset();
                await this.loadAgents();
                this.updateStats();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to create agent');
            }
        } catch (error) {
            console.error('Failed to create agent:', error);
            this.showToast(error.message, 'error');
        }
    }

    async deployAgent(agentName) {
        try {
            this.showToast(`Deploying agent ${agentName}...`, 'info');
            
            // This would typically call a deployment endpoint
            // For now, we'll simulate the process
            setTimeout(() => {
                this.showToast(`Agent ${agentName} deployed successfully`, 'success');
                this.loadAgents(); // Refresh the list
            }, 2000);
        } catch (error) {
            console.error('Failed to deploy agent:', error);
            this.showToast('Failed to deploy agent', 'error');
        }
    }

    async editAgent(agentName) {
        // For now, we'll just show the agent details
        // In a real implementation, you might want to show an edit form
        this.showAgentDetails(agentName);
    }

    async loadAgentDetails(agentName) {
        try {
            const response = await fetch(`${API_BASE_URL}/agents/${agentName}`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (response.ok) {
                const agent = await response.json();
                this.renderAgentDetails(agent);
            } else {
                throw new Error('Failed to load agent details');
            }
        } catch (error) {
            console.error('Failed to load agent details:', error);
            this.showToast('Failed to load agent details', 'error');
        }
    }

    renderAgentDetails(agent) {
        const container = document.getElementById('agent-details-content');
        const title = document.getElementById('agent-details-title');
        
        if (title) title.textContent = agent.display_name || agent.name;
        
        if (container) {
            container.innerHTML = `
                <div class="agent-details">
                    <div class="detail-section">
                        <h3>Configuration</h3>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Name:</label>
                                <span>${agent.name}</span>
                            </div>
                            <div class="detail-item">
                                <label>Display Name:</label>
                                <span>${agent.display_name || 'Not set'}</span>
                            </div>
                            <div class="detail-item">
                                <label>LLM Model:</label>
                                <span>${agent.llm_model}</span>
                            </div>
                            <div class="detail-item">
                                <label>Embedding Model:</label>
                                <span>${agent.embedding_model}</span>
                            </div>
                            <div class="detail-item">
                                <label>Persona:</label>
                                <span>${agent.persona_prompt || 'Not set'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h3>Tools</h3>
                        <div class="tools-list">
                            ${(agent.tools || []).map(tool => `
                                <span class="tool-tag">${tool}</span>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h3>Data Sources</h3>
                        <div class="sources-list">
                            ${(agent.sources || []).length > 0 ? 
                                agent.sources.map(source => `
                                    <div class="source-item">
                                        <span class="source-type">${source.type}</span>
                                        <span class="source-path">${source.path || source.url || 'N/A'}</span>
                                    </div>
                                `).join('') : 
                                '<p class="text-muted">No data sources configured</p>'
                            }
                        </div>
                    </div>
                </div>
            `;
        }
    }

    // Workflow Management
    async createWorkflow() {
        const form = document.getElementById('create-workflow-form');
        const formData = new FormData(form);
        
        const workflowData = {
            name: formData.get('name'),
            description: formData.get('description'),
            agent_name: formData.get('agent_name') || null,
            steps: [
                {
                    name: 'start',
                    type: 'start',
                    description: 'Workflow start'
                }
            ],
            trigger: {
                type: 'manual',
                description: 'Manual trigger'
            }
        };

        try {
            const response = await fetch(`${API_BASE_URL}/workflows`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify(workflowData)
            });

            if (response.ok) {
                this.showToast('Workflow created successfully', 'success');
                this.hideCreateWorkflowModal();
                form.reset();
                await this.loadWorkflows();
                this.updateStats();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to create workflow');
            }
        } catch (error) {
            console.error('Failed to create workflow:', error);
            this.showToast(error.message, 'error');
        }
    }

    async triggerWorkflow(workflowName) {
        try {
            this.showToast(`Triggering workflow ${workflowName}...`, 'info');
            
            const response = await fetch(`${API_BASE_URL}/workflows/${workflowName}/trigger`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify({})
            });

            if (response.ok) {
                this.showToast(`Workflow ${workflowName} triggered successfully`, 'success');
            } else {
                throw new Error('Failed to trigger workflow');
            }
        } catch (error) {
            console.error('Failed to trigger workflow:', error);
            this.showToast('Failed to trigger workflow', 'error');
        }
    }

    async loadWorkflowDetails(workflowName) {
        try {
            const response = await fetch(`${API_BASE_URL}/workflows/${workflowName}`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (response.ok) {
                const workflow = await response.json();
                this.renderWorkflowDetails(workflow);
            } else {
                throw new Error('Failed to load workflow details');
            }
        } catch (error) {
            console.error('Failed to load workflow details:', error);
            this.showToast('Failed to load workflow details', 'error');
        }
    }

    renderWorkflowDetails(workflow) {
        const container = document.getElementById('workflow-details-content');
        const title = document.getElementById('workflow-details-title');
        
        if (title) title.textContent = workflow.name;
        
        if (container) {
            container.innerHTML = `
                <div class="workflow-details">
                    <div class="detail-section">
                        <h3>Description</h3>
                        <p>${workflow.description || 'No description provided'}</p>
                    </div>
                    
                    <div class="detail-section">
                        <h3>Configuration</h3>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Agent:</label>
                                <span>${workflow.agent_name || 'None'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Steps:</label>
                                <span>${workflow.steps?.length || 0}</span>
                            </div>
                            <div class="detail-item">
                                <label>Trigger:</label>
                                <span>${workflow.trigger?.type || 'Manual'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h3>Steps</h3>
                        <div class="steps-list">
                            ${(workflow.steps || []).map((step, index) => `
                                <div class="step-item">
                                    <div class="step-number">${index + 1}</div>
                                    <div class="step-info">
                                        <div class="step-name">${step.name}</div>
                                        <div class="step-type">${step.type}</div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        }
    }

    populateWorkflowAgentSelect() {
        const select = document.getElementById('workflow-agent');
        if (select) {
            select.innerHTML = '<option value="">No default agent</option>';
            this.agents.forEach(agent => {
                const option = document.createElement('option');
                option.value = agent.name;
                option.textContent = agent.display_name || agent.name;
                select.appendChild(option);
            });
        }
    }

    // Utility Methods
    getAgentStatus(agent) {
        // This is a simplified status check
        // In a real implementation, you'd check the actual deployment status
        return 'online';
    }

    formatTime(timestamp) {
        if (!timestamp) return 'Unknown';
        
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-notification');
        if (!container) {
            console.warn('Toast notification container not found');
            return;
        }

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-icon">
                ${type === 'success' ? '✓' : type === 'error' ? '✗' : type === 'warning' ? '⚠' : 'ℹ'}
            </div>
            <div class="toast-message">${message}</div>
        `;

        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
}

// Global functions for onclick handlers
function switchView(view) {
    dashboard.switchView(view);
}

function showCreateAgentModal() {
    dashboard.showCreateAgentModal();
}

function hideCreateAgentModal() {
    dashboard.hideCreateAgentModal();
}

function showCreateWorkflowModal() {
    dashboard.showCreateWorkflowModal();
}

function hideCreateWorkflowModal() {
    dashboard.hideCreateWorkflowModal();
}

function showAgentDetails(agentName) {
    dashboard.showAgentDetails(agentName);
}

function hideAgentDetailsModal() {
    dashboard.hideAgentDetailsModal();
}

function showWorkflowDetails(workflowName) {
    dashboard.showWorkflowDetails(workflowName);
}

function hideWorkflowDetailsModal() {
    dashboard.hideWorkflowDetailsModal();
}

function logout() {
    localStorage.removeItem('ragnetic_user_token');
    localStorage.removeItem('ragnetic_db_user_id');
    localStorage.removeItem('ragnetic_username');
    window.location.href = '/login';
}

// toggleSidebar function removed - no longer needed

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Ensure all required elements exist before initializing
    const requiredElements = [
        'toast-notification',
        'create-agent-form',
        'create-workflow-form'
    ];
    
    const missingElements = requiredElements.filter(id => !document.getElementById(id));
    
    if (missingElements.length > 0) {
        console.warn('Some required elements are missing:', missingElements);
    }
    
    try {
        window.dashboard = new Dashboard();
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
    }
});
