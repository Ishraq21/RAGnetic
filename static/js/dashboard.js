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

        const editAgentForm = document.getElementById('edit-agent-form');
        if (editAgentForm) {
            editAgentForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.updateAgent();
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
                    <button class="btn-danger" onclick="event.stopPropagation(); dashboard.deleteAgent('${agent.name}')">
                        Delete
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

    showEditAgentModal(agent) {
        this.populateEditForm(agent);
        this.showModal('edit-agent-modal');
    }

    hideEditAgentModal() {
        this.hideModal(document.getElementById('edit-agent-modal'));
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
    populateEditForm(agent) {
        const form = document.getElementById('edit-agent-form');
        if (!form) return;

        // Populate basic fields
        form.querySelector('#edit-agent-name').value = agent.name;
        form.querySelector('#edit-agent-display-name').value = agent.display_name || '';
        form.querySelector('#edit-agent-description').value = agent.description || '';
        form.querySelector('#edit-agent-persona').value = agent.persona_prompt || '';
        form.querySelector('#edit-agent-execution-prompt').value = agent.execution_prompt || '';
        form.querySelector('#edit-agent-llm-model').value = agent.llm_model;
        form.querySelector('#edit-agent-embedding-model').value = agent.embedding_model;
        
        // Populate vector store settings
        if (agent.vector_store) {
            form.querySelector('#edit-vector-store-type').value = agent.vector_store.type || 'faiss';
            form.querySelector('#edit-bm25-k').value = agent.vector_store.bm25_k || 5;
            form.querySelector('#edit-semantic-k').value = agent.vector_store.semantic_k || 5;
            form.querySelector('#edit-rerank-top-n').value = agent.vector_store.rerank_top_n || 5;
            form.querySelector('#edit-retrieval-strategy').value = agent.vector_store.retrieval_strategy || 'hybrid';
        }

        // Populate chunking settings
        if (agent.chunking) {
            form.querySelector('#edit-chunking-mode').value = agent.chunking.mode || 'default';
            form.querySelector('#edit-chunk-size').value = agent.chunking.chunk_size || 1000;
            form.querySelector('#edit-chunk-overlap').value = agent.chunking.chunk_overlap || 100;
        }

        // Populate model parameters
        if (agent.model_params) {
            form.querySelector('#edit-temperature').value = agent.model_params.temperature || 0.7;
            form.querySelector('#edit-max-tokens').value = agent.model_params.max_tokens || 2000;
        }

        // Populate scaling settings
        if (agent.scaling) {
            form.querySelector('#edit-parallel-ingestion').checked = agent.scaling.parallel_ingestion || false;
            form.querySelector('#edit-ingestion-workers').value = agent.scaling.num_ingestion_workers || 4;
        }

        // Populate tools
        const toolCheckboxes = form.querySelectorAll('input[name="edit-tools"]');
        toolCheckboxes.forEach(checkbox => {
            checkbox.checked = agent.tools && agent.tools.includes(checkbox.value);
        });

        // Populate data policies
        if (agent.data_policies && agent.data_policies.length > 0) {
            const piiPolicy = agent.data_policies.find(p => p.type === 'pii_redaction');
            if (piiPolicy) {
                const piiTypes = piiPolicy.pii_config?.types || [];
                if (piiTypes.includes('email') && piiTypes.includes('phone') && piiTypes.includes('ssn')) {
                    form.querySelector('#edit-pii-redaction').value = 'comprehensive';
                } else if (piiTypes.includes('email') && piiTypes.includes('phone')) {
                    form.querySelector('#edit-pii-redaction').value = 'basic';
                } else {
                    form.querySelector('#edit-pii-redaction').value = 'none';
                }
            }

            const keywordPolicy = agent.data_policies.find(p => p.type === 'keyword_filter');
            if (keywordPolicy) {
                form.querySelector('#edit-keyword-filtering').value = keywordPolicy.keyword_filter_config?.keywords?.join(', ') || '';
            }
        }

        // Update temperature display
        const tempSlider = form.querySelector('#edit-temperature');
        const tempValue = form.querySelector('#edit-temperature-value');
        if (tempSlider && tempValue) {
            tempValue.textContent = tempSlider.value;
        }
    }

    async updateAgent() {
        const form = document.getElementById('edit-agent-form');
        const formData = new FormData(form);
        
        // Get selected tools
        const selectedTools = [];
        const toolCheckboxes = form.querySelectorAll('input[name="edit-tools"]:checked');
        toolCheckboxes.forEach(checkbox => {
            selectedTools.push(checkbox.value);
        });
        
        // Get PII redaction settings
        const piiRedaction = formData.get('pii_redaction');
        let dataPolicies = [];
        
        if (piiRedaction && piiRedaction !== 'none') {
            const piiTypes = piiRedaction === 'basic' 
                ? ['email', 'phone'] 
                : ['email', 'phone', 'ssn', 'credit_card', 'name'];
            
            dataPolicies.push({
                type: 'pii_redaction',
                pii_config: {
                    types: piiTypes,
                    redaction_char: '*',
                    redaction_placeholder: null
                }
            });
        }
        
        // Get keyword filtering
        const keywordFiltering = formData.get('keyword_filtering');
        if (keywordFiltering && keywordFiltering.trim()) {
            const keywords = keywordFiltering.split(',').map(k => k.trim()).filter(k => k);
            if (keywords.length > 0) {
                dataPolicies.push({
                    type: 'keyword_filter',
                    keyword_filter_config: {
                        keywords: keywords,
                        action: 'redact',
                        redaction_char: '*',
                        redaction_placeholder: null
                    }
                });
            }
        }
        
        const agentData = {
            name: formData.get('name'),
            display_name: formData.get('display_name'),
            description: formData.get('description'),
            persona_prompt: formData.get('persona_prompt') || 'You are a helpful assistant.',
            execution_prompt: formData.get('execution_prompt') || null,
            llm_model: formData.get('llm_model'),
            embedding_model: formData.get('embedding_model'),
            tools: selectedTools,
            sources: [], // Keep existing sources for now
            vector_store: {
                type: formData.get('vector_store_type') || 'faiss',
                bm25_k: parseInt(formData.get('bm25_k')) || 5,
                semantic_k: parseInt(formData.get('semantic_k')) || 5,
                rerank_top_n: parseInt(formData.get('rerank_top_n')) || 5,
                hit_rate_k_value: 5,
                retrieval_strategy: formData.get('retrieval_strategy') || 'hybrid'
            },
            chunking: {
                mode: formData.get('chunking_mode') || 'default',
                chunk_size: parseInt(formData.get('chunk_size')) || 1000,
                chunk_overlap: parseInt(formData.get('chunk_overlap')) || 100,
                breakpoint_percentile_threshold: 95
            },
            model_params: {
                temperature: parseFloat(formData.get('temperature')) || 0.7,
                max_tokens: parseInt(formData.get('max_tokens')) || 2000,
                top_p: null
            },
            scaling: {
                parallel_ingestion: formData.get('parallel_ingestion') === 'on',
                num_ingestion_workers: parseInt(formData.get('ingestion_workers')) || 4
            },
            data_policies: dataPolicies
        };

        try {
            const response = await fetch(`${API_BASE_URL}/agents/${agentData.name}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify(agentData)
            });

            if (response.ok) {
                this.showToast('Agent updated successfully', 'success');
                this.hideEditAgentModal();
                await this.loadAgents();
                this.updateStats();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update agent');
            }
        } catch (error) {
            console.error('Failed to update agent:', error);
            this.showToast(error.message, 'error');
        }
    }

    async createAgent() {
        const form = document.getElementById('create-agent-form');
        const formData = new FormData(form);
        
        // Get selected tools
        const selectedTools = [];
        const toolCheckboxes = form.querySelectorAll('input[name="tools"]:checked');
        toolCheckboxes.forEach(checkbox => {
            selectedTools.push(checkbox.value);
        });
        
        // Get PII redaction settings
        const piiRedaction = formData.get('pii_redaction');
        let dataPolicies = [];
        
        if (piiRedaction && piiRedaction !== 'none') {
            const piiTypes = piiRedaction === 'basic' 
                ? ['email', 'phone'] 
                : ['email', 'phone', 'ssn', 'credit_card', 'name'];
            
            dataPolicies.push({
                type: 'pii_redaction',
                pii_config: {
                    types: piiTypes,
                    redaction_char: '*',
                    redaction_placeholder: null
                }
            });
        }
        
        // Get keyword filtering
        const keywordFiltering = formData.get('keyword_filtering');
        if (keywordFiltering && keywordFiltering.trim()) {
            const keywords = keywordFiltering.split(',').map(k => k.trim()).filter(k => k);
            if (keywords.length > 0) {
                dataPolicies.push({
                    type: 'keyword_filter',
                    keyword_filter_config: {
                        keywords: keywords,
                        action: 'redact',
                        redaction_char: '*',
                        redaction_placeholder: null
                    }
                });
            }
        }
        
        const agentData = {
            name: formData.get('name'),
            display_name: formData.get('display_name'),
            description: formData.get('description'),
            persona_prompt: formData.get('persona_prompt') || 'You are a helpful assistant.',
            execution_prompt: formData.get('execution_prompt') || null,
            llm_model: formData.get('llm_model'),
            embedding_model: formData.get('embedding_model'),
            tools: selectedTools,
            sources: [], // Will be populated when data sources are added
            vector_store: {
                type: formData.get('vector_store_type') || 'faiss',
                bm25_k: parseInt(formData.get('bm25_k')) || 5,
                semantic_k: parseInt(formData.get('semantic_k')) || 5,
                rerank_top_n: parseInt(formData.get('rerank_top_n')) || 5,
                hit_rate_k_value: 5,
                retrieval_strategy: formData.get('retrieval_strategy') || 'hybrid'
            },
            chunking: {
                mode: formData.get('chunking_mode') || 'default',
                chunk_size: parseInt(formData.get('chunk_size')) || 1000,
                chunk_overlap: parseInt(formData.get('chunk_overlap')) || 100,
                breakpoint_percentile_threshold: 95
            },
            model_params: {
                temperature: parseFloat(formData.get('temperature')) || 0.7,
                max_tokens: parseInt(formData.get('max_tokens')) || 2000,
                top_p: null
            },
            scaling: {
                parallel_ingestion: formData.get('parallel_ingestion') === 'on',
                num_ingestion_workers: parseInt(formData.get('ingestion_workers')) || 4
            },
            data_policies: dataPolicies
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
            this.showToast(`Checking agent ${agentName} deployment status...`, 'info');
            
            // Check agent inspection to see deployment status
            const response = await fetch(`${API_BASE_URL}/agents/${agentName}/inspection`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (response.ok) {
                const inspection = await response.json();
                if (inspection.is_deployed) {
                    this.showToast(`Agent ${agentName} is already deployed and ready`, 'success');
                } else {
                    this.showToast(`Agent ${agentName} is not yet deployed. Data embedding may still be in progress.`, 'warning');
                }
                this.loadAgents(); // Refresh the list
            } else {
                throw new Error('Failed to check deployment status');
            }
        } catch (error) {
            console.error('Failed to check agent deployment:', error);
            this.showToast('Failed to check agent deployment status', 'error');
        }
    }

    async editAgent(agentName) {
        // Load agent data and show edit form
        try {
            const response = await fetch(`${API_BASE_URL}/agents/${agentName}`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (response.ok) {
                const agent = await response.json();
                this.showEditAgentModal(agent);
            } else {
                throw new Error('Failed to load agent details');
            }
        } catch (error) {
            console.error('Failed to load agent details:', error);
            this.showToast('Failed to load agent details', 'error');
        }
    }

    async deleteAgent(agentName) {
        this.showDeleteConfirmationModal(agentName);
    }

    showDeleteConfirmationModal(agentName) {
        const modal = document.getElementById('delete-confirmation-modal');
        const agentNameElement = document.getElementById('delete-agent-name');
        
        if (agentNameElement) {
            agentNameElement.textContent = `This action cannot be undone and will permanently remove the agent "${agentName}" and all its associated data.`;
        }
        
        // Store the agent name for the confirmation
        this.agentToDelete = agentName;
        
        if (modal) {
            modal.classList.add('show');
        }
    }

    hideDeleteConfirmationModal() {
        const modal = document.getElementById('delete-confirmation-modal');
        if (modal) {
            modal.classList.remove('show');
        }
        this.agentToDelete = null;
    }

    async confirmDeleteAgent() {
        if (!this.agentToDelete) return;
        
        try {
            this.showToast(`Deleting agent ${this.agentToDelete}...`, 'info');
            this.hideDeleteConfirmationModal();
            
            const response = await fetch(`${API_BASE_URL}/agents/${this.agentToDelete}`, {
                method: 'DELETE',
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (response.ok) {
                this.showToast('Agent deleted successfully', 'success');
                await this.loadAgents();
                this.updateStats();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete agent');
            }
        } catch (error) {
            console.error('Failed to delete agent:', error);
            this.showToast(error.message, 'error');
        }
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

    // Advanced Settings Toggle
    toggleAdvancedSettings() {
        const advancedSettings = document.getElementById('advanced-settings');
        const toggleText = document.getElementById('advanced-toggle-text');
        const toggleIcon = document.getElementById('advanced-toggle-icon');
        
        if (advancedSettings.classList.contains('hidden')) {
            advancedSettings.classList.remove('hidden');
            toggleText.textContent = 'Hide Advanced';
            toggleIcon.style.transform = 'rotate(180deg)';
        } else {
            advancedSettings.classList.add('hidden');
            toggleText.textContent = 'Show Advanced';
            toggleIcon.style.transform = 'rotate(0deg)';
        }
    }

    // Data Source Management
    addDataSource() {
        const container = document.getElementById('data-sources-container');
        const emptyState = container.querySelector('.empty-state');
        
        if (emptyState) {
            emptyState.remove();
        }
        
        const dataSourceId = Date.now();
        const dataSourceHtml = `
            <div class="data-source-form" id="data-source-${dataSourceId}">
                <div class="data-source-header">
                    <h4>Data Source ${container.children.length + 1}</h4>
                    <button type="button" class="remove-data-source" onclick="dashboard.removeDataSource(${dataSourceId})">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="source-type-${dataSourceId}">Source Type</label>
                        <select id="source-type-${dataSourceId}" name="source_type_${dataSourceId}" required>
                            <option value="local">Local Files</option>
                            <option value="url">Web URL</option>
                            <option value="api">API Endpoint</option>
                            <option value="db">Database</option>
                            <option value="code_repository">Code Repository</option>
                            <option value="gdoc">Google Docs</option>
                            <option value="web_crawler">Web Crawler</option>
                            <option value="notebook">Jupyter Notebook</option>
                            <option value="parquet">Parquet Files</option>
                            <option value="csv">CSV Files</option>
                            <option value="pdf">PDF Documents</option>
                            <option value="txt">Text Files</option>
                            <option value="docx">Word Documents</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="source-path-${dataSourceId}">Path/URL</label>
                        <input type="text" id="source-path-${dataSourceId}" name="source_path_${dataSourceId}" 
                               placeholder="Enter file path, URL, or connection string">
                    </div>
                </div>
                <div class="form-group">
                    <label for="source-description-${dataSourceId}">Description</label>
                    <input type="text" id="source-description-${dataSourceId}" name="source_description_${dataSourceId}" 
                           placeholder="Brief description of this data source">
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', dataSourceHtml);
    }

    removeDataSource(dataSourceId) {
        const dataSource = document.getElementById(`data-source-${dataSourceId}`);
        if (dataSource) {
            dataSource.remove();
            
            // Check if we need to show the empty state
            const container = document.getElementById('data-sources-container');
            if (container.children.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14,2 14,8 20,8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <polyline points="10,9 9,9 8,9"></polyline>
                        </svg>
                        <p>No data sources configured</p>
                        <small>Your agent will start with general knowledge only</small>
                    </div>
                `;
            }
        }
    }

    // Initialize form event listeners
    initializeFormListeners() {
        // Temperature range input for create form
        const temperatureInput = document.getElementById('temperature');
        if (temperatureInput) {
            const temperatureValue = document.getElementById('temperature-value');
            temperatureInput.addEventListener('input', (e) => {
                if (temperatureValue) {
                    temperatureValue.textContent = e.target.value;
                }
            });
        }

        // Temperature range input for edit form
        const editTemperatureInput = document.getElementById('edit-temperature');
        if (editTemperatureInput) {
            const editTemperatureValue = document.getElementById('edit-temperature-value');
            editTemperatureInput.addEventListener('input', (e) => {
                if (editTemperatureValue) {
                    editTemperatureValue.textContent = e.target.value;
                }
            });
        }

        // Toggle switch for parallel ingestion (create form)
        const parallelIngestionToggle = document.getElementById('parallel-ingestion');
        if (parallelIngestionToggle) {
            parallelIngestionToggle.addEventListener('change', (e) => {
                const workersInput = document.getElementById('ingestion-workers');
                if (workersInput) {
                    workersInput.disabled = !e.target.checked;
                    if (!e.target.checked) {
                        workersInput.value = 4;
                    }
                }
            });
        }

        // Toggle switch for parallel ingestion (edit form)
        const editParallelIngestionToggle = document.getElementById('edit-parallel-ingestion');
        if (editParallelIngestionToggle) {
            editParallelIngestionToggle.addEventListener('change', (e) => {
                const workersInput = document.getElementById('edit-ingestion-workers');
                if (workersInput) {
                    workersInput.disabled = !e.target.checked;
                    if (!e.target.checked) {
                        workersInput.value = 4;
                    }
                }
            });
        }
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

function showEditAgentModal(agentName) {
    dashboard.editAgent(agentName);
}

function hideEditAgentModal() {
    dashboard.hideEditAgentModal();
}

function deleteAgent(agentName) {
    dashboard.deleteAgent(agentName);
}

function hideDeleteConfirmationModal() {
    dashboard.hideDeleteConfirmationModal();
}

function confirmDeleteAgent() {
    dashboard.confirmDeleteAgent();
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

// Global functions for the form
function toggleAdvancedSettings() {
    dashboard.toggleAdvancedSettings();
}

function addDataSource() {
    dashboard.addDataSource();
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    dashboard = new Dashboard();
    dashboard.initializeFormListeners();
});
