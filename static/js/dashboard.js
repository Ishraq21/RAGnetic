// Dashboard JavaScript functionality
console.log('Dashboard.js loaded successfully');

class Dashboard {
    constructor() {
        this.currentView = 'overview';
        this.agents = [];
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
                console.log('Loaded agents:', this.agents);
                this.renderAgents();
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('Failed to load agents:', error);
            this.showToast('Failed to load agents', 'error');
        }
    }


    async loadRecentActivity() {
        try {
            // Load recent agent runs from audit API
            const auditUrl = `${API_BASE_URL}/audit/runs?limit=5`;
            console.log('Loading recent agent runs from:', auditUrl, '(FIXED VERSION)');
            
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

        } catch (error) {
            console.error('Failed to load recent activity:', error);
            // Show no recent activity on error
            this.renderRecentAgentActivity([]);
        }
    }

    updateStats() {
        const totalAgentsElement = document.getElementById('total-agents');
        const totalRunsElement = document.getElementById('total-runs');
        const successRateElement = document.getElementById('success-rate');
        const totalTrainingJobsElement = document.getElementById('total-training-jobs');
        
        if (totalAgentsElement) totalAgentsElement.textContent = this.agents.length;
        
        // Calculate total runs and success rate (placeholder for now)
        if (totalRunsElement) totalRunsElement.textContent = '0';
        if (successRateElement) successRateElement.textContent = '0%';
        
        // Load training statistics
        this.loadTrainingStats();
    }

    async loadTrainingStats() {
        try {
            const response = await fetch('/api/v1/training/stats', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (response.ok) {
                const stats = await response.json();
                const totalTrainingJobsElement = document.getElementById('total-training-jobs');
                if (totalTrainingJobsElement) {
                    totalTrainingJobsElement.textContent = stats.total_jobs || 0;
                }
            }
        } catch (error) {
            console.error('Failed to load training stats:', error);
            const totalTrainingJobsElement = document.getElementById('total-training-jobs');
            if (totalTrainingJobsElement) {
                totalTrainingJobsElement.textContent = '0';
            }
        }
    }

    async loadAvailableModels() {
        try {
            const response = await fetch('/api/v1/training/models/available', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const models = await response.json();
            this.populateModelDropdowns(models);
        } catch (error) {
            console.error('Failed to load available models:', error);
        }
    }

    populateModelDropdowns(models) {
        const createDropdown = document.getElementById('agent-llm-model');
        const editDropdown = document.getElementById('edit-agent-llm-model');
        
        if (!createDropdown && !editDropdown) return;

        // Group models by type
        const fineTunedModels = models.filter(m => m.type === 'fine_tuned');
        const baseModels = models.filter(m => m.type === 'base');

        const createDropdownHTML = this.buildModelDropdownHTML(fineTunedModels, baseModels);
        const editDropdownHTML = this.buildModelDropdownHTML(fineTunedModels, baseModels);

        if (createDropdown) {
            createDropdown.innerHTML = createDropdownHTML;
        }
        if (editDropdown) {
            editDropdown.innerHTML = editDropdownHTML;
        }
    }

    buildModelDropdownHTML(fineTunedModels, baseModels) {
        let html = '';
        
        // Add fine-tuned models first
        if (fineTunedModels.length > 0) {
            html += '<optgroup label="Fine-tuned Models">';
            fineTunedModels.forEach(model => {
                html += `<option value="${model.value}">${model.label}</option>`;
            });
            html += '</optgroup>';
        }

        // Add base models
        html += '<optgroup label="OpenAI Models">';
        baseModels.filter(m => m.value.startsWith('gpt')).forEach(model => {
            html += `<option value="${model.value}">${model.label}</option>`;
        });
        html += '</optgroup>';

        html += '<optgroup label="Anthropic Models">';
        baseModels.filter(m => m.value.startsWith('claude')).forEach(model => {
            html += `<option value="${model.value}">${model.label}</option>`;
        });
        html += '</optgroup>';

        html += '<optgroup label="Google Models">';
        baseModels.filter(m => m.value.startsWith('gemini')).forEach(model => {
            html += `<option value="${model.value}">${model.label}</option>`;
        });
        html += '</optgroup>';

        html += '<optgroup label="Ollama Models">';
        baseModels.filter(m => m.value.startsWith('ollama')).forEach(model => {
            html += `<option value="${model.value}">${model.label}</option>`;
        });
        html += '</optgroup>';

        return html;
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

        const agentsHtml = this.agents.map(agent => this.renderAgentCard(agent)).join('');
        container.innerHTML = `
            <div class="list-header">Agents</div>
            ${agentsHtml}
        `;
    }

    renderAgentCard(agent) {
        console.log('renderAgentCard called with agent:', agent);
        if (!agent || !agent.name) {
            console.error('Invalid agent data in renderAgentCard:', agent);
            return '';
        }
        
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
                    <button class="btn-danger" onclick="event.stopPropagation(); console.log('Delete button clicked for agent:', '${agent.name}'); dashboard.deleteAgent('${agent.name}')">
                        Delete
                    </button>
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
                    ${run.status === 'completed' ? '' : run.status === 'failed' ? '' : '⋯'}
                </div>
                <div class="activity-content">
                    <div class="activity-title">${run.agent_name || 'Unknown Agent'}</div>
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
        } else if (view === 'training') {
            // Initialize training dashboard if not already done
            if (typeof trainingDashboard !== 'undefined') {
                trainingDashboard.loadTrainingJobs();
            } else {
                // Wait for training dashboard to be initialized
                const checkTrainingDashboard = setInterval(() => {
                    if (typeof trainingDashboard !== 'undefined') {
                        trainingDashboard.loadTrainingJobs();
                        clearInterval(checkTrainingDashboard);
                    }
                }, 100);
                
                // Clear interval after 5 seconds to prevent infinite checking
                setTimeout(() => clearInterval(checkTrainingDashboard), 5000);
            }
        } else if (view === 'models') {
            // Load fine-tuned models when the models view is activated
            console.log('Switching to models view, loading models...');
            if (window.fineTunedModelsManager) {
                console.log('FineTunedModelsManager found, loading models...');
                fineTunedModelsManager.loadModels();
            } else {
                console.error('FineTunedModelsManager not found!');
            }
        } else if (view === 'overview') {
            this.loadOverviewData();
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


    showAgentDetails(agentName) {
        this.showModal('agent-details-modal');
        this.loadAgentDetails(agentName);
    }

    hideAgentDetailsModal() {
        this.hideModal(document.getElementById('agent-details-modal'));
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
            
            // Populate advanced vector store settings
            if (agent.vector_store.qdrant_host) {
                form.querySelector('#edit-qdrant-host').value = agent.vector_store.qdrant_host;
            }
            if (agent.vector_store.qdrant_port) {
                form.querySelector('#edit-qdrant-port').value = agent.vector_store.qdrant_port;
            }
            if (agent.vector_store.pinecone_index_name) {
                form.querySelector('#edit-pinecone-index-name').value = agent.vector_store.pinecone_index_name;
            }
            if (agent.vector_store.mongodb_db_name) {
                form.querySelector('#edit-mongodb-db-name').value = agent.vector_store.mongodb_db_name;
            }
            if (agent.vector_store.mongodb_collection_name) {
                form.querySelector('#edit-mongodb-collection-name').value = agent.vector_store.mongodb_collection_name;
            }
            if (agent.vector_store.mongodb_index_name) {
                form.querySelector('#edit-mongodb-index-name').value = agent.vector_store.mongodb_index_name;
            }
            
            // Show relevant advanced settings
            this.toggleVectorStoreSettings(agent.vector_store.type || 'faiss', 'edit-');
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
            form.querySelector('#edit-top-p').value = agent.model_params.top_p || 1.0;
        }
        
        // Populate LLM retry and timeout settings
        form.querySelector('#edit-llm-retries').value = agent.llm_retries || 0;
        form.querySelector('#edit-llm-timeout').value = agent.llm_timeout || 60;
        
        // Populate evaluation LLM model
        if (agent.evaluation_llm_model) {
            form.querySelector('#edit-evaluation-llm-model').value = agent.evaluation_llm_model;
        }

        // Populate scaling settings
        if (agent.scaling) {
            form.querySelector('#edit-parallel-ingestion').checked = agent.scaling.parallel_ingestion || false;
            form.querySelector('#edit-ingestion-workers').value = agent.scaling.num_ingestion_workers || 4;
        }
        
        // Populate advanced configuration
        if (agent.fine_tuned_model_id) {
            form.querySelector('#edit-fine-tuned-model-id').value = agent.fine_tuned_model_id;
        }
        
        if (agent.benchmark) {
            form.querySelector('#edit-max-context-docs').value = agent.benchmark.max_context_docs || 20;
        }
        
        form.querySelector('#edit-reproducible-ids').checked = agent.reproducible_ids !== false;

        // Populate tools
        const toolCheckboxes = form.querySelectorAll('input[name="edit-tools"]');
        toolCheckboxes.forEach(checkbox => {
            checkbox.checked = agent.tools && agent.tools.includes(checkbox.value);
        });

        // Populate data sources
        this.populateEditDataSources(agent.sources || []);

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
        
        // Update top-p display
        const topPSlider = form.querySelector('#edit-top-p');
        const topPValue = form.querySelector('#edit-top-p-value');
        if (topPSlider && topPValue) {
            topPValue.textContent = topPSlider.value;
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
        
        // Process data sources (file uploads and other sources)
        const dataSources = await this.processDataSources(form);
        
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
            sources: dataSources,
            vector_store: {
                type: formData.get('vector_store_type') || 'faiss',
                bm25_k: parseInt(formData.get('bm25_k')) || 5,
                semantic_k: parseInt(formData.get('semantic_k')) || 5,
                rerank_top_n: parseInt(formData.get('rerank_top_n')) || 5,
                hit_rate_k_value: 5,
                retrieval_strategy: formData.get('retrieval_strategy') || 'hybrid',
                // Advanced vector store settings
                qdrant_host: formData.get('qdrant_host') || null,
                qdrant_port: parseInt(formData.get('qdrant_port')) || null,
                pinecone_index_name: formData.get('pinecone_index_name') || null,
                mongodb_db_name: formData.get('mongodb_db_name') || null,
                mongodb_collection_name: formData.get('mongodb_collection_name') || null,
                mongodb_index_name: formData.get('mongodb_index_name') || null
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
                top_p: parseFloat(formData.get('top_p')) || 1.0
            },
            llm_retries: parseInt(formData.get('llm_retries')) || 0,
            llm_timeout: parseInt(formData.get('llm_timeout')) || 60,
            evaluation_llm_model: formData.get('evaluation_llm_model') || null,
            fine_tuned_model_id: formData.get('fine_tuned_model_id') || null,
            benchmark: {
                context_window_tokens: parseInt(formData.get('context_window_tokens')) || 8000,
                context_budget_ratio: parseFloat(formData.get('context_budget_ratio')) || 0.70,
                answer_reserve_tokens: parseInt(formData.get('answer_reserve_tokens')) || 1024,
                enable_doc_truncation: formData.get('enable_doc_truncation') === 'true',
                max_context_docs: parseInt(formData.get('max_context_docs')) || 20
            },
            reproducible_ids: formData.get('reproducible_ids') === 'on',
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
        
        // Process data sources (file uploads and other sources)
        const dataSources = await this.processDataSources(form);
        
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
            sources: dataSources,
            vector_store: {
                type: formData.get('vector_store_type') || 'faiss',
                bm25_k: parseInt(formData.get('bm25_k')) || 5,
                semantic_k: parseInt(formData.get('semantic_k')) || 5,
                rerank_top_n: parseInt(formData.get('rerank_top_n')) || 5,
                hit_rate_k_value: 5,
                retrieval_strategy: formData.get('retrieval_strategy') || 'hybrid',
                // Advanced vector store settings
                qdrant_host: formData.get('qdrant_host') || null,
                qdrant_port: parseInt(formData.get('qdrant_port')) || null,
                pinecone_index_name: formData.get('pinecone_index_name') || null,
                mongodb_db_name: formData.get('mongodb_db_name') || null,
                mongodb_collection_name: formData.get('mongodb_collection_name') || null,
                mongodb_index_name: formData.get('mongodb_index_name') || null
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
                top_p: parseFloat(formData.get('top_p')) || 1.0
            },
            llm_retries: parseInt(formData.get('llm_retries')) || 0,
            llm_timeout: parseInt(formData.get('llm_timeout')) || 60,
            evaluation_llm_model: formData.get('evaluation_llm_model') || null,
            fine_tuned_model_id: formData.get('fine_tuned_model_id') || null,
            benchmark: {
                context_window_tokens: parseInt(formData.get('context_window_tokens')) || 8000,
                context_budget_ratio: parseFloat(formData.get('context_budget_ratio')) || 0.70,
                answer_reserve_tokens: parseInt(formData.get('answer_reserve_tokens')) || 1024,
                enable_doc_truncation: formData.get('enable_doc_truncation') === 'true',
                max_context_docs: parseInt(formData.get('max_context_docs')) || 20
            },
            reproducible_ids: formData.get('reproducible_ids') === 'on',
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
        console.log('deleteAgent called with:', agentName);
        if (!agentName || agentName === 'null' || agentName === 'undefined') {
            console.error('Invalid agent name:', agentName);
            this.showToast('Invalid agent name', 'error');
            return;
        }
        this.showDeleteConfirmationModal(agentName);
    }

    showDeleteConfirmationModal(agentName) {
        console.log('showDeleteConfirmationModal called with:', agentName);
        if (!agentName || agentName === 'null' || agentName === 'undefined') {
            console.error('Invalid agent name in showDeleteConfirmationModal:', agentName);
            this.showToast('Invalid agent name', 'error');
            return;
        }
        
        const modal = document.getElementById('delete-confirmation-modal');
        const agentNameElement = document.getElementById('delete-agent-name');
        
        if (agentNameElement) {
            agentNameElement.textContent = agentName;
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
        console.log('confirmDeleteAgent called, agentToDelete:', this.agentToDelete);
        if (!this.agentToDelete || this.agentToDelete === 'null' || this.agentToDelete === 'undefined') {
            console.error('No agent to delete or invalid agent name:', this.agentToDelete);
            this.showToast('No agent selected for deletion', 'error');
            this.hideDeleteConfirmationModal();
            return;
        }
        
        // Store the agent name before hiding the modal
        const agentToDelete = this.agentToDelete;
        
        try {
            this.showToast(`Deleting agent ${agentToDelete}...`, 'info');
            this.hideDeleteConfirmationModal();
            
            const response = await fetch(`${API_BASE_URL}/agents/${encodeURIComponent(agentToDelete)}`, {
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
                                <label>Evaluation Model:</label>
                                <span>${agent.evaluation_llm_model || 'Same as main LLM'}</span>
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
                ${type === 'success' ? '' : type === 'error' ? '' : type === 'warning' ? '' : 'ℹ'}
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
                        <select id="source-type-${dataSourceId}" name="source_type_${dataSourceId}" required onchange="dashboard.toggleDataSourceInput(${dataSourceId})">
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
                </div>
                <div class="form-group" id="file-upload-group-${dataSourceId}">
                    <label>Upload Files</label>
                    <div class="file-upload-area" id="file-upload-${dataSourceId}">
                        <div class="file-upload-content">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7,10 12,15 17,10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                            <h4>Drop files here or click to browse</h4>
                            <p>Supports PDF, DOCX, TXT, CSV, Parquet, and more</p>
                            <input type="file" id="file-input-${dataSourceId}" name="files_${dataSourceId}" multiple accept=".pdf,.docx,.txt,.csv,.parquet,.json,.md,.py,.ipynb" style="display: none;">
                        </div>
                        <div class="file-list" id="file-list-${dataSourceId}" style="display: none;">
                            <!-- Uploaded files will be listed here -->
                        </div>
                    </div>
                </div>
                <div class="form-group" id="path-input-group-${dataSourceId}" style="display: none;">
                    <label for="source-path-${dataSourceId}">Path/URL</label>
                    <input type="text" id="source-path-${dataSourceId}" name="source_path_${dataSourceId}" 
                           placeholder="Enter file path, URL, or connection string">
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', dataSourceHtml);
        
        // Initialize file upload functionality for this data source
        this.initializeFileUpload(dataSourceId);
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

    toggleDataSourceInput(dataSourceId) {
        const sourceType = document.getElementById(`source-type-${dataSourceId}`).value;
        const fileUploadGroup = document.getElementById(`file-upload-group-${dataSourceId}`);
        const pathInputGroup = document.getElementById(`path-input-group-${dataSourceId}`);
        
        if (sourceType === 'local' || sourceType === 'pdf' || sourceType === 'txt' || 
            sourceType === 'docx' || sourceType === 'csv' || sourceType === 'parquet' || 
            sourceType === 'notebook') {
            fileUploadGroup.style.display = 'block';
            pathInputGroup.style.display = 'none';
        } else {
            fileUploadGroup.style.display = 'none';
            pathInputGroup.style.display = 'block';
        }
    }

    initializeFileUpload(dataSourceId) {
        const fileUploadArea = document.getElementById(`file-upload-${dataSourceId}`);
        const fileInput = document.getElementById(`file-input-${dataSourceId}`);
        const fileList = document.getElementById(`file-list-${dataSourceId}`);
        
        // Click to browse files
        fileUploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop functionality
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('drag-over');
        });
        
        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('drag-over');
        });
        
        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            this.handleFileSelection(dataSourceId, files);
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(dataSourceId, e.target.files);
        });
    }

    handleFileSelection(dataSourceId, files) {
        const fileList = document.getElementById(`file-list-${dataSourceId}`);
        const fileUploadArea = document.getElementById(`file-upload-${dataSourceId}`);
        const fileUploadContent = fileUploadArea ? fileUploadArea.querySelector('.file-upload-content') : null;
        
        if (files.length > 0) {
            if (fileUploadContent) {
                fileUploadContent.style.display = 'none';
            }
            if (fileList) {
                fileList.style.display = 'block';
            }
            
            let fileListHtml = '<div class="file-list-header"><h5>Selected Files</h5></div>';
            
            Array.from(files).forEach((file, index) => {
                const fileSize = this.formatFileSize(file.size);
                const fileIcon = this.getFileIcon(file.type, file.name);
                
                fileListHtml += `
                    <div class="file-item">
                        <div class="file-info">
                            <div class="file-icon">${fileIcon}</div>
                            <div class="file-details">
                                <span class="file-name">${file.name}</span>
                                <span class="file-size">${fileSize}</span>
                            </div>
                        </div>
                        <button type="button" class="remove-file" onclick="dashboard.removeFile(${dataSourceId}, ${index})">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                `;
            });
            
            fileList.innerHTML = fileListHtml;
        }
    }

    removeFile(dataSourceId, fileIndex) {
        const fileInput = document.getElementById(`file-input-${dataSourceId}`);
        const fileList = document.getElementById(`file-list-${dataSourceId}`);
        const fileUploadArea = document.getElementById(`file-upload-${dataSourceId}`);
        const fileUploadContent = fileUploadArea ? fileUploadArea.querySelector('.file-upload-content') : null;
        
        // Create new FileList without the removed file
        const dt = new DataTransfer();
        const files = Array.from(fileInput.files);
        files.splice(fileIndex, 1);
        
        files.forEach(file => dt.items.add(file));
        fileInput.files = dt.files;
        
        if (fileInput.files.length === 0) {
            if (fileList) {
                fileList.style.display = 'none';
            }
            if (fileUploadContent) {
                fileUploadContent.style.display = 'block';
            }
        } else {
            this.handleFileSelection(dataSourceId, fileInput.files);
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getFileIcon(type, filename) {
        const extension = filename.split('.').pop().toLowerCase();
        
        const icons = {
            'pdf': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'docx': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'txt': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'csv': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'json': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'py': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'ipynb': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>',
            'md': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14,2 14,8 20,8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>'
        };
        
        return icons[extension] || icons['txt'];
    }

    async processDataSources(form) {
        const dataSources = [];
        
        // Find all data source forms (both create and edit)
        const dataSourceForms = form.querySelectorAll('.data-source-form');
        
        for (const dataSourceForm of dataSourceForms) {
            const sourceTypeSelect = dataSourceForm.querySelector('select[name*="source_type"]');
            
            if (!sourceTypeSelect) continue;
            
            const sourceType = sourceTypeSelect.value;
            
            if (sourceType === 'local' || sourceType === 'pdf' || sourceType === 'txt' || 
                sourceType === 'docx' || sourceType === 'csv' || sourceType === 'parquet' || 
                sourceType === 'notebook') {
                // Handle file uploads
                const fileInput = dataSourceForm.querySelector('input[type="file"]');
                if (fileInput && fileInput.files.length > 0) {
                    // Upload files and create data sources
                    for (const file of fileInput.files) {
                        try {
                            const uploadedPath = await this.uploadFile(file);
                            dataSources.push({
                                type: 'local',
                                path: uploadedPath
                            });
                        } catch (error) {
                            console.error('Failed to upload file:', error);
                            this.showToast(`Failed to upload ${file.name}: ${error.message}`, 'error');
                        }
                    }
                }
            } else {
                // Handle other source types (URL, API, etc.)
                const pathInput = dataSourceForm.querySelector('input[name*="source_path"]');
                if (pathInput && pathInput.value.trim()) {
                    const sourceConfig = {
                        type: sourceType
                    };
                    
                    // Add appropriate field based on source type
                    if (sourceType === 'url') {
                        sourceConfig.url = pathInput.value.trim();
                    } else if (sourceType === 'api') {
                        sourceConfig.url = pathInput.value.trim();
                    } else if (sourceType === 'db') {
                        sourceConfig.db_connection = pathInput.value.trim();
                    } else if (sourceType === 'gdoc') {
                        sourceConfig.folder_id = pathInput.value.trim();
                    } else {
                        sourceConfig.path = pathInput.value.trim();
                    }
                    
                    dataSources.push(sourceConfig);
                }
            }
        }
        
        return dataSources;
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/agents/upload-file`, {
            method: 'POST',
            body: formData,
            headers: {
                'X-API-Key': loggedInUserToken
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
        }
        
        const result = await response.json();
        return result.file_path;
    }

    populateEditDataSources(sources) {
        const container = document.getElementById('edit-data-sources-container');
        if (!container) return;

        // Clear existing data sources
        container.innerHTML = '';

        if (sources.length === 0) {
            // Show empty state
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
            return;
        }

        // Add each data source
        sources.forEach((source, index) => {
            this.addEditDataSource(source, index + 1);
        });
    }

    addEditDataSource(source = null, sourceNumber = null) {
        const container = document.getElementById('edit-data-sources-container');
        const emptyState = container.querySelector('.empty-state');
        
        if (emptyState) {
            emptyState.remove();
        }
        
        const dataSourceId = Date.now() + Math.floor(Math.random() * 1000);
        const sourceType = source ? source.type : 'local';
        const sourcePath = source ? (source.path || source.url || source.db_connection || source.folder_id || '') : '';
        
        const dataSourceHtml = `
            <div class="data-source-form" id="edit-data-source-${dataSourceId}">
                <div class="data-source-header">
                    <h4>Data Source ${sourceNumber || container.children.length + 1}</h4>
                    <button type="button" class="remove-data-source" onclick="dashboard.removeEditDataSource(${dataSourceId})">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="edit-source-type-${dataSourceId}">Source Type</label>
                        <select id="edit-source-type-${dataSourceId}" name="edit_source_type_${dataSourceId}" required onchange="dashboard.toggleEditDataSourceInput(${dataSourceId})">
                            <option value="local" ${sourceType === 'local' ? 'selected' : ''}>Local Files</option>
                            <option value="url" ${sourceType === 'url' ? 'selected' : ''}>Web URL</option>
                            <option value="api" ${sourceType === 'api' ? 'selected' : ''}>API Endpoint</option>
                            <option value="db" ${sourceType === 'db' ? 'selected' : ''}>Database</option>
                            <option value="code_repository" ${sourceType === 'code_repository' ? 'selected' : ''}>Code Repository</option>
                            <option value="gdoc" ${sourceType === 'gdoc' ? 'selected' : ''}>Google Docs</option>
                            <option value="web_crawler" ${sourceType === 'web_crawler' ? 'selected' : ''}>Web Crawler</option>
                            <option value="notebook" ${sourceType === 'notebook' ? 'selected' : ''}>Jupyter Notebook</option>
                            <option value="parquet" ${sourceType === 'parquet' ? 'selected' : ''}>Parquet Files</option>
                            <option value="csv" ${sourceType === 'csv' ? 'selected' : ''}>CSV Files</option>
                            <option value="pdf" ${sourceType === 'pdf' ? 'selected' : ''}>PDF Documents</option>
                            <option value="txt" ${sourceType === 'txt' ? 'selected' : ''}>Text Files</option>
                            <option value="docx" ${sourceType === 'docx' ? 'selected' : ''}>Word Documents</option>
                        </select>
                    </div>
                </div>
                <div class="form-group" id="edit-file-upload-group-${dataSourceId}" style="display: ${sourceType === 'local' || sourceType === 'pdf' || sourceType === 'txt' || sourceType === 'docx' || sourceType === 'csv' || sourceType === 'parquet' || sourceType === 'notebook' ? 'block' : 'none'}">
                    <label>Upload Files</label>
                    <div class="file-upload-area" id="edit-file-upload-${dataSourceId}">
                        <div class="file-upload-content">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7,10 12,15 17,10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                            <h4>Drop files here or click to browse</h4>
                            <p>Supports PDF, DOCX, TXT, CSV, Parquet, and more</p>
                            <input type="file" id="edit-file-input-${dataSourceId}" name="edit_files_${dataSourceId}" multiple accept=".pdf,.docx,.txt,.csv,.parquet,.json,.md,.py,.ipynb" style="display: none;">
                        </div>
                        <div class="file-list" id="edit-file-list-${dataSourceId}" style="display: none;">
                            <!-- Uploaded files will be listed here -->
                        </div>
                    </div>
                </div>
                <div class="form-group" id="edit-path-input-group-${dataSourceId}" style="display: ${sourceType === 'local' || sourceType === 'pdf' || sourceType === 'txt' || sourceType === 'docx' || sourceType === 'csv' || sourceType === 'parquet' || sourceType === 'notebook' ? 'none' : 'block'}">
                    <label for="edit-source-path-${dataSourceId}">Path/URL</label>
                    <input type="text" id="edit-source-path-${dataSourceId}" name="edit_source_path_${dataSourceId}" 
                           placeholder="Enter file path, URL, or connection string" value="${sourcePath}">
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', dataSourceHtml);
        
        // Initialize file upload functionality for this data source
        this.initializeEditFileUpload(dataSourceId);
    }

    removeEditDataSource(dataSourceId) {
        const dataSource = document.getElementById(`edit-data-source-${dataSourceId}`);
        if (dataSource) {
            dataSource.remove();
            
            // Check if we need to show the empty state
            const container = document.getElementById('edit-data-sources-container');
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

    toggleEditDataSourceInput(dataSourceId) {
        const sourceType = document.getElementById(`edit-source-type-${dataSourceId}`).value;
        const fileUploadGroup = document.getElementById(`edit-file-upload-group-${dataSourceId}`);
        const pathInputGroup = document.getElementById(`edit-path-input-group-${dataSourceId}`);
        
        if (sourceType === 'local' || sourceType === 'pdf' || sourceType === 'txt' || 
            sourceType === 'docx' || sourceType === 'csv' || sourceType === 'parquet' || 
            sourceType === 'notebook') {
            fileUploadGroup.style.display = 'block';
            pathInputGroup.style.display = 'none';
        } else {
            fileUploadGroup.style.display = 'none';
            pathInputGroup.style.display = 'block';
        }
    }

    initializeEditFileUpload(dataSourceId) {
        const fileUploadArea = document.getElementById(`edit-file-upload-${dataSourceId}`);
        const fileInput = document.getElementById(`edit-file-input-${dataSourceId}`);
        const fileList = document.getElementById(`edit-file-list-${dataSourceId}`);
        
        // Click to browse files
        fileUploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop functionality
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('drag-over');
        });
        
        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('drag-over');
        });
        
        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            this.handleEditFileSelection(dataSourceId, files);
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleEditFileSelection(dataSourceId, e.target.files);
        });
    }

    handleEditFileSelection(dataSourceId, files) {
        const fileList = document.getElementById(`edit-file-list-${dataSourceId}`);
        const fileUploadArea = document.getElementById(`edit-file-upload-${dataSourceId}`);
        const fileUploadContent = fileUploadArea ? fileUploadArea.querySelector('.file-upload-content') : null;
        
        if (files.length > 0) {
            if (fileUploadContent) {
                fileUploadContent.style.display = 'none';
            }
            if (fileList) {
                fileList.style.display = 'block';
            }
            
            let fileListHtml = '<div class="file-list-header"><h5>Selected Files</h5></div>';
            
            Array.from(files).forEach((file, index) => {
                const fileSize = this.formatFileSize(file.size);
                const fileIcon = this.getFileIcon(file.type, file.name);
                
                fileListHtml += `
                    <div class="file-item">
                        <div class="file-info">
                            <div class="file-icon">${fileIcon}</div>
                            <div class="file-details">
                                <span class="file-name">${file.name}</span>
                                <span class="file-size">${fileSize}</span>
                            </div>
                        </div>
                        <button type="button" class="remove-file" onclick="dashboard.removeEditFile(${dataSourceId}, ${index})">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                `;
            });
            
            fileList.innerHTML = fileListHtml;
        }
    }

    removeEditFile(dataSourceId, fileIndex) {
        const fileInput = document.getElementById(`edit-file-input-${dataSourceId}`);
        const fileList = document.getElementById(`edit-file-list-${dataSourceId}`);
        const fileUploadArea = document.getElementById(`edit-file-upload-${dataSourceId}`);
        const fileUploadContent = fileUploadArea ? fileUploadArea.querySelector('.file-upload-content') : null;
        
        // Create new FileList without the removed file
        const dt = new DataTransfer();
        const files = Array.from(fileInput.files);
        files.splice(fileIndex, 1);
        
        files.forEach(file => dt.items.add(file));
        fileInput.files = dt.files;
        
        if (fileInput.files.length === 0) {
            if (fileList) {
                fileList.style.display = 'none';
            }
            if (fileUploadContent) {
                fileUploadContent.style.display = 'block';
            }
        } else {
            this.handleEditFileSelection(dataSourceId, fileInput.files);
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

        // Top-p range input for create form
        const topPInput = document.getElementById('top-p');
        if (topPInput) {
            const topPValue = document.getElementById('top-p-value');
            topPInput.addEventListener('input', (e) => {
                if (topPValue) {
                    topPValue.textContent = e.target.value;
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

        // Top-p range input for edit form
        const editTopPInput = document.getElementById('edit-top-p');
        if (editTopPInput) {
            const editTopPValue = document.getElementById('edit-top-p-value');
            editTopPInput.addEventListener('input', (e) => {
                if (editTopPValue) {
                    editTopPValue.textContent = e.target.value;
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

        // Vector store type change handlers
        this.setupVectorStoreTypeHandlers();
    }

    setupVectorStoreTypeHandlers() {
        // Create form vector store type handler
        const vectorStoreTypeSelect = document.getElementById('vector-store-type');
        if (vectorStoreTypeSelect) {
            vectorStoreTypeSelect.addEventListener('change', (e) => {
                this.toggleVectorStoreSettings(e.target.value, '');
            });
        }

        // Edit form vector store type handler
        const editVectorStoreTypeSelect = document.getElementById('edit-vector-store-type');
        if (editVectorStoreTypeSelect) {
            editVectorStoreTypeSelect.addEventListener('change', (e) => {
                this.toggleVectorStoreSettings(e.target.value, 'edit-');
            });
        }
    }

    toggleVectorStoreSettings(vectorStoreType, prefix) {
        // Hide all advanced settings
        const qdrantSettings = document.getElementById(prefix + 'qdrant-settings');
        const pineconeSettings = document.getElementById(prefix + 'pinecone-settings');
        const mongodbSettings = document.getElementById(prefix + 'mongodb-settings');

        if (qdrantSettings) qdrantSettings.style.display = 'none';
        if (pineconeSettings) pineconeSettings.style.display = 'none';
        if (mongodbSettings) mongodbSettings.style.display = 'none';

        // Show relevant settings based on vector store type
        switch (vectorStoreType) {
            case 'qdrant':
                if (qdrantSettings) qdrantSettings.style.display = 'block';
                break;
            case 'pinecone':
                if (pineconeSettings) pineconeSettings.style.display = 'block';
                break;
            case 'mongodb_atlas':
                if (mongodbSettings) mongodbSettings.style.display = 'block';
                break;
        }
    }
}

// Fine-tuned Models Management
console.log('About to define FineTunedModelsManager class');

class FineTunedModelsManager {
    constructor() {
        console.log('FineTunedModelsManager constructor called');
        this.models = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        // Don't load models immediately - wait for the view to be activated
    }

    setupEventListeners() {
        // No search or filter functionality needed
    }

    async loadModels() {
        console.log('Loading fine-tuned models...');
        try {
            const response = await fetch('/api/v1/training/models?status_filter=completed', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.models = await response.json();
            console.log('Loaded models:', this.models);
            this.renderModels();
        } catch (error) {
            console.error('Failed to load fine-tuned models:', error);
            this.showError('Failed to load fine-tuned models');
        }
    }

    renderModels() {
        console.log('Rendering models...', this.models);
        const grid = document.getElementById('models-grid');
        if (!grid) {
            console.error('models-grid element not found!');
            return;
        }

        if (this.models.length === 0) {
            grid.innerHTML = `
                <div class="list-header">Fine-tuned Models</div>
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z"></path>
                        <path d="M2 17L12 22L22 17"></path>
                        <path d="M2 12L12 17L22 12"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                    <h3>No Fine-tuned Models</h3>
                    <p>Create your first training job to get started with fine-tuning</p>
                </div>
            `;
            return;
        }

        const modelsHtml = this.models.map(model => this.renderModelCard(model)).join('');
        grid.innerHTML = `
            <div class="list-header">Fine-tuned Models</div>
            ${modelsHtml}
        `;
    }

    renderModelCard(model) {
        const status = this.getModelStatus(model);
        const statusClass = this.getStatusClass(status);
        const createdDate = new Date(model.created_at).toLocaleDateString();
        const size = this.formatModelSize(model);

        return `
            <div class="model-card" onclick="fineTunedModelsManager.showModelDetails('${model.adapter_id}')">
                <div class="model-header">
                    <div class="model-info">
                        <h3>${model.job_name}</h3>
                        <p>${model.base_model_name}</p>
                    </div>
                    <span class="model-status ${statusClass}">${status}</span>
                </div>
                
                <div class="model-meta">
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12,6 12,12 16,14"></polyline>
                        </svg>
                        ${createdDate}
                    </span>
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                            <line x1="8" y1="21" x2="16" y2="21"></line>
                            <line x1="12" y1="17" x2="12" y2="21"></line>
                        </svg>
                        ${size}
                    </span>
                    ${model.final_loss ? `
                        <span>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 3v18h18"></path>
                                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path>
                            </svg>
                            Loss: ${model.final_loss.toFixed(4)}
                        </span>
                    ` : ''}
                </div>
                
                <div class="model-actions">
                    <button class="btn-secondary" onclick="event.stopPropagation(); fineTunedModelsManager.showModelDetails('${model.adapter_id}')">
                        View Details
                    </button>
                    ${this.getActionButton(model)}
                </div>
            </div>
        `;
    }

    getModelStatus(model) {
        return model.training_status || 'unknown';
    }

    getStatusClass(status) {
        const statusClasses = {
            'pending': 'pending',
            'running': 'running',
            'completed': 'completed',
            'failed': 'failed',
            'paused': 'paused'
        };
        return statusClasses[status] || 'unknown';
    }

    formatModelSize(model) {
        // This would need to be implemented based on actual model size calculation
        return 'N/A';
    }

    getActionButton(model) {
        const status = model.training_status;
        
        switch (status) {
            case 'completed':
                return `<button class="btn-primary" onclick="event.stopPropagation(); fineTunedModelsManager.useModelInAgent('${model.adapter_id}')">Use in Agent</button>`;
            case 'failed':
                return `<button class="btn-danger" onclick="event.stopPropagation(); fineTunedModelsManager.deleteModel('${model.adapter_id}')">Delete</button>`;
            default:
                return `<button class="btn-text" onclick="event.stopPropagation(); fineTunedModelsManager.deleteModel('${model.adapter_id}')">Delete</button>`;
        }
    }


    async showModelDetails(adapterId) {
        try {
            const response = await fetch(`/api/v1/training/jobs/${adapterId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const model = await response.json();
            this.renderModelDetailsModal(model);
            this.showModal('model-details-modal');
        } catch (error) {
            console.error('Failed to load model details:', error);
            this.showError('Failed to load model details');
        }
    }

    renderModelDetailsModal(model) {
        const status = this.getModelStatus(model);
        const statusClass = this.getStatusClass(status);
        const createdDate = new Date(model.created_at).toLocaleString();
        const updatedDate = new Date(model.updated_at).toLocaleString();

        document.getElementById('model-details-title').textContent = model.job_name;
        document.getElementById('model-details-subtitle').textContent = `${model.base_model_name} • ${status}`;

        document.getElementById('model-details-content').innerHTML = `
            <div class="model-details-grid">
                <div class="detail-section">
                    <h3>Model Information</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>Job Name</label>
                            <span>${model.job_name}</span>
                        </div>
                        <div class="detail-item">
                            <label>Base Model</label>
                            <span>${model.base_model_name}</span>
                        </div>
                        <div class="detail-item">
                            <label>Status</label>
                            <span class="model-status ${statusClass}">${status}</span>
                        </div>
                        <div class="detail-item">
                            <label>Adapter ID</label>
                            <span class="code-text">${model.adapter_id}</span>
                        </div>
                        <div class="detail-item">
                            <label>Created</label>
                            <span>${createdDate}</span>
                        </div>
                        <div class="detail-item">
                            <label>Last Updated</label>
                            <span>${updatedDate}</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h3>Training Metrics</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>Final Loss</label>
                            <span>${model.final_loss ? model.final_loss.toFixed(4) : 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Validation Loss</label>
                            <span>${model.validation_loss ? model.validation_loss.toFixed(4) : 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>GPU Hours</label>
                            <span>${model.gpu_hours_consumed ? model.gpu_hours_consumed.toFixed(2) : 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Estimated Cost</label>
                            <span>${model.estimated_training_cost_usd ? `$${model.estimated_training_cost_usd.toFixed(2)}` : 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Device</label>
                            <span>${model.device || 'CPU'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Mixed Precision</label>
                            <span>${model.mixed_precision || 'None'}</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h3>Hyperparameters</h3>
                    <div class="hyperparameters-grid">
                        ${model.hyperparameters ? Object.entries(model.hyperparameters).map(([key, value]) => `
                            <div class="hyperparameter-item">
                                <label>${this.formatHyperparameterName(key)}</label>
                                <span>${value}</span>
                            </div>
                        `).join('') : '<p>No hyperparameters available</p>'}
                    </div>
                </div>
            </div>
        `;
    }

    formatHyperparameterName(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    async deleteModel(adapterId) {
        if (!confirm('Are you sure you want to delete this fine-tuned model? This action cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`/api/v1/training/jobs/${adapterId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.showSuccess('Fine-tuned model deleted successfully');
            this.loadModels(); // Reload the models list
        } catch (error) {
            console.error('Failed to delete model:', error);
            this.showError('Failed to delete fine-tuned model');
        }
    }

    useModelInAgent(adapterId) {
        // Switch to agents view and pre-select this model
        dashboard.switchView('agents');
        // You could also open the create agent modal with this model pre-selected
        setTimeout(() => {
            dashboard.showCreateAgentModal();
            // Pre-select the fine-tuned model in the dropdown
            const modelSelect = document.getElementById('agent-llm-model');
            if (modelSelect) {
                modelSelect.value = `fine_tuned:${adapterId}`;
            }
        }, 100);
    }

    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
    }

    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('show');
            document.body.style.overflow = '';
        }
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-icon">
                ${this.getToastIcon(type)}
            </div>
            <div class="toast-message">${message}</div>
        `;

        const container = document.getElementById('toast-notification');
        if (container) {
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.remove();
            }, 5000);
        }
    }

    getToastIcon(type) {
        const icons = {
            success: '',
            error: '',
            warning: '',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
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


function showAgentDetails(agentName) {
    dashboard.showAgentDetails(agentName);
}

function hideAgentDetailsModal() {
    dashboard.hideAgentDetailsModal();
}


function logout() {
    localStorage.removeItem('ragnetic_user_token');
    localStorage.removeItem('ragnetic_db_user_id');
    localStorage.removeItem('ragnetic_username');
    window.location.href = '/login';
}

// toggleSidebar function removed - no longer needed

// Global functions for the form

function addDataSource() {
    dashboard.addDataSource();
}

function addEditDataSource() {
    dashboard.addEditDataSource();
}

function updateRangeValue(id, value) {
    const valueElement = document.getElementById(id + '-value');
    if (valueElement) {
        valueElement.textContent = value;
    }
}

// Global functions for fine-tuned models
function hideModelDetailsModal() {
    fineTunedModelsManager.hideModal('model-details-modal');
}

// Initialize dashboard when DOM is loaded
console.log('Setting up DOMContentLoaded event listener');
document.addEventListener('DOMContentLoaded', function() {
    dashboard = new Dashboard();
    dashboard.initializeFormListeners();
    dashboard.loadAvailableModels(); // Load fine-tuned models for agent configuration
    
    // Initialize fine-tuned models manager
    console.log('Initializing FineTunedModelsManager...');
    fineTunedModelsManager = new FineTunedModelsManager();
});
