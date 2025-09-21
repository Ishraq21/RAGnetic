// Dashboard JavaScript functionality
console.log('Dashboard.js loaded successfully');

class Dashboard {
    constructor() {
        this.currentView = 'home';
        this.agents = [];
        this.agentList = [];
        this.agentsPage = 1;
        this.agentsPerPage = 8;
        this._isLoadingAgents = false;
        this._actionLocks = new Map();
        this.recentActivityPage = 1;
        this.recentActivityPageSize = 5;
        this.lastDataHash = null;
        this.eventSource = null;
        this.init();
    }

    // Agent detail and action methods
    openAgentDetail(agentName) {
        console.log('Opening agent detail for:', agentName);
        this.currentAgentName = agentName;
        this.switchView('agent-detail');
        this.loadAgentDetailData(agentName);
        
        // Start auto-refresh for logs
        this.startLogAutoRefresh();
    }
    
    startLogAutoRefresh() {
        // Clear any existing interval
        if (this.logRefreshInterval) {
            clearInterval(this.logRefreshInterval);
        }
        
        // Start new interval
        this.logRefreshInterval = setInterval(() => {
            if (this.currentAgentName) {
                this.loadAgentLogs(this.currentAgentName);
            }
        }, 30000); // 30 seconds
    }
    
    stopLogAutoRefresh() {
        if (this.logRefreshInterval) {
            clearInterval(this.logRefreshInterval);
            this.logRefreshInterval = null;
        }
    }

    cloneAgent(agentName) {
        console.log('Cloning agent:', agentName);
        this.showToast('Agent cloning coming soon!', 'info');
    }

    testAgent(agentName) {
        console.log('Testing agent:', agentName);
        this.showToast('Agent testing coming soon!', 'info');
    }

    deployAgent(agentName) {
        console.log('Deploying agent:', agentName);
        this.checkAgentDeployment(agentName);
    }

    async init() {
        this.setupEventListeners();
        this.loadUserInfo();
        this.initOverviewControls();
        this.updateRangeLabels();
        await this.loadOverviewData();
    }

    setupEventListeners() {
        // Basic event listeners
        console.log('Setting up event listeners');
        // Tab navigation click handlers
        const tabButtons = document.querySelectorAll('.tab-nav-item');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const view = btn.getAttribute('data-view');
                if (view) {
                    this.switchView(view);
                }
            });
        });
    }

    loadUserInfo() {
        // Load user information
        console.log('Loading user info');
    }

    initOverviewControls() {
        // Initialize overview controls
        console.log('Initializing overview controls');
    }

    updateRangeLabels() {
        // Update range labels
        console.log('Updating range labels');
    }

    async loadOverviewData() {
        // Load overview data
        console.log('Loading overview data');
    }

    async checkAgentDeployment(agentName) {
        console.log('Deploying agent:', agentName);
        try {
            const response = await fetch(`/api/v1/agents/${agentName}/deploy-state`, {
                method: 'POST',
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                this.showToast(`Agent ${agentName} deployed successfully!`, 'success');
                // Reload agents to update status
                if (typeof loadAgentsInline === 'function') {
                    loadAgentsInline();
                }
            } else {
                const error = await response.json().catch(() => ({}));
                this.showToast(`Failed to deploy agent: ${error.detail || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            this.showToast(`Deploy error: ${error.message}`, 'error');
        }
    }

    async getAgentActualStatus(agentName) {
        try {
            const response = await fetch(`/api/v1/agents/${agentName}/status`, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const status = await response.json();
                return status.actual_status;
            } else {
                console.error('Failed to get agent status:', response.status);
                return 'unknown';
            }
        } catch (error) {
            console.error('Error getting agent status:', error);
            return 'unknown';
        }
    }

    getApiKey() {
        return localStorage.getItem('ragnetic_api_key') || 'c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691';
    }

    showToast(message, type = 'info') {
        console.log(`Toast: ${message} (${type})`);
    }

    // View switching functionality
    switchView(view) {
        console.log('Switching to view:', view);
        this.currentView = view;
        
        // Hide all views
        const views = document.querySelectorAll('.view-content');
        views.forEach(view => {
            view.style.display = 'none';
        });
        
        // Show the selected view
        const targetView = document.getElementById(`${view}-view`);
        if (targetView) {
            targetView.style.display = 'block';
        }
        
        // Update navigation active state
        this.updateNavigationState(view);
        
        // Load data for specific views
        if (view === 'agents') {
            this.loadAgents();
        } else if (view === 'overview') {
            this.loadOverviewData();
        } else if (view === 'agent-detail') {
            // Agent detail view is handled by openAgentDetail
            console.log('Switched to agent detail view');
        }
    }
    
    updateNavigationState(activeView) {
        // Remove active class from all nav items
        const navItems = document.querySelectorAll('.tab-nav-item');
        navItems.forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to current nav item
        const activeNavItem = document.querySelector(`[data-view="${activeView}"]`);
        if (activeNavItem) {
            activeNavItem.classList.add('active');
        }
    }
    
    loadAgents() {
        console.log('Loading agents...');
        // This will be called when switching to agents tab
        if (typeof loadAgentsInline === 'function') {
            loadAgentsInline();
        }
    }

    // Load agent detail data
    async loadAgentDetailData(agentName) {
        console.log('Loading agent detail data for:', agentName);
        
        try {
            // Load basic agent data
            const response = await fetch(`/api/v1/agents/${agentName}`, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const agent = await response.json();
                this.populateAgentDetailData(agent);
                
                // Load additional data in parallel
                this.loadAgentStatusData(agentName);
                this.loadAgentCostData(agentName);
                this.loadAgentUsageData(agentName);
                this.loadAgentYAML(agentName);
                this.loadAgentLogs(agentName);
            } else {
                this.showToast('Failed to load agent data', 'error');
            }
        } catch (error) {
            console.error('Error loading agent detail:', error);
            this.showToast('Error loading agent data', 'error');
        }
    }

    populateAgentDetailData(agent) {
        // Update breadcrumb
        document.getElementById('agent-breadcrumb-name').textContent = agent.name || 'Unknown';
        
        // Update enhanced header
        document.getElementById('agent-detail-title').textContent = agent.name || 'Unknown';
        document.getElementById('agent-detail-subtitle').textContent = agent.display_name || 'AI Agent';
        
        // Update meta information
        const createdDate = agent.created_at ? new Date(agent.created_at) : null;
        const updatedDate = agent.updated_at ? new Date(agent.updated_at) : null;
        
        if (createdDate) {
            document.getElementById('agent-created-date').textContent = this.formatRelativeTime(createdDate);
        }
        if (updatedDate) {
            document.getElementById('agent-updated-date').textContent = this.formatRelativeTime(updatedDate);
        }
        
        // Update status indicator
        this.updateAgentStatus(agent.status || 'stopped');
        
        // Basic information
        document.getElementById('agent-name').textContent = agent.name || 'N/A';
        document.getElementById('agent-display-name').textContent = agent.display_name || agent.name || 'N/A';
        document.getElementById('agent-description').textContent = agent.description || 'No description provided';
        document.getElementById('agent-created').textContent = agent.created_at ? new Date(agent.created_at).toLocaleDateString() : 'N/A';
        document.getElementById('agent-updated').textContent = agent.updated_at ? new Date(agent.updated_at).toLocaleDateString() : 'N/A';
        
        // AI Configuration
        document.getElementById('agent-llm-model').textContent = agent.llm_model || 'N/A';
        document.getElementById('agent-embedding-model').textContent = agent.embedding_model || 'N/A';
        document.getElementById('agent-vector-store').textContent = agent.vector_store?.type || 'N/A';
        document.getElementById('agent-temperature').textContent = agent.model_params?.temperature || 'Default (0.7)';
        document.getElementById('agent-max-tokens').textContent = agent.model_params?.max_tokens || 'Default (2000)';
        
        // Data Sources & Tools
        const sourcesCount = Array.isArray(agent.sources) ? agent.sources.length : 0;
        document.getElementById('agent-sources-count').textContent = `${sourcesCount} configured`;
        document.getElementById('agent-tools').textContent = this.formatToolsList(agent.tools);
        document.getElementById('agent-chunk-size').textContent = `${agent.chunking?.chunk_size || '1000'} tokens`;
        document.getElementById('agent-chunk-overlap').textContent = `${agent.chunking?.chunk_overlap || '100'} tokens`;
    }

    async loadAgentStatusData(agentName) {
        try {
            const response = await fetch(`/api/v1/agents/${agentName}/status`, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const status = await response.json();
                this.updateAgentStatus(status);
            }
        } catch (error) {
            console.log('Could not fetch status data:', error);
        }
    }

    updateAgentStatus(status) {
        // Handle both string and object status
        const statusText = typeof status === 'string' ? status : (status.actual_status || 'unknown');
        const formattedStatus = statusText.charAt(0).toUpperCase() + statusText.slice(1);
        
        document.getElementById('agent-status-text').textContent = formattedStatus;
        document.getElementById('agent-status-description').textContent = this.getStatusDescription(statusText);
        
        // Update status pill
        const statusPill = document.getElementById('agent-detail-status');
        statusPill.className = 'status-pill';
        statusPill.classList.add(`status-${statusText}`);
        
        // Update avatar status indicator based on backend actual_status
        const statusIndicator = document.getElementById('agent-status-indicator');
        if (statusIndicator) {
            // Remove all existing status classes
            statusIndicator.className = 'status-indicator';
            
            // Use the actual_status from backend (deployed/stopped)
            if (statusText === 'deployed') {
                statusIndicator.classList.add('status-online');
            } else if (statusText === 'stopped') {
                statusIndicator.classList.add('status-offline');
            } else {
                statusIndicator.classList.add('status-unknown');
            }
        }
        
        // Update system status if status is an object
        if (typeof status === 'object') {
            document.getElementById('agent-db-status').textContent = status.database_status || 'Unknown';
            document.getElementById('agent-vector-status').textContent = status.vectorstore_exists ? 'Ready' : 'Not Initialized';
            document.getElementById('agent-offline-status').textContent = status.offline_marker_exists ? 'Enabled' : 'Disabled';
        }
    }

    async loadAgentCostData(agentName, timeframe = 30) {
        console.log('loadAgentCostData called with:', agentName, timeframe);
        try {
            // Calculate date range based on timeframe
            const endDate = new Date();
            const startDate = new Date();
            startDate.setDate(startDate.getDate() - timeframe);
            
            console.log('Date range:', startDate.toISOString(), 'to', endDate.toISOString());
            
            // Use the existing analytics endpoint with time filtering
            const url = `/api/v1/analytics/usage-summary?agent_name=${agentName}&start_time=${startDate.toISOString()}&end_time=${endDate.toISOString()}&limit=1`;
            console.log('Fetching from URL:', url);
            
            const response = await fetch(url, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const analyticsData = await response.json();
                console.log('Analytics response:', analyticsData);
                const costData = analyticsData.length > 0 ? analyticsData[0] : null;
                
                if (costData) {
                    const costValue = (costData.total_estimated_cost_usd || 0).toFixed(4);
                    console.log('Updating cost data:', costValue, costData);
                    // Update both header and analytics sections
                    document.getElementById('agent-monthly-cost').textContent = `$${costValue}`;
                    document.getElementById('agent-total-cost').textContent = `$${costValue}`;
                    document.getElementById('quick-cost').textContent = `$${costValue}`;
                    document.getElementById('agent-token-usage').textContent = `${costData.total_tokens || '0'} tokens`;
                    document.getElementById('agent-api-calls').textContent = `${costData.total_requests || '0'} calls`;
                } else {
                    console.log('No cost data found');
                    // No data found - update both sections
                    document.getElementById('agent-monthly-cost').textContent = '$0.00';
                    document.getElementById('agent-total-cost').textContent = '$0.00';
                    document.getElementById('quick-cost').textContent = '$0.00';
                    document.getElementById('agent-token-usage').textContent = '0 tokens';
                    document.getElementById('agent-api-calls').textContent = '0 calls';
                }
            } else {
                console.log('Response not ok:', response.status, response.statusText);
            }
        } catch (error) {
            console.log('Could not fetch cost data:', error);
            // Set default values on error - update both sections
            document.getElementById('agent-monthly-cost').textContent = '$0.00';
            document.getElementById('agent-total-cost').textContent = '$0.00';
            document.getElementById('quick-cost').textContent = '$0.00';
            document.getElementById('agent-token-usage').textContent = '0 tokens';
            document.getElementById('agent-api-calls').textContent = '0 calls';
        }
    }

    async loadAgentUsageData(agentName) {
        try {
            // Use the existing analytics endpoint instead of custom agent endpoint
            const response = await fetch(`/api/v1/analytics/usage-summary?agent_name=${agentName}&limit=1`, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const analyticsData = await response.json();
                const usageData = analyticsData.length > 0 ? analyticsData[0] : null;
                
                if (usageData) {
                    // Update both header and analytics sections
                    document.getElementById('agent-total-queries').textContent = usageData.total_requests || '0';
                    document.getElementById('quick-total-queries').textContent = usageData.total_requests || '0';
                    
                    // Calculate success rate (all analytics entries are successful)
                    const successRate = usageData.total_requests > 0 ? 100 : 0;
                    document.getElementById('agent-success-rate').textContent = `${successRate.toFixed(1)}%`;
                    document.getElementById('quick-success-rate').textContent = `${successRate.toFixed(1)}%`;
                    
                    document.getElementById('agent-avg-response').textContent = usageData.avg_generation_time_s ? `${(usageData.avg_generation_time_s * 1000).toFixed(0)}ms` : 'N/A';
                    // For last activity, we'll need to get this from a different endpoint or calculate it
                    document.getElementById('agent-last-activity').textContent = 'Recently'; // Placeholder for now
                } else {
                    // No data found - update both sections
                    document.getElementById('agent-total-queries').textContent = '0';
                    document.getElementById('quick-total-queries').textContent = '0';
                    document.getElementById('agent-success-rate').textContent = 'N/A';
                    document.getElementById('quick-success-rate').textContent = 'N/A';
                    document.getElementById('agent-avg-response').textContent = 'N/A';
                    document.getElementById('agent-last-activity').textContent = 'Never';
                }
            }
        } catch (error) {
            console.log('Could not fetch usage data:', error);
            // Set default values on error - update both sections
            document.getElementById('agent-total-queries').textContent = '0';
            document.getElementById('quick-total-queries').textContent = '0';
            document.getElementById('agent-success-rate').textContent = 'N/A';
            document.getElementById('quick-success-rate').textContent = 'N/A';
            document.getElementById('agent-avg-response').textContent = 'N/A';
            document.getElementById('agent-last-activity').textContent = 'Never';
        }
    }

    async loadAgentYAML(agentName) {
        try {
            const response = await fetch(`/api/v1/agents/${agentName}/yaml`, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const yaml = await response.text();
                document.getElementById('agent-yaml-content').textContent = yaml;
            } else {
                // Fallback: generate YAML from agent data
                const agentResponse = await fetch(`/api/v1/agents/${agentName}`, {
                    headers: { 'X-API-Key': this.getApiKey() }
                });
                if (agentResponse.ok) {
                    const agent = await agentResponse.json();
                    const yaml = this.generateYAMLFromAgent(agent);
                    document.getElementById('agent-yaml-content').textContent = yaml;
                } else {
                    document.getElementById('agent-yaml-content').textContent = 'YAML configuration not available';
                }
            }
        } catch (error) {
            console.log('Could not fetch YAML:', error);
            // Try fallback as last resort
            try {
                const agentResponse = await fetch(`/api/v1/agents/${agentName}`, {
                    headers: { 'X-API-Key': this.getApiKey() }
                });
                if (agentResponse.ok) {
                    const agent = await agentResponse.json();
                    const yaml = this.generateYAMLFromAgent(agent);
                    document.getElementById('agent-yaml-content').textContent = yaml;
                    return;
                }
            } catch (_) {}
            document.getElementById('agent-yaml-content').textContent = 'Error loading YAML configuration';
        }
    }

    generateYAMLFromAgent(agent) {
        const tools = (agent.tools || ['retriever']).map(t => `  - ${t}`).join('\n');
        const sources = (agent.sources || []).map(s => `  - type: ${s.type}\n    path: ${s.path || s.url || ''}\n    max_depth: ${s.max_depth ?? 2}`).join('\n');
        return `name: ${agent.name}
display_name: ${agent.display_name || agent.name}
description: |\n  ${(agent.description || 'No description provided').replace(/\n/g, '\n  ')}

# AI Configuration
llm_model: ${agent.llm_model || 'gpt-4o-mini'}
embedding_model: ${agent.embedding_model || 'text-embedding-3-small'}

# Model Parameters
model_params:
  temperature: ${agent.model_params?.temperature ?? 0.7}
  max_tokens: ${agent.model_params?.max_tokens ?? 2000}

# Vector Store Configuration
vector_store:
  type: ${agent.vector_store?.type || 'faiss'}
  retrieval_strategy: ${agent.vector_store?.retrieval_strategy || 'hybrid'}

# Chunking Configuration
chunking:
  chunk_size: ${agent.chunking?.chunk_size ?? 1000}
  chunk_overlap: ${agent.chunking?.chunk_overlap ?? 100}

# Tools
tools:
${tools || '  - retriever'}

# Data Sources
sources:
${sources || '  # none'}

`;
    }

    async loadAgentLogs(agentName) {
        try {
            const response = await fetch(`/api/v1/agents/${agentName}/logs?lines=200`, {
                headers: { 'X-API-Key': this.getApiKey() }
            });
            
            if (response.ok) {
                const logs = await response.text();
                this.displayAgentLogs(logs);
            } else if (response.status === 404) {
                document.getElementById('agent-logs-content').innerHTML = `
                    <div class="empty-logs">
                        <div class="empty-icon">LOG</div>
                        <span class="hint">No logs available for this agent yet.</span>
                        <p class="empty-description">Logs will appear here once the agent starts processing requests.</p>
                        <div class="actions">
                            <button class="btn-text" onclick="refreshLogs()">Refresh</button>
                        </div>
                    </div>`;
            } else {
                document.getElementById('agent-logs-content').innerHTML = `
                    <div class="error-logs">
                        <div class="error-icon">ERROR</div>
                        <span>Failed to load logs. Please try again.</span>
                        <div class="actions">
                            <button class="btn-text" onclick="refreshLogs()">Retry</button>
                        </div>
                    </div>`;
            }
        } catch (error) {
            console.error('Could not fetch logs:', error);
            document.getElementById('agent-logs-content').innerHTML = `
                <div class="error-logs">
                    <div class="error-icon">ERROR</div>
                    <span>Failed to load logs. Please check your connection and try again.</span>
                    <div class="actions">
                        <button class="btn-text" onclick="refreshLogs()">Retry</button>
                    </div>
                </div>`;
        }
    }

    displayAgentLogs(logs) {
        const logsContent = document.getElementById('agent-logs-content');
        
        // Check if the response is the "no logs" message
        if (logs.trim() === "No logs available for this agent yet.") {
            logsContent.innerHTML = `
                <div class="empty-logs">
                    <div class="empty-icon">LOG</div>
                    <span class="hint">No logs available for this agent yet.</span>
                    <p class="empty-description">Logs will appear here once the agent starts processing requests.</p>
                    <div class="actions">
                        <button class="btn-text" onclick="refreshLogs()">Refresh</button>
                    </div>
                </div>`;
            return;
        }
        
        const logLines = logs.split('\n').filter(line => line.trim());
        
        if (logLines.length === 0) {
            logsContent.innerHTML = `
                <div class="empty-logs">
                    <div class="empty-icon">LOG</div>
                    <span class="hint">No logs available for this agent yet.</span>
                    <p class="empty-description">Logs will appear here once the agent starts processing requests.</p>
                    <div class="actions">
                        <button class="btn-text" onclick="refreshLogs()">Refresh</button>
                    </div>
                </div>`;
            return;
        }
        
        const logEntries = logLines.map((line, index) => {
            let logClass = 'info';
            let timestamp = '';
            let message = line;
            
            // Extract timestamp if present
            const timestampMatch = line.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)/);
            if (timestampMatch) {
                timestamp = new Date(timestampMatch[1]).toLocaleTimeString();
                message = line.replace(timestampMatch[0], '').trim();
            }
            
            // Determine log level
            if (message.toLowerCase().includes('error') || message.toLowerCase().includes('failed')) {
                logClass = 'error';
            } else if (message.toLowerCase().includes('warning') || message.toLowerCase().includes('warn')) {
                logClass = 'warning';
            } else if (message.toLowerCase().includes('debug')) {
                logClass = 'info';
            }
            
            // Add log level indicator
            const levelIndicator = logClass === 'error' ? 'ERROR' : 
                                 logClass === 'warning' ? 'WARN' : 
                                 logClass === 'info' ? 'INFO' : 'LOG';
            
            return `<div class="log-entry ${logClass}" data-timestamp="${timestamp}">
                <span class="log-level">${levelIndicator}</span>
                <span class="log-message">${message}</span>
            </div>`;
        }).join('');
        
        logsContent.innerHTML = logEntries;
        
        // Auto-scroll to bottom for new logs
        const logsContainer = document.querySelector('.logs-container');
        if (logsContainer) {
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }

    formatToolsList(tools) {
        if (!tools || !Array.isArray(tools) || tools.length === 0) return 'None';
        
        const toolDescriptions = {
            'retriever': 'Document Retrieval',
            'calculator': 'Mathematical Calculations',
            'web_search': 'Web Search',
            'code_executor': 'Code Execution',
            'file_manager': 'File Management',
            'api_toolkit': 'API Toolkit',
            'search_engine': 'Search Engine'
        };
        
        return tools.map(tool => toolDescriptions[tool] || tool).join(', ');
    }

    getStatusDescription(status) {
        const descriptions = {
            'created': 'Agent is created but not yet deployed',
            'deployed': 'Agent is running and ready to handle requests',
            'stopped': 'Agent is stopped and not processing requests',
            'idle': 'Agent is deployed but not actively processing',
            'error': 'Agent encountered an error and needs attention',
            'unknown': 'Status is unknown or not available'
        };
        return descriptions[status] || 'Status information not available';
    }

    // Helper method to format relative time
    formatRelativeTime(date) {
        const now = new Date();
        const diffInSeconds = Math.floor((now - date) / 1000);
        
        if (diffInSeconds < 60) {
            return 'just now';
        } else if (diffInSeconds < 3600) {
            const minutes = Math.floor(diffInSeconds / 60);
            return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        } else if (diffInSeconds < 86400) {
            const hours = Math.floor(diffInSeconds / 3600);
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        } else {
            const days = Math.floor(diffInSeconds / 86400);
            return `${days} day${days > 1 ? 's' : ''} ago`;
        }
    }

    // Update agent status with enhanced visual feedback
    updateAgentStatusEnhanced(status) {
        const statusPill = document.getElementById('agent-detail-status');
        const statusText = document.getElementById('agent-status-text');
        const statusIndicator = document.getElementById('agent-status-indicator');
        
        if (statusPill && statusText) {
            // Remove existing status classes
            statusPill.className = 'status-pill';
            
            // Add new status class
            statusPill.classList.add(`status-${status}`);
            
            // Update status text
            const statusLabels = {
                'active': 'Active',
                'running': 'Running',
                'stopped': 'Stopped',
                'deploying': 'Deploying',
                'idle': 'Idle',
                'error': 'Error'
            };
            
            statusText.textContent = statusLabels[status] || 'Unknown';
        }
        
        // Update status indicator color
        if (statusIndicator) {
            const statusColors = {
                'active': 'var(--success-color)',
                'running': 'var(--success-color)',
                'stopped': 'var(--error-color)',
                'deploying': 'var(--warning-color)',
                'idle': 'var(--text-secondary)',
                'error': 'var(--error-color)'
            };
            
            statusIndicator.style.background = statusColors[status] || 'var(--text-secondary)';
        }
    }
}

// Helper function to get status description
function getStatusDescription(status) {
    const descriptions = {
        'created': 'Agent is created but not yet deployed',
        'deployed': 'Agent is running and ready to handle requests',
        'stopped': 'Agent is stopped and not processing requests',
        'idle': 'Agent is deployed but not actively processing',
        'error': 'Agent encountered an error and needs attention',
        'unknown': 'Status is unknown or not available'
    };
    return descriptions[status] || 'Status information not available';
}

// Helper function to format tools list (using Dashboard class method)
function formatToolsList(tools) {
    const dashboard = new Dashboard();
    return dashboard.formatToolsList(tools);
}

// Load actions for agent details modal
async function loadAgentDetailsActions(agentName, status) {
    const actionsElement = document.getElementById('agent-details-actions');
    if (!actionsElement) return;
    
    const actions = getAgentActionsSync(status, agentName);
    actionsElement.innerHTML = actions;
}

// Global functions for onclick handlers
function switchView(view) {
    if (window.dashboard) {
        window.dashboard.switchView(view);
    }
}

function showCreateAgentModal() {
    console.log('Show create agent modal');
    showModalById('create-agent-modal');
}

function hideCreateAgentModal() {
    console.log('Hide create agent modal');
    hideModalById('create-agent-modal');
}

function showEditAgentModal(agentName) {
    console.log('Show edit agent modal for:', agentName);
    // Load agent data and populate form
    loadAgentForEdit(agentName);
    showModalById('edit-agent-modal');
}

function hideEditAgentModal() {
    console.log('Hide edit agent modal');
    hideModalById('edit-agent-modal');
}

// Consistent modal helpers
function showModalById(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) {
        console.error(`Modal not found: ${modalId}`);
        return;
    }
    modal.classList.add('show');
    modal.style.removeProperty('display');
    modal.style.removeProperty('visibility');
    document.body.style.overflow = 'hidden';
}

function hideModalById(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    modal.classList.remove('show');
    modal.style.removeProperty('visibility');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

async function loadAgentForEdit(agentName) {
    console.log('Loading agent data for edit:', agentName);
    try {
        const response = await fetch(`/api/v1/agents/${agentName}`, {
            headers: { 'X-API-Key': getApiKey() }
        });
        
        console.log('Agent API response status:', response.status);
        
        if (response.ok) {
            const agent = await response.json();
            console.log('Agent data loaded:', agent);
            
            // Populate edit form
            const nameField = document.getElementById('edit-agent-name');
            const displayNameField = document.getElementById('edit-agent-display-name');
            const descriptionField = document.getElementById('edit-agent-description');
            
            console.log('Form fields found:', {
                name: !!nameField,
                displayName: !!displayNameField,
                description: !!descriptionField
            });
            
            if (nameField) nameField.value = agent.name || '';
            if (displayNameField) displayNameField.value = agent.display_name || '';
            if (descriptionField) descriptionField.value = agent.description || '';
            
            // Populate other fields as needed
            if (agent.llm_model) {
                const llmSelect = document.getElementById('edit-agent-llm-model');
                if (llmSelect) {
                    llmSelect.value = agent.llm_model;
                    console.log('Set LLM model to:', agent.llm_model);
                } else {
                    console.error('LLM model select not found');
                }
            }
            
            if (agent.embedding_model) {
                const embeddingSelect = document.getElementById('edit-agent-embedding-model');
                if (embeddingSelect) {
                    embeddingSelect.value = agent.embedding_model;
                    console.log('Set embedding model to:', agent.embedding_model);
                } else {
                    console.error('Embedding model select not found');
                }
            }
            
            // Populate persona prompt
            if (agent.persona_prompt) {
                const personaField = document.getElementById('edit-agent-persona');
                if (personaField) {
                    personaField.value = agent.persona_prompt;
                    console.log('Set persona prompt');
                }
            }
            
            // Populate execution prompt
            if (agent.execution_prompt) {
                const executionField = document.getElementById('edit-agent-execution-prompt');
                if (executionField) {
                    executionField.value = agent.execution_prompt;
                    console.log('Set execution prompt');
                }
            }
            
            console.log('Form populated successfully');

            // Populate existing sources
            try {
                const editContainer = document.getElementById('edit-data-sources-container');
                if (editContainer) {
                    editContainer.innerHTML = '';
                    const sources = Array.isArray(agent.sources) ? agent.sources : [];
                    for (const s of sources) {
                        const id = `eds-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                        const block = document.createElement('div');
                        block.className = 'data-source-block';
                        block.dataset.sourceId = id;
                        block.innerHTML = createDataSourceHTML(id, 'edit');
                        editContainer.appendChild(block);
                        
                        // Populate the fields with existing data
                        setTimeout(() => {
                            populateDataSourceFields(id, s);
                        }, 100);
                    }
                }
            } catch (e) {
                console.warn('Failed to populate edit sources', e);
            }
        } else {
            console.error('Failed to load agent data:', response.status);
            alert('Failed to load agent data');
        }
    } catch (error) {
        console.error('Error loading agent:', error);
        alert('Error loading agent data');
    }
}

function deleteAgent(agentName) {
    console.log('Showing delete confirmation for agent:', agentName);
    // Set the agent name in the modal
    const agentNameElement = document.getElementById('delete-agent-name');
    if (agentNameElement) {
        agentNameElement.textContent = agentName;
    }
    
    // Store the agent name for the confirm function
    window.pendingDeleteAgent = agentName;
    
    // Show the delete confirmation modal
    showModalById('delete-confirmation-modal');
}

async function confirmDeleteAgent() {
    const agentName = window.pendingDeleteAgent;
    if (!agentName) {
        console.error('No agent name for deletion');
        return;
    }
    
    console.log('Deleting agent:', agentName);
    try {
        const response = await fetch(`/api/v1/agents/${agentName}`, {
            method: 'DELETE',
            headers: { 'X-API-Key': getApiKey() }
        });
        
        if (response.ok) {
            console.log('Agent deleted successfully');
            showToast('Agent deleted successfully', 'success');
            // Reload agents list
            if (window.loadAgentsInline) {
                window.loadAgentsInline();
            }
        } else {
            const error = await response.json().catch(() => ({}));
            console.error('Delete failed:', error);
            showToast('Failed to delete agent: ' + (error.detail || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast('Failed to delete agent: ' + error.message, 'error');
    } finally {
        // Hide the modal and clear the pending delete
        hideDeleteConfirmationModal();
        window.pendingDeleteAgent = null;
    }
}

// Form submission handlers
document.addEventListener('DOMContentLoaded', function() {
    // Create agent form
    const createForm = document.getElementById('create-agent-form');
    if (createForm) {
        createForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            await handleCreateAgent();
        });
    }
    
    // Edit agent form
    const editForm = document.getElementById('edit-agent-form');
    console.log('Edit form found during setup:', !!editForm);
    if (editForm) {
        editForm.addEventListener('submit', async function(e) {
            console.log('Edit form submitted');
            e.preventDefault();
            await handleEditAgent();
        });
        console.log('Edit form event listener added');
    } else {
        console.error('Edit form not found during setup!');
    }
});

async function handleCreateAgent() {
    console.log('handleCreateAgent called');
    const form = document.getElementById('create-agent-form');
    if (!form) {
        console.error('Create form not found!');
        showToast('Create form not found!', 'error');
        return;
    }
    
    const formData = new FormData(form);
    let sources = collectSources('#data-sources-container');
    
    // Validate required fields
    const name = formData.get('name');
    if (!name || name.trim() === '') {
        showToast('Agent name is required', 'error');
        return;
    }
    
    // Handle file uploads first
    const uploadedSources = await handleFileUploads(sources);
    if (uploadedSources === null) {
        return; // Error already shown
    }
    sources = uploadedSources;
    
    const agentData = {
        name: name.trim(),
        display_name: formData.get('display_name') || name.trim(),
        description: formData.get('description') || '',
        persona_prompt: formData.get('persona_prompt') || 'You are a helpful assistant.',
        llm_model: formData.get('llm_model') || 'gpt-4o-mini',
        embedding_model: formData.get('embedding_model') || 'text-embedding-3-small',
        sources,
        tools: ['retriever'],
        vector_store: {
            type: 'faiss',
            retrieval_strategy: 'hybrid'
        },
    };
    
    console.log('Agent data to submit:', agentData);
    
    try {
        const response = await fetch('/api/v1/agents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': getApiKey()
            },
            body: JSON.stringify(agentData)
        });
        
        console.log('Response status:', response.status);
        
        if (response.ok) {
            const result = await response.json();
            console.log('Agent created:', result);
            showToast('Agent created successfully', 'success');
            hideCreateAgentModal();
            // Reload agents list
            if (window.loadAgentsInline) {
                window.loadAgentsInline();
            }
        } else {
            const errorText = await response.text();
            console.error('Create failed:', response.status, errorText);
            let errorMessage = 'Unknown error';
            try {
                const error = JSON.parse(errorText);
                errorMessage = error.detail || error.message || errorText;
            } catch (e) {
                errorMessage = errorText;
            }
            showToast('Failed to create agent: ' + errorMessage, 'error');
        }
    } catch (error) {
        console.error('Create error:', error);
        showToast('Failed to create agent: ' + error.message, 'error');
    }
}

// Handle file uploads and return processed sources
async function handleFileUploads(sources) {
    const processedSources = [];
    
    for (const source of sources) {
        if (source && source._isFileUpload) {
            // Find the file input for this source
            const container = document.querySelector(`[data-source-id]`);
            if (!container) continue;
            
            const fileInput = container.querySelector('input[type="file"]');
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                showToast('Please select files to upload', 'error');
                return null;
            }
            
            // Upload files
            const uploadedPaths = await uploadFiles(fileInput.files);
            if (uploadedPaths === null) {
                return null; // Error already shown
            }
            
            // Create local sources for uploaded files
            for (const path of uploadedPaths) {
                processedSources.push({
                    type: 'local',
                    path: path,
                    max_depth: source.max_depth,
                    file_types: source.file_types,
                    headers: null,
                    params: null,
                    method: 'GET',
                    payload: null,
                    json_pointer: null
                });
            }
        } else if (source) {
            processedSources.push(source);
        }
    }
    
    return processedSources;
}

// Upload files to server and return paths
async function uploadFiles(files) {
    const uploadedPaths = [];
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/v1/agents/upload-file', {
                method: 'POST',
                headers: {
                    'X-API-Key': getApiKey()
                },
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                uploadedPaths.push(result.file_path);
                console.log('File uploaded:', result.file_path);
            } else {
                const error = await response.text();
                console.error('Upload failed:', error);
                showToast(`Failed to upload ${file.name}: ${error}`, 'error');
                return null;
            }
        } catch (error) {
            console.error('Upload error:', error);
            showToast(`Failed to upload ${file.name}: ${error.message}`, 'error');
            return null;
        }
    }
    
    return uploadedPaths;
}

async function handleEditAgent() {
    console.log('handleEditAgent called');
    const form = document.getElementById('edit-agent-form');
    console.log('Edit form found:', !!form);
    
    if (!form) {
        console.error('Edit form not found!');
        alert('Edit form not found!');
        return;
    }
    
    const formData = new FormData(form);
    const agentName = formData.get('name');
    console.log('Agent name from form:', agentName);
    
    const sources = collectSources('#edit-data-sources-container');
    const agentData = {
        name: agentName,
        display_name: formData.get('display_name'),
        description: formData.get('description'),
        persona_prompt: formData.get('persona_prompt') || 'You are a helpful assistant.',
        llm_model: formData.get('llm_model') || 'gpt-4o-mini',
        embedding_model: formData.get('embedding_model') || 'text-embedding-3-small',
        sources,
        tools: ['retriever'],
        vector_store: {
            type: 'faiss',
            retrieval_strategy: 'hybrid'
        },
    };
    
    console.log('Agent data to submit:', agentData);
    
    try {
        const response = await fetch(`/api/v1/agents/${agentName}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': getApiKey()
            },
            body: JSON.stringify(agentData)
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('Agent updated:', result);
            showToast('Agent updated successfully', 'success');
            hideEditAgentModal();
            // Reload agents list
            if (window.loadAgentsInline) {
                window.loadAgentsInline();
            }
        } else {
            const error = await response.json().catch(() => ({}));
            console.error('Update failed:', error);
            showToast('Failed to update agent: ' + (error.detail || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Update error:', error);
        showToast('Failed to update agent: ' + error.message, 'error');
    }
}

function collectSources(containerSelector) {
    const container = document.querySelector(containerSelector);
    if (!container) return [];
    const blocks = Array.from(container.querySelectorAll('.data-source-block'));
    return blocks.map(block => {
        const sourceId = block.dataset.sourceId;
        const type = block.querySelector('.ds-type')?.value || 'local';
        const depth = parseInt(block.querySelector('.ds-depth')?.value || '2', 10);
        
        // Base source object
        const source = {
            type,
            max_depth: Number.isFinite(depth) ? depth : 2,
            file_types: null, // Automatic file type detection
            headers: null,
            params: null,
            method: 'GET',
            payload: null,
            json_pointer: null
        };
        
        // Add type-specific fields
        switch (type) {
            case 'local':
                source.path = block.querySelector(`#ds-path-${sourceId}`)?.value || null;
                break;
                
            case 'file_upload':
                // For file uploads, we'll handle them separately and create local sources
                const fileInput = block.querySelector(`#ds-upload-${sourceId}`);
                if (fileInput && fileInput.files && fileInput.files.length > 0) {
                    // This will be handled by the upload process
                    source.type = 'local'; // Convert to local after upload
                    source.path = 'uploaded_files'; // Placeholder
                    source._isFileUpload = true; // Flag for special handling
                } else {
                    return null; // Skip if no files selected
                }
                break;
                
            case 'url':
            case 'web_crawler':
                source.url = block.querySelector(`#ds-url-${sourceId}`)?.value || null;
                source.method = block.querySelector(`#ds-method-${sourceId}`)?.value || 'GET';
                const headersText = block.querySelector(`#ds-headers-${sourceId}`)?.value;
                if (headersText) {
                    try {
                        source.headers = JSON.parse(headersText);
                    } catch (e) {
                        console.warn('Invalid headers JSON:', headersText);
                    }
                }
                break;
                
            case 'code_repository':
                source.path = block.querySelector(`#ds-repo-${sourceId}`)?.value || null;
                const branch = block.querySelector(`#ds-branch-${sourceId}`)?.value;
                const token = block.querySelector(`#ds-token-${sourceId}`)?.value;
                if (branch) source.params = { branch };
                if (token) source.headers = { Authorization: `Bearer ${token}` };
                break;
                
            case 'db':
                source.db_connection = block.querySelector(`#ds-connection-${sourceId}`)?.value || null;
                const query = block.querySelector(`#ds-query-${sourceId}`)?.value;
                if (query) source.params = { query };
                break;
                
            case 'gdoc':
                source.folder_id = block.querySelector(`#ds-folder-${sourceId}`)?.value || null;
                const docIdsText = block.querySelector(`#ds-doc-ids-${sourceId}`)?.value;
                if (docIdsText) {
                    source.document_ids = docIdsText.split('\n').filter(id => id.trim());
                }
                break;
                
            case 'api':
                source.url = block.querySelector(`#ds-api-url-${sourceId}`)?.value || null;
                source.method = block.querySelector(`#ds-api-method-${sourceId}`)?.value || 'GET';
                const apiHeadersText = block.querySelector(`#ds-api-headers-${sourceId}`)?.value;
                if (apiHeadersText) {
                    try {
                        source.headers = JSON.parse(apiHeadersText);
                    } catch (e) {
                        console.warn('Invalid API headers JSON:', apiHeadersText);
                    }
                }
                const payloadText = block.querySelector(`#ds-api-payload-${sourceId}`)?.value;
                if (payloadText) {
                    try {
                        source.payload = JSON.parse(payloadText);
                    } catch (e) {
                        console.warn('Invalid API payload JSON:', payloadText);
                    }
                }
                break;
                
        }
        
        return source;
    });
}

function hideDeleteConfirmationModal() {
    console.log('Hide delete confirmation modal');
    hideModalById('delete-confirmation-modal');
    // Clear the pending delete
    window.pendingDeleteAgent = null;
}

function viewAgent(agentName) {
    console.log('Viewing agent:', agentName);
    // Navigate to agent detail page
    if (window.dashboard) {
        window.dashboard.openAgentDetail(agentName);
    }
}

function editAgent(agentName) {
    console.log('Editing agent:', agentName);
    console.log('Calling showEditAgentModal with:', agentName);
    showEditAgentModal(agentName);
}


async function showAgentDetails(agentName) {
    console.log('Show agent details modal for:', agentName);
    const modal = document.getElementById('agent-details-modal');
    const title = document.getElementById('agent-details-title');
    const content = document.getElementById('agent-details-content');
    
    if (modal && title && content) {
        // Set the title
        title.textContent = `Agent Details: ${agentName}`;
        
        // Show loading state
        content.innerHTML = `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p>Loading agent details...</p>
            </div>
        `;
        
        // Show the modal
        showModalById('agent-details-modal');
        
        // Load agent data
        try {
            const response = await fetch(`/api/v1/agents/${agentName}`, {
                headers: { 'X-API-Key': getApiKey() }
            });
            
            if (response.ok) {
                const agent = await response.json();
                console.log('Agent details loaded:', agent);
                
                // Get additional data from database
                let dbData = {};
                try {
                    const dbResponse = await fetch(`/api/v1/agents/${agentName}/status`, {
                        headers: { 'X-API-Key': getApiKey() }
                    });
                    if (dbResponse.ok) {
                        dbData = await dbResponse.json();
                    }
                } catch (error) {
                    console.log('Could not fetch database data:', error);
                }
                
                // Get cost and usage data from analytics
                let analyticsData = {};
                try {
                    const analyticsResponse = await fetch(`/api/v1/analytics/usage-summary?agent_name=${agentName}&limit=1`, {
                        headers: { 'X-API-Key': getApiKey() }
                    });
                    if (analyticsResponse.ok) {
                        const analyticsArray = await analyticsResponse.json();
                        analyticsData = analyticsArray.length > 0 ? analyticsArray[0] : {};
                    }
                } catch (error) {
                    console.log('Could not fetch analytics data:', error);
                }

                // Format status with proper capitalization
                const status = dbData.actual_status || 'unknown';
                const formattedStatus = status.charAt(0).toUpperCase() + status.slice(1);

                // Populate the modal content
                content.innerHTML = `
                    <div class="agent-details">
                        <!-- Header with Status and Actions -->
                        <div class="agent-details-header">
                            <div class="agent-status-section">
                                <div class="status-display">
                                    <div class="status-pill ${status}">
                                        <div class="status-dot"></div>
                                        ${formattedStatus}
                                    </div>
                                    <div class="status-description">
                                        ${getStatusDescription(status)}
                                    </div>
                                </div>
                            </div>
                            <div class="agent-actions-section" id="agent-details-actions">
                                <!-- Actions will be loaded here -->
                            </div>
                        </div>

                        <div class="detail-section">
                            <h3>Basic Information</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>Name:</label>
                                    <span class="agent-name-display">${agent.name || 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Display Name:</label>
                                    <span>${agent.display_name || agent.name || 'N/A'}</span>
                                </div>
                                <div class="detail-item full-width">
                                    <label>Description:</label>
                                    <span class="description-text">${agent.description || 'No description provided'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Created:</label>
                                    <span>${agent.created_at ? new Date(agent.created_at).toLocaleDateString() : 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="detail-section">
                            <h3>AI Configuration</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>LLM Model:</label>
                                    <span class="model-badge">${agent.llm_model || 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Embedding Model:</label>
                                    <span class="model-badge">${agent.embedding_model || 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Vector Store:</label>
                                    <span class="tech-badge">${agent.vector_store?.type || 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Temperature:</label>
                                    <span>${agent.model_params?.temperature || 'Default (0.7)'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Max Tokens:</label>
                                    <span>${agent.model_params?.max_tokens || 'Default (2000)'}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="detail-section">
                            <h3>Data Sources & Tools</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>Data Sources:</label>
                                    <span class="sources-info">
                                        ${agent.sources?.length || 0} configured
                                        ${agent.sources?.length === 0 ? '<span class="warning-text"> - No data sources configured</span>' : ''}
                                    </span>
                                </div>
                                <div class="detail-item">
                                    <label>Tools:</label>
                                    <span class="tools-list">${formatToolsList(agent.tools)}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Chunk Size:</label>
                                    <span>${agent.chunking?.chunk_size || '1000'} tokens</span>
                                </div>
                                <div class="detail-item">
                                    <label>Chunk Overlap:</label>
                                    <span>${agent.chunking?.chunk_overlap || '100'} tokens</span>
                                </div>
                            </div>
                        </div>

                        <div class="detail-section">
                            <h3>Usage & Performance</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>Total Queries:</label>
                                    <span class="metric-value">${analyticsData.total_requests || '0'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Success Rate:</label>
                                    <span class="metric-value">${analyticsData.total_requests > 0 ? '100.0%' : 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Avg Response Time:</label>
                                    <span class="metric-value">${analyticsData.avg_generation_time_s ? (analyticsData.avg_generation_time_s * 1000).toFixed(0) + 'ms' : 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Last Activity:</label>
                                    <span>Recently</span>
                                </div>
                            </div>
                        </div>

                        <div class="detail-section">
                            <h3>Cost & Billing</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>This Month:</label>
                                    <span class="cost-value">$${(analyticsData.total_estimated_cost_usd || 0).toFixed(4)}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Total Cost:</label>
                                    <span class="cost-value">$${(analyticsData.total_estimated_cost_usd || 0).toFixed(4)}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Token Usage:</label>
                                    <span class="metric-value">${analyticsData.total_tokens || '0'} tokens</span>
                                </div>
                                <div class="detail-item">
                                    <label>API Calls:</label>
                                    <span class="metric-value">${analyticsData.total_requests || '0'} calls</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="detail-section">
                            <h3>System Status</h3>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <label>Database Status:</label>
                                    <span class="status-badge ${dbData.database_status || 'unknown'}">${dbData.database_status || 'Unknown'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Vector Store:</label>
                                    <span class="status-badge ${dbData.vectorstore_exists ? 'success' : 'warning'}">${dbData.vectorstore_exists ? 'Ready' : 'Not Initialized'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Offline Mode:</label>
                                    <span class="status-badge ${dbData.offline_marker_exists ? 'info' : 'success'}">${dbData.offline_marker_exists ? 'Enabled' : 'Disabled'}</span>
                                </div>
                                <div class="detail-item">
                                    <label>Last Updated:</label>
                                    <span>${dbData.last_updated ? new Date(dbData.last_updated).toLocaleString() : 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                // Load actions for the details modal
                loadAgentDetailsActions(agentName, status);
            } else {
                content.innerHTML = `
                    <div class="error-state">
                        <p>Failed to load agent details. Please try again.</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error loading agent details:', error);
            content.innerHTML = `
                <div class="error-state">
                    <p>Error loading agent details: ${error.message}</p>
                </div>
            `;
        }
    } else {
        console.error('Agent details modal not found!');
    }
}

function hideAgentDetailsModal() {
    console.log('Hide agent details modal');
    hideModalById('agent-details-modal');
}

function hideRunDetailsModal() {
    console.log('Hide run details modal');
}

function logout() {
    console.log('Logout');
    localStorage.removeItem('ragnetic_username');
    window.location.href = '/login';
}

function addDataSource() {
    const container = document.getElementById('data-sources-container');
    if (!container) return;
    const id = `ds-${Date.now()}`;
    const block = document.createElement('div');
    block.className = 'data-source-block';
    block.dataset.sourceId = id;
    block.innerHTML = createDataSourceHTML(id, 'create');
    container.appendChild(block);
    
    // Add event listeners for dynamic behavior
    setupDataSourceEventListeners(block);
}

function addEditDataSource() {
    const container = document.getElementById('edit-data-sources-container');
    if (!container) return;
    const id = `eds-${Date.now()}`;
    const block = document.createElement('div');
    block.className = 'data-source-block';
    block.dataset.sourceId = id;
    block.innerHTML = createDataSourceHTML(id, 'edit');
    container.appendChild(block);
    
    // Add event listeners for dynamic behavior
    setupDataSourceEventListeners(block);
}

function updateRangeValue(id, value) {
    console.log('Update range value:', id, value);
}

// Comprehensive data source HTML generation
function createDataSourceHTML(id, mode = 'create') {
    return `
        <div class="data-source-block-content">
            <div class="data-source-header">
                <h4>Data Source Configuration</h4>
                <button type="button" class="btn-text btn-danger" onclick="removeDataSource('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3,6 5,6 21,6"></polyline>
                        <path d="m19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                    Remove Source
                </button>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="ds-type-${id}">Source Type *</label>
                    <select id="ds-type-${id}" name="source_type" class="ds-type" onchange="updateSourceFields('${id}')">
                        <option value="">Select source type...</option>
                        <option value="local">Local File Path (server)</option>
                        <option value="file_upload">File Upload (browser)</option>
                        <option value="url">Web URL</option>
                        <option value="web_crawler">Web Crawler</option>
                        <option value="code_repository">Code Repository (Git)</option>
                        <option value="db">Database (SQL)</option>
                        <option value="gdoc">Google Docs</option>
                        <option value="api">API Endpoint</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="ds-max-depth-${id}">Max Depth</label>
                    <input type="number" id="ds-max-depth-${id}" class="ds-depth" value="2" min="0" max="10" />
                    <small>How deep to crawl (0 = no limit)</small>
                </div>
            </div>
            
            <div id="ds-fields-${id}" class="source-specific-fields">
                <div class="source-placeholder">
                    <p>Select a source type to configure specific options</p>
                </div>
            </div>
            
        </div>
    `;
}

// Update source-specific fields based on type selection
function updateSourceFields(sourceId) {
    const typeSelect = document.getElementById(`ds-type-${sourceId}`);
    const fieldsContainer = document.getElementById(`ds-fields-${sourceId}`);
    const sourceType = typeSelect.value;
    
    if (!fieldsContainer) return;
    
    let fieldsHTML = '';
    
    switch (sourceType) {
        case 'local':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-path-${sourceId}">File/Directory Path *</label>
                    <input type="text" id="ds-path-${sourceId}" class="ds-path" placeholder="/path/to/file/or/directory" required />
                    <small>Server-relative path to file or directory</small>
                </div>
            `;
            break;
            
        case 'file_upload':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-upload-${sourceId}">Upload Files *</label>
                    <input type="file" id="ds-upload-${sourceId}" class="ds-upload" multiple accept="*/*" />
                    <small>Select one or more files to upload (supports all file types: PDF, DOCX, TXT, CSV, JSON, YAML, MD, HTML, Jupyter, Log, Terraform, HCL, and more)</small>
                </div>
            `;
            break;
            
        case 'url':
        case 'web_crawler':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-url-${sourceId}">URL *</label>
                    <input type="url" id="ds-url-${sourceId}" class="ds-url" placeholder="https://example.com" required />
                    <small>Web URL to crawl or scrape</small>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="ds-method-${sourceId}">HTTP Method</label>
                        <select id="ds-method-${sourceId}" class="ds-method">
                            <option value="GET">GET</option>
                            <option value="POST">POST</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="ds-headers-${sourceId}">Custom Headers (JSON)</label>
                        <textarea id="ds-headers-${sourceId}" class="ds-headers" placeholder='{"Authorization": "Bearer token"}' rows="2"></textarea>
                    </div>
                </div>
            `;
            break;
            
        case 'code_repository':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-repo-${sourceId}">Repository URL *</label>
                    <input type="url" id="ds-repo-${sourceId}" class="ds-repo" placeholder="https://github.com/user/repo" required />
                    <small>Git repository URL (GitHub, GitLab, etc.)</small>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="ds-branch-${sourceId}">Branch</label>
                        <input type="text" id="ds-branch-${sourceId}" class="ds-branch" placeholder="main" />
                    </div>
                    <div class="form-group">
                        <label for="ds-token-${sourceId}">Access Token</label>
                        <input type="password" id="ds-token-${sourceId}" class="ds-token" placeholder="Optional: for private repos" />
                    </div>
                </div>
            `;
            break;
            
        case 'db':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-connection-${sourceId}">Database Connection *</label>
                    <input type="text" id="ds-connection-${sourceId}" class="ds-connection" placeholder="postgresql://user:pass@host:port/db" required />
                    <small>Database connection string</small>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="ds-query-${sourceId}">SQL Query</label>
                        <textarea id="ds-query-${sourceId}" class="ds-query" placeholder="SELECT * FROM table" rows="3"></textarea>
                    </div>
                </div>
            `;
            break;
            
        case 'gdoc':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-folder-${sourceId}">Google Drive Folder ID</label>
                    <input type="text" id="ds-folder-${sourceId}" class="ds-folder" placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms" />
                    <small>Google Drive folder ID (optional)</small>
                </div>
                <div class="form-group">
                    <label for="ds-doc-ids-${sourceId}">Document IDs</label>
                    <textarea id="ds-doc-ids-${sourceId}" class="ds-doc-ids" placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms&#10;1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms" rows="3"></textarea>
                    <small>One document ID per line (optional)</small>
                </div>
            `;
            break;
            
        case 'api':
            fieldsHTML = `
                <div class="form-group">
                    <label for="ds-api-url-${sourceId}">API Endpoint *</label>
                    <input type="url" id="ds-api-url-${sourceId}" class="ds-api-url" placeholder="https://api.example.com/data" required />
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="ds-api-method-${sourceId}">HTTP Method</label>
                        <select id="ds-api-method-${sourceId}" class="ds-api-method">
                            <option value="GET">GET</option>
                            <option value="POST">POST</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="ds-api-headers-${sourceId}">Headers (JSON)</label>
                        <textarea id="ds-api-headers-${sourceId}" class="ds-api-headers" placeholder='{"Authorization": "Bearer token"}' rows="2"></textarea>
                    </div>
                </div>
                <div class="form-group">
                    <label for="ds-api-payload-${sourceId}">Request Payload (JSON)</label>
                    <textarea id="ds-api-payload-${sourceId}" class="ds-api-payload" placeholder='{"query": "data"}' rows="3"></textarea>
                </div>
            `;
            break;
            
            
        default:
            fieldsHTML = `
                <div class="source-placeholder">
                    <p>Select a source type to configure specific options</p>
                </div>
            `;
    }
    
    fieldsContainer.innerHTML = fieldsHTML;
}

// Setup event listeners for data source blocks
function setupDataSourceEventListeners(block) {
    const typeSelect = block.querySelector('.ds-type');
    if (typeSelect) {
        typeSelect.addEventListener('change', () => {
            const sourceId = block.dataset.sourceId;
            updateSourceFields(sourceId);
        });
    }
}

// Remove data source block
function removeDataSource(sourceId) {
    const block = document.querySelector(`[data-source-id="${sourceId}"]`);
    if (block) {
        block.remove();
    }
}

// Populate data source fields with existing data
function populateDataSourceFields(sourceId, sourceData) {
    // Set source type first
    const typeSelect = document.getElementById(`ds-type-${sourceId}`);
    if (typeSelect && sourceData.type) {
        typeSelect.value = sourceData.type;
        updateSourceFields(sourceId);
        
        // Wait for fields to be rendered, then populate them
        setTimeout(() => {
            // Set max depth
            const depthInput = document.getElementById(`ds-max-depth-${sourceId}`);
            if (depthInput && sourceData.max_depth !== undefined) {
                depthInput.value = sourceData.max_depth;
            }
            
            
            // Set type-specific fields
            switch (sourceData.type) {
                case 'local':
                    const pathInput = document.getElementById(`ds-path-${sourceId}`);
                    if (pathInput && sourceData.path) pathInput.value = sourceData.path;
                    break;
                    
                case 'url':
                case 'web_crawler':
                    const urlInput = document.getElementById(`ds-url-${sourceId}`);
                    if (urlInput && sourceData.url) urlInput.value = sourceData.url;
                    const methodSelect = document.getElementById(`ds-method-${sourceId}`);
                    if (methodSelect && sourceData.method) methodSelect.value = sourceData.method;
                    const headersTextarea = document.getElementById(`ds-headers-${sourceId}`);
                    if (headersTextarea && sourceData.headers) {
                        headersTextarea.value = JSON.stringify(sourceData.headers, null, 2);
                    }
                    break;
                    
                case 'code_repository':
                    const repoInput = document.getElementById(`ds-repo-${sourceId}`);
                    if (repoInput && sourceData.path) repoInput.value = sourceData.path;
                    const branchInput = document.getElementById(`ds-branch-${sourceId}`);
                    if (branchInput && sourceData.params?.branch) branchInput.value = sourceData.params.branch;
                    const tokenInput = document.getElementById(`ds-token-${sourceId}`);
                    if (tokenInput && sourceData.headers?.Authorization) {
                        tokenInput.value = sourceData.headers.Authorization.replace('Bearer ', '');
                    }
                    break;
                    
                case 'db':
                    const connectionInput = document.getElementById(`ds-connection-${sourceId}`);
                    if (connectionInput && sourceData.db_connection) connectionInput.value = sourceData.db_connection;
                    const queryTextarea = document.getElementById(`ds-query-${sourceId}`);
                    if (queryTextarea && sourceData.params?.query) queryTextarea.value = sourceData.params.query;
                    break;
                    
                case 'gdoc':
                    const folderInput = document.getElementById(`ds-folder-${sourceId}`);
                    if (folderInput && sourceData.folder_id) folderInput.value = sourceData.folder_id;
                    const docIdsTextarea = document.getElementById(`ds-doc-ids-${sourceId}`);
                    if (docIdsTextarea && sourceData.document_ids) {
                        docIdsTextarea.value = sourceData.document_ids.join('\n');
                    }
                    break;
                    
                case 'api':
                    const apiUrlInput = document.getElementById(`ds-api-url-${sourceId}`);
                    if (apiUrlInput && sourceData.url) apiUrlInput.value = sourceData.url;
                    const apiMethodSelect = document.getElementById(`ds-api-method-${sourceId}`);
                    if (apiMethodSelect && sourceData.method) apiMethodSelect.value = sourceData.method;
                    const apiHeadersTextarea = document.getElementById(`ds-api-headers-${sourceId}`);
                    if (apiHeadersTextarea && sourceData.headers) {
                        apiHeadersTextarea.value = JSON.stringify(sourceData.headers, null, 2);
                    }
                    const payloadTextarea = document.getElementById(`ds-api-payload-${sourceId}`);
                    if (payloadTextarea && sourceData.payload) {
                        payloadTextarea.value = JSON.stringify(sourceData.payload, null, 2);
                    }
                    break;
                    
            }
        }, 200);
    }
}

// Agent state management global functions
function deployAgentState(agentName) {
    if (window.dashboard) {
        window.dashboard.deployAgent(agentName);
    }
}

async function stopAgentState(agentName) {
    console.log('Stopping agent:', agentName);
    try {
        const response = await fetch(`/api/v1/agents/${agentName}/shutdown`, {
            method: 'POST',
            headers: { 'X-API-Key': getApiKey() }
        });
        
        if (response.ok) {
            showToast(`Agent ${agentName} stopped successfully!`, 'success');
            // Reload agents to update status
            if (typeof loadAgentsInline === 'function') {
                loadAgentsInline();
            }
        } else if (response.status === 409) {
            showToast(`Agent ${agentName} is not currently deployed`, 'info');
        } else {
            const error = await response.json().catch(() => ({}));
            showToast(`Failed to stop agent: ${error.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Stop error: ${error.message}`, 'error');
    }
}

async function resumeAgentState(agentName) {
    console.log('Resuming agent:', agentName);
    try {
        const response = await fetch(`/api/v1/agents/${agentName}/deploy-state`, {
            method: 'POST',
            headers: { 'X-API-Key': getApiKey() }
        });
        
        if (response.ok) {
            showToast(`Agent ${agentName} resumed successfully!`, 'success');
            if (typeof loadAgentsInline === 'function') {
                loadAgentsInline();
            }
        } else {
            const error = await response.json().catch(() => ({}));
            showToast(`Failed to resume agent: ${error.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Resume error: ${error.message}`, 'error');
    }
}

async function retryAgentDeployment(agentName) {
    console.log('Retrying agent deployment:', agentName);
    try {
        const response = await fetch(`/api/v1/agents/${agentName}/deploy-state`, {
            method: 'POST',
            headers: { 'X-API-Key': getApiKey() }
        });
        
        if (response.ok) {
            showToast(`Agent ${agentName} retry initiated!`, 'success');
            if (typeof loadAgentsInline === 'function') {
                loadAgentsInline();
            }
        } else {
            const error = await response.json().catch(() => ({}));
            showToast(`Failed to retry agent: ${error.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Retry error: ${error.message}`, 'error');
    }
}


function getApiKey() {
    return localStorage.getItem('ragnetic_api_key') || 'c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691';
}

function showToast(message, type = 'info') {
    console.log(`Toast: ${message} (${type})`);
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 6px;
        z-index: 10000;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Agent Detail Page Functions
function editAgentFromDetail() {
    const agentName = window.dashboard?.currentAgentName;
    if (agentName) {
        editAgent(agentName);
    }
}


function copyYAML() {
    const yamlContent = document.getElementById('agent-yaml-content').textContent;
    navigator.clipboard.writeText(yamlContent).then(() => {
        showToast('YAML copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy YAML', 'error');
    });
}

function downloadYAML() {
    const agentName = window.dashboard?.currentAgentName || 'agent';
    const yamlContent = document.getElementById('agent-yaml-content').textContent;
    
    const blob = new Blob([yamlContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${agentName}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('YAML downloaded', 'success');
}

function refreshLogs() {
    const agentName = window.dashboard?.currentAgentName;
    if (agentName && window.dashboard) {
        // Show loading state
        const logsContent = document.getElementById('agent-logs-content');
        logsContent.innerHTML = `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p>Refreshing logs...</p>
            </div>`;
        
        window.dashboard.loadAgentLogs(agentName);
        showToast('Logs refreshed', 'success');
    }
}

// Auto-refresh logs every 30 seconds
function startLogAutoRefresh() {
    const agentName = window.dashboard?.currentAgentName;
    if (agentName) {
        setInterval(() => {
            if (window.dashboard && window.dashboard.currentAgentName === agentName) {
                window.dashboard.loadAgentLogs(agentName);
            }
        }, 30000); // 30 seconds
    }
}

// Stop auto-refresh when leaving the page
function stopLogAutoRefresh() {
    if (window.logRefreshInterval) {
        clearInterval(window.logRefreshInterval);
        window.logRefreshInterval = null;
    }
}

function downloadLogs() {
    const agentName = window.dashboard?.currentAgentName || 'agent';
    const logsContent = document.getElementById('agent-logs-content').textContent;
    
    const blob = new Blob([logsContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${agentName}-logs.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Logs downloaded', 'success');
}

// Global function to update cost timeframe - defined in global scope
window.updateCostTimeframe = function() {
    console.log('updateCostTimeframe called');
    const timeframe = document.getElementById('cost-timeframe').value;
    const agentName = document.getElementById('agent-detail-title').textContent;
    
    console.log('Timeframe:', timeframe, 'Agent:', agentName);
    
    // Update the period label based on timeframe
    const periodLabels = {
        '7': 'Last 7 days',
        '30': 'Last 30 days', 
        '90': 'Last 90 days',
        '365': 'Last year'
    };
    
    const periodLabel = document.getElementById('cost-period-label');
    if (periodLabel) {
        periodLabel.textContent = periodLabels[timeframe] || 'This Month';
    }
    
    // Reload cost data with new timeframe
    if (window.dashboardManager) {
        console.log('Reloading cost data with timeframe:', timeframe);
        window.dashboardManager.loadAgentCostData(agentName, parseInt(timeframe));
    } else {
        console.log('Dashboard manager not available');
    }
};

// Initialize dashboard when DOM is loaded
console.log('Setting up DOMContentLoaded event listener');
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new Dashboard();
    window.dashboardManager = window.dashboard; // Make it accessible globally
    console.log('Dashboard initialized');
});
