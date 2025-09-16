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
        this.showToast('Agent detail page coming soon!', 'info');
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
}

// Global functions for onclick handlers
function switchView(view) {
    if (window.dashboard) {
        window.dashboard.switchView(view);
    }
}

function showCreateAgentModal() {
    console.log('Show create agent modal');
}

function hideCreateAgentModal() {
    console.log('Hide create agent modal');
}

function showEditAgentModal(agentName) {
    console.log('Show edit agent modal for:', agentName);
}

function hideEditAgentModal() {
    console.log('Hide edit agent modal');
}

async function deleteAgent(agentName) {
    if (!confirm(`Are you sure you want to delete agent ${agentName}? This action cannot be undone.`)) {
        return;
    }
    
    console.log('Deleting agent:', agentName);
    try {
        const response = await fetch(`/api/v1/agents/${agentName}`, {
            method: 'DELETE',
            headers: { 'X-API-Key': getApiKey() }
        });
        
        if (response.ok) {
            showToast(`Agent ${agentName} deleted successfully!`, 'success');
            // Reload agents to update list
            if (typeof loadAgentsInline === 'function') {
                loadAgentsInline();
            }
        } else {
            const error = await response.json().catch(() => ({}));
            showToast(`Failed to delete agent: ${error.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Delete error: ${error.message}`, 'error');
    }
}

function hideDeleteConfirmationModal() {
    console.log('Hide delete confirmation modal');
}

function confirmDeleteAgent() {
    console.log('Confirm delete agent');
}

function showAgentDetails(agentName) {
    console.log('Show agent details for:', agentName);
}

function hideAgentDetailsModal() {
    console.log('Hide agent details modal');
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
    console.log('Add data source');
}

function addEditDataSource() {
    console.log('Add edit data source');
}

function updateRangeValue(id, value) {
    console.log('Update range value:', id, value);
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

function viewAgentLogs(agentName) {
    console.log('View agent logs:', agentName);
    showToast(`Viewing logs for ${agentName} - feature coming soon!`, 'info');
}

function viewAgent(agentName) {
    console.log('View agent:', agentName);
    showToast(`Viewing details for ${agentName} - feature coming soon!`, 'info');
}

function editAgent(agentName) {
    console.log('Edit agent:', agentName);
    showToast(`Editing ${agentName} - feature coming soon!`, 'info');
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

// Initialize dashboard when DOM is loaded
console.log('Setting up DOMContentLoaded event listener');
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new Dashboard();
    console.log('Dashboard initialized');
});
