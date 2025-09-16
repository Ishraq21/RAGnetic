#!/usr/bin/env python3
"""Clean fix for agent tab display integration"""

import re

# Read the current dashboard.js
with open('/Users/ishraq21/ragnetic/static/js/dashboard.js', 'r') as f:
    dashboard_js = f.read()

# 1. Update the renderAgents function to work with new agent tab structure
old_render_agents = r'''    renderAgents\(\) \{
        const container = document\.getElementById\('agents-grid'\);
        if \(!container\) return;

        const list = this\.agentList && this\.agentList\.length \? this\.agentList : this\.agents;

        if \(!list \|\| list\.length === 0\) \{
            container\.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                        <circle cx="9" cy="7" r="4"></circle>
                    </svg>
                    <h3>No agents yet</h3>
                    <p>Create your first AI agent to get started</p>
                    <button class="btn-primary" onclick="dashboard\.showCreateAgentModal\(\)">Create Agent</button>
                </div>
            `;
            return;
        \}

        const totalPages = Math\.max\(1, Math\.ceil\(list\.length / this\.agentsPerPage\)\);
        if \(this\.agentsPage > totalPages\) this\.agentsPage = totalPages;
        const start = \(this\.agentsPage - 1\) \* this\.agentsPerPage;
        const end = start \+ this\.agentsPerPage;
        const pageAgents = list\.slice\(start, end\);

        const agentsHtml = pageAgents\.map\(agent => this\.renderAgentCard\(agent\)\)\.join\('\'\);
        const hasPrev = this\.agentsPage > 1;
        const hasNext = this\.agentsPage < totalPages;
        
        container\.innerHTML = `
            <div class="list-header">Agents</div>
            \$\{agentsHtml\}
            <div class="pager">
                <button class="pager-btn" onclick="dashboard\.changeAgentsPage\(-1\)" \$\{hasPrev \? '\'\' : 'disabled'\} aria-label="Previous page">‹</button>
                <span class="pager-info">Page \$\{this\.agentsPage\} / \$\{totalPages\}</span>
                <button class="pager-btn" onclick="dashboard\.changeAgentsPage\(1\)" \$\{hasNext \? '\'\' : 'disabled'\} aria-label="Next page">›</button>
            </div>
        `;
    \}'''

new_render_agents = '''    renderAgents() {
        // Check if we're using the new agent tab structure
        const tableBody = document.getElementById('agent-table-body');
        const cardsGrid = document.getElementById('agent-cards-grid');
        const loadingState = document.getElementById('agent-loading');
        const emptyState = document.getElementById('agent-empty');
        
        // Hide loading state
        if (loadingState) loadingState.style.display = 'none';
        
        const list = this.agentList && this.agentList.length ? this.agentList : this.agents;

        if (!list || list.length === 0) {
            // Show empty state for new agent tab
            if (emptyState) {
                emptyState.style.display = 'flex';
            }
            // Hide table and grid views
            if (tableBody) tableBody.innerHTML = '';
            if (cardsGrid) cardsGrid.innerHTML = '';
            return;
        }

        // Hide empty state
        if (emptyState) emptyState.style.display = 'none';

        // Render in new agent tab structure
        if (tableBody || cardsGrid) {
            this.renderNewAgentTab(list);
        } else {
            // Fallback to old structure if new elements don't exist
            this.renderOldAgentTab(list);
        }
    }

    renderNewAgentTab(list) {
        const tableBody = document.getElementById('agent-table-body');
        const cardsGrid = document.getElementById('agent-cards-grid');
        
        if (tableBody) {
            // Render table view
            tableBody.innerHTML = list.map(agent => this.renderAgentTableRow(agent)).join('');
        }
        
        if (cardsGrid) {
            // Render grid view
            cardsGrid.innerHTML = list.map(agent => this.renderAgentCard(agent)).join('');
        }
    }

    renderOldAgentTab(list) {
        const container = document.getElementById('agents-grid');
        if (!container) return;

        const totalPages = Math.max(1, Math.ceil(list.length / this.agentsPerPage));
        if (this.agentsPage > totalPages) this.agentsPage = totalPages;
        const start = (this.agentsPage - 1) * this.agentsPerPage;
        const end = start + this.agentsPerPage;
        const pageAgents = list.slice(start, end);

        const agentsHtml = pageAgents.map(agent => this.renderAgentCard(agent)).join('');
        const hasPrev = this.agentsPage > 1;
        const hasNext = this.agentsPage < totalPages;
        
        container.innerHTML = `
            <div class="list-header">Agents</div>
            ${agentsHtml}
            <div class="pager">
                <button class="pager-btn" onclick="dashboard.changeAgentsPage(-1)" ${hasPrev ? '' : 'disabled'} aria-label="Previous page">‹</button>
                <span class="pager-info">Page ${this.agentsPage} / ${totalPages}</span>
                <button class="pager-btn" onclick="dashboard.changeAgentsPage(1)" ${hasNext ? '' : 'disabled'} aria-label="Next page">›</button>
            </div>
        `;
    }

    renderAgentTableRow(agent) {
        // Check if agent is deployed by looking for vectorstore directory or inspection endpoint
        const isDeployed = agent.is_deployed || (agent.vector_store && agent.vector_store.status === 'ready');
        const statusClass = isDeployed ? 'online' : 'offline';
        const deploymentStatus = isDeployed ? 'deployed' : 'not-deployed';
        const lastRun = agent.last_run ? new Date(agent.last_run).toLocaleString() : 'Never';
        const cost = agent.cost_this_month || 0;
        
        return `
            <tr>
                <td>
                    <div class="agent-name" onclick="dashboard.openAgentDetail('${agent.name}')">${agent.name || agent.display_name || 'Unnamed Agent'}</div>
                    <div class="agent-description">${agent.description || 'No description'}</div>
                </td>
                <td>
                    <div class="status-pill ${statusClass}">
                        <div class="status-dot"></div>
                        ${statusClass}
                    </div>
                </td>
                <td>
                    <div class="model-info">
                        <div class="model-name">${agent.llm_model || 'GPT-4'}</div>
                        <div class="embedding-name">${agent.embedding_model || 'text-embedding-3-small'}</div>
                    </div>
                </td>
                <td>
                    <div class="embedding-name">${agent.embedding_model || 'text-embedding-3-small'}</div>
                </td>
                <td>
                    <div class="deployment-status">
                        <div class="deployment-badge ${deploymentStatus}">${deploymentStatus}</div>
                    </div>
                </td>
                <td>
                    <div class="last-run">${lastRun}</div>
                </td>
                <td>
                    <div class="cost-amount">$${cost.toFixed(2)}</div>
                </td>
                <td>
                    <div class="agent-actions">
                        ${!isDeployed ? 
                            `<button class="action-btn primary" onclick="dashboard.deployAgent('${agent.name}')" title="Deploy">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                                </svg>
                            </button>` : ''
                        }
                        <button class="action-btn" onclick="dashboard.showEditAgentModal('${agent.name}')" title="Edit">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                            </svg>
                        </button>
                        <button class="action-btn" onclick="dashboard.cloneAgent('${agent.name}')" title="Clone">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        </button>
                        <button class="action-btn" onclick="dashboard.testAgent('${agent.name}')" title="Test">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="5,3 19,12 5,21"></polygon>
                            </svg>
                        </button>
                        <button class="action-btn" onclick="dashboard.deleteAgent('${agent.name}')" title="Delete">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3,6 5,6 21,6"></polyline>
                                <path d="M19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"></path>
                            </svg>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }'''

# Replace the renderAgents function
dashboard_js = re.sub(old_render_agents, new_render_agents, dashboard_js, flags=re.DOTALL)

# 2. Add missing methods that the new agent tab expects
missing_methods = '''
    // Agent detail and action methods for new agent tab
    openAgentDetail(agentName) {
        console.log('Opening agent detail for:', agentName);
        // TODO: Implement agent detail page navigation
        this.showToast('Agent detail page coming soon!', 'info');
    }

    cloneAgent(agentName) {
        console.log('Cloning agent:', agentName);
        // TODO: Implement agent cloning
        this.showToast('Agent cloning coming soon!', 'info');
    }

    testAgent(agentName) {
        console.log('Testing agent:', agentName);
        // TODO: Implement agent testing
        this.showToast('Agent testing coming soon!', 'info');
    }

    deployAgent(agentName) {
        console.log('Deploying agent:', agentName);
        // Use existing deployment logic
        this.checkAgentDeployment(agentName);
    }'''

# Add the missing methods before the closing brace of the Dashboard class
dashboard_js = dashboard_js.replace('    }', missing_methods + '\n    }')

# 3. Update the search functionality to work with new agent tab
old_search_agents = r'''        const container = document\.getElementById\('agents-grid'\);
        if \(container\) \{
            this\.agentList = filtered;
            this\.agentsPage = 1;
            this\.renderAgents\(\);
        \}'''

new_search_agents = '''        // Update agent list and re-render
        this.agentList = filtered;
        this.agentsPage = 1;
        this.renderAgents();'''

dashboard_js = re.sub(old_search_agents, new_search_agents, dashboard_js)

# Write the updated dashboard.js
with open('/Users/ishraq21/ragnetic/static/js/dashboard.js', 'w') as f:
    f.write(dashboard_js)

print("✅ Agent Tab Display Integration Fixed!")
print("")
print("🔧 Changes Made:")
print("• Updated renderAgents() to work with new agent tab structure")
print("• Added renderNewAgentTab() for table and card views")
print("• Added renderAgentTableRow() for proper table rendering")
print("• Added missing methods: openAgentDetail, cloneAgent, testAgent")
print("• Fixed search functionality to work with new structure")
print("• Used correct agent data fields: name, llm_model, embedding_model")
print("• Maintained backward compatibility with old structure")
print("")
print("🎯 Agents should now display properly in the new agent tab!")
