// Projects Management JavaScript
console.log('Projects.js loaded successfully');

class ProjectsManager {
    constructor() {
        this.projects = [];
        this.agents = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Create project form
        const createProjectForm = document.getElementById('create-project-form');
        if (createProjectForm) {
            createProjectForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.createProject();
            });
        }

        // Attach agent form
        const attachAgentForm = document.getElementById('attach-agent-form');
        if (attachAgentForm) {
            attachAgentForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.attachAgent();
            });
        }
    }

    async loadProjects() {
        try {
            // Ensure token is available
            const token = loggedInUserToken || localStorage.getItem('ragnetic_user_token');
            
            if (!token) {
                console.error('No authentication token available for projects request');
                return;
            }
            
            const response = await fetch(`${API_BASE_URL}/projects`, {
                headers: {
                    'X-API-Key': token,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.projects = data;
            this.renderProjects();
        } catch (error) {
            console.error('Error loading projects:', error);
            this.showError('Failed to load projects');
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

    renderProjects() {
        const container = document.querySelector('.projects-content');
        if (!container) return;

        if (this.projects.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                            <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                        </svg>
                    </div>
                    <h3>No Projects Yet</h3>
                    <p>Create your first project to organize your agents and deployments.</p>
                    <button class="btn-primary" onclick="projectsManager.showCreateProjectModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Create Project
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="projects-header">
                <div class="section-actions">
                    <button class="btn-primary" onclick="projectsManager.showCreateProjectModal()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 5v14M5 12h14"></path>
                        </svg>
                        Create Project
                    </button>
                </div>
            </div>
            <div class="projects-grid">
                ${this.projects.map(project => this.renderProjectCard(project)).join('')}
            </div>
        `;
    }

    renderProjectCard(project) {
        const createdDate = new Date(project.created_at).toLocaleDateString();
        const agentCount = project.agents ? project.agents.length : 0;
        
        return `
            <div class="project-card">
                <div class="project-header">
                    <div class="project-title">
                        <h3>${this.escapeHtml(project.name)}</h3>
                        <span class="project-status status-${project.status}">${project.status}</span>
                    </div>
                    <div class="project-actions">
                        <button class="btn-icon" onclick="projectsManager.showAttachAgentModal(${project.id})" title="Attach Agent">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                                <circle cx="8.5" cy="7" r="4"></circle>
                                <line x1="20" y1="8" x2="20" y2="14"></line>
                                <line x1="23" y1="11" x2="17" y2="11"></line>
                            </svg>
                        </button>
                        <button class="btn-icon" onclick="projectsManager.editProject(${project.id})" title="Edit Project">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                            </svg>
                        </button>
                        <button class="btn-icon btn-danger" onclick="projectsManager.deleteProject(${project.id})" title="Delete Project">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3,6 5,6 21,6"></polyline>
                                <path d="M19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="project-content">
                    <p class="project-description">${this.escapeHtml(project.description || 'No description provided')}</p>
                    <div class="project-stats">
                        <div class="stat">
                            <span class="stat-label">Agents</span>
                            <span class="stat-value">${agentCount}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Created</span>
                            <span class="stat-value">${createdDate}</span>
                        </div>
                    </div>
                </div>
                ${agentCount > 0 ? `
                    <div class="project-agents">
                        <h4>Attached Agents</h4>
                        <div class="agents-list">
                            ${project.agents.map(agent => `
                                <div class="agent-item">
                                    <span class="agent-name">${this.escapeHtml(agent.name)}</span>
                                    <span class="agent-status status-${agent.status}">${agent.status}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    showCreateProjectModal() {
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Create New Project</h2>
                    <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                </div>
                <form id="create-project-form">
                    <div class="form-group">
                        <label for="project-name">Project Name</label>
                        <input type="text" id="project-name" name="name" required placeholder="Enter project name">
                    </div>
                    <div class="form-group">
                        <label for="project-description">Description</label>
                        <textarea id="project-description" name="description" rows="3" placeholder="Enter project description"></textarea>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                        <button type="submit" class="btn-primary">Create Project</button>
                    </div>
                </form>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Setup form event listener
        modal.querySelector('#create-project-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.createProject();
        });
    }

    showAttachAgentModal(projectId) {
        this.loadAgents().then(() => {
            const modal = document.createElement('div');
            modal.className = 'modal show';
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>Attach Agent to Project</h2>
                        <button class="modal-close" onclick="this.closest('.modal').remove()">&times;</button>
                    </div>
                    <form id="attach-agent-form">
                        <input type="hidden" name="project_id" value="${projectId}">
                        <div class="form-group">
                            <label for="agent-select">Select Agent</label>
                            <select id="agent-select" name="agent_id" required>
                                <option value="">Choose an agent...</option>
                                ${this.agents.map(agent => `
                                    <option value="${agent.id}">${this.escapeHtml(agent.name)}</option>
                                `).join('')}
                            </select>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn-secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                            <button type="submit" class="btn-primary">Attach Agent</button>
                        </div>
                    </form>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Setup form event listener
            modal.querySelector('#attach-agent-form').addEventListener('submit', (e) => {
                e.preventDefault();
                this.attachAgent();
            });
        });
    }

    async createProject() {
        const form = document.getElementById('create-project-form');
        if (!form) return;

        const formData = new FormData(form);
        const projectData = {
            name: formData.get('name'),
            description: formData.get('description')
        };

        try {
            const response = await fetch(`${API_BASE_URL}/projects`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(projectData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showSuccess('Project created successfully');
            document.querySelector('.modal').remove();
            this.loadProjects();
        } catch (error) {
            console.error('Error creating project:', error);
            this.showError('Failed to create project');
        }
    }

    async attachAgent() {
        const form = document.getElementById('attach-agent-form');
        if (!form) return;

        const formData = new FormData(form);
        const projectId = formData.get('project_id');
        const agentId = formData.get('agent_id');

        try {
            const response = await fetch(`${API_BASE_URL}/projects/${projectId}/agents`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ agent_id: agentId })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.showSuccess('Agent attached successfully');
            document.querySelector('.modal').remove();
            this.loadProjects();
        } catch (error) {
            console.error('Error attaching agent:', error);
            this.showError('Failed to attach agent');
        }
    }

    async deleteProject(projectId) {
        if (!confirm('Are you sure you want to delete this project? This action cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
                method: 'DELETE',
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.showSuccess('Project deleted successfully');
            this.loadProjects();
        } catch (error) {
            console.error('Error deleting project:', error);
            this.showError('Failed to delete project');
        }
    }

    editProject(projectId) {
        // TODO: Implement project editing
        this.showInfo('Project editing feature coming soon');
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

// Initialize projects manager when DOM is loaded
let projectsManager;
document.addEventListener('DOMContentLoaded', () => {
    projectsManager = new ProjectsManager();
});
