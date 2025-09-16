// GPU Configuration Form Component
class GPUConfigForm {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            showAdvanced: false,
            onConfigChange: null,
            ...options
        };
        this.gpuOptions = null;
        this.currentConfig = {
            platform: 'native',
            enabled: false,
            gpu_type: null,
            provider: null,
            max_hours: 2.0,
            auto_provision: true,
            min_memory_gb: null,
            min_cuda_cores: null,
            container_disk_gb: 50,
            volume_gb: 0,
            volume_mount_path: '/workspace',
            ports: '8000/http',
            environment_vars: {},
            docker_args: '',
            start_jupyter: false,
            start_ssh: true,
            purpose: 'inference',
            max_cost_per_hour: null,
            schedule_start: null,
            schedule_stop: null
        };
        
        this.init();
    }
    
    async init() {
        await this.loadGPUOptions();
        this.render();
        this.attachEventListeners();
    }
    
    async loadGPUOptions() {
        try {
            const response = await fetch('/api/v1/agents/gpu-options', {
                headers: {
                    'Authorization': `Bearer ${this.getApiKey()}`
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.gpuOptions = await response.json();
        } catch (error) {
            console.error('Error loading GPU options:', error);
            this.showError('Failed to load GPU options. Please refresh the page.');
        }
    }
    
    getApiKey() {
        // Get API key from localStorage or cookie
        return localStorage.getItem('api_key') || this.getCookie('api_key');
    }
    
    getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }
    
    render() {
        if (!this.gpuOptions) {
            this.container.innerHTML = '<div class="loading-spinner">Loading GPU options...</div>';
            return;
        }
        
        this.container.innerHTML = `
            <div class="gpu-config-form">
                ${this.renderPlatformSelection()}
                ${this.renderRunPodConfig()}
                ${this.renderAdvancedOptions()}
                ${this.renderCostEstimate()}
            </div>
        `;
    }
    
    renderPlatformSelection() {
        return `
            <div class="form-section">
                <div class="section-header">
                    <h3>Platform Selection</h3>
                    <p>Choose between native platform or RunPod GPU cloud</p>
                </div>
                
                <div class="platform-options">
                    <div class="platform-option ${this.currentConfig.platform === 'native' ? 'selected' : ''}" 
                         data-platform="native">
                        <div class="platform-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                                <line x1="8" y1="21" x2="16" y2="21"></line>
                                <line x1="12" y1="17" x2="12" y2="21"></line>
                            </svg>
                        </div>
                        <div class="platform-info">
                            <h4>${this.gpuOptions.platforms.native.name}</h4>
                            <p>${this.gpuOptions.platforms.native.description}</p>
                            <ul class="feature-list">
                                ${this.gpuOptions.platforms.native.features.map(feature => `<li>${feature}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="platform-option ${this.currentConfig.platform === 'runpod' ? 'selected' : ''}" 
                         data-platform="runpod">
                        <div class="platform-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                                <polyline points="3.27,6.96 12,12.01 20.73,6.96"></polyline>
                                <line x1="12" y1="22.08" x2="12" y2="12"></line>
                            </svg>
                        </div>
                        <div class="platform-info">
                            <h4>${this.gpuOptions.platforms.runpod.name}</h4>
                            <p>${this.gpuOptions.platforms.runpod.description}</p>
                            <ul class="feature-list">
                                ${this.gpuOptions.platforms.runpod.features.map(feature => `<li>${feature}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderRunPodConfig() {
        if (this.currentConfig.platform !== 'runpod') {
            return '';
        }
        
        return `
            <div class="form-section">
                <div class="section-header">
                    <h3>RunPod GPU Configuration</h3>
                    <p>Configure GPU requirements and instance settings</p>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="gpu-enabled" ${this.currentConfig.enabled ? 'checked' : ''}>
                            Enable GPU Acceleration
                        </label>
                        <small>Enable GPU acceleration for this agent</small>
                    </div>
                </div>
                
                <div class="gpu-config-details" style="display: ${this.currentConfig.enabled ? 'block' : 'none'}">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="gpu-type">GPU Type</label>
                            <select id="gpu-type" ${!this.currentConfig.enabled ? 'disabled' : ''}>
                                <option value="">Select GPU Type</option>
                                ${this.gpuOptions.gpu_types.map(gpu => `
                                    <option value="${gpu.id}" ${this.currentConfig.gpu_type === gpu.id ? 'selected' : ''}>
                                        ${gpu.display_name} (${gpu.memory_gb}GB, ${gpu.cuda_cores.toLocaleString()} CUDA cores)
                                    </option>
                                `).join('')}
                            </select>
                            <small>Choose the GPU type based on your requirements</small>
                        </div>
                        
                        <div class="form-group">
                            <label for="gpu-provider">Provider</label>
                            <select id="gpu-provider" ${!this.currentConfig.enabled ? 'disabled' : ''}>
                                <option value="">Select Provider</option>
                            </select>
                            <small>Choose the provider tier for cost and availability</small>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="max-hours">Maximum Hours</label>
                            <input type="number" id="max-hours" min="0.1" max="168" step="0.1" 
                                   value="${this.currentConfig.max_hours}" ${!this.currentConfig.enabled ? 'disabled' : ''}>
                            <small>Maximum hours to run the GPU instance (0.1 to 168 hours)</small>
                        </div>
                        
                        <div class="form-group">
                            <label for="purpose">Purpose</label>
                            <select id="purpose" ${!this.currentConfig.enabled ? 'disabled' : ''}>
                                <option value="inference" ${this.currentConfig.purpose === 'inference' ? 'selected' : ''}>Inference</option>
                                <option value="training" ${this.currentConfig.purpose === 'training' ? 'selected' : ''}>Training</option>
                                <option value="both" ${this.currentConfig.purpose === 'both' ? 'selected' : ''}>Both</option>
                                <option value="development" ${this.currentConfig.purpose === 'development' ? 'selected' : ''}>Development</option>
                            </select>
                            <small>Primary purpose for GPU usage</small>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="min-memory">Minimum Memory (GB)</label>
                            <input type="number" id="min-memory" min="4" max="320" 
                                   value="${this.currentConfig.min_memory_gb || ''}" ${!this.currentConfig.enabled ? 'disabled' : ''}>
                            <small>Minimum GPU memory requirement (optional)</small>
                        </div>
                        
                        <div class="form-group">
                            <label for="max-cost">Max Cost per Hour ($)</label>
                            <input type="number" id="max-cost" min="0.1" max="10" step="0.01" 
                                   value="${this.currentConfig.max_cost_per_hour || ''}" ${!this.currentConfig.enabled ? 'disabled' : ''}>
                            <small>Maximum cost per hour willing to pay (optional)</small>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="auto-provision" ${this.currentConfig.auto_provision ? 'checked' : ''} ${!this.currentConfig.enabled ? 'disabled' : ''}>
                            Auto-provision GPU on deployment
                        </label>
                        <small>Automatically provision GPU when agent is deployed</small>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderAdvancedOptions() {
        if (this.currentConfig.platform !== 'runpod' || !this.currentConfig.enabled) {
            return '';
        }
        
        return `
            <div class="form-section">
                <div class="section-header">
                    <h3>Advanced Configuration</h3>
                    <button type="button" class="btn-secondary" id="toggle-advanced">
                        ${this.options.showAdvanced ? 'Hide' : 'Show'} Advanced Options
                    </button>
                </div>
                
                <div class="advanced-options" style="display: ${this.options.showAdvanced ? 'block' : 'none'}">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="container-disk">Container Disk (GB)</label>
                            <input type="number" id="container-disk" min="10" max="500" 
                                   value="${this.currentConfig.container_disk_gb}">
                            <small>Container disk size (10-500 GB)</small>
                        </div>
                        
                        <div class="form-group">
                            <label for="volume-size">Volume Size (GB)</label>
                            <input type="number" id="volume-size" min="0" max="1000" 
                                   value="${this.currentConfig.volume_gb}">
                            <small>Additional volume size for persistent storage (0-1000 GB)</small>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="volume-mount">Volume Mount Path</label>
                            <input type="text" id="volume-mount" value="${this.currentConfig.volume_mount_path}">
                            <small>Path where the volume will be mounted</small>
                        </div>
                        
                        <div class="form-group">
                            <label for="ports">Ports</label>
                            <input type="text" id="ports" value="${this.currentConfig.ports}">
                            <small>Ports to expose (e.g., "8000/http,8080/http")</small>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="docker-args">Docker Arguments</label>
                        <textarea id="docker-args" rows="2">${this.currentConfig.docker_args}</textarea>
                        <small>Additional Docker arguments (optional)</small>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="start-jupyter" ${this.currentConfig.start_jupyter ? 'checked' : ''}>
                                Start Jupyter Lab
                            </label>
                            <small>Start Jupyter Lab on the instance</small>
                        </div>
                        
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="start-ssh" ${this.currentConfig.start_ssh ? 'checked' : ''}>
                                Enable SSH Access
                            </label>
                            <small>Enable SSH access to the instance</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderCostEstimate() {
        if (this.currentConfig.platform !== 'runpod' || !this.currentConfig.enabled) {
            return '';
        }
        
        const selectedGpu = this.gpuOptions.gpu_types.find(gpu => gpu.id === this.currentConfig.gpu_type);
        const selectedProvider = this.getSelectedProvider();
        
        if (!selectedGpu || !selectedProvider) {
            return '';
        }
        
        const costPerHour = selectedProvider.cost_per_hour;
        const totalCost = costPerHour * this.currentConfig.max_hours;
        
        return `
            <div class="form-section cost-estimate">
                <div class="section-header">
                    <h3>Cost Estimate</h3>
                </div>
                
                <div class="cost-breakdown">
                    <div class="cost-item">
                        <span class="cost-label">GPU Type:</span>
                        <span class="cost-value">${selectedGpu.display_name}</span>
                    </div>
                    <div class="cost-item">
                        <span class="cost-label">Provider:</span>
                        <span class="cost-value">${selectedProvider.name}</span>
                    </div>
                    <div class="cost-item">
                        <span class="cost-label">Cost per Hour:</span>
                        <span class="cost-value">$${costPerHour.toFixed(2)}</span>
                    </div>
                    <div class="cost-item">
                        <span class="cost-label">Max Hours:</span>
                        <span class="cost-value">${this.currentConfig.max_hours}</span>
                    </div>
                    <div class="cost-item total">
                        <span class="cost-label">Estimated Total Cost:</span>
                        <span class="cost-value">$${totalCost.toFixed(2)}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    getSelectedProvider() {
        if (!this.currentConfig.gpu_type || !this.currentConfig.provider) {
            return null;
        }
        
        const providers = this.gpuOptions.providers_by_gpu[this.currentConfig.gpu_type] || [];
        return providers.find(p => p.name === this.currentConfig.provider);
    }
    
    attachEventListeners() {
        // Platform selection
        this.container.addEventListener('click', (e) => {
            if (e.target.closest('.platform-option')) {
                const platform = e.target.closest('.platform-option').dataset.platform;
                this.setPlatform(platform);
            }
        });
        
        // GPU enabled toggle
        const gpuEnabled = this.container.querySelector('#gpu-enabled');
        if (gpuEnabled) {
            gpuEnabled.addEventListener('change', (e) => {
                this.currentConfig.enabled = e.target.checked;
                this.render();
                this.attachEventListeners();
                this.notifyConfigChange();
            });
        }
        
        // GPU type selection
        const gpuType = this.container.querySelector('#gpu-type');
        if (gpuType) {
            gpuType.addEventListener('change', (e) => {
                this.currentConfig.gpu_type = e.target.value;
                this.updateProviderOptions();
                this.renderCostEstimate();
                this.notifyConfigChange();
            });
        }
        
        // Provider selection
        const gpuProvider = this.container.querySelector('#gpu-provider');
        if (gpuProvider) {
            gpuProvider.addEventListener('change', (e) => {
                this.currentConfig.provider = e.target.value;
                this.renderCostEstimate();
                this.notifyConfigChange();
            });
        }
        
        // Other form inputs
        const inputs = this.container.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                input.addEventListener('change', (e) => {
                    this.updateConfigFromInput(e.target);
                });
            } else {
                input.addEventListener('input', (e) => {
                    this.updateConfigFromInput(e.target);
                });
            }
        });
        
        // Advanced options toggle
        const toggleAdvanced = this.container.querySelector('#toggle-advanced');
        if (toggleAdvanced) {
            toggleAdvanced.addEventListener('click', () => {
                this.options.showAdvanced = !this.options.showAdvanced;
                this.render();
                this.attachEventListeners();
            });
        }
    }
    
    setPlatform(platform) {
        this.currentConfig.platform = platform;
        if (platform === 'native') {
            this.currentConfig.enabled = false;
        }
        this.render();
        this.attachEventListeners();
        this.notifyConfigChange();
    }
    
    updateProviderOptions() {
        const providerSelect = this.container.querySelector('#gpu-provider');
        if (!providerSelect || !this.currentConfig.gpu_type) {
            return;
        }
        
        const providers = this.gpuOptions.providers_by_gpu[this.currentConfig.gpu_type] || [];
        providerSelect.innerHTML = '<option value="">Select Provider</option>';
        
        providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.name;
            option.textContent = `${provider.name} - $${provider.cost_per_hour.toFixed(2)}/hour`;
            if (provider.name === this.currentConfig.provider) {
                option.selected = true;
            }
            providerSelect.appendChild(option);
        });
    }
    
    updateConfigFromInput(input) {
        const value = input.type === 'checkbox' ? input.checked : input.value;
        const field = input.id.replace('-', '_');
        
        if (field in this.currentConfig) {
            if (input.type === 'number') {
                this.currentConfig[field] = value ? parseFloat(value) : null;
            } else {
                this.currentConfig[field] = value;
            }
            
            this.notifyConfigChange();
        }
    }
    
    renderCostEstimate() {
        const costSection = this.container.querySelector('.cost-estimate');
        if (costSection) {
            costSection.outerHTML = this.renderCostEstimate();
        }
    }
    
    notifyConfigChange() {
        if (this.options.onConfigChange) {
            this.options.onConfigChange(this.currentConfig);
        }
    }
    
    getConfig() {
        return { ...this.currentConfig };
    }
    
    setConfig(config) {
        this.currentConfig = { ...this.currentConfig, ...config };
        this.render();
        this.attachEventListeners();
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.container.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Export for use in other modules
window.GPUConfigForm = GPUConfigForm;
