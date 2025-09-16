// Training Dashboard JavaScript functionality
class TrainingDashboard {
    constructor() {
        this.trainingJobs = [];
        this.currentJob = null;
        this.refreshInterval = null;
        this.trainingPage = 1;
        this.trainingPerPage = 10;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTrainingJobs();
        this.startAutoRefresh();
    }

    setupEventListeners() {

        // Modal event listeners
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.hideAllModals();
            }
        });

        // Form validation
        const form = document.getElementById('create-training-job-form');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitTrainingJob();
            });
        }

        // GPU mode toggle
        const gpuModeToggle = document.getElementById('gpu-mode-toggle');
        if (gpuModeToggle) {
            gpuModeToggle.addEventListener('change', (e) => {
                this.toggleGPUMode(e.target.checked);
            });
        }

        // File upload event listeners
        this.setupFileUploadListeners();
    }

    setupFileUploadListeners() {
        // Training dataset upload
        const datasetUpload = document.getElementById('dataset-upload');
        const datasetArea = datasetUpload?.parentElement.querySelector('.file-upload-area');
        
        if (datasetUpload && datasetArea) {
            this.setupFileUpload(datasetUpload, datasetArea, 'dataset');
        }

        // Evaluation dataset upload
        const evalDatasetUpload = document.getElementById('eval-dataset-upload');
        const evalDatasetArea = evalDatasetUpload?.parentElement.querySelector('.file-upload-area');
        
        if (evalDatasetUpload && evalDatasetArea) {
            this.setupFileUpload(evalDatasetUpload, evalDatasetArea, 'eval-dataset');
        }
    }

    setupFileUpload(fileInput, uploadArea, type) {
        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File selection
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelection(e.target.files[0], type);
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0], type);
            }
        });
    }

    async handleFileSelection(file, type) {
        // Validate file type
        if (!file.name.endsWith('.jsonl') && !file.name.endsWith('.json')) {
            this.showToast('Please select a JSONL or JSON file', 'error');
            return;
        }

        // Show upload progress
        this.showFileUploadProgress(type, file.name);

        try {
            // Upload file
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/v1/training/upload-dataset', {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken
                },
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Upload failed: ${response.status}`);
            }

            const result = await response.json();
            
            // Update UI to show selected file
            this.showFileSelected(type, file.name, result.file_path);
            
            // Set hidden input value
            const pathInput = document.getElementById(`${type}-path`);
            if (pathInput) {
                pathInput.value = result.file_path;
            }

            this.showToast(`Dataset uploaded successfully: ${file.name}`, 'success');

        } catch (error) {
            console.error('File upload error:', error);
            this.showToast(`Upload failed: ${error.message}`, 'error');
            this.resetFileUpload(type);
        }
    }

    showFileUploadProgress(type, filename) {
        const uploadArea = document.getElementById(`${type}-upload`).parentElement.querySelector('.file-upload-area');
        const uploadContent = uploadArea.querySelector('.file-upload-content');
        const uploadSelected = uploadArea.querySelector('.file-upload-selected');
        
        if (uploadContent) uploadContent.style.display = 'none';
        if (uploadSelected) uploadSelected.style.display = 'none';

        // Create progress indicator
        let progressDiv = uploadArea.querySelector('.upload-progress');
        if (!progressDiv) {
            progressDiv = document.createElement('div');
            progressDiv.className = 'upload-progress';
            progressDiv.innerHTML = `
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <div class="upload-status">Uploading ${filename}...</div>
            `;
            uploadArea.appendChild(progressDiv);
        }
        
        progressDiv.classList.add('show');
        
        // Simulate progress (since we can't track real progress easily)
        const progressFill = progressDiv.querySelector('.progress-fill');
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = `${progress}%`;
        }, 200);

        // Store interval for cleanup
        uploadArea._progressInterval = interval;
    }

    showFileSelected(type, filename, filePath) {
        const uploadArea = document.getElementById(`${type}-upload`).parentElement.querySelector('.file-upload-area');
        const uploadContent = uploadArea.querySelector('.file-upload-content');
        const uploadSelected = uploadArea.querySelector('.file-upload-selected');
        const progressDiv = uploadArea.querySelector('.upload-progress');
        
        // Clean up progress
        if (uploadArea._progressInterval) {
            clearInterval(uploadArea._progressInterval);
            delete uploadArea._progressInterval;
        }
        if (progressDiv) {
            progressDiv.classList.remove('show');
            setTimeout(() => progressDiv.remove(), 300);
        }

        // Show selected file
        if (uploadContent) uploadContent.style.display = 'none';
        if (uploadSelected) {
            uploadSelected.style.display = 'flex';
            uploadSelected.querySelector('.file-name').textContent = filename;
        }
    }

    resetFileUpload(type) {
        const uploadArea = document.getElementById(`${type}-upload`).parentElement.querySelector('.file-upload-area');
        const uploadContent = uploadArea.querySelector('.file-upload-content');
        const uploadSelected = uploadArea.querySelector('.file-upload-selected');
        const progressDiv = uploadArea.querySelector('.upload-progress');
        const fileInput = document.getElementById(`${type}-upload`);
        const pathInput = document.getElementById(`${type}-path`);
        
        // Clean up progress
        if (uploadArea._progressInterval) {
            clearInterval(uploadArea._progressInterval);
            delete uploadArea._progressInterval;
        }
        if (progressDiv) {
            progressDiv.classList.remove('show');
            setTimeout(() => progressDiv.remove(), 300);
        }

        // Reset UI
        if (uploadContent) uploadContent.style.display = 'flex';
        if (uploadSelected) uploadSelected.style.display = 'none';
        if (fileInput) fileInput.value = '';
        if (pathInput) pathInput.value = '';
    }

    toggleGPUMode(enabled) {
        const gpuConfigSection = document.getElementById('gpu-config-section');
        if (gpuConfigSection) {
            gpuConfigSection.style.display = enabled ? 'block' : 'none';
        }

        if (enabled) {
            this.loadGPUProviders();
        }
    }

    async loadGPUProviders() {
        try {
            const response = await fetch(`${API_BASE_URL}/gpu/providers`, {
                headers: {
                    'X-API-Key': loggedInUserToken,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const providers = await response.json();
            this.populateGPUOptions(providers);
        } catch (error) {
            console.error('Error loading GPU providers:', error);
            this.showToast('Failed to load GPU providers', 'error');
        }
    }

    populateGPUOptions(providers) {
        const gpuTypeSelect = document.getElementById('gpu-type');
        const gpuProviderSelect = document.getElementById('gpu-provider');
        
        if (gpuTypeSelect) {
            gpuTypeSelect.innerHTML = '<option value="">Select GPU type...</option>';
            providers.forEach(provider => {
                provider.gpu_types.forEach(gpu => {
                    const option = document.createElement('option');
                    option.value = gpu.gpu_type;
                    option.textContent = `${gpu.gpu_type} - $${gpu.cost_per_hour}/hour`;
                    option.dataset.provider = provider.name;
                    option.dataset.cost = gpu.cost_per_hour;
                    gpuTypeSelect.appendChild(option);
                });
            });
        }

        if (gpuProviderSelect) {
            gpuProviderSelect.innerHTML = '<option value="">Select provider...</option>';
            providers.forEach(provider => {
                const option = document.createElement('option');
                option.value = provider.name;
                option.textContent = provider.name;
                gpuProviderSelect.appendChild(option);
            });
        }

        // Update cost estimate when GPU type changes
        if (gpuTypeSelect) {
            gpuTypeSelect.addEventListener('change', () => {
                this.updateCostEstimate();
            });
        }

        const maxHoursInput = document.getElementById('max-hours');
        if (maxHoursInput) {
            maxHoursInput.addEventListener('input', () => {
                this.updateCostEstimate();
            });
        }
    }

    updateCostEstimate() {
        const gpuTypeSelect = document.getElementById('gpu-type');
        const maxHoursInput = document.getElementById('max-hours');
        const costEstimate = document.getElementById('cost-estimate');

        if (!gpuTypeSelect || !maxHoursInput || !costEstimate) return;

        const selectedOption = gpuTypeSelect.options[gpuTypeSelect.selectedIndex];
        if (selectedOption && selectedOption.value) {
            const costPerHour = parseFloat(selectedOption.dataset.cost);
            const maxHours = parseFloat(maxHoursInput.value) || 0;
            const totalCost = costPerHour * maxHours;

            costEstimate.innerHTML = `
                <div class="estimate-card">
                    <h4>Cost Estimate</h4>
                    <div class="estimate-details">
                        <div class="estimate-item">
                            <span>GPU Cost:</span>
                            <span>$${costPerHour.toFixed(2)}/hour</span>
                        </div>
                        <div class="estimate-item">
                            <span>Total Estimated:</span>
                            <span>$${totalCost.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            `;
            costEstimate.style.display = 'block';
        } else {
            costEstimate.style.display = 'none';
        }
    }

    async loadTrainingJobs() {
        try {
            const response = await fetch('/api/v1/training/jobs', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.trainingJobs = await response.json();
            this.renderTrainingJobs();
        } catch (error) {
            console.error('Failed to load training jobs:', error);
            this.showToast('Failed to load training jobs', 'error');
        }
    }

    renderTrainingJobs() {
        const grid = document.getElementById('training-grid');
        if (!grid) return;

        if (this.trainingJobs.length === 0) {
            grid.innerHTML = `
                <div class="list-header">Training Jobs</div>
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                        <circle cx="9" cy="7" r="4"></circle>
                    </svg>
                    <h3>No training jobs</h3>
                    <p>Create a job to see it here</p>
                </div>
            `;
            return;
        }

        const totalPages = Math.max(1, Math.ceil(this.trainingJobs.length / this.trainingPerPage));
        if (this.trainingPage > totalPages) this.trainingPage = totalPages;
        const start = (this.trainingPage - 1) * this.trainingPerPage;
        const end = start + this.trainingPerPage;
        const pageJobs = this.trainingJobs.slice(start, end);

        const jobsHtml = pageJobs.map(job => this.renderTrainingJobCard(job)).join('');
        const hasPrev = this.trainingPage > 1;
        const hasNext = this.trainingPage < totalPages;

        grid.innerHTML = `
            <div class="list-header">Training Jobs</div>
            ${jobsHtml}
            <div class="pager">
                <button class="pager-btn" onclick="trainingDashboard.changeTrainingPage(-1)" ${hasPrev ? '' : 'disabled'} aria-label="Previous page">‹</button>
                <span class="pager-info">Page ${this.trainingPage} / ${totalPages}</span>
                <button class="pager-btn" onclick="trainingDashboard.changeTrainingPage(1)" ${hasNext ? '' : 'disabled'} aria-label="Next page">›</button>
            </div>
        `;
    }

    changeTrainingPage(delta) {
        const totalPages = Math.max(1, Math.ceil(this.trainingJobs.length / this.trainingPerPage));
        const nextPage = Math.min(Math.max(1, this.trainingPage + delta), totalPages);
        if (nextPage === this.trainingPage) return;
        this.trainingPage = nextPage;
        this.renderTrainingJobs();
    }

    renderTrainingJobCard(job) {
        const status = this.getJobStatus(job);
        const statusClass = this.getStatusClass(status);
        const progress = this.calculateProgress(job);
        const duration = this.formatDuration(job);
        const cost = job.estimated_training_cost_usd ? `$${job.estimated_training_cost_usd.toFixed(2)}` : 'N/A';

        return `
            <div class="training-job-card" onclick="trainingDashboard.showJobDetails('${job.adapter_id}')">
                <div class="job-header">
                    <div class="job-info">
                        <h3>${job.job_name}</h3>
                        <p>${job.base_model_name}</p>
                    </div>
                    <span class="job-status ${statusClass}">${status}</span>
                </div>
                
                <div class="job-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <span class="progress-text">${progress}%</span>
                </div>
                
                <div class="job-meta">
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12,6 12,12 16,14"></polyline>
                        </svg>
                        ${duration}
                    </span>
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="12" y1="1" x2="12" y2="23"></line>
                            <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                        </svg>
                        ${cost}
                    </span>
                    <span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                            <line x1="8" y1="21" x2="16" y2="21"></line>
                            <line x1="12" y1="17" x2="12" y2="21"></line>
                        </svg>
                        ${job.device || 'CPU'}
                    </span>
                </div>
                
                <div class="job-actions">
                    <button class="btn-secondary" onclick="event.stopPropagation(); trainingDashboard.showJobDetails('${job.adapter_id}')">
                        View Details
                    </button>
                    ${this.getActionButton(job)}
                </div>
            </div>
        `;
    }

    getJobStatus(job) {
        return job.training_status || 'unknown';
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

    calculateProgress(job) {
        if (job.training_status === 'completed') return 100;
        if (job.training_status === 'failed') return 0;
        if (job.training_status === 'pending') return 0;
        
        if (job.current_step && job.max_steps && job.max_steps > 0) {
            const progress = Math.round((job.current_step / job.max_steps) * 100);
            return Math.min(progress, 99); // Cap at 99% until completed
        }
        
        // For running jobs without step info, estimate based on duration
        if (job.training_status === 'running' && job.created_at) {
            const startTime = new Date(job.created_at);
            const now = new Date();
            const durationMinutes = (now - startTime) / (1000 * 60);
            
            // Estimate progress based on typical training duration (adjust as needed)
            const estimatedTotalMinutes = 30; // Default estimate
            const estimatedProgress = Math.min((durationMinutes / estimatedTotalMinutes) * 100, 95);
            return Math.round(estimatedProgress);
        }
        
        return 0;
    }

    formatDuration(job) {
        if (!job.created_at) return 'Unknown';
        
        const start = new Date(job.created_at);
        const end = job.updated_at ? new Date(job.updated_at) : new Date();
        const diffMs = end - start;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
        
        if (diffHours > 0) {
            return `${diffHours}h ${diffMinutes}m`;
        }
        return `${diffMinutes}m`;
    }

    getActionButton(job) {
        const status = job.training_status;
        
        switch (status) {
            case 'running':
                return `<button class="btn-danger" onclick="event.stopPropagation(); trainingDashboard.cancelJob('${job.adapter_id}')">Cancel</button>`;
            case 'completed':
                return `<button class="btn-danger" onclick="event.stopPropagation(); trainingDashboard.deleteJob('${job.adapter_id}')">Delete</button>`;
            case 'failed':
                return `<button class="btn-danger" onclick="event.stopPropagation(); trainingDashboard.deleteJob('${job.adapter_id}')">Delete</button>`;
            default:
                return `<button class="btn-text" onclick="event.stopPropagation(); trainingDashboard.deleteJob('${job.adapter_id}')">Delete</button>`;
        }
    }


    async showJobDetails(adapterId) {
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

            this.currentJob = await response.json();
            this.renderJobDetailsModal();
            this.showModal('training-job-details-modal');
        } catch (error) {
            console.error('Failed to load job details:', error);
            this.showToast('Failed to load job details', 'error');
        }
    }

    renderJobDetailsModal() {
        if (!this.currentJob) return;

        const job = this.currentJob;
        const status = this.getJobStatus(job);
        const statusClass = this.getStatusClass(status);
        const progress = this.calculateProgress(job);
        const duration = this.formatDuration(job);
        const cost = job.estimated_training_cost_usd ? `$${job.estimated_training_cost_usd.toFixed(2)}` : 'N/A';
        const gpuHours = job.gpu_hours_consumed ? job.gpu_hours_consumed.toFixed(2) : 'N/A';

        document.getElementById('job-details-title').textContent = job.job_name;
        document.getElementById('job-details-subtitle').textContent = `${job.base_model_name} • ${status}`;

        // Action button removed - no longer needed
        const actionButton = document.getElementById('job-action-button');
        if (actionButton) {
            actionButton.innerHTML = this.getActionButton(job);
            actionButton.onclick = () => this.performJobAction(job);
        }

        document.getElementById('job-details-content').innerHTML = `
            <div class="job-details-grid">
                <div class="detail-section">
                    <h3>Job Information</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>Job Name</label>
                            <span>${job.job_name}</span>
                        </div>
                        <div class="detail-item">
                            <label>Base Model</label>
                            <span>${job.base_model_name}</span>
                        </div>
                        <div class="detail-item">
                            <label>Status</label>
                            <span class="job-status ${statusClass}">${status}</span>
                        </div>
                        <div class="detail-item">
                            <label>Created</label>
                            <span>${new Date(job.created_at).toLocaleString()}</span>
                        </div>
                        <div class="detail-item">
                            <label>Last Updated</label>
                            <span>${new Date(job.updated_at).toLocaleString()}</span>
                        </div>
                        <div class="detail-item">
                            <label>Duration</label>
                            <span>${duration}</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h3>Training Progress</h3>
                    <div class="progress-section">
                        <div class="progress-bar large">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                        <div class="progress-info">
                            <span class="progress-text">${progress}% Complete</span>
                            ${job.current_step && job.max_steps ? 
                                `<span class="progress-steps">Step ${job.current_step} of ${job.max_steps}</span>` : 
                                ''
                            }
                        </div>
                    </div>
                    
                    ${job.eta_seconds ? `
                        <div class="eta-info">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12,6 12,12 16,14"></polyline>
                            </svg>
                            <span>ETA: ${this.formatETA(job.eta_seconds)}</span>
                        </div>
                    ` : ''}
                </div>

                <div class="detail-section">
                    <h3>Training Metrics</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>Final Loss</label>
                            <span>${job.final_loss ? job.final_loss.toFixed(4) : 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Validation Loss</label>
                            <span>${job.validation_loss ? job.validation_loss.toFixed(4) : 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <label>GPU Hours</label>
                            <span>${gpuHours}</span>
                        </div>
                        <div class="detail-item">
                            <label>Estimated Cost</label>
                            <span>${cost}</span>
                        </div>
                        <div class="detail-item">
                            <label>Device</label>
                            <span>${job.device || 'CPU'}</span>
                        </div>
                        <div class="detail-item">
                            <label>Mixed Precision</label>
                            <span>${job.mixed_precision || 'None'}</span>
                        </div>
                    </div>
                </div>

                <div class="detail-section">
                    <h3>Hyperparameters</h3>
                    <div class="hyperparameters-grid">
                        ${job.hyperparameters ? Object.entries(job.hyperparameters).map(([key, value]) => `
                            <div class="hyperparameter-item">
                                <label>${this.formatHyperparameterName(key)}</label>
                                <span>${value}</span>
                            </div>
                        `).join('') : '<p>No hyperparameters available</p>'}
                    </div>
                </div>

                <div class="detail-section">
                    <h3>Training Logs</h3>
                    <div class="logs-container">
                        <div class="logs-header">
                            <button class="btn-text" onclick="trainingDashboard.refreshLogs('${job.adapter_id}')">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="23,4 23,10 17,10"></polyline>
                                    <polyline points="1,20 1,14 7,14"></polyline>
                                    <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path>
                                </svg>
                                Refresh Logs
                            </button>
                        </div>
                        <div class="logs-content" id="logs-content-${job.adapter_id}">
                            <div class="loading-state">
                                <div class="loading-spinner"></div>
                                <p>Loading logs...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Load logs immediately
        this.loadJobLogs(job.adapter_id);
    }

    formatETA(etaSeconds) {
        if (!etaSeconds) return 'Unknown';
        
        const hours = Math.floor(etaSeconds / 3600);
        const minutes = Math.floor((etaSeconds % 3600) / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        return `${minutes}m`;
    }

    formatHyperparameterName(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    async loadJobLogs(adapterId) {
        try {
            const response = await fetch(`/api/v1/training/jobs/${adapterId}/logs`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const logsContainer = document.getElementById(`logs-content-${adapterId}`);
            
            if (logsContainer) {
                // Sanitize logs content to prevent XSS
                const sanitizedLogs = this.sanitizeLogs(data.logs || 'No logs available');
                const logSource = data.source || 'local';
                
                logsContainer.innerHTML = `
                    <div class="logs-source">
                        <span class="source-badge source-${logSource}">${logSource === 'gpu_provider' ? 'GPU Provider' : 'Local'}</span>
                    </div>
                    <pre class="logs-text">${sanitizedLogs}</pre>
                `;
            }
        } catch (error) {
            console.error('Failed to load logs:', error);
            const logsContainer = document.getElementById(`logs-content-${adapterId}`);
            if (logsContainer) {
                logsContainer.innerHTML = `
                    <div class="error-state">
                        <p>Failed to load logs: ${error.message}</p>
                    </div>
                `;
            }
        }
    }

    async refreshLogs(adapterId) {
        await this.loadJobLogs(adapterId);
        this.showToast('Logs refreshed', 'success');
    }

    async performJobAction(job) {
        const status = job.training_status;
        
        switch (status) {
            case 'running':
                await this.cancelJob(job.adapter_id);
                break;
            case 'completed':
                await this.deleteJob(job.adapter_id);
                break;
            case 'failed':
                await this.deleteJob(job.adapter_id);
                break;
            default:
                await this.deleteJob(job.adapter_id);
                break;
        }
    }

    async cancelJob(adapterId) {
        if (!confirm('Are you sure you want to cancel this training job?')) {
            return;
        }

        try {
            const response = await fetch(`/api/v1/training/jobs/${adapterId}/cancel`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.showToast('Training job cancellation requested', 'success');
            this.loadTrainingJobs();
            this.hideModal('training-job-details-modal');
        } catch (error) {
            console.error('Failed to cancel job:', error);
            this.showToast('Failed to cancel training job', 'error');
        }
    }

    async downloadModel(adapterId) {
        this.showToast('Model download feature coming soon', 'info');
        // TODO: Implement model download functionality
    }

    async deleteJob(adapterId) {
        if (!confirm('Are you sure you want to delete this training job? This action cannot be undone.')) {
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

            this.showToast('Training job deleted successfully', 'success');
            this.loadTrainingJobs();
            this.hideModal('training-job-details-modal');
        } catch (error) {
            console.error('Failed to delete job:', error);
            this.showToast('Failed to delete training job', 'error');
        }
    }

    async submitTrainingJob() {
        const form = document.getElementById('create-training-job-form');
        if (!form) return;

        // Validate required fields
        const requiredFields = ['job_name', 'base_model_name', 'output_base_dir'];
        const missingFields = [];
        
        for (const fieldName of requiredFields) {
            const field = form.querySelector(`[name="${fieldName}"]`);
            if (!field || !field.value.trim()) {
                missingFields.push(fieldName.replace('_', ' '));
            }
        }
        
        // Check if training dataset is uploaded
        const datasetPath = form.querySelector('[name="dataset_path"]').value.trim();
        if (!datasetPath) {
            this.showToast('Please upload a training dataset file', 'error');
            return;
        }
        
        if (missingFields.length > 0) {
            this.showToast(`Please fill in required fields: ${missingFields.join(', ')}`, 'error');
            return;
        }

        const formData = new FormData(form);
        const jobConfig = {};

        // Convert form data to object
        for (let [key, value] of formData.entries()) {
            if (key === 'lora_rank' || key === 'lora_alpha' || key === 'epochs' || key === 'batch_size' || 
                key === 'gradient_accumulation_steps' || key === 'logging_steps' || key === 'save_steps' || 
                key === 'save_total_limit') {
                jobConfig[key] = parseInt(value);
            } else if (key === 'learning_rate' || key === 'lora_dropout' || key === 'cost_per_gpu_hour') {
                jobConfig[key] = parseFloat(value);
            } else if (key === 'target_modules' && value.trim()) {
                jobConfig[key] = value.split(',').map(s => s.trim());
            } else if (value.trim()) {
                jobConfig[key] = value;
            }
        }

        // Create hyperparameters object
        const hyperparameters = {
            lora_rank: jobConfig.lora_rank || 8,
            learning_rate: jobConfig.learning_rate || 0.0002,
            epochs: jobConfig.epochs || 3,
            batch_size: jobConfig.batch_size || 4,
            lora_alpha: jobConfig.lora_alpha || 16,
            target_modules: jobConfig.target_modules || null,
            lora_dropout: jobConfig.lora_dropout || 0.05,
            gradient_accumulation_steps: jobConfig.gradient_accumulation_steps || 1,
            logging_steps: jobConfig.logging_steps || 10,
            save_steps: jobConfig.save_steps || 500,
            save_total_limit: jobConfig.save_total_limit || 1,
            cost_per_gpu_hour: jobConfig.cost_per_gpu_hour || 0.5,
            mixed_precision_dtype: jobConfig.mixed_precision_dtype || 'no'
        };

        const trainingJobConfig = {
            job_name: jobConfig.job_name,
            base_model_name: jobConfig.base_model_name,
            dataset_path: jobConfig.dataset_path,
            eval_dataset_path: jobConfig.eval_dataset_path || null,
            output_base_dir: jobConfig.output_base_dir,
            device: jobConfig.device || 'auto',
            hyperparameters: hyperparameters
        };

        // Add GPU configuration if GPU mode is enabled
        const gpuModeToggle = document.getElementById('gpu-mode-toggle');
        if (gpuModeToggle && gpuModeToggle.checked) {
            const gpuType = document.getElementById('gpu-type')?.value;
            const gpuProvider = document.getElementById('gpu-provider')?.value;
            const maxHours = document.getElementById('max-hours')?.value;

            if (!gpuType || !gpuProvider || !maxHours) {
                this.showToast('Please fill in all GPU configuration fields', 'error');
                return;
            }

            trainingJobConfig.use_gpu = true;
            trainingJobConfig.gpu_type = gpuType;
            trainingJobConfig.gpu_provider = gpuProvider;
            trainingJobConfig.max_hours = parseFloat(maxHours);
        }

        try {
            const endpoint = trainingJobConfig.use_gpu ? '/api/v1/training/jobs/gpu' : '/api/v1/training/apply';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify(trainingJobConfig)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showToast('Training job created successfully', 'success');
            this.hideModal('create-training-job-modal');
            this.loadTrainingJobs();
            
            // Reset form
            form.reset();
            this.resetRangeValues();
        } catch (error) {
            console.error('Failed to create training job:', error);
            this.showToast(`Failed to create training job: ${error.message}`, 'error');
        }
    }

    startAutoRefresh() {
        // Refresh every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.loadTrainingJobs();
        }, 30000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    cleanup() {
        // Clean up intervals and timeouts
        this.stopAutoRefresh();
        
        // Clear current job data
        this.currentJob = null;
        this.trainingJobs = [];
    }

    showModal(modalId) {
        // Close all other modals first
        this.hideAllModals();
        
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

    hideAllModals() {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            modal.classList.remove('show');
        });
        document.body.style.overflow = '';
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

    resetRangeValues() {
        const ranges = [
            'lora-rank', 'lora-alpha', 'lora-dropout', 'learning-rate', 
            'epochs', 'batch-size', 'gradient-accumulation'
        ];
        
        ranges.forEach(id => {
            const range = document.getElementById(id);
            if (range) {
                // Reset to default value
                const defaultValue = range.getAttribute('value') || range.defaultValue;
                range.value = defaultValue;
                updateRangeValue(id, defaultValue);
            }
        });
    }

    sanitizeLogs(logs) {
        // Basic XSS prevention for logs
        return logs
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#x27;');
    }

    async saveTrainingConfig(jobConfig) {
        try {
            // Create YAML config structure matching CLI format
            const yamlConfig = {
                job_name: jobConfig.job_name,
                base_model_name: jobConfig.base_model_name,
                dataset_path: jobConfig.dataset_path,
                eval_dataset_path: jobConfig.eval_dataset_path || null,
                output_base_dir: jobConfig.output_base_dir,
                device: jobConfig.device || 'auto',
                hyperparameters: {
                    lora_rank: parseInt(jobConfig.lora_rank) || 8,
                    learning_rate: parseFloat(jobConfig.learning_rate) || 0.0002,
                    epochs: parseInt(jobConfig.epochs) || 3,
                    batch_size: parseInt(jobConfig.batch_size) || 4,
                    lora_alpha: parseInt(jobConfig.lora_alpha) || 16,
                    lora_dropout: parseFloat(jobConfig.lora_dropout) || 0.05,
                    gradient_accumulation_steps: parseInt(jobConfig.gradient_accumulation_steps) || 1,
                    logging_steps: parseInt(jobConfig.logging_steps) || 10,
                    save_steps: parseInt(jobConfig.save_steps) || 500,
                    save_total_limit: parseInt(jobConfig.save_total_limit) || 1,
                    cost_per_gpu_hour: parseFloat(jobConfig.cost_per_gpu_hour) || 0.5,
                    mixed_precision_dtype: jobConfig.mixed_precision_dtype || 'no'
                }
            };

            // Save to training_configs directory
            const response = await fetch('/api/v1/training/save-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify({
                    config: yamlConfig,
                    filename: `${jobConfig.job_name.replace(/[^a-zA-Z0-9-_]/g, '_')}_config.yaml`
                })
            });

            if (!response.ok) {
                console.warn('Failed to save training config file:', response.status);
            }
        } catch (error) {
            console.warn('Failed to save training config file:', error);
        }
    }

    async loadTrainingConfigs() {
        try {
            const response = await fetch('/api/v1/training/configs', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                }
            });

            if (response.ok) {
                const configs = await response.json();
                this.populateConfigDropdown(configs);
            }
        } catch (error) {
            console.error('Failed to load training configs:', error);
        }
    }

    populateConfigDropdown(configs) {
        const configSelect = document.getElementById('load-config-select');
        if (!configSelect) return;

        configSelect.innerHTML = '<option value="">Load from existing config...</option>';
        configs.forEach(config => {
            const option = document.createElement('option');
            option.value = config.filename;
            option.textContent = config.job_name || config.filename;
            configSelect.appendChild(option);
        });
    }
}

// Global functions for modal interactions
function showCreateTrainingJobModal() {
    trainingDashboard.showModal('create-training-job-modal');
}

function hideCreateTrainingJobModal() {
    trainingDashboard.hideModal('create-training-job-modal');
}

function hideTrainingJobDetailsModal() {
    trainingDashboard.hideModal('training-job-details-modal');
}

function submitTrainingJob() {
    trainingDashboard.submitTrainingJob();
}

function toggleAdvancedSettings() {
    const settings = document.getElementById('advanced-settings');
    const button = event.target.closest('button');
    
    if (settings) {
        settings.classList.toggle('hidden');
        
        const icon = button.querySelector('svg');
        if (icon) {
            icon.style.transform = settings.classList.contains('hidden') ? 'rotate(0deg)' : 'rotate(180deg)';
        }
    }
}

function updateRangeValue(id, value) {
    const valueElement = document.getElementById(`${id}-value`);
    if (valueElement) {
        if (id === 'lora-dropout' || id === 'learning-rate') {
            valueElement.textContent = parseFloat(value).toFixed(4);
        } else {
            valueElement.textContent = value;
        }
    }
}

// Global functions for file upload
function removeUploadedFile(type) {
    if (trainingDashboard) {
        trainingDashboard.resetFileUpload(type);
    }
}

// Initialize training dashboard when DOM is loaded
let trainingDashboard;
document.addEventListener('DOMContentLoaded', () => {
    trainingDashboard = new TrainingDashboard();
});
