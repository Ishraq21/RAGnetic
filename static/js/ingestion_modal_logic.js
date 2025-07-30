// app/static/js/ingestion_modal_logic.js

// Define the IngestionModal global object immediately upon script parse
window.IngestionModal = (function() {
    // Private variables for the module scope
    let ingestionForm;
    let ingestionAgentNameSelect;
    let ingestionSourceTypeSelect;
    let ingestionSourceFieldsDiv;
    let ingestionStatusMessagesDiv;
    let ingestionProgressBar;
    let startIngestionBtn;
    let parallelIngestionCheckbox;
    let numWorkersGroup;
    let numIngestionWorkersInput;
    let closeIngestionModalBtn;

    // UI elements for panel switching
    let ingestionFormPanel;
    let ingestionStatusPanel;
    let backToIngestionFormBtn;


    /**
     * Initializes the ingestion modal's JavaScript logic and elements.
     * This function should be called by agent_interface.html after `ingestion_modal.html`
     * has been loaded and inserted into the DOM.
     */
    function init() {
        // Basic check for required globals from agent_interface.html
        if (typeof API_BASE_URL === 'undefined' || typeof WS_URL === 'undefined' ||
            typeof loggedInUserToken === 'undefined' || typeof loggedInDbUserId === 'undefined' ||
            typeof agentSelect === 'undefined' || typeof showToast === 'undefined' ||
            typeof ingestionModal === 'undefined' || typeof openIngestionModalBtn === 'undefined'
        ) {
            console.error("IngestionModal.init: Required global variables/elements from agent_interface.html are not defined. Essential elements might be missing.");
            return;
        }

        // Query elements *inside* the ingestionModal's loaded content
        ingestionForm = ingestionModal.querySelector('#ingestion-form');
        ingestionAgentNameSelect = ingestionModal.querySelector('#ingestionAgentName');
        ingestionSourceTypeSelect = ingestionModal.querySelector('#ingestionSourceType');
        ingestionSourceFieldsDiv = ingestionModal.querySelector('#ingestion-source-fields');
        ingestionStatusMessagesDiv = ingestionModal.querySelector('#ingestion-status-messages');
        ingestionProgressBar = ingestionModal.querySelector('#ingestionProgressBar');
        startIngestionBtn = ingestionModal.querySelector('#startIngestionBtn');
        parallelIngestionCheckbox = ingestionModal.querySelector('#parallelIngestion');
        numWorkersGroup = ingestionModal.querySelector('#numWorkersGroup');
        numIngestionWorkersInput = ingestionModal.querySelector('#numIngestionWorkers');
        closeIngestionModalBtn = ingestionModal.querySelector('#close-ingestion-modal');

        // Panel elements
        ingestionFormPanel = ingestionModal.querySelector('#ingestion-form-panel');
        ingestionStatusPanel = ingestionModal.querySelector('#ingestion-status-panel');
        backToIngestionFormBtn = ingestionModal.querySelector('#backToIngestionFormBtn');


        // Attach all event listeners
        if (closeIngestionModalBtn) closeIngestionModalBtn.addEventListener('click', closeIngestionModal);
        if (ingestionSourceTypeSelect) {
            ingestionSourceTypeSelect.addEventListener('change', (event) => {
                renderIngestionSourceFields(event.target.value);
            });
        }
        if (parallelIngestionCheckbox) {
            parallelIngestionCheckbox.addEventListener('change', (event) => {
                if (['local', 'code_repository', 'directory', 'web_crawler', 'local_upload'].includes(ingestionSourceTypeSelect.value)) {
                    if (numWorkersGroup) numWorkersGroup.style.display = event.target.checked ? 'block' : 'none';
                }
            });
        }
        if (ingestionForm) {
            ingestionForm.addEventListener('submit', handleIngestionFormSubmit);
        }
        if (backToIngestionFormBtn) {
            backToIngestionFormBtn.addEventListener('click', showIngestionFormPanel);
        }

        // Initial setup for the modal state
        resetIngestionModalState();
    }

    // Function to switch to the form panel
    function showIngestionFormPanel() {
        if (ingestionFormPanel) {
            ingestionFormPanel.style.display = 'block'; // Or 'flex' if you style it as flex
        }
        if (ingestionStatusPanel) {
            ingestionStatusPanel.style.display = 'none';
        }
        // Do NOT call resetIngestionModalState here, as it calls showIngestionFormPanel,
        // which would create an infinite loop. resetIngestionModalState should be the entry point for resetting.
    }

    // Function to switch to the status panel
    function showIngestionStatusPanel() {
        if (ingestionFormPanel) {
            ingestionFormPanel.style.display = 'none';
        }
        if (ingestionStatusPanel) {
            ingestionStatusPanel.style.display = 'block'; // Or 'flex'
        }
    }

    function resetIngestionModalState() {
        if (ingestionForm) ingestionForm.reset();
        if (ingestionStatusMessagesDiv) ingestionStatusMessagesDiv.innerHTML = '<p class="info-msg">Real-time ingestion progress and logs will appear here:</p>';
        if (ingestionProgressBar) {
            ingestionProgressBar.style.width = '0%';
            ingestionProgressBar.textContent = '';
        }
        if (startIngestionBtn) {
            startIngestionBtn.disabled = false;
            startIngestionBtn.textContent = "Start Ingestion";
        }
        if (ingestionSourceTypeSelect) {
            if (!ingestionSourceTypeSelect.value) {
                ingestionSourceTypeSelect.value = 'local_upload';
            }
            renderIngestionSourceFields(ingestionSourceTypeSelect.value);
        }
        // Ensure form panel is visible and status panel is hidden on reset/initial open
        showIngestionFormPanel(); // This is the correct place to set the initial state
    }

    function closeIngestionModal() {
        if (ingestionModal) ingestionModal.style.display = 'none';
        resetIngestionModalState(); // Ensures it resets to the form view
    }

    // Helper for ingestion modal status messages
    function addIngestionStatusMessage(message, type = 'info') {
        if (!ingestionStatusMessagesDiv) {
            console.warn("ingestionStatusMessagesDiv not found. Cannot add status message:", message);
            return;
        }
        const p = document.createElement('p');
        p.className = `status-message ${type}-msg`;
        p.textContent = message;
        ingestionStatusMessagesDiv.appendChild(p);
        ingestionStatusMessagesDiv.scrollTop = ingestionStatusMessagesDiv.scrollHeight;
    }

    // Public method: loadIngestionAgents
    async function loadIngestionAgents(currentChatAgentName) {
        if (!loggedInUserToken) {
            addIngestionStatusMessage('Authentication token missing. Cannot load agents.', 'error');
            return;
        }
        try {
            const response = await fetch(`${API_BASE_URL}/agents`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            if (response.ok) {
                const agents = await response.json();
                if (ingestionAgentNameSelect) {
                    ingestionAgentNameSelect.innerHTML = '<option value="">-- Select an Agent --</option>';
                    agents.forEach(agentItem => {
                        const option = document.createElement('option');
                        option.value = agentItem.name;
                        option.textContent = agentItem.display_name || agentItem.name;
                        ingestionAgentNameSelect.appendChild(option);
                    });
                    if (agents.length > 0) {
                        ingestionAgentNameSelect.value = currentChatAgentName || agents[0].name;
                    }
                }
            } else {
                addIngestionStatusMessage(`Error loading agents: ${response.status} - ${response.statusText}`, 'error');
                console.error('Failed to load agents for ingestion modal:', response.status, response.statusText);
            }
        } catch (error) {
            addIngestionStatusMessage('Network error while loading agents for ingestion. Check server connection.', 'error');
            console.error('Network error during ingestion agent load:', error);
        }
    }

    // Public method: renderIngestionSourceFields
    function renderIngestionSourceFields(sourceType) {
        if (!ingestionSourceFieldsDiv) {
            console.warn("ingestionSourceFieldsDiv not found. Cannot render source fields.");
            return;
        }
        ingestionSourceFieldsDiv.innerHTML = '';
        let fieldsHtml = '';
        if (numWorkersGroup) numWorkersGroup.style.display = 'none';

        switch (sourceType) {
            case 'local_upload':
                fieldsHtml = `
                    <div class="form-group file-upload-group">
                        <label for="fileInput">Choose File(s) from your computer:</label>
                        <input type="file" id="fileInput" name="fileInput" multiple required>
                        <p class="info">Select one or more files to upload directly from your local machine. Supported types include PDF, DOCX, CSV, TXT, Jupyter Notebooks, Parquet files, and IaC files (YAML/Terraform).</p>
                    </div>
                `;
                if (numWorkersGroup && parallelIngestionCheckbox) numWorkersGroup.style.display = parallelIngestionCheckbox.checked ? 'block' : 'none';
                break;
            case 'local':
                fieldsHtml = `
                    <div class="form-group file-upload-group">
                        <label for="filePath">File Path (relative to project root):</label>
                        <input type="text" id="filePath" name="filePath" placeholder="e.g., data/my_documents/doc.txt" required>
                        <p class="info">Ensure this path is within your RAGnetic project's configured allowed data directories.</p>
                    </div>
                `;
                if (numWorkersGroup && parallelIngestionCheckbox) numWorkersGroup.style.display = parallelIngestionCheckbox.checked ? 'block' : 'none';
                break;
            case 'url':
            case 'web_crawler':
                fieldsHtml = `
                    <div class="form-group">
                        <label for="urlPath">URL:</label>
                        <input type="url" id="urlPath" name="urlPath" placeholder="e.g., https://example.com/docs" required>
                    </div>
                `;
                if (sourceType === 'web_crawler') {
                    fieldsHtml += `
                        <div class="form-group">
                            <label for="maxDepth">Max Crawl Depth:</label>
                            <input type="number" id="maxDepth" name="maxDepth" value="2" min="1">
                        </div>
                    `;
                }
                if (numWorkersGroup && parallelIngestionCheckbox) numWorkersGroup.style.display = parallelIngestionCheckbox.checked ? 'block' : 'none';
                break;
            case 'db':
                fieldsHtml = `
                    <div class="form-group">
                        <label for="dbConnectionString">Database Connection String:</label>
                        <input type="text" id="dbConnectionString" name="dbConnectionString" placeholder="e.g., sqlite:///data.db or postgresql://user:pass@host:port/dbname" required>
                        <p class="info">This must be a connection string configured in your server's .ragnetic/config.ini or .env.</p>
                    </div>
                `;
                break;
            case 'code_repository':
                fieldsHtml = `
                    <div class="form-group">
                        <label for="repoPath">Repository URL or Local Path:</label>
                        <input type="text" id="repoPath" name="repoPath" placeholder="e.g., https://github.com/org/repo.git or /local/path/to/repo" required>
                        <p class="info">For local paths, ensure it's within allowed data directories.</p>
                    </div>
                `;
                if (numWorkersGroup && parallelIngestionCheckbox) numWorkersGroup.style.display = parallelIngestionCheckbox.checked ? 'block' : 'none';
                break;
            case 'gdoc':
                fieldsHtml = `
                    <div class="form-group">
                        <label for="gdocFolderId">Google Drive Folder ID (optional):</label>
                        <input type="text" id="gdocFolderId" name="gdocFolderId" placeholder="Folder ID">
                    </div>
                    <div class="form-group">
                        <label for="gdocDocumentIds">Google Drive Document IDs (comma-separated, optional):</label>
                        <input type="text" id="gdocDocumentIds" name="gdocDocumentIds" placeholder="doc_id1,doc_id2">
                    </div>
                    <div class="form-group">
                        <label for="gdocFileTypes">File Types (comma-separated, optional, e.g., 'document', 'spreadsheet'):</label>
                        <input type="text" id="gdocFileTypes" name="gdocFileTypes" placeholder="document,spreadsheet">
                        <p class="info">Ensure Google Drive API credentials are configured on the server ('ragnetic auth gdrive').</p>
                    </div>
                `;
                break;
            case 'api':
                fieldsHtml = `
                    <div class="form-group">
                        <label for="apiUrl">API Endpoint URL:</label>
                        <input type="url" id="apiUrl" name="apiUrl" required>
                    </div>
                    <div class="form-group">
                        <label for="apiMethod">Method:</label>
                        <select id="apiMethod" name="apiMethod">
                            <option value="GET">GET</option>
                            <option value="POST">POST</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="apiHeaders">Headers (JSON string, optional):</label>
                        <textarea id="apiHeaders" name="apiHeaders" rows="3" placeholder='{"Authorization": "Bearer token"}'></textarea>
                    </div>
                    <div class="form-group">
                        <label for="apiParams">Parameters (JSON string for GET, optional):</label>
                        <textarea id="apiParams" name="apiParams" rows="3" placeholder='{"query": "value"}'></textarea>
                    </div>
                    <div class="form-group">
                        <label for="apiPayload">Payload (JSON string for POST, optional):</label>
                        <textarea id="apiPayload" name="apiPayload" rows="5" placeholder='{"data": "value"}'></textarea>
                    </div>
                    <div class="form-group">
                        <label for="jsonPointer">JSON Pointer (optional, e.g., '/items/0'):</label>
                        <input type="text" id="jsonPointer" name="jsonPointer" placeholder="/path/to/records">
                        <p class="info">JSON Pointer extracts specific data from the API response.</p>
                    </div>
                `;
                break;
            default:
                fieldsHtml = `<p class="info-msg">Select a source type to configure its specific ingestion options.</p>`;
        }
        ingestionSourceFieldsDiv.innerHTML = fieldsHtml;
    }

    async function handleIngestionFormSubmit(event) {
        event.preventDefault();

        // Immediately switch to the status panel
        showIngestionStatusPanel();

        addIngestionStatusMessage('Ingestion process started:', 'info'); // Clear and reset
        ingestionProgressBar.style.width = '0%';
        ingestionProgressBar.textContent = '';
        // startIngestionBtn is part of the form, so its visibility is handled by the panel.
        // Its state (disabled/text) will be handled when returning to the form.

        const selectedAgentName = ingestionAgentNameSelect.value;
        const sourceType = ingestionSourceTypeSelect.value;
        // The reproducibleIds checkbox is part of the form, its value is read once on submit
        const reproducibleIds = document.getElementById('reproducibleIds').checked;
        const parallelIngestion = document.getElementById('parallelIngestion').checked;
        const numIngestionWorkers = document.getElementById('numIngestionWorkers').value ? parseInt(document.getElementById('numIngestionWorkers').value) : undefined;

        let sourcesToAdd = [];

        try {
            if (!loggedInUserToken || !loggedInDbUserId) {
                throw new Error('Not authenticated. Please log in.');
            }
            if (!selectedAgentName) {
                throw new Error('Please select an agent.');
            }

            if (sourceType === 'local_upload') {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput ? fileInput.files : [];
                if (files.length === 0) throw new Error('No files selected for upload.');

                addIngestionStatusMessage(`Uploading ${files.length} file(s)...`, 'info');
                ingestionProgressBar.style.width = '0%';
                ingestionProgressBar.textContent = '';

                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    const formData = new FormData();
                    formData.append('file', file);

                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', `${API_BASE_URL}/agents/upload-file`, true);
                    xhr.setRequestHeader('X-API-Key', loggedInUserToken);

                    xhr.upload.onprogress = (e) => {
                        if (e.lengthComputable) {
                            const percentComplete = (e.loaded / e.total) * 100;
                            ingestionProgressBar.style.width = percentComplete + '%';
                            ingestionProgressBar.textContent = `Uploading: ${percentComplete.toFixed(0)}%`;
                        }
                    };

                    const uploadPromise = new Promise((resolve, reject) => {
                        xhr.onload = () => {
                            if (xhr.status >= 200 && xhr.status < 300) {
                                resolve(JSON.parse(xhr.responseText));
                            } else {
                                let errorDetail = xhr.responseText;
                                try { errorDetail = JSON.parse(xhr.responseText).detail; } catch (e) {}
                                reject(new Error(`Server responded with status ${xhr.status}: ${errorDetail || xhr.statusText}`));
                            }
                        };
                        xhr.onerror = () => reject(new Error('Network error during file upload.'));
                        xhr.send(formData);
                    });

                    const uploadResult = await uploadPromise;
                    sourcesToAdd.push({ type: 'local', path: uploadResult.file_path });
                    addIngestionStatusMessage(`Uploaded '${file.name}' to '${uploadResult.file_path}'`, 'success');
                }
                ingestionProgressBar.textContent = 'Uploads Complete!';
                ingestionProgressBar.style.width = '100%';
            } else {
                let newSource = {};
                switch (sourceType) {
                    case 'local':
                        const filePath = document.getElementById('filePath');
                        if (!filePath || !filePath.value) throw new Error('File Path is required.');
                        newSource = { type: sourceType, path: filePath.value };
                        break;
                    case 'url':
                    case 'web_crawler':
                        const urlPath = document.getElementById('urlPath');
                        const maxDepth = document.getElementById('maxDepth');
                        if (!urlPath || !urlPath.value) throw new Error('URL is required.');
                        newSource = { type: sourceType, url: urlPath.value };
                        if (sourceType === 'web_crawler' && maxDepth && maxDepth.value !== undefined) {
                            newSource.max_depth = parseInt(maxDepth.value);
                        }
                        break;
                    case 'db':
                        const dbConnectionString = document.getElementById('dbConnectionString');
                        if (!dbConnectionString || !dbConnectionString.value) throw new Error('Database connection string is required.');
                        newSource = { type: 'db', db_connection: dbConnectionString.value };
                        break;
                    case 'code_repository':
                        const repoPath = document.getElementById('repoPath');
                        if (!repoPath || !repoPath.value) throw new Error('Repository path is required.');
                        newSource = { type: 'code_repository', path: repoPath.value };
                        break;
                    case 'gdoc':
                        const gdocFolderId = document.getElementById('gdocFolderId');
                        const gdocDocumentIds = document.getElementById('gdocDocumentIds');
                        const gdocFileTypes = document.getElementById('gdocFileTypes');
                        if (!gdocFolderId.value && !gdocDocumentIds.value) throw new Error('Either Folder ID or Document IDs are required for Google Drive.');
                        newSource = {
                            type: 'gdoc',
                            folder_id: gdocFolderId.value || undefined,
                            document_ids: gdocDocumentIds.value ? gdocDocumentIds.value.split(',').map(id => id.trim()) : undefined,
                            file_types: gdocFileTypes.value ? gdocFileTypes.value.split(',').map(type => type.trim()) : undefined
                        };
                        break;
                    case 'api':
                        const apiUrl = document.getElementById('apiUrl');
                        const apiMethod = document.getElementById('apiMethod');
                        const apiHeaders = document.getElementById('apiHeaders');
                        const apiParams = document.getElementById('apiParams');
                        const apiPayload = document.getElementById('apiPayload');
                        const jsonPointer = document.getElementById('jsonPointer');
                        if (!apiUrl || !apiUrl.value) throw new Error('API URL is required.');
                        newSource = {
                            type: 'api',
                            url: apiUrl.value,
                            method: apiMethod.value,
                            headers: apiHeaders && apiHeaders.value ? JSON.parse(apiHeaders.value) : undefined,
                            params: apiParams && apiParams.value ? JSON.parse(apiParams.value) : undefined,
                            payload: apiPayload && apiPayload.value ? JSON.parse(apiPayload.value) : undefined,
                            json_pointer: jsonPointer && jsonPointer.value ? jsonPointer.value : undefined
                        };
                        break;
                    default:
                        throw new Error('Unsupported source type selected.');
                }
                sourcesToAdd.push(newSource);
            }

            const existingAgentResponse = await fetch(`${API_BASE_URL}/agents/${selectedAgentName}`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            if (!existingAgentResponse.ok) {
                const errorData = await existingAgentResponse.json();
                throw new Error(`Failed to fetch existing agent config: ${errorData.detail || existingAgentResponse.statusText}`);
            }
            const existingAgentConfig = await existingAgentResponse.json();

            let agentConfigPayload = JSON.parse(JSON.stringify(existingAgentConfig));

            agentConfigPayload.reproducible_ids = reproducibleIds;
            agentConfigPayload.scaling = {
                parallel_ingestion: parallelIngestion,
                num_ingestion_workers: numIngestionWorkers
            };

            const mergedSources = [];
            const existingSourcesMap = new Map();

            (existingAgentConfig.sources || []).forEach(src => {
                const identifier = src.path || src.url || src.db_connection;
                if (identifier) {
                    existingSourcesMap.set(`${src.type}-${identifier}`, src);
                }
                mergedSources.push(src);
            });

            sourcesToAdd.forEach(newSrc => {
                const identifier = newSrc.path || newSrc.url || newSrc.db_connection;
                if (identifier && !existingSourcesMap.has(`${newSrc.type}-${identifier}`)) {
                    mergedSources.push(newSrc);
                } else if (identifier) {
                    addIngestionStatusMessage(`Warning: Source '${newSrc.type}' with identifier '${identifier}' already exists. Skipping addition.`, 'warn');
                }
            });
            agentConfigPayload.sources = mergedSources;

            addIngestionStatusMessage(`Sending ingestion request to backend for agent '${selectedAgentName}'...`, 'info');
            const response = await fetch(`${API_BASE_URL}/agents/${selectedAgentName}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': loggedInUserToken
                },
                body: JSON.stringify(agentConfigPayload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`API request to trigger ingestion failed: ${errorData.detail || response.statusText}`);
            }

            const result = await response.json();
            addIngestionStatusMessage(`Ingestion triggered successfully for agent '${selectedAgentName}'. Status: ${result.status}.`, 'success');
            ingestionProgressBar.style.width = '100%';
            ingestionProgressBar.textContent = 'Ingestion Triggered!';

        } catch (error) {
            addIngestionStatusMessage(`Ingestion process error: ${error.message}`, 'error');
            console.error('Ingestion process error:', error);
            ingestionProgressBar.style.width = '0%';
            ingestionProgressBar.textContent = '';
            // If an error occurs, it's generally good to show the "Ingest More Data" button
            // so the user can go back and fix the form.
        }
    }

    // Expose public methods
    return {
        init: init,
        loadAgents: loadIngestionAgents,
        renderIngestionSourceFields: renderIngestionSourceFields,
        closeIngestionModal: closeIngestionModal,
        // showIngestionFormPanel is exposed to allow agent_interface.html to reset to form view on modal open
        showIngestionFormPanel: showIngestionFormPanel
    };

})();