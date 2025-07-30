// Define window.quickUploads as an immediately invoked function expression (IIFE)
// to encapsulate its variables and expose only public methods.
window.quickUploads = (function() {
    // Private variable to store File objects attached by the user.
    let attachedFiles = [];

    // Constants for client-side file validation.
    const MAX_FILE_SIZE_MB = 25;
    const SUPPORTED_FILE_TYPES = [
        '.pdf', '.docx', '.txt', '.csv', '.json', '.yaml', '.yml', '.hcl', '.tf', '.ipynb', '.md', '.log'
    ];

    /**
     * Initializes the quickUploads module.
     * Called automatically when the DOM content is loaded.
     */
    function init() {
        console.log("Quick Uploads module initialized.");
        // Re-render to ensure display state is correct on load (e.g., if files were somehow persisted - though they aren't yet)
        renderAttachedFiles();
    }

    /**
     * Handles the selection of files from the file input element.
     * Validates each selected file and adds valid ones to the `attachedFiles` array.
     * @param {FileList} files - The FileList object from an <input type="file"> change event.
     */
    function handleFileSelection(files) {
        if (!files || files.length === 0) {
            return;
        }

        let validFiles = [];
        let invalidCount = 0;

        // Iterate over the selected files for validation
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const validationErrors = validateFile(file); // Perform client-side validation

            if (validationErrors.length === 0) {
                validFiles.push(file); // Add valid files to a temporary list
            } else {
                displayInvalidFile(file, validationErrors); // Show error for invalid files
                invalidCount++;
            }
        }

        // Concatenate newly valid files with existing attached files
        attachedFiles = [...attachedFiles, ...validFiles];
        renderAttachedFiles(); // Update the UI to show all attached files

        if (invalidCount > 0) {
            // Notify the user about any invalid files that were skipped
            window.showToast(`Skipped ${invalidCount} invalid file(s). Check attachment list for details.`, true);
        }
    }

    /**
     * Performs client-side validation for a single file.
     * @param {File} file - The File object to validate.
     * @returns {string[]} An array of error messages; empty if the file is valid.
     */
    function validateFile(file) {
        const errors = [];

        // 1. File size check
        if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            errors.push(`File exceeds maximum size of ${MAX_FILE_SIZE_MB}MB.`);
        }

        // 2. File type/extension check
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!SUPPORTED_FILE_TYPES.includes(fileExtension)) {
            errors.push(`Unsupported file type: '${fileExtension}'. Supported types: ${SUPPORTED_FILE_TYPES.join(', ')}.`);
        }

        // Future: More advanced validation like magic bytes check could be added here
        // (requires FileReader API and more complex logic).

        return errors;
    }

    /**
     * Renders the current list of `attachedFiles` in the UI (`quickUploadFileList`).
     */
    function renderAttachedFiles() {
        // `quickUploadFileList` is a global element from `agent_interface.html`
        // It's accessed directly here because this module is designed to interact with that specific UI.
        if (!window.quickUploadFileList) {
            console.error("quickUploadFileList element not found in the DOM.");
            return;
        }

        window.quickUploadFileList.innerHTML = ''; // Clear previous entries

        // Show/hide the attachment list container based on whether there are files.
        if (attachedFiles.length === 0) {
            window.quickUploadFileList.style.display = 'none';
        } else {
            window.quickUploadFileList.style.display = 'flex'; // Use flex display for styling
        }

        // Create and append a visual item for each attached file.
        attachedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'attachment-item';
            fileItem.dataset.index = index; // Store original index for removal

            const fileNameSpan = document.createElement('span');
            fileNameSpan.className = 'file-name';
            fileNameSpan.textContent = file.name; // Display the file name

            const removeButton = document.createElement('button');
            removeButton.className = 'remove-btn';
            removeButton.innerHTML = '&times;'; // 'x' character for a close button
            removeButton.title = `Remove ${file.name}`;
            // Attach click event listener to remove the file.
            removeButton.onclick = () => removeAttachedFile(index);

            fileItem.appendChild(fileNameSpan);
            fileItem.appendChild(removeButton);
            window.quickUploadFileList.appendChild(fileItem);
        });
    }

    /**
     * Displays an invalid file in the attachment list with an error message.
     * This is separate from `renderAttachedFiles` because invalid files are not added
     * to `attachedFiles` but should still provide immediate user feedback.
     * @param {File} file - The invalid File object.
     * @param {string[]} errors - Array of validation error messages.
     */
    function displayInvalidFile(file, errors) {
        if (!window.quickUploadFileList) return;

        window.quickUploadFileList.style.display = 'flex'; // Ensure the list is visible

        const fileItem = document.createElement('div');
        fileItem.className = 'attachment-item error'; // Apply error styling
        fileItem.title = `Validation Error: ${errors.join('\n')}`; // Show errors on hover

        const fileNameSpan = document.createElement('span');
        fileNameSpan.className = 'file-name';
        fileNameSpan.textContent = `${file.name} (Error!)`; // Indicate error in text

        const removeButton = document.createElement('button');
        removeButton.className = 'remove-btn';
        removeButton.innerHTML = '&times;';
        removeButton.title = `Remove ${file.name}`;
        // For invalid files, removal simply removes the visual element.
        removeButton.onclick = () => {
            fileItem.remove();
            // If all items (including errors) are removed, hide the container.
            if (window.quickUploadFileList.children.length === 0) {
                window.quickUploadFileList.style.display = 'none';
            }
        };

        fileItem.appendChild(fileNameSpan);
        fileItem.appendChild(removeButton);
        window.quickUploadFileList.appendChild(fileItem);
    }

    /**
     * Removes a file from the `attachedFiles` array by its index
     * and triggers a re-render of the attachment list.
     * @param {number} indexToRemove - The index of the file to remove.
     */
    function removeAttachedFile(indexToRemove) {
        // Filter out the file at the specified index.
        attachedFiles = attachedFiles.filter((_, index) => index !== indexToRemove);
        renderAttachedFiles(); // Update the UI.
    }

    /**
     * Public method to get all currently attached raw File objects.
     * This is used by `sendMessage` to get files *before* they are uploaded.
     * @returns {File[]} An array of attached File objects.
     */
    function getAttachedFiles() {
        return attachedFiles;
    }

    /**
     * Public method to clear all attached files from the internal array and the UI.
     * This is typically called after files have been successfully sent/processed.
     */
    function clearAttachedFiles() {
        attachedFiles = [];
        renderAttachedFiles(); // Clear the UI.
    }

    /**
     * Public method to upload a single File object to the backend's temporary document endpoint.
     * This function is crucial for getting the `temp_doc_id` from the backend.
     * @param {File} file - The File object to upload.
     * @param {string} threadId - The current chat session's thread ID.
     * @returns {Promise<Object>} A Promise that resolves with the backend's response:
     * `{file_name: string, file_size: number, temp_doc_id: string}` on success.
     * Throws an error on failure.
     */
    async function uploadFileToBackend(file, threadId) {
        // `API_BASE_URL` and `loggedInUserToken` are global variables defined in `agent_interface.html`.
        // They are directly accessible here due to the script loading order.
        const formData = new FormData();
        formData.append('file', file);
        formData.append('thread_id', threadId); // Pass the chat thread_id for backend scoping.
        try {
            // Perform the fetch (HTTP POST) request to your backend endpoint.
            const response = await fetch(`${API_BASE_URL}/chat/upload-temp-document`, {
                method: 'POST',
                headers: {
                    'X-API-Key': loggedInUserToken, // Authenticate the request.
                },
                body: formData // Use FormData for multipart/form-data upload.
            });

            // Check if the HTTP response indicates an error (e.g., 4xx, 5xx).
            if (!response.ok) {
                let errorDetail = await response.text(); // Get raw error text from response body.
                try {
                    // Attempt to parse error detail if it's JSON (e.g., FastAPI's HTTPException).
                    errorDetail = JSON.parse(errorDetail).detail || errorDetail;
                } catch (e) {
                    /* If JSON parsing fails, use the raw text. */
                }
                // Construct and throw a user-friendly error message.
                throw new Error(`Upload failed for '${file.name}': ${response.status} - ${errorDetail}`);
            }

            // If upload is successful, parse the JSON response.
            const result = await response.json();
            // Notify the user about the successful upload.
            window.showToast(`Uploaded '${file.name}' successfully.`, false);
            // Return the crucial information (especially `temp_doc_id`) from the backend.
            return {
                file_name: result.file_name,
                file_size: result.file_size,
                temp_doc_id: result.temp_doc_id
            };
        } catch (error) {
            // Display a toast notification for the upload error.
            window.showToast(`Error uploading '${file.name}': ${error.message}`, true);
            // Re-throw the error so that the `sendMessage` function in `agent_interface.html`
            // can catch it and prevent the chat message from being sent if any upload fails.
            throw error;
        }
    }

    // Return the public interface of the module.
    // This is the crucial part that exposes the functions.
    return {
        init: init,
        handleFileSelection: handleFileSelection,
        getAttachedFiles: getAttachedFiles,
        clearAttachedFiles: clearAttachedFiles,
        uploadFileToBackend: uploadFileToBackend // EXPOSED: This was the missing piece causing the TypeError
    };
})();

// Automatically initialize the quickUploads module when the DOM is ready.
document.addEventListener('DOMContentLoaded', () => {
    // Ensure `window.quickUploads` is available globally after definition.
    if (window.quickUploads) {
        window.quickUploads.init();
    }
});