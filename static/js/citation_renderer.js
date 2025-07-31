window.citationRenderer = (function() {
    const CITATION_HIGHLIGHT_CLASS = 'citation-marker';
    const CITATION_POPUP_CLASS = 'citation-popup';

    // A simple cache to avoid re-fetching snippets for the same chunk.
    const snippetsCache = {};

    /**
     * Initializes the citation renderer by adding event listeners to the main messages container.
     */
    function init() {
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.addEventListener('mouseover', handleCitationHover);
            messagesContainer.addEventListener('mouseout', hideAllCitationPopups);
            console.log("Citation Renderer initialized.");
        } else {
            console.warn("Could not find 'messages' container. Citation rendering will not be active.");
        }
    }

    /**
     * Finds and replaces citation markers in a message element's HTML with interactive spans.
     * @param {HTMLElement} messageElement - The DOM element containing the AI message.
     * @param {Array<Object>} citations - A list of citation objects. Each object must have `marker_text` and `chunk_id`.
     */
    function renderCitations(messageElement, citations) {
        if (!citations || citations.length === 0) {
            return;
        }

        const contentElement = messageElement.querySelector('.content');
        if (!contentElement) {
            console.error("Message element is missing a '.content' child. Cannot render citations.");
            return;
        }

        let contentHtml = contentElement.innerHTML;

        citations.forEach(citation => {
            const markerText = citation.marker_text;
            const chunkId = citation.chunk_id;

            if (!markerText || chunkId === undefined) {
                console.warn("Skipping invalid citation object:", citation);
                return;
            }

            // Create the interactive span with the chunk_id stored in a data attribute.
            const replacementHtml = `<span class="${CITATION_HIGHLIGHT_CLASS}" data-chunk-id="${chunkId}" data-doc-name="${citation.doc_name || ''}">${markerText}</span>`;

            // Use a regex for safer replacement to avoid conflicts with HTML attributes.
            // This ensures we only replace the text content.
            const markerRegex = new RegExp(markerText.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'), 'g');
            contentHtml = contentHtml.replace(markerRegex, replacementHtml);
        });

        contentElement.innerHTML = contentHtml;
    }

    /**
     * Handles the mouse hover event on a citation marker to show the snippet popup.
     * @param {MouseEvent} event - The mouse event.
     */
    function handleCitationHover(event) {
        const target = event.target.closest(`.${CITATION_HIGHLIGHT_CLASS}`);
        if (target) {
            showCitationPopup(target);
        }
    }

    /**
     * Displays the citation popup for a given marker element.
     * @param {HTMLElement} markerElement - The citation span element that was hovered.
     */
    function showCitationPopup(markerElement) {
        hideAllCitationPopups(); // Ensure only one popup is visible at a time.

        const { chunkId, docName } = markerElement.dataset;

        if (!chunkId) {
            console.error("Cannot show popup: chunkId is missing from data attribute.");
            return;
        }

        // Create and display the popup immediately with a "Loading..." state.
        const popup = createAndDisplayPopup(markerElement, 'Fetching snippet...', docName);

        // Fetch from cache or API.
        if (snippetsCache[chunkId]) {
            updatePopupContent(popup, snippetsCache[chunkId]);
        } else {
            fetchCitationSnippet(chunkId)
                .then(snippet => {
                    snippetsCache[chunkId] = snippet;
                    updatePopupContent(popup, snippet);
                })
                .catch(error => {
                    console.error("Error fetching citation snippet:", error);
                    updatePopupContent(popup, `Failed to load snippet: ${error.message}`);
                });
        }
    }

    /**
     * Creates the popup element and inserts it into the DOM.
     * @param {HTMLElement} markerElement - The element to anchor the popup to.
     * @param {string} content - The initial content for the popup.
     * @param {string} docName - The name of the source document for the header.
     * @returns {HTMLElement} The created popup element.
     */
    function createAndDisplayPopup(markerElement, content, docName) {
        const popup = document.createElement('div');
        popup.className = CITATION_POPUP_CLASS;

        popup.innerHTML = `
            <div class="popup-header">${docName || 'Source Document'}</div>
            <div class="popup-content loading">${content}</div>
        `;

        // Position the popup near the marker.
        markerElement.parentNode.insertBefore(popup, markerElement.nextSibling);
        return popup;
    }

    /**
     * Updates the content of an existing popup and removes the loading state.
     * @param {HTMLElement} popupElement - The popup element to update.
     * @param {string} newContent - The new text content.
     */
    function updatePopupContent(popupElement, newContent) {
        const contentDiv = popupElement?.querySelector('.popup-content');
        if (contentDiv) {
            contentDiv.textContent = newContent;
            contentDiv.classList.remove('loading');
        }
    }

    /**
     * Removes all citation popups from the DOM.
     */
    function hideAllCitationPopups() {
        document.querySelectorAll(`.${CITATION_POPUP_CLASS}`).forEach(popup => popup.remove());
    }

    /**
     * Fetches a citation snippet from the backend API using its chunk ID.
     * @param {string} chunkId - The unique ID of the document chunk.
     * @returns {Promise<string>} A promise that resolves with the snippet text.
     */
    async function fetchCitationSnippet(chunkId) {

        const apiUrl = `${API_BASE_URL}/chat/citation-snippet?chunk_id=${chunkId}`;

        try {
            const response = await fetch(apiUrl, {
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `HTTP Error ${response.status}` }));
                throw new Error(errorData.detail || 'Unknown server error');
            }

            const data = await response.json();
            return data.snippet; // Expects API to return { "snippet": "..." }
        } catch (error) {
            console.error("API call to fetch citation snippet failed:", error);
            throw error; // Re-throw to be caught by the caller.
        }
    }

    // Public API for the module.
    return {
        init: init,
        renderCitations: renderCitations,
    };
})();

// Initialize the module when the DOM is ready.
document.addEventListener('DOMContentLoaded', () => {
    window.citationRenderer?.init();
});