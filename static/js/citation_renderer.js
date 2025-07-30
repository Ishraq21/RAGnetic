window.citationRenderer = (function() {
    const CITATION_HIGHLIGHT_CLASS = 'citation-marker';
    const CITATION_POPUP_CLASS = 'citation-popup';

    // A simple cache to avoid re-fetching snippets for the same citation.
    const documentSnippetsCache = {};

    function init() {
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.addEventListener('click', handleCitationClick);
            messagesContainer.addEventListener('mouseover', handleCitationHover);
            messagesContainer.addEventListener('mouseout', handleCitationOut);
            console.log("Citation Renderer initialized.");
        }
    }

    /**
     * Finds and replaces citation markers in a message element's HTML with interactive spans.
     * This method is more robust than using character offsets.
     * @param {HTMLElement} messageElement - The DOM element containing the AI message.
     * @param {Array<Object>} citations - A list of citation objects from the message's metadata.
     */
    function renderCitations(messageElement, citations) {
        if (!citations || citations.length === 0) {
            return;
        }

        const contentElement = messageElement.querySelector('.content');
        if (!contentElement) return;

        let contentHtml = contentElement.innerHTML;

        citations.forEach(citation => {
            const markerText = citation.text; // The full text of the citation marker, e.g., "[↩:doc.pdf:1]"

            // Construct the replacement HTML with data attributes for the popup
            let dataAttrs = `data-doc-name="${citation.doc_name || ''}"`;
            if (citation.page_number) dataAttrs += ` data-page="${citation.page_number}"`;
            if (citation.temp_doc_id) dataAttrs += ` data-temp-doc-id="${citation.temp_doc_id}"`;

            const replacementHtml = `<span class="${CITATION_HIGHLIGHT_CLASS}" ${dataAttrs}>${markerText}</span>`;

            // Replace all occurrences of the marker text with the interactive span
            contentHtml = contentHtml.replaceAll(markerText, replacementHtml);
        });

        contentElement.innerHTML = contentHtml;
    }

    function handleCitationClick(event) {
        const target = event.target.closest(`.${CITATION_HIGHLIGHT_CLASS}`);
        if (target) {
            console.log("Citation clicked:", target.dataset);
            // Future functionality can be added here, like pinning the popup.
        }
    }

    function handleCitationHover(event) {
        const target = event.target.closest(`.${CITATION_HIGHLIGHT_CLASS}`);
        if (target) {
            showCitationPopup(target);
        }
    }

    function handleCitationOut(event) {
        const target = event.target.closest(`.${CITATION_HIGHLIGHT_CLASS}`);
        if (target) {
            hideAllCitationPopups();
        }
    }

    function showCitationPopup(markerElement) {
        hideAllCitationPopups(); // Ensure only one popup is visible at a time

        const { docName, page, tempDocId } = markerElement.dataset;
        const cacheKey = `${tempDocId || 'perm'}_${docName}_${page || 'all'}`;

        // Create and display the popup immediately with a "Loading..." state
        const popup = createAndDisplayPopup(markerElement, 'Fetching snippet...', docName, page, true);

        // Fetch from cache or API
        if (documentSnippetsCache[cacheKey]) {
            updatePopupContent(popup, documentSnippetsCache[cacheKey]);
        } else {
            fetchCitationSnippet(tempDocId, docName, page)
                .then(snippet => {
                    documentSnippetsCache[cacheKey] = snippet;
                    updatePopupContent(popup, snippet);
                })
                .catch(error => {
                    console.error("Error fetching citation snippet:", error);
                    updatePopupContent(popup, `Failed to load snippet: ${error.message}`);
                });
        }
    }

    function createAndDisplayPopup(markerElement, content, docName, page, isLoading = false) {
        const popup = document.createElement('div');
        popup.className = CITATION_POPUP_CLASS;

        popup.innerHTML = `
            <div class="popup-header">${docName || 'Document'}${page ? ` (Page ${page})` : ''}</div>
            <div class="popup-content ${isLoading ? 'loading' : ''}">${content}</div>
        `;

        markerElement.parentNode.insertBefore(popup, markerElement.nextSibling);
        return popup;
    }

    function updatePopupContent(popupElement, newContent) {
        if (!popupElement) return;
        const contentDiv = popupElement.querySelector('.popup-content');
        if (contentDiv) {
            contentDiv.textContent = newContent;
            contentDiv.classList.remove('loading');
        }
    }

    function hideAllCitationPopups() {
        document.querySelectorAll(`.${CITATION_POPUP_CLASS}`).forEach(popup => popup.remove());
    }

    async function fetchCitationSnippet(tempDocId, docName, page) {
        const apiUrl = `${API_BASE_URL}/chat/citation-snippet`;
        const params = new URLSearchParams({
            doc_name: docName,
        });

        if (page) params.append('page', page);
        if (tempDocId) params.append('temp_doc_id', tempDocId);

        try {
            const response = await fetch(`${apiUrl}?${params.toString()}`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `HTTP Error ${response.status}` }));
                throw new Error(errorData.detail);
            }

            const data = await response.json();
            return data.snippet; // Expects API to return { "snippet": "..." }
        } catch (error) {
            console.error("Error fetching citation snippet:", error);
            throw error;
        }
    }

    // Public API for the module
    return {
        init: init,
        renderCitations: renderCitations,
    };
})();

// Initialize the module when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.citationRenderer?.init();
});