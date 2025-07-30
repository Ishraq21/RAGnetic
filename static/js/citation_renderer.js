window.citationRenderer = (function() {
    const CITATION_HIGHLIGHT_CLASS = 'citation-marker';
    const CITATION_POPUP_CLASS = 'citation-popup';

    // This is a temporary cache for document snippets. In a production system,
    // you might fetch these on demand via an API endpoint that provides snippets
    // based on temp_doc_id, doc_name, and page.
    let documentSnippetsCache = {}; // Key: temp_doc_id_page, Value: {content: "...", doc_name: "...", page: N}

    function init() {
        // Event delegation for dynamically added messages
        document.getElementById('messages').addEventListener('click', handleCitationClick);
        document.getElementById('messages').addEventListener('mouseover', handleCitationHover);
        document.getElementById('messages').addEventListener('mouseout', handleCitationOut);
        console.log("Citation Renderer initialized.");
    }

    /**
     * Renders citations in a given message element based on its stored meta data.
     * This should be called after a message's content is rendered (e.g., after marked.parse).
     *
     * @param {HTMLElement} messageElement The DOM element containing the AI message.
     * @param {Array<Object>} citations A list of citation objects from the message's meta.
     */
    function renderCitations(messageElement, citations) {
        if (!citations || citations.length === 0) {
            return;
        }

        let contentHtml = messageElement.querySelector('.content').innerHTML;
        let offset = 0; // Keep track of offset due to HTML insertions

        // Sort citations by start_char to avoid issues with offset
        citations.sort((a, b) => a.start_char - b.start_char);

        citations.forEach(citation => {
            const markerText = citation.marker_text;
            const docName = citation.doc_name;
            const page = citation.page;
            const tempDocId = citation.temp_doc_id;
            const originalStart = citation.start_char;
            const originalEnd = citation.end_char;

            if (!markerText || originalStart === undefined || originalEnd === undefined) {
                console.warn("Invalid citation data, skipping:", citation);
                return;
            }

            // Construct data attributes for lookup
            let dataAttrs = `data-doc-name="${docName || ''}"`;
            if (page !== null) dataAttrs += ` data-page="${page}"`;
            if (tempDocId) dataAttrs += ` data-temp-doc-id="${tempDocId}"`;

            const replacementHtml = `<span class="${CITATION_HIGHLIGHT_CLASS}" ${dataAttrs}>${markerText}</span>`;

            // Insert the span into the HTML string, adjusting for previous insertions
            const currentStart = originalStart + offset;
            const currentEnd = originalEnd + offset;

            if (currentEnd > contentHtml.length) {
                console.warn(`Citation marker bounds out of range for "${markerText}". HTML length: ${contentHtml.length}, start: ${currentStart}, end: ${currentEnd}`);
                return;
            }

            contentHtml = contentHtml.substring(0, currentStart) +
                          replacementHtml +
                          contentHtml.substring(currentEnd);

            // Update offset based on the length of the HTML inserted
            offset += (replacementHtml.length - markerText.length);
        });

        messageElement.querySelector('.content').innerHTML = contentHtml;
    }

    function handleCitationClick(event) {
        const target = event.target;
        if (target.classList.contains(CITATION_HIGHLIGHT_CLASS)) {
            // Future: Implement deep-linking or more complex pop-up.
            // For now, just logging click.
            console.log("Citation clicked:", target.dataset);
            showCitationPopup(target);
        }
    }

    function handleCitationHover(event) {
        const target = event.target;
        if (target.classList.contains(CITATION_HIGHLIGHT_CLASS)) {
            showCitationPopup(target);
        }
    }

    function handleCitationOut(event) {
        const target = event.target;
        if (target.classList.contains(CITATION_HIGHLIGHT_CLASS)) {
            hideAllCitationPopups();
        }
    }

    function showCitationPopup(citationMarkerElement) {
        hideAllCitationPopups(); // Hide any other open popups

        const docName = citationMarkerElement.dataset.docName;
        const page = citationMarkerElement.dataset.page;
        const tempDocId = citationMarkerElement.dataset.tempDocId;

        // Construct a unique key for the cache
        const cacheKey = `${tempDocId || 'perm'}_${docName || 'unknown'}_${page || 'no-page'}`;

        let snippetContent = `Fetching snippet for ${docName}`;
        if (page) snippetContent += ` (Page ${page})`;
        snippetContent += '...';

        const existingSnippet = documentSnippetsCache[cacheKey];

        if (existingSnippet) {
            snippetContent = existingSnippet.content;
            createAndDisplayPopup(citationMarkerElement, snippetContent, docName, page);
        } else {
            // Show a "Loading..." popup immediately
            createAndDisplayPopup(citationMarkerElement, snippetContent, docName, page, true);
            // Fetch snippet from backend API
            fetchCitationSnippet(tempDocId, docName, page)
                .then(snippet => {
                    documentSnippetsCache[cacheKey] = {
                        content: snippet,
                        doc_name: docName,
                        page: page
                    };
                    // Update the popup with actual content
                    if (citationMarkerElement.nextElementSibling &&
                        citationMarkerElement.nextElementSibling.classList.contains(CITATION_POPUP_CLASS)) {
                        citationMarkerElement.nextElementSibling.querySelector('.popup-content').textContent = snippet;
                    } else {
                        // If popup was hidden/removed due to quick mouse movement, re-display
                        createAndDisplayPopup(citationMarkerElement, snippet, docName, page);
                    }
                })
                .catch(error => {
                    console.error("Error fetching citation snippet:", error);
                    const errorMessage = `Failed to load snippet for ${docName}.`;
                     if (citationMarkerElement.nextElementSibling &&
                        citationMarkerElement.nextElementSibling.classList.contains(CITATION_POPUP_CLASS)) {
                        citationMarkerElement.nextElementSibling.querySelector('.popup-content').textContent = errorMessage;
                    } else {
                        createAndDisplayPopup(citationMarkerElement, errorMessage, docName, page);
                    }
                });
        }
    }

    function createAndDisplayPopup(markerElement, content, docName, page, isLoading = false) {
        const popup = document.createElement('div');
        popup.className = CITATION_POPUP_CLASS;

        const header = document.createElement('div');
        header.className = 'popup-header';
        header.textContent = `${docName || 'Document'}${page ? ` (Page ${page})` : ''}`;
        popup.appendChild(header);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'popup-content';
        contentDiv.textContent = content;
        if (isLoading) contentDiv.classList.add('loading');
        popup.appendChild(contentDiv);

        // Position the popup
        // Append it as a sibling to the marker, for easier positioning
        markerElement.parentNode.insertBefore(popup, markerElement.nextSibling);

        // Basic positioning (adjust as needed with CSS)
        // You might need to calculate more precise position based on markerElement's rect
        // For simplicity, CSS will handle absolute positioning relative to the message bubble.
    }

    function hideAllCitationPopups() {
        document.querySelectorAll(`.${CITATION_POPUP_CLASS}`).forEach(popup => popup.remove());
    }

    async function fetchCitationSnippet(tempDocId, docName, page) {
        // This is a placeholder for a new API endpoint you'd create in app/main.py
        // This endpoint would retrieve a small snippet of the document content
        // based on temp_doc_id, doc_name, and page.
        // For now, it returns a dummy snippet.
        const apiUrl = `${API_BASE_URL}/chat/citation-snippet`; // Example API endpoint

        const params = new URLSearchParams({
            doc_name: docName,
            page: page || '', // Send empty string if no page
            temp_doc_id: tempDocId || '' // Send empty string if no temp_doc_id (for permanent docs)
        });

        // In a real scenario, you'd send `temp_doc_id` or `doc_name` to identify the source.
        // For simplicity, this mock returns a generic snippet.
        // You might need to add current_user.id and current_thread_id to API request if needed for backend lookup.

        try {
            // For now, return a dummy string.
            // In Phase 3, Step 5, we will create the backend endpoint to fetch real snippets.
            return `Snippet from '${docName}' ${page ? `(page ${page})` : ''}. This is a placeholder snippet.`;
            /*
            const response = await fetch(`${apiUrl}?${params.toString()}`, {
                headers: { 'X-API-Key': loggedInUserToken }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data.snippet; // Assume API returns { "snippet": "..." }
            */
        } catch (error) {
            console.error("Error fetching citation snippet:", error);
            throw error;
        }
    }


    // Public methods
    return {
        init: init,
        renderCitations: renderCitations,
        // Expose fetchCitationSnippet for testing or direct use if needed
        fetchCitationSnippet: fetchCitationSnippet
    };

})();

// Initialize the citationRenderer module when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (window.citationRenderer) {
        window.citationRenderer.init();
    }
});