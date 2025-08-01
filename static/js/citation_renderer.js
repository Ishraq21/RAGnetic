window.citationRenderer = (function() {
  const CITATION_HIGHLIGHT_CLASS = 'citation-marker';
  const API_BASE_URL = window.location.origin + '/api/v1';

  const CHAT_API_ENDPOINT = `${API_BASE_URL}/chat/citation-snippet`;

  let snippetCache = {};

  function showCitationPopup(citation, snippet, targetElement) {
      console.log("Attempting to show popup with snippet:", snippet);

      let popup = document.getElementById('citation-popup');
      if (!popup) {
          popup = document.createElement('div');
          popup.id = 'citation-popup';
          popup.className = 'citation-popup';
          document.body.appendChild(popup);
          popup.addEventListener('click', (e) => e.stopPropagation());
      }

      const docName = citation.doc_name || 'Document';
      const pageNumber = citation.page_number ? `(Page ${citation.page_number})` : '';
      const chunkContent = snippet ? `<div class="popup-snippet">${snippet}</div>` : `<div class="popup-snippet-error">Source not available. This may be a hallucinated citation.</div>`;

      popup.innerHTML = `
          <div class="popup-header">
              <span class="popup-title">${docName} ${pageNumber}</span>
              <button class="popup-close-btn">&times;</button>
          </div>
          <div class="popup-content">
              ${chunkContent}
          </div>
      `;

      const closeBtn = popup.querySelector('.popup-close-btn');
      closeBtn.addEventListener('click', () => {
          popup.style.display = 'none';
      });

      const rect = targetElement.getBoundingClientRect();
      const popupWidth = 400;
      let top = rect.bottom + window.scrollY + 8;
      let left = rect.left + window.scrollX;

      if (left + popupWidth > window.innerWidth) {
          left = window.innerWidth - popupWidth - 10;
      }

      if (popup.style.display === 'block' && popup.dataset.chunkId === String(citation.chunk_id)) {
          popup.style.display = 'none';
          return;
      }

      // CRITICAL FIX: Check if the element exists before trying to access its style
      const chatContainer = document.getElementById('chat-container');
      if (chatContainer) {
          popup.style.top = `${top}px`;
          popup.style.left = `${left}px`;
          popup.style.display = 'block';

          // Store the chunk_id on the popup element to track which citation it's for
          popup.dataset.chunkId = String(citation.chunk_id);
      } else {
          console.error("Chat container not found. Cannot position popup.");
      }

      const hidePopup = (e) => {
          if (!popup.contains(e.target) && e.target !== targetElement) {
              popup.style.display = 'none';
              document.removeEventListener('click', hidePopup);
          }
      };
      document.addEventListener('click', hidePopup);
  }

  /**
   * Fetches a single snippet for a given chunk ID.
   * Caches results to avoid duplicate requests.
   * @param {number} chunkId - The ID of the chunk to fetch.
   * @returns {Promise<Object>} A promise that resolves to the snippet object.
   */
  async function fetchSnippet(chunkId) {
      const loggedInUserToken = localStorage.getItem('ragnetic_user_token');
      if (!loggedInUserToken) {
          console.error("No API key found in local storage.");
          window.showToast("Authentication required to view citations. Please log in.", true);
          return null;
      }

      // Check cache first
      if (snippetCache[chunkId]) {
          console.log(`Cache hit for chunk ID: ${chunkId}`);
          return snippetCache[chunkId];
      }

      try {
          const url = new URL(CHAT_API_ENDPOINT);
          url.searchParams.append('chunk_id', chunkId);

          const response = await fetch(url.toString(), {
              headers: {
                  'X-API-Key': loggedInUserToken
              }
          });

          if (!response.ok) {
              console.error("API response not ok:", response);
              throw new Error('Failed to fetch citation snippet.');
          }

          const snippetData = await response.json();

          snippetCache[chunkId] = snippetData;
          return snippetData;

      } catch (error) {
          console.error(`Error fetching snippet for chunk ID ${chunkId}:`, error);
          return null;
      }
  }

  /**
   * Renders inline citation markers in a message element.
   * @param {HTMLElement} messageElement - The parent message element.
   * @param {Array} citations - An array of citation objects.
   */
  function renderCitations(messageElement, citations) {
    if (!citations?.length) return;
    const contentEl = messageElement.querySelector('.content');
    if (!contentEl) return;
    let html = contentEl.innerHTML;

    const citationMap = new Map();
    citations.forEach(c => citationMap.set(c.marker_text, c));

    html = html.replace(/\[(\d+)\]/g, (match) => {
        const citation = citationMap.get(match);
        if (citation) {
            const chunkId = citation.chunk_id;
            const isHallucinated = chunkId === -1 ? 'data-hallucinated="true"' : '';

            const dataAttrs = `data-chunk-id="${chunkId}" data-marker-text="${citation.marker_text}" data-doc-name="${citation.doc_name || ''}" data-page-number="${citation.page_number || ''}"`;

            return `<span class="${CITATION_HIGHLIGHT_CLASS}" ${isHallucinated} ${dataAttrs}>${match}</span>`;
        }
        return match;
    });

    contentEl.innerHTML = html;

    const citationSpans = messageElement.querySelectorAll(`.${CITATION_HIGHLIGHT_CLASS}`);
    citationSpans.forEach(span => {
      const citationData = {
          chunk_id: parseInt(span.dataset.chunkId),
          marker_text: span.dataset.markerText,
          doc_name: span.dataset.docName,
          page_number: span.dataset.pageNumber ? parseInt(span.dataset.pageNumber) : null
      };

      span.addEventListener('click', async (event) => {
          event.stopPropagation();
          if (citationData.chunk_id === -1) {
              showCitationPopup(citationData, null, event.target);
          } else {
              const snippetData = await fetchSnippet(citationData.chunk_id);
              const snippet = snippetData ? snippetData.snippet : null;
              showCitationPopup(citationData, snippet, event.target);
          }
      });
    });
  }

  return {
    renderCitations
  };
})();