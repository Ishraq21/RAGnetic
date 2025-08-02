window.citationRenderer = (function() {
  const CITATION_CLASS   = 'citation-marker';
  const POPUP_ID         = 'citation-popup';
  const API_BASE         = window.location.origin + '/api/v1';
  const SNIPPET_ENDPOINT = `${API_BASE}/chat/citation-snippet`;
  const MARGIN           = 8;  // px

  let snippetCache       = {};
  let hideOnClickOutside = null;

  async function showCitationPopup(citation, snippet, anchor) {
    let popup = document.getElementById(POPUP_ID);
    if (!popup) {
      popup = document.createElement('div');
      popup.id = POPUP_ID;
      popup.className = 'citation-popup';
      document.body.appendChild(popup);
      popup.addEventListener('click', e => e.stopPropagation());
    }

    const title   = citation.doc_name     || 'Document';
    const hasPage = citation.page_number !== undefined && citation.page_number !== null;
    const page    = hasPage ? ` • Page ${citation.page_number}` : '';
    const chunk = ` • Chunk ${citation.chunk_id}`;
    popup.innerHTML = `
      <div class="popup-header">
        <span class="popup-title">${title}${page}${chunk}</span>
        <button class="popup-close-btn">&times;</button>
      </div>
      <div class="popup-content">
        ${ snippet
            ? `<div class="popup-snippet">${snippet}</div>`
            : `<div class="popup-snippet-error">Source unavailable. This may be hallucinatory.</div>`
        }
      </div>
    `;
    popup.querySelector('.popup-close-btn')
         .onclick = () => popup.classList.remove('visible');

    if (popup.dataset.chunkId === String(citation.chunk_id) &&
        popup.classList.contains('visible')) {
      return popup.classList.remove('visible');
    }
    popup.dataset.chunkId = citation.chunk_id;

    popup.style.position   = 'fixed';
    popup.style.visibility = 'hidden';
    popup.classList.add('visible');
    const { width: W, height: H } = popup.getBoundingClientRect();
    const popupW = Math.min(W, 400);
    const markerRect = anchor.getBoundingClientRect();

    let left = markerRect.left;
    if (left + popupW + MARGIN > window.innerWidth) {
      left = window.innerWidth - popupW - MARGIN;
    } else if (left < MARGIN) {
      left = MARGIN;
    }

    let top;
    if (markerRect.bottom + H + MARGIN <= window.innerHeight) {
      top = markerRect.bottom + MARGIN;
    } else {
      top = markerRect.top - H - MARGIN;
    }

    Object.assign(popup.style, {
      top:       `${top}px`,
      left:      `${left}px`,
      visibility: ''
    });
    popup.classList.add('visible');

    if (hideOnClickOutside) {
      document.removeEventListener('click', hideOnClickOutside);
    }
    hideOnClickOutside = e => {
      if (!popup.contains(e.target) && e.target !== anchor) {
        popup.classList.remove('visible');
        document.removeEventListener('click', hideOnClickOutside);
        hideOnClickOutside = null;
      }
    };
    document.addEventListener('click', hideOnClickOutside);
  }

  async function fetchSnippet(chunkId) {
    const token = localStorage.getItem('ragnetic_user_token');
    if (!token) {
      window.showToast("Please log in to view citations.", true);
      return null;
    }
    if (snippetCache[chunkId]) {
      return snippetCache[chunkId];
    }
    try {
      const url = new URL(SNIPPET_ENDPOINT);
      url.searchParams.set('chunk_id', chunkId);
      const res = await fetch(url, { headers: { 'X-API-Key': token } });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      // Expecting { snippet: "...", page_number: 5 } from your API
      snippetCache[chunkId] = data;
      return data;
    } catch (err) {
      console.error("Citation snippet error:", err);
      return null;
    }
  }

  function renderCitations(msgEl, citations) {
    if (!citations?.length) return;
    const content = msgEl.querySelector('.content');
    if (!content) return;

    // Build lookup: "[n]" → citation
    const lookup = new Map(citations.map(c => [c.marker_text, c]));

    // Wrap each marker in a span
    content.innerHTML = content.innerHTML.replace(/\[(\d+)\]/g, match => {
      const c = lookup.get(match);
      if (!c) return match;
      return `
        <span class="${CITATION_CLASS}"
              data-chunk-id="${c.chunk_id}"
              data-marker-text="${c.marker_text}"
              data-doc-name="${c.doc_name||''}"
              data-page-number="${c.page_number||''}">
          ${match}
        </span>`;
    });

    // Attach click handlers
    content.querySelectorAll(`.${CITATION_CLASS}`).forEach(span => {
      const cd = {
        chunk_id:    +span.dataset.chunkId,
        marker_text: span.dataset.markerText,
        doc_name:    span.dataset.docName,
        page_number: span.dataset.pageNumber ? +span.dataset.pageNumber : null
      };
      span.onclick = async e => {
        e.stopPropagation();
        if (cd.chunk_id === -1) {
          return showCitationPopup(cd, null, span);
        }
        const data = await fetchSnippet(cd.chunk_id);
        const finalPage = data?.page_number ?? cd.page_number;
        showCitationPopup(
          { ...cd, page_number: finalPage },
          data?.snippet,
          span
        );
      };
    });
  }

  return { renderCitations };
})();
