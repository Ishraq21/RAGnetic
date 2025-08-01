window.citationRenderer = (function() {
  const CITATION_HIGHLIGHT_CLASS = 'citation-marker';
  const API_BASE_URL            = window.location.origin + '/api/v1';
  const CHAT_API_ENDPOINT       = `${API_BASE_URL}/chat/citation-snippet`;
  const POPUP_MARGIN            = 8;    // px

  let snippetCache = {};
  let currentHideCallback = null;

  async function showCitationPopup(citation, snippet, target) {
    // create or reuse popup container
    let popup = document.getElementById('citation-popup');
    if (!popup) {
      popup = document.createElement('div');
      popup.id        = 'citation-popup';
      popup.className = 'citation-popup';
      document.body.appendChild(popup);
      // prevent clicks inside popup from bubbling out
      popup.addEventListener('click', e => e.stopPropagation());
    }

    // fill content
    const title = citation.doc_name || 'Document';
    const page  = citation.page_number ? ` (Page ${citation.page_number})` : '';
    popup.innerHTML = `
      <div class="popup-header">
        <span class="popup-title">${title}${page}</span>
        <button class="popup-close-btn">&times;</button>
      </div>
      <div class="popup-content">
        ${ snippet
            ? `<div class="popup-snippet">${snippet}</div>`
            : `<div class="popup-snippet-error">Source unavailable. This may be hallucinatory.</div>`
        }
      </div>
    `;

    // wire up close button
    popup.querySelector('.popup-close-btn')
         .onclick = () => popup.classList.remove('visible');

    // if clicking the same marker again, just hide
    if (popup.dataset.chunkId === String(citation.chunk_id) && popup.classList.contains('visible')) {
      return popup.classList.remove('visible');
    }
    popup.dataset.chunkId = citation.chunk_id;

    // Make it visible (but invisible to user) so we can measure
    popup.style.position   = 'fixed';
    popup.style.visibility = 'hidden';
    popup.classList.add('visible');

    // measure
    const popupRect   = popup.getBoundingClientRect();
    const popupH      = popupRect.height;
    const popupW      = Math.min(popupRect.width, 400);
    const markerRect  = target.getBoundingClientRect();

    // horizontal: clamp within viewport
    let left = markerRect.left;
    if (left + popupW + POPUP_MARGIN > window.innerWidth) {
      left = window.innerWidth - popupW - POPUP_MARGIN;
    } else if (left < POPUP_MARGIN) {
      left = POPUP_MARGIN;
    }

    // vertical: flip if needed
    let top;
    if (markerRect.bottom + popupH + POPUP_MARGIN <= window.innerHeight) {
      // enough space below
      top = markerRect.bottom + POPUP_MARGIN;
    } else {
      // render above
      top = markerRect.top - popupH - POPUP_MARGIN;
    }

    // apply final positioning and show
    popup.style.top        = `${top}px`;
    popup.style.left       = `${left}px`;
    popup.style.visibility = '';
    popup.classList.add('visible');

    // cleanup old outside-click handler
    if (currentHideCallback) {
      document.removeEventListener('click', currentHideCallback);
    }
    // hide when clicking anywhere else
    currentHideCallback = e => {
      if (!popup.contains(e.target) && e.target !== target) {
        popup.classList.remove('visible');
        document.removeEventListener('click', currentHideCallback);
        currentHideCallback = null;
      }
    };
    document.addEventListener('click', currentHideCallback);
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
      const url = new URL(CHAT_API_ENDPOINT);
      url.searchParams.append('chunk_id', chunkId);
      const res = await fetch(url, { headers: { 'X-API-Key': token } });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      snippetCache[chunkId] = data;
      return data;
    } catch (err) {
      console.error("Citation snippet error:", err);
      return null;
    }
  }

  function renderCitations(messageEl, citations) {
    if (!citations?.length) return;
    const content = messageEl.querySelector('.content');
    if (!content) return;

    // build map for quick lookup by "[n]"
    const map = new Map(citations.map(c => [c.marker_text, c]));

    // wrap markers in <span>
    content.innerHTML = content.innerHTML.replace(/\[(\d+)\]/g, m => {
      const c = map.get(m);
      return c
        ? `<span class="${CITATION_HIGHLIGHT_CLASS}"
                  data-chunk-id="${c.chunk_id}"
                  data-marker-text="${c.marker_text}"
                  data-doc-name="${c.doc_name||''}"
                  data-page-number="${c.page_number||''}">${m}</span>`
        : m;
    });

    // attach click handlers
    content.querySelectorAll(`.${CITATION_HIGHLIGHT_CLASS}`)
      .forEach(span => {
        const cd = {
          chunk_id:    parseInt(span.dataset.chunkId),
          marker_text: span.dataset.markerText,
          doc_name:    span.dataset.docName,
          page_number: span.dataset.pageNumber ? parseInt(span.dataset.pageNumber) : null
        };
        span.onclick = async e => {
          e.stopPropagation();
          if (cd.chunk_id === -1) {
            showCitationPopup(cd, null, span);
          } else {
            const sn = await fetchSnippet(cd.chunk_id);
            showCitationPopup(cd, sn?.snippet, span);
          }
        };
      });
  }

  return { renderCitations };
})();
