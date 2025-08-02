window.citationRenderer = (function() {
  const CITATION_CLASS   = 'citation-marker';
  const POPUP_ID         = 'citation-popup';
  const API_BASE         = window.location.origin + '/api/v1';
  const SNIPPET_ENDPOINT = `${API_BASE}/chat/citation-snippet`;
  const MARGIN           = 8;      // px
  const MAX_SNIPPET_LEN  = 300;    // characters before truncation

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
    const hasPage = citation.page_number != null;
    const page    = hasPage ? ` • Page ${citation.page_number}` : '';
    const chunk   = ` • Chunk ${citation.chunk_id}`;

    // Determine truncated vs full text
    const fullText = snippet || '';
    const isLong   = fullText.length > MAX_SNIPPET_LEN;
    const teaser   = isLong
      ? fullText.slice(0, MAX_SNIPPET_LEN) + '…'
      : fullText;

    popup.innerHTML = `
      <div class="popup-header">
        <span class="popup-title">${title}${page}${chunk}</span>
        <button class="popup-close-btn">&times;</button>
      </div>
      <div class="popup-content">
        <div class="popup-snippet">${teaser}</div>
        ${isLong ? `<a href="#" class="toggle-snippet">Show more</a>` : ''}
      </div>
    `;
    // close
    popup.querySelector('.popup-close-btn')
         .onclick = () => popup.classList.remove('visible');

    // toggle logic
    if (isLong) {
      const toggle = popup.querySelector('.toggle-snippet');
      const snippetDiv = popup.querySelector('.popup-snippet');
      toggle.onclick = e => {
        e.preventDefault();
        const expanded = toggle.dataset.expanded === 'true';
        if (expanded) {
          snippetDiv.textContent = teaser;
          toggle.textContent = 'Show more';
          toggle.dataset.expanded = 'false';
        } else {
          snippetDiv.textContent = fullText;
          toggle.textContent = 'Show less';
          toggle.dataset.expanded = 'true';
        }
      };
    }

    // if same citation clicked twice, hide
    if (popup.dataset.chunkId === String(citation.chunk_id)
        && popup.classList.contains('visible')) {
      return popup.classList.remove('visible');
    }
    popup.dataset.chunkId = citation.chunk_id;

    // position
    popup.style.position   = 'fixed';
    popup.style.visibility = 'hidden';
    popup.classList.add('visible');
    const { width: W, height: H } = popup.getBoundingClientRect();
    const popupW = Math.min(W, 400);
    const markerRect = anchor.getBoundingClientRect();

    let left = Math.max(MARGIN,
      Math.min(markerRect.left, window.innerWidth - popupW - MARGIN)
    );
    let top  = markerRect.bottom + H + MARGIN <= window.innerHeight
      ? markerRect.bottom + MARGIN
      : markerRect.top - H - MARGIN;

    Object.assign(popup.style, {
      top:       `${top}px`,
      left:      `${left}px`,
      visibility: ''
    });
    popup.classList.add('visible');

    // click outside to hide
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
    if (snippetCache[chunkId]) return snippetCache[chunkId];
    try {
      const url = new URL(SNIPPET_ENDPOINT);
      url.searchParams.set('chunk_id', chunkId);
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

  function renderCitations(msgEl, citations) {
    if (!citations?.length) return;
    const content = msgEl.querySelector('.content');
    if (!content) return;

    const lookup = new Map(citations.map(c => [c.marker_text, c]));
    content.innerHTML = content.innerHTML.replace(/\[(\d+)\]/g, match => {
      const c = lookup.get(match);
      if (!c) return match;
      return `<span class="${CITATION_CLASS}"
                    data-chunk-id="${c.chunk_id}"
                    data-marker-text="${c.marker_text}"
                    data-doc-name="${c.doc_name||''}"
                    data-page-number="${c.page_number||''}"
              >${match}</span>`;
    });

    content.querySelectorAll(`.${CITATION_CLASS}`).forEach(span => {
      const cd = {
        chunk_id:    +span.dataset.chunkId,
        marker_text: span.dataset.markerText,
        doc_name:    span.dataset.docName,
        page_number: span.dataset.pageNumber ? +span.dataset.pageNumber : null
      };
      span.onclick = async e => {
        e.stopPropagation();
        const data = cd.chunk_id === -1
          ? { snippet: null, page_number: null }
          : await fetchSnippet(cd.chunk_id);
        const snippet = data?.snippet ?? null;
        showCitationPopup(
          { ...cd, page_number: data?.page_number ?? cd.page_number },
          snippet,
          span
        );
      };
    });
  }

  return { renderCitations };
})();
