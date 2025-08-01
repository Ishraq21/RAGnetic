window.citationRenderer = (function() {
  const CITATION_HIGHLIGHT_CLASS = 'citation-marker';
  // Note: Don't get DOM elements here. They might not exist yet.
  /**
   * Renders inline citation markers in a message element.
   * Finds plain text markers like `[1]` and wraps them in a span.
   *
   * @param {HTMLElement} messageElement - The parent message element containing the content.
   * @param {Array} citations - An array of citation objects.
   */
  function renderCitations(messageElement, citations) {
    if (!citations?.length) return;
    const contentEl = messageElement.querySelector('.content');
    if (!contentEl) return;
    let html = contentEl.innerHTML;
    html = html.replace(/\[(\d+)\]/g, (match, p1) => {
      const citationIndex = parseInt(p1) - 1;
      const citation = citations[citationIndex];
      if (citation) {
        return `<span class="${CITATION_HIGHLIGHT_CLASS}" data-chunk-id="${citation.chunk_id}">${match}</span>`;
      }
      return match;
    });
    contentEl.innerHTML = html;
  }
  /**
   * Renders a "Sources" button and populates the sources panel when clicked.
   *
   * @param {HTMLElement} messageElement - The parent message element.
   * @param {Array} citations - An array of citation objects.
   */
function renderSourcesButton(messageElement, citations) {
  const btn = messageElement.querySelector('.sources-button');
  if (!btn) return;

  if (!citations || citations.length === 0) {
    // If no citations, explicitly hide the button by adding the 'hidden' class
    btn.classList.add('hidden');
    // Ensure 'visible' is not present
    btn.classList.remove('visible');
    return;
  }

  // If citations exist, explicitly show the button by removing the 'hidden' class
  btn.classList.remove('hidden');
  // Add 'visible' class if it exists and is used in your CSS,
  // otherwise this line is not strictly needed but good practice.
  btn.classList.add('visible');

  // Get the sources panel and list at the time of the click handler being set.
  const sourcesPanel = document.getElementById('sources-panel');
  const sourcesList  = document.getElementById('sources-list');
  if (!sourcesPanel || !sourcesList) {
      console.warn("Sources panel not found. Cannot render sources.");
      return;
  }

  btn.onclick = () => {
    sourcesList.innerHTML = '';
    citations.forEach((c, i) => {
      const li = document.createElement('li');
      li.innerHTML = `
        <div class="source-title">[${i + 1}] ${c.doc_name || 'Document ' + c.doc_id}</div>
        <div class="source-snippet">${c.snippet || 'Snippet not available.'}</div>
        <div class="source-meta">
          Chunk ${c.chunk_id}
          ${c.sentences_range ? `, sentences ${c.sentences_range}` : ''}
        </div>
      `;
      sourcesList.appendChild(li);
    });
    sourcesPanel.classList.add('open');
  };
}
  // No init function needed if elements are grabbed inside the function
  return {
    renderCitations,
    renderSourcesButton
  };
})();