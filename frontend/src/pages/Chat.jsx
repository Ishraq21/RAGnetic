import React, { useEffect, useState } from 'react';
import api from '../api';

const newThreadId = () => {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return Math.random().toString(36).substring(2);
};

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [isSourcesOpen, setIsSourcesOpen] = useState(false);
  const [input, setInput] = useState('');
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [threadId, setThreadId] = useState(() => {
    return localStorage.getItem('ragnetic_thread_id') || newThreadId();
  });

  const renderMessageContent = (content, citations = []) => {
    const citationMap = {};
    citations.forEach((c) => {
      citationMap[c.marker_text] = c;
    });

    const parts = [];
    let lastIndex = 0;
    const regex = /\[(\d+)\]/g;
    let match;

    while ((match = regex.exec(content)) !== null) {
      const marker = match[0];
      const citation = citationMap[marker];
      parts.push(content.slice(lastIndex, match.index));
      if (citation) {
        parts.push(
          <span
            key={`${match.index}-${marker}`}
            className="citation-marker"
            onClick={() => openSources([citation])}
          >
            {marker}
          </span>
        );
      } else {
        parts.push(marker);
      }
      lastIndex = match.index + marker.length;
    }

    parts.push(content.slice(lastIndex));

    return parts.map((p, i) => (
      <React.Fragment key={i}>{p}</React.Fragment>
    ));
  };

  useEffect(() => {
    const loadAgents = async () => {
      try {
        const res = await api.get('/agents');
        setAgents(res.data);
        if (res.data.length) setSelectedAgent(res.data[0].name);
      } catch (e) {
        console.error('Failed to load agents', e);
      }
    };
    loadAgents();
  }, []);

  useEffect(() => {
    localStorage.setItem('ragnetic_thread_id', threadId);
  }, [threadId]);

  useEffect(() => {
    // reset thread when agent changes
    if (selectedAgent) {
      setThreadId(newThreadId());
      setMessages([]);
    }
  }, [selectedAgent]);

  const sendMessage = async () => {
    if (!input || !selectedAgent) return;
    const userMsg = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    try {
        const res = await api.post(`/agents/${selectedAgent}/query`, { query: userMsg.content, thread_id: threadId });
        const aiMsg = { role: 'assistant', content: res.data.response, citations: res.data.citations || [] };
        setMessages((prev) => [...prev, aiMsg]);
      } catch (e) {
        console.error('Query failed', e);
      }
  };

  const openSources = async (citations) => {
    try {
      const chunkIds = citations.map(c => c.chunk_id).join(',');
      const res = await api.get(`/chat/citation-snippets`, { params: { chunk_ids: chunkIds } });
      setSources(res.data);
      setIsSourcesOpen(true);
    } catch (e) {
      console.error('Failed to load citation snippets', e);
    }
  };

  const closeSources = () => {
    setIsSourcesOpen(false);
    setSources([]);
  };

    return (
      <div id="chat-container">
        <div style={{ marginBottom: '0.5rem' }}>
          <label>Agent:&nbsp;</label>
          <select value={selectedAgent} onChange={(e) => setSelectedAgent(e.target.value)}>
            {agents.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))}
          </select>
        </div>
        <div id="messages">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role === 'assistant' ? 'agent' : 'user'}`}>
              <div className="content">
                {m.role === 'assistant'
                  ? renderMessageContent(m.content, m.citations)
                  : m.content}
              </div>
              {m.role === 'assistant' && m.citations && m.citations.length > 0 && (
                <button
                  className="sources-button visible"
                  onClick={() => openSources(m.citations)}
                >
                  Sources
                </button>
              )}
            </div>
          ))}
        </div>
        <div id="input-form-container">
          <div id="input-form">
            <textarea
              id="query-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={1}
            />
            <button id="send-stop-btn" onClick={sendMessage}>Send</button>
          </div>
        </div>
        <div id="sources-panel" className={isSourcesOpen ? 'open' : ''}>
          <div className="sources-panel-header">
            <span>Sources</span>
            <button id="sources-panel-close" onClick={closeSources}>×</button>
          </div>
          <ul id="sources-list">
            {sources.map((s) => (
              <li key={s.id}>
                <div className="source-title">{s.document_name}{s.page_number ? ` (p.${s.page_number})` : ''}</div>
                <div className="source-snippet">{s.snippet}</div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

export default Chat;
