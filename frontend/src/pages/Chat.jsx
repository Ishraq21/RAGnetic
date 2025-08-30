import React, { useEffect, useState } from 'react';
import api from '../api';

const newThreadId = () => {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return Math.random().toString(36).substring(2);
};

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState('');
  const [threadId, setThreadId] = useState(() => {
    return localStorage.getItem('ragnetic_thread_id') || newThreadId();
  });

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
      const aiMsg = { role: 'assistant', content: res.data.response };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e) {
      console.error('Query failed', e);
    }
  };

  return (
    <div>
      <h1>Chat</h1>
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
      <div style={{ border: `1px solid var(--border-color)`, padding: '1rem', marginBottom: '1rem', height: '300px', overflowY: 'auto', background: 'var(--card-background)' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: '0.5rem' }}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
      </div>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        style={{ width: '80%', padding: '0.5rem' }}
      />
      <button onClick={sendMessage} style={{ marginLeft: '0.5rem' }}>
        Send
      </button>
    </div>
  );
};

export default Chat;
