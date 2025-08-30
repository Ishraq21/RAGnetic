import React, { useEffect, useState } from 'react';
import api from '../api';

const Connectors = () => {
  const [connectors, setConnectors] = useState([]);
  const [form, setForm] = useState({ name: '', type: '' });
  const [error, setError] = useState(null);

  const fetchConnectors = async () => {
    try {
      const res = await api.get('/connectors');
      setConnectors(res.data);
      setError(null);
    } catch (e) {
      console.error('Failed to load connectors', e);
      setError('Connectors API unavailable');
    }
  };

  useEffect(() => {
    fetchConnectors();
  }, []);

  const createConnector = async (e) => {
    e.preventDefault();
    try {
      await api.post('/connectors', form);
      setForm({ name: '', type: '' });
      fetchConnectors();
    } catch (e) {
      console.error('Failed to create connector', e);
    }
  };

  return (
    <div>
      <h1>Connectors</h1>
      {error && <p style={{ color: 'var(--error-color)' }}>{error}</p>}
      <form onSubmit={createConnector} style={{ marginBottom: '1rem' }}>
        <input
          placeholder="Name"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          placeholder="Type"
          value={form.type}
          onChange={(e) => setForm({ ...form, type: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <button type="submit">Add</button>
      </form>
      <ul>
        {connectors.map((c) => (
          <li key={c.id || c.name}>{c.name} - {c.type}</li>
        ))}
      </ul>
    </div>
  );
};

export default Connectors;
