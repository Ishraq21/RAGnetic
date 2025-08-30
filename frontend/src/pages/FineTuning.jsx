import React, { useEffect, useState } from 'react';
import api from '../api';

const FineTuning = () => {
  const [models, setModels] = useState([]);
  const [form, setForm] = useState({ job_name: '', base_model_name: '', dataset_path: '' });
  const [loading, setLoading] = useState(false);

  const fetchModels = async () => {
    try {
      const res = await api.get('/training/models');
      setModels(res.data);
    } catch (e) {
      console.error('Failed to load fine-tuned models', e);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const submitJob = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await api.post('/training/apply', form);
      setForm({ job_name: '', base_model_name: '', dataset_path: '' });
      fetchModels();
    } catch (e) {
      console.error('Failed to submit job', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Fine-Tuning</h1>
      <form onSubmit={submitJob} style={{ marginBottom: '1rem' }}>
        <input
          placeholder="Job Name"
          value={form.job_name}
          onChange={(e) => setForm({ ...form, job_name: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          placeholder="Base Model"
          value={form.base_model_name}
          onChange={(e) => setForm({ ...form, base_model_name: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          placeholder="Dataset Path"
          value={form.dataset_path}
          onChange={(e) => setForm({ ...form, dataset_path: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <button type="submit" disabled={loading}>Submit</button>
      </form>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>Job Name</th>
            <th>Base Model</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.adapter_id} style={{ textAlign: 'left', borderBottom: `1px solid var(--border-color)` }}>
              <td>{m.job_name}</td>
              <td>{m.base_model_name}</td>
              <td>{m.training_status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default FineTuning;
