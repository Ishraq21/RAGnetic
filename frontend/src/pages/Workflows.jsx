import React, { useEffect, useState } from 'react';
import api from '../api';

const Workflows = () => {
  const [workflows, setWorkflows] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchWorkflows = async () => {
    try {
      const res = await api.get('/workflows');
      setWorkflows(res.data);
    } catch (e) {
      console.error('Failed to load workflows', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const addWorkflow = async () => {
    const name = prompt('Workflow name');
    const agent = prompt('Agent name');
    if (!name || !agent) return;
    try {
      await api.post('/workflows', { name, agent_name: agent, steps: [] });
      fetchWorkflows();
    } catch (e) {
      console.error('Failed to create workflow', e);
    }
  };

  if (loading) return <p>Loading workflows...</p>;

  return (
    <div>
      <h1>Workflows</h1>
      <button onClick={addWorkflow}>New Workflow</button>
      <ul>
        {workflows.map((wf) => (
          <li key={wf.name}>{wf.name} - {wf.agent_name}</li>
        ))}
      </ul>
    </div>
  );
};

export default Workflows;
