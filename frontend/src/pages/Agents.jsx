import React, { useEffect, useState } from 'react';
import api from '../api';

const Agents = () => {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newAgent, setNewAgent] = useState({ name: '', llm_model: '' });

  const fetchAgents = async () => {
    try {
      const res = await api.get('/agents');
      const withStatus = await Promise.all(
        res.data.map(async (agent) => {
          try {
            const insp = await api.get(`/agents/${agent.name}/inspection`);
            return { ...agent, is_deployed: insp.data.is_deployed };
          } catch (e) {
            console.error(e);
            return { ...agent, is_deployed: false };
          }
        })
      );
      setAgents(withStatus);
    } catch (e) {
      console.error('Failed to load agents', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAgents();
  }, []);

  const toggleDeploy = async (agent) => {
    try {
      if (agent.is_deployed) {
        await api.delete(`/agents/${agent.name}/deploy`);
      } else {
        await api.post(`/agents/${agent.name}/deploy`);
      }
      const insp = await api.get(`/agents/${agent.name}/inspection`);
      setAgents((prev) =>
        prev.map((a) =>
          a.name === agent.name ? { ...a, is_deployed: insp.data.is_deployed } : a
        )
      );
    } catch (e) {
      console.error('Failed to toggle deploy', e);
    }
  };

  const deleteAgent = async (name) => {
    try {
      await api.delete(`/agents/${name}`);
      setAgents((prev) => prev.filter((a) => a.name !== name));
    } catch (e) {
      console.error('Failed to delete agent', e);
    }
  };

  const createAgent = async (e) => {
    e.preventDefault();
    try {
      await api.post('/agents', newAgent);
      setNewAgent({ name: '', llm_model: '' });
      fetchAgents();
    } catch (e) {
      console.error('Failed to create agent', e);
    }
  };

  if (loading) return <p>Loading agents...</p>;

  return (
    <div>
      <h1>Agents</h1>
      <form onSubmit={createAgent} style={{ marginBottom: '1rem' }}>
        <input
          placeholder="Name"
          value={newAgent.name}
          onChange={(e) => setNewAgent({ ...newAgent, name: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          placeholder="LLM Model"
          value={newAgent.llm_model}
          onChange={(e) => setNewAgent({ ...newAgent, llm_model: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <button type="submit">Create</button>
      </form>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>Name</th>
            <th>Model</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {agents.map((agent) => (
            <tr key={agent.name} style={{ textAlign: 'left', borderBottom: `1px solid var(--border-color)` }}>
              <td>{agent.name}</td>
              <td>{agent.llm_model}</td>
              <td>{agent.is_deployed ? 'Deployed' : 'Undeployed'}</td>
              <td>
                <button onClick={() => toggleDeploy(agent)}>
                  {agent.is_deployed ? 'Undeploy' : 'Deploy'}
                </button>
                <button style={{ marginLeft: '0.5rem' }} onClick={() => deleteAgent(agent.name)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Agents;
