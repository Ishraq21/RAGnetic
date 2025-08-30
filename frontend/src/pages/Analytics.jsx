import React, { useEffect, useState } from 'react';
import api from '../api';

const Analytics = () => {
  const [usage, setUsage] = useState([]);

  useEffect(() => {
    const loadUsage = async () => {
      try {
        const res = await api.get('/analytics/usage-summary');
        setUsage(res.data);
      } catch (e) {
        console.error('Failed to load usage summary', e);
      }
    };
    loadUsage();
  }, []);

  return (
    <div>
      <h1>Analytics</h1>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>Agent</th>
            <th>Total Tokens</th>
            <th>Estimated Cost (USD)</th>
          </tr>
        </thead>
        <tbody>
          {usage.map((u, idx) => (
            <tr key={idx} style={{ textAlign: 'left', borderBottom: `1px solid var(--border-color)` }}>
              <td>{u.agent_name || 'All'}</td>
              <td>{u.total_tokens}</td>
              <td>{u.total_estimated_cost_usd}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Analytics;
