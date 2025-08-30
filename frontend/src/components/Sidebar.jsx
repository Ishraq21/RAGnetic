import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../AuthContext';

const Sidebar = () => {
  const { logout } = useAuth();
  return (
    <div className="sidebar">
      <h2>RAGnetic</h2>
      <NavLink to="/">Dashboard</NavLink>
      <NavLink to="/chat">Chat</NavLink>
      <NavLink to="/agents">Agents</NavLink>
      <NavLink to="/workflows">Workflows</NavLink>
      <NavLink to="/analytics">Analytics</NavLink>
      <NavLink to="/fine-tuning">Fine-Tuning</NavLink>
      <NavLink to="/connectors">Connectors</NavLink>
      <NavLink to="/security">Security</NavLink>
      <NavLink to="/settings">Settings</NavLink>
      <button onClick={logout} style={{ marginTop: 'auto' }}>Logout</button>
    </div>
  );
};

export default Sidebar;
