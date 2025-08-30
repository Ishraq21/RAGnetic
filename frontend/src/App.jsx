import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import Agents from './pages/Agents';
import Workflows from './pages/Workflows';
import Analytics from './pages/Analytics';
import FineTuning from './pages/FineTuning';
import Security from './pages/Security';
import Connectors from './pages/Connectors';
import Settings from './pages/Settings';
import Login from './pages/Login';
import { useAuth } from './AuthContext';

const RequireAuth = ({ children }) => {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
};

function App() {
  const { user } = useAuth();
  return (
    <div className="app">
      {user && <Sidebar />}
      <div className="content">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<RequireAuth><Dashboard /></RequireAuth>} />
          <Route path="/chat" element={<RequireAuth><Chat /></RequireAuth>} />
          <Route path="/agents" element={<RequireAuth><Agents /></RequireAuth>} />
          <Route path="/workflows" element={<RequireAuth><Workflows /></RequireAuth>} />
          <Route path="/analytics" element={<RequireAuth><Analytics /></RequireAuth>} />
          <Route path="/fine-tuning" element={<RequireAuth><FineTuning /></RequireAuth>} />
          <Route path="/security" element={<RequireAuth><Security /></RequireAuth>} />
          <Route path="/connectors" element={<RequireAuth><Connectors /></RequireAuth>} />
          <Route path="/settings" element={<RequireAuth><Settings /></RequireAuth>} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
