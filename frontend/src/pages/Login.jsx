import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      await login(username, password);
      navigate('/');
    } catch (err) {
      const detail = err.response?.data?.detail || 'Login failed';
      setError(detail);
    }
  };

  return (
    <div style={{ maxWidth: '400px', margin: '4rem auto' }}>
      <h1>Login</h1>
      {error && <p style={{ color: 'var(--error-color)' }}>{error}</p>}
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          style={{ width: '100%', padding: '0.5rem', marginBottom: '0.5rem' }}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={{ width: '100%', padding: '0.5rem', marginBottom: '0.5rem' }}
        />
        <button type="submit" style={{ width: '100%' }}>
          Login
        </button>
      </form>
    </div>
  );
};

export default Login;
