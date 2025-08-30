import React, { useEffect, useState } from 'react';
import api from '../api';

const Security = () => {
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({ username: '', email: '', password: '' });

  const fetchUsers = async () => {
    try {
      const res = await api.get('/security/users');
      setUsers(res.data);
    } catch (e) {
      console.error('Failed to load users', e);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const createUser = async (e) => {
    e.preventDefault();
    try {
      await api.post('/security/users', form);
      setForm({ username: '', email: '', password: '' });
      fetchUsers();
    } catch (e) {
      console.error('Failed to create user', e);
    }
  };

  return (
    <div>
      <h1>Security</h1>
      <form onSubmit={createUser} style={{ marginBottom: '1rem' }}>
        <input
          placeholder="Username"
          value={form.username}
          onChange={(e) => setForm({ ...form, username: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          placeholder="Email"
          value={form.email}
          onChange={(e) => setForm({ ...form, email: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          placeholder="Password"
          type="password"
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <button type="submit">Create</button>
      </form>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>Username</th>
            <th>Email</th>
          </tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.id} style={{ textAlign: 'left', borderBottom: `1px solid var(--border-color)` }}>
              <td>{u.username}</td>
              <td>{u.email}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Security;
