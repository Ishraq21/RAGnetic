import React, { useEffect, useState } from 'react';
import api from '../api';
import { useAuth } from '../AuthContext';

const Settings = () => {
  const { user } = useAuth();
  const [profile, setProfile] = useState(null);
  const [form, setForm] = useState({ username: '', email: '' });

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await api.get('/security/me');
        setProfile(res.data);
        setForm({ username: res.data.username, email: res.data.email || '' });
      } catch (e) {
        console.error('Failed to load profile', e);
      }
    };
    fetchProfile();
  }, []);

  const save = async (e) => {
    e.preventDefault();
    try {
      await api.put(`/security/users/${user.id}`, form);
      const res = await api.get('/security/me');
      setProfile(res.data);
    } catch (e) {
      console.error('Failed to update profile', e);
    }
  };

  if (!profile) return <p>Loading settings...</p>;

  return (
    <div>
      <h1>Settings</h1>
      <form onSubmit={save} style={{ maxWidth: '400px' }}>
        <label>Username</label>
        <input
          value={form.username}
          onChange={(e) => setForm({ ...form, username: e.target.value })}
          required
          style={{ display: 'block', marginBottom: '0.5rem', padding: '0.5rem' }}
        />
        <label>Email</label>
        <input
          value={form.email}
          onChange={(e) => setForm({ ...form, email: e.target.value })}
          style={{ display: 'block', marginBottom: '0.5rem', padding: '0.5rem' }}
        />
        <button type="submit">Save</button>
      </form>
    </div>
  );
};

export default Settings;
