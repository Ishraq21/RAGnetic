import React, { createContext, useContext, useState } from 'react';
import api from './api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const token = localStorage.getItem('ragnetic_user_token');
    const id = localStorage.getItem('ragnetic_db_user_id');
    return token ? { token, id } : null;
  });

  const login = async (username, password) => {
    const res = await api.post('/security/login', { username, password });
    const token = res.data.access_token;
    localStorage.setItem('ragnetic_user_token', token);
    const me = await api.get('/security/me', { headers: { 'X-API-Key': token } });
    localStorage.setItem('ragnetic_db_user_id', me.data.id);
    setUser({ token, id: me.data.id, username: me.data.username });
  };

  const logout = () => {
    localStorage.removeItem('ragnetic_user_token');
    localStorage.removeItem('ragnetic_db_user_id');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
