import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
});

// Attach API key from localStorage on each request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('ragnetic_user_token');
  if (token) {
    config.headers['X-API-Key'] = token;
  }
  return config;
});

export default api;
