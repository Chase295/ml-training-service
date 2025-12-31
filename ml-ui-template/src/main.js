// Einfache React App Loader
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './style.css';

const app = document.getElementById('app');
if (app) {
  const root = ReactDOM.createRoot(app);
  root.render(
    React.createElement(React.StrictMode, null,
      React.createElement(App)
    )
  );
  console.log('🚀 Pump Service UI: React erfolgreich geladen!');
} else {
  console.error('App element not found!');
}
