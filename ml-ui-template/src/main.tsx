// Einfacher Test ohne React
import './style.css'

const app = document.getElementById('app')
if (app) {
  app.innerHTML = `
    <div style="padding: 20px; text-align: center; font-family: Arial, sans-serif;">
      <h1 style="color: #1976d2;">🚀 Pump Service UI</h1>
      <p>JavaScript läuft! React wird geladen...</p>
      <div style="margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 4px;">
        <p>✅ Vite Dev Server: Port 5173</p>
        <p>✅ JavaScript: Aktiv</p>
        <p>🔄 React: Wird geladen...</p>
      </div>
    </div>
  `
}

// Nach 1 Sekunde React laden
setTimeout(() => {
  import('react').then((React) => {
    import('react-dom/client').then((ReactDOM) => {
      import('./App.tsx').then((AppModule) => {
        const root = ReactDOM.createRoot(document.getElementById('app')!)
        root.render(React.createElement(React.StrictMode, null, React.createElement(AppModule.default)))
      })
    })
  }).catch((error) => {
    console.error('React loading failed:', error)
    if (app) {
      app.innerHTML += '<p style="color: red;">❌ React-Fehler: ' + error.message + '</p>'
    }
  })
}, 1000)
