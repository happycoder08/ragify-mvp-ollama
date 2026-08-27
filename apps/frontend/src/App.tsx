import { BrowserRouter, Routes, Route, Link, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { isLoggedIn, clearToken } from './utils/auth';
import Login from './pages/Login';
import Docs from './pages/Docs';
import Query from './pages/Query';
import Demo from './pages/Demo';
import './App.css';

function ClearToken() {
  useEffect(() => {
    clearToken();
    window.location.href = '/login';
  }, []);
  return null;
}

function Navigation() {
  const loggedIn = isLoggedIn();

  const handleLogout = () => {
    clearToken();
    window.location.href = '/login';
  };

  if (!loggedIn) return null;

  return (
    <nav className="app-nav">
      <div className="nav-content">
        <div className="nav-brand">
          <span className="logo">🔍</span>
          <span className="brand-name">RAGify</span>
        </div>
        <div className="nav-links">
          <Link to="/docs" className="nav-link">Documents</Link>
          <Link to="/query" className="nav-link">Chat</Link>
        </div>
        <button onClick={handleLogout} className="logout-button">
          Logout
        </button>
      </div>
    </nav>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const loggedIn = isLoggedIn();
  return loggedIn ? <>{children}</> : <Navigate to="/login" replace />;
}

function App() {
  return (
    <BrowserRouter>
      <Navigation />
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/clear-token" element={<ClearToken />} />
        <Route
          path="/query"
          element={
            <ProtectedRoute>
              <Query />
            </ProtectedRoute>
          }
        />
        <Route
          path="/docs"
          element={
            <ProtectedRoute>
              <Docs />
            </ProtectedRoute>
          }
        />
        <Route
          path="/demo"
          element={
            <ProtectedRoute>
              <Demo />
            </ProtectedRoute>
          }
        />
        <Route path="/" element={<Navigate to="/docs" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
