import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AnimatePresence } from 'framer-motion';
import { GameStateProvider } from './store/gameStore';
import Navigation from './components/common/Navigation';
import LoadingScreen from './components/common/LoadingScreen';
import ErrorBoundary from './components/common/ErrorBoundary';
import './styles/globals.css';

// Lazy load pages
const Home = lazy(() => import('./pages/Home'));
const Games = lazy(() => import('./pages/Games'));
const Sandbox = lazy(() => import('./pages/Sandbox'));
const Learn = lazy(() => import('./pages/Learn'));
const Visualizations = lazy(() => import('./pages/Visualizations'));

// Dark theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4169E1',
    },
    secondary: {
      main: '#FDB813',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '3rem',
      fontWeight: 700,
    },
  },
});

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <GameStateProvider>
          <Router>
            <div className="app">
              <Navigation />
              <AnimatePresence mode="wait">
                <Suspense fallback={<LoadingScreen />}>
                  <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/games/*" element={<Games />} />
                    <Route path="/sandbox" element={<Sandbox />} />
                    <Route path="/learn/*" element={<Learn />} />
                    <Route path="/visualizations" element={<Visualizations />} />
                  </Routes>
                </Suspense>
              </AnimatePresence>
            </div>
          </Router>
        </GameStateProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;