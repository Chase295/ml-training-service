import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  CssBaseline,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useMediaQuery,
  IconButton,
  Container,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Settings as SettingsIcon,
  Article as LogsIcon,
  Analytics as MetricsIcon,
  Info as InfoIcon,
  ViewList as ModelsIcon,
  Add as TrainingIcon,
  Science as TestingIcon,
  Assessment as TestResultsIcon,
  Compare as CompareIcon,
  TableChart as ComparisonsIcon,
  Work as JobsIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';

// Pages
import Dashboard from './pages/Dashboard';
import Jobs from './pages/Jobs';
import Training from './pages/Training';
import Compare from './pages/Compare';
import Details from './pages/Details';
import Config from './pages/Config';
import Metrics from './pages/Metrics';
import Info from './pages/Info';
import Models from './pages/Models';

// Theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00d4ff',
    },
    secondary: {
      main: '#ff4081',
    },
    background: {
      default: '#0f0f23',
      paper: 'rgba(255, 255, 255, 0.05)',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backdropFilter: 'blur(10px)',
        },
      },
    },
  },
});

// Navigation Items - Alle 12 Tabs für ML Training
const navItems = [
  { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
  { path: '/config', label: 'Konfiguration', icon: <SettingsIcon /> },
  { path: '/logs', label: 'Logs', icon: <LogsIcon /> },
  { path: '/metrics', label: 'Metriken', icon: <MetricsIcon /> },
  { path: '/info', label: 'Info', icon: <InfoIcon /> },
  { path: '/models', label: 'Modelle', icon: <ModelsIcon /> },
  { path: '/training', label: 'Training', icon: <TrainingIcon /> },
  { path: '/test', label: 'Testen', icon: <TestingIcon /> },
  { path: '/test-results', label: 'Test-Ergebnisse', icon: <TestResultsIcon /> },
  { path: '/compare', label: 'Vergleichen', icon: <CompareIcon /> },
  { path: '/comparisons', label: 'Vergleichs-Übersicht', icon: <ComparisonsIcon /> },
  { path: '/jobs', label: 'Jobs', icon: <JobsIcon /> },
];

// Layout Component
const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [drawerOpen, setDrawerOpen] = React.useState(false);

  const drawerContent = (
    <Box sx={{ width: 250, pt: 2 }}>
      <Typography variant="h6" sx={{ px: 2, pb: 1, fontWeight: 'bold', color: '#00d4ff' }}>
        🤖 ML Training Service
      </Typography>
      <List>
        {navItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              onClick={() => isMobile && setDrawerOpen(false)}
              sx={{
                borderRadius: 2,
                mx: 1,
                '&.Mui-selected': {
                  backgroundColor: 'rgba(0, 212, 255, 0.2)',
                  color: '#00d4ff',
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 212, 255, 0.3)',
                  },
                  '& .MuiListItemIcon-root': {
                    color: '#00d4ff',
                  },
                },
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <Box sx={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%)',
      color: 'white',
      overflow: 'auto'
    }}>
      {/* Mobile Drawer */}
      {isMobile ? (
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: 250,
              background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          {drawerContent}
        </Drawer>
      ) : (
        /* Desktop Drawer */
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: 250,
              background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
            },
          }}
          open
        >
          {drawerContent}
        </Drawer>
      )}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          width: { xs: '100%', md: 'calc(100% - 250px)' },
          ml: { xs: 0, md: '250px' },
        }}
      >
        {/* Top Bar */}
        <AppBar
          position="static"
          sx={{
            background: 'rgba(26, 26, 46, 0.8)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: 'none',
          }}
        >
          <Toolbar>
            {isMobile && (
              <IconButton
                color="inherit"
                edge="start"
                onClick={() => setDrawerOpen(true)}
                sx={{ mr: 2 }}
              >
                <MenuIcon />
              </IconButton>
            )}
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#00d4ff' }}>
              🤖 ML Training Service Management
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              v1.0.0
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Page Content */}
        <Box sx={{ py: { xs: 2, sm: 3, md: 4 } }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
};

// Coming Soon Component
const ComingSoon: React.FC<{ title: string }> = ({ title }) => (
  <Container maxWidth="md" sx={{ py: 8, textAlign: 'center' }}>
    <Typography variant="h4" sx={{ color: '#00d4ff', mb: 2 }}>
      🚧 {title}
    </Typography>
    <Typography variant="body1" color="textSecondary">
      Diese Funktion wird in Kürze implementiert.
    </Typography>
  </Container>
);

// Main App Component
function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/config" element={<Config />} />
            <Route path="/logs" element={<ComingSoon title="Logs" />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/info" element={<Info />} />
            <Route path="/models" element={<Models />} />
            <Route path="/training" element={<Training />} />
            <Route path="/test" element={<ComingSoon title="Testen" />} />
            <Route path="/test-results" element={<ComingSoon title="Test-Ergebnisse" />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/comparisons" element={<ComingSoon title="Vergleichs-Übersicht" />} />
            <Route path="/jobs" element={<Jobs />} />
            <Route path="/details/:id" element={<Details />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
