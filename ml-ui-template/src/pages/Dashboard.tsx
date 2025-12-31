import React, { useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Timeline as TimelineIcon,
  Storage as StorageIcon,
  SwapHoriz as SwapHorizIcon,
} from '@mui/icons-material';
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'; // Für zukünftige Charts
import { usePumpStore } from '../stores/pumpStore';

const Dashboard: React.FC = () => {
  const {
    health,
    config,
    isLoading,
    error,
    fetchHealth,
    fetchConfig,
    startPolling,
    stopPolling,
  } = usePumpStore();

  // Berechne isServiceHealthy direkt in der Komponente für Reaktivität
  const isServiceHealthy = health ? health.status === 'healthy' : false;

  // Berechne Uptime direkt in der Komponente
  const uptimeFormatted = React.useMemo(() => {
    if (!health?.uptime_seconds) return 'N/A';

    const uptime = health.uptime_seconds;
    const hours = Math.floor(uptime / 3600);
    const minutes = Math.floor((uptime % 3600) / 60);
    const seconds = Math.floor(uptime % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }, [health?.uptime_seconds]);

  useEffect(() => {
    // Initial data load
    fetchHealth();
    fetchConfig();

    // Start polling
    startPolling();

    // Cleanup
    return () => {
      stopPolling();
    };
  }, [fetchHealth, fetchConfig, startPolling, stopPolling]);

  // Debug: Health-Daten prüfen
  useEffect(() => {
    console.log('🔍 Dashboard: Health-Daten geändert:', health);
    console.log('🔍 Dashboard: Cache-Stats verfügbar:', !!health?.cache_stats);
    if (health?.cache_stats) {
      console.log('🔍 Dashboard: Cache-Stats Details:', {
        total: health.cache_stats.total_coins,
        activated: health.cache_stats.activated_coins,
        expired: health.cache_stats.expired_coins
      });
    }
  }, [health]);

  if (isLoading && !health) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Lade Service-Status...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        🚀 Pump Service Dashboard
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Service Status Overview */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 3 }}>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Service Status
                </Typography>
                <Typography variant="h5">
                  {isServiceHealthy ? 'Healthy' : 'Degraded'}
                </Typography>
              </Box>
              {isServiceHealthy ? (
                <CheckCircleIcon color="success" sx={{ fontSize: 40 }} />
              ) : (
                <ErrorIcon color="error" sx={{ fontSize: 40 }} />
              )}
            </Box>
          </CardContent>
        </Card>

        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Uptime
                </Typography>
                <Typography variant="h5">
                  {uptimeFormatted}
                </Typography>
              </Box>
              <TimelineIcon color="primary" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>

        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  WebSocket
                </Typography>
                <Chip
                  label={health?.ws_connected ? 'Connected' : 'Disconnected'}
                  color={health?.ws_connected ? 'success' : 'error'}
                  size="small"
                />
              </Box>
              <SwapHorizIcon color="primary" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>

        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Datenbank
                </Typography>
                <Chip
                  label={health?.db_connected ? 'Connected' : 'Disconnected'}
                  color={health?.db_connected ? 'success' : 'error'}
                  size="small"
                />
              </Box>
              <StorageIcon color="primary" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* Cache & Discovery Stats */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 3 }}>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            💾 Cache Statistiken
          </Typography>
          {health ? (
            <Box>
              <Typography>
                Gesamt Coins: <strong>{health.cache_stats?.total_coins || 0}</strong>
              </Typography>
              <Typography>
                Aktivierte: <strong>{health.cache_stats?.activated_coins || 0}</strong>
              </Typography>
              <Typography>
                Abgelaufen: <strong>{health.cache_stats?.expired_coins || 0}</strong>
              </Typography>
              <Typography>
                Ältester Cache: <strong>{health.cache_stats?.oldest_age_seconds || 0}s</strong>
              </Typography>
            </Box>
          ) : (
            <Typography color="textSecondary">
              Daten werden geladen...
            </Typography>
          )}
        </Paper>

        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            🔍 Discovery Statistiken
          </Typography>
          {health?.discovery_stats && (
            <Box>
              <Typography>
                Entdeckte Coins: <strong>{health.discovery_stats.total_coins_discovered}</strong>
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Typography sx={{ mr: 1 }}>
                  n8n Status:
                </Typography>
                <Chip
                  label={health.discovery_stats.n8n_available ? 'Verfügbar' : 'Nicht verfügbar'}
                  color={health.discovery_stats.n8n_available ? 'success' : 'warning'}
                  size="small"
                />
              </Box>
              <Typography sx={{ mt: 1 }}>
                n8n Buffer: <strong>{health.discovery_stats.n8n_buffer_size || 0}</strong> Coins warten
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>

      {/* Tracking Stats */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 3 }}>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            📊 Tracking Statistiken
          </Typography>
          {health?.tracking_stats && (
            <Box>
              <Typography>
                Aktive Coins: <strong>{health.tracking_stats.active_coins}</strong>
              </Typography>
              <Typography>
                Trades Gesamt: <strong>{health.tracking_stats.total_trades.toLocaleString()}</strong>
              </Typography>
              <Typography>
                Metriken gespeichert: <strong>{health.tracking_stats.total_metrics_saved.toLocaleString()}</strong>
              </Typography>
            </Box>
          )}
        </Paper>

        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="h6" gutterBottom>
            ⚙️ Service Konfiguration
          </Typography>
          {config && (
            <Box>
              <Typography variant="body2" color="textSecondary">
                Cache-Zeit: <strong>{config.coin_cache_seconds}s</strong>
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Batch-Größe: <strong>{config.batch_size}</strong>
              </Typography>
              <Typography variant="body2" color="textSecondary">
                DB Refresh: <strong>{config.db_refresh_interval}s</strong>
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>

      {/* Status Indicators */}
      {health?.last_error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>Letzter Fehler:</strong> {health.last_error}
          </Typography>
        </Alert>
      )}

      <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
        Letzte Aktualisierung: {health ? new Date().toLocaleTimeString() : 'N/A'}
      </Typography>
    </Container>
  );
};

export default Dashboard;
