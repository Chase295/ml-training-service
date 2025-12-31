import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Alert,
  LinearProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { pumpApi } from '../services/api';

// interface MetricData {
//   name: string;
//   value: number;
//   timestamp?: string;
// }

interface ParsedMetrics {
  coins_received_total: number;
  coins_sent_total: number;
  cache_size_current: number;
  active_coins_current: number;
  trades_received_total: number;
  ws_reconnects_total: number;
  db_connected: number;
  n8n_available: number;
  uptime_seconds: number;
  errors_total: number;
  response_time_avg: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  // Filter-Metriken
  coins_filtered_bad_name: number;
  coins_filtered_spam_burst: number;
  coins_filtered_total: number;
}

const COLORS = ['#00d4ff', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#ff5722'];

const Metrics: React.FC = () => {
  const [metrics, setMetrics] = useState<string>('');
  const [parsedMetrics, setParsedMetrics] = useState<ParsedMetrics | null>(null);
  const [n8nStatus, setN8nStatus] = useState<boolean | null>(null);
  const [streamStats, setStreamStats] = useState<any>(null);
  const [phases, setPhases] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // Mock data for charts (in a real app, this would come from historical data)
  const [chartData] = useState([
    { time: '00:00', coins: 120, cache: 25, trades: 450 },
    { time: '04:00', coins: 135, cache: 28, trades: 520 },
    { time: '08:00', coins: 148, cache: 32, trades: 680 },
    { time: '12:00', coins: 162, cache: 35, trades: 750 },
    { time: '16:00', coins: 175, cache: 38, trades: 820 },
    { time: '20:00', coins: 168, cache: 36, trades: 790 },
  ]);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError('');
      const metricsData = await pumpApi.getMetrics();
      setMetrics(metricsData);
      setParsedMetrics(parsePrometheusMetrics(metricsData));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
    } finally {
      setLoading(false);
    }
  };

  const fetchN8nStatus = async () => {
    try {
      const health = await pumpApi.getHealth();
      setN8nStatus(health.discovery_stats?.n8n_available || false);
    } catch (err) {
      setN8nStatus(false);
    }
  };

  const fetchStreamStats = async () => {
    try {
      const data = await pumpApi.getStreamStats();
      setStreamStats(data);
    } catch (err) {
      console.error('Failed to fetch stream stats:', err);
      setStreamStats({});
    }
  };

  const fetchPhases = async () => {
    try {
      const data = await pumpApi.getPhases();
      setPhases(data.phases || []);
    } catch (err) {
      console.error('Failed to fetch phases:', err);
      setPhases([]);
    }
  };

  const parsePrometheusMetrics = (rawMetrics: string): ParsedMetrics => {
    const lines = rawMetrics.split('\n');
    const parsed: any = {
      coins_received_total: 0,
      coins_sent_total: 0,
      cache_size_current: 0,
      active_coins_current: 0,
      trades_received_total: 0,
      ws_reconnects_total: 0,
      db_connected: 0,
      n8n_available: 0,
      uptime_seconds: 0,
      errors_total: 0,
      response_time_avg: 0,
      memory_usage_mb: 0,
      cpu_usage_percent: 0,
      // Filter-Metriken
      coins_filtered_bad_name: 0,
      coins_filtered_spam_burst: 0,
      coins_filtered_total: 0,
    };

    lines.forEach(line => {
      if (line.startsWith('#') || !line.trim()) return;

      const [name, value] = line.split(' ');
      const numValue = parseFloat(value);

      if (name.includes('coins_received_total')) parsed.coins_received_total = numValue;
      if (name.includes('coins_sent_total')) parsed.coins_sent_total = numValue;
      if (name.includes('cache_size')) parsed.cache_size_current = numValue;
      if (name.includes('coins_tracked')) parsed.active_coins_current = numValue;
      if (name.includes('trades_received_total')) parsed.trades_received_total = numValue;
      if (name.includes('ws_reconnects_total')) parsed.ws_reconnects_total = numValue;
      if (name.includes('db_connected')) parsed.db_connected = numValue;
      if (name.includes('n8n_available')) parsed.n8n_available = numValue;
      if (name.includes('uptime_seconds')) parsed.uptime_seconds = numValue;
      if (name.includes('errors_total')) parsed.errors_total = numValue;

      // Filter-Metriken parsen
      if (name.includes('coins_filtered_total')) {
        if (line.includes('reason="bad_name"')) {
          parsed.coins_filtered_bad_name = numValue;
        } else if (line.includes('reason="spam_burst"')) {
          parsed.coins_filtered_spam_burst = numValue;
        }
        // Gesamt gefilterte Coins
        parsed.coins_filtered_total += numValue;
      }
    });

    return parsed;
  };

  useEffect(() => {
    fetchMetrics();
    fetchN8nStatus();
    fetchStreamStats();
    fetchPhases();
    const interval = setInterval(() => {
      fetchMetrics();
      fetchN8nStatus();
      fetchStreamStats();
      fetchPhases();
    }, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const statusData = parsedMetrics ? [
    { name: 'Coins Empfangen', value: parsedMetrics.coins_received_total, color: COLORS[0] },
    { name: 'Coins Gesendet', value: parsedMetrics.coins_sent_total, color: COLORS[1] },
    { name: 'Trades', value: parsedMetrics.trades_received_total, color: COLORS[2] },
    { name: 'Cache Size', value: parsedMetrics.cache_size_current, color: COLORS[3] },
  ] : [];

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          📊 Service Metriken
        </Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={fetchMetrics}
          disabled={loading}
        >
          {loading ? 'Aktualisiere...' : 'Aktualisieren'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Key Metrics Overview */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 4 }}>
        <Box sx={{ flex: 1 }}>
          <Card sx={{
            bgcolor: 'rgba(0, 212, 255, 0.1)',
            border: '1px solid rgba(0, 212, 255, 0.3)',
            backdropFilter: 'blur(10px)'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Coins Empfangen
                  </Typography>
                  <Typography variant="h4" sx={{ color: '#00d4ff' }}>
                    {parsedMetrics?.coins_received_total.toLocaleString() || '0'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                    Neue Coins von PumpPortal entdeckt
                  </Typography>
                </Box>
                <TimelineIcon sx={{ fontSize: 40, color: '#00d4ff' }} />
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Card sx={{
            bgcolor: 'rgba(76, 175, 80, 0.1)',
            border: '1px solid rgba(76, 175, 80, 0.3)',
            backdropFilter: 'blur(10px)'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Cache Größe
                  </Typography>
                  <Typography variant="h4" sx={{ color: '#4caf50' }}>
                    {parsedMetrics?.cache_size_current || '0'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                    Aktive Coins im 120s Cache
                  </Typography>
                </Box>
                <MemoryIcon sx={{ fontSize: 40, color: '#4caf50' }} />
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Card sx={{
            bgcolor: 'rgba(255, 152, 0, 0.1)',
            border: '1px solid rgba(255, 152, 0, 0.3)',
            backdropFilter: 'blur(10px)'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Trades Gesamt
                  </Typography>
                  <Typography variant="h4" sx={{ color: '#ff9800' }}>
                    {parsedMetrics?.trades_received_total.toLocaleString() || '0'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                    Trade-Events von PumpPortal empfangen
                  </Typography>
                </Box>
                <SpeedIcon sx={{ fontSize: 40, color: '#ff9800' }} />
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Card sx={{
            bgcolor: 'rgba(244, 67, 54, 0.1)',
            border: '1px solid rgba(244, 67, 54, 0.3)',
            backdropFilter: 'blur(10px)'
          }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Fehler Gesamt
                  </Typography>
                  <Typography variant="h4" sx={{ color: '#f44336' }}>
                    {parsedMetrics?.errors_total || '0'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                    Verbindungs- und API-Fehler
                  </Typography>
                </Box>
                <ErrorIcon sx={{ fontSize: 40, color: '#f44336' }} />
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Database Statistics */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ mt: 4, mb: 2 }}>
          🗄️ Datenbank Statistiken
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Card sx={{
              bgcolor: 'rgba(156, 39, 176, 0.1)',
              border: '1px solid rgba(156, 39, 176, 0.3)',
              backdropFilter: 'blur(10px)'
            }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Aktive Streams
                    </Typography>
                    <Typography variant="h4" sx={{ color: '#9c27b0' }}>
                      {streamStats?.active_streams || '0'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                      Live getrackte Coins
                    </Typography>
                  </Box>
                  <Typography sx={{ fontSize: 40, color: '#9c27b0' }}>📊</Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Card sx={{
              bgcolor: 'rgba(33, 150, 243, 0.1)',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              backdropFilter: 'blur(10px)'
            }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Gesamt Streams
                    </Typography>
                    <Typography variant="h4" sx={{ color: '#2196f3' }}>
                      {streamStats?.total_streams || '0'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                      Alle jemals getrackten Coins
                    </Typography>
                  </Box>
                  <Typography sx={{ fontSize: 40, color: '#2196f3' }}>📈</Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Card sx={{
              bgcolor: 'rgba(76, 175, 80, 0.1)',
              border: '1px solid rgba(76, 175, 80, 0.3)',
              backdropFilter: 'blur(10px)'
            }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Phasen
                    </Typography>
                    <Typography variant="h4" sx={{ color: '#4caf50' }}>
                      {phases.length}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                      Tracking-Phasen definiert
                    </Typography>
                  </Box>
                  <Typography sx={{ fontSize: 40, color: '#4caf50' }}>🏷️</Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>
        </Box>

        {/* Phase Distribution */}
        {streamStats?.streams_by_phase && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ mt: 2, mb: 2 }}>
              📊 Phasen-Verteilung
            </Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(5, 1fr)' }, gap: 2 }}>
              {phases.map((phase: any) => {
                const count = streamStats.streams_by_phase[phase.id] || 0;
                const percentage = streamStats.total_streams > 0 ? (count / streamStats.total_streams * 100).toFixed(1) : '0';

                return (
                  <Card key={phase.id} sx={{
                    bgcolor: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(10px)'
                  }}>
                    <CardContent sx={{ textAlign: 'center', py: 2 }}>
                      <Typography variant="h6" sx={{ color: '#00d4ff', fontWeight: 'bold' }}>
                        {count}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {phase.name}
                      </Typography>
                      <Typography variant="caption" sx={{ color: '#b8c5d6' }}>
                        {percentage}%
                      </Typography>
                    </CardContent>
                  </Card>
                );
              })}
            </Box>
          </Box>
        )}
      </Box>

      {/* Filter Statistics */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ mt: 4, mb: 2 }}>
          🛡️ Coin Filter Statistiken
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Card sx={{
              bgcolor: 'rgba(255, 193, 7, 0.1)',
              border: '1px solid rgba(255, 193, 7, 0.3)',
              backdropFilter: 'blur(10px)'
            }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Bad Name Filter
                    </Typography>
                    <Typography variant="h5">
                      {parsedMetrics?.coins_filtered_bad_name.toLocaleString() || '0'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Coins mit schlechten Namen gefiltert
                    </Typography>
                  </Box>
                  <ErrorIcon sx={{ fontSize: 40, color: '#ff9800' }} />
                </Box>
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Card sx={{
              bgcolor: 'rgba(244, 67, 54, 0.1)',
              border: '1px solid rgba(244, 67, 54, 0.3)',
              backdropFilter: 'blur(10px)'
            }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Spam-Burst Filter
                    </Typography>
                    <Typography variant="h5">
                      {parsedMetrics?.coins_filtered_spam_burst.toLocaleString() || '0'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Coins wegen Spam-Burst gefiltert
                    </Typography>
                  </Box>
                  <ErrorIcon sx={{ fontSize: 40, color: '#f44336' }} />
                </Box>
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Card sx={{
              bgcolor: 'rgba(156, 39, 176, 0.1)',
              border: '1px solid rgba(156, 39, 176, 0.3)',
              backdropFilter: 'blur(10px)'
            }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Gesamt Gefiltert
                    </Typography>
                    <Typography variant="h5">
                      {parsedMetrics?.coins_filtered_total.toLocaleString() || '0'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Insgesamt gefilterte Coins
                    </Typography>
                  </Box>
                  <MemoryIcon sx={{ fontSize: 40, color: '#9c27b0' }} />
                </Box>
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>

      {/* Charts */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, gap: 3, mb: 4 }}>
        <Box sx={{ flex: 2 }}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom>
              📈 Performance über Zeit
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="time" stroke="#b8c5d6" />
                <YAxis stroke="#b8c5d6" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(26, 26, 46, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="coins"
                  stroke="#00d4ff"
                  strokeWidth={2}
                  name="Coins"
                />
                <Line
                  type="monotone"
                  dataKey="cache"
                  stroke="#4caf50"
                  strokeWidth={2}
                  name="Cache"
                />
                <Line
                  type="monotone"
                  dataKey="trades"
                  stroke="#ff9800"
                  strokeWidth={2}
                  name="Trades"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom>
              🥧 Metriken-Verteilung
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Box>
      </Box>

      {/* System Status */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3, mb: 4 }}>
        <Box sx={{ flex: 1 }}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom>
              🔗 Service Status
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>WebSocket:</Typography>
                <Chip
                  label="Connected"
                  color="success"
                  size="small"
                  variant="outlined"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>Datenbank:</Typography>
                <Chip
                  label={parsedMetrics?.db_connected ? "Connected" : "Disconnected"}
                  color={parsedMetrics?.db_connected ? "success" : "error"}
                  size="small"
                  variant="outlined"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>n8n Service:</Typography>
                <Chip
                  label={n8nStatus ? "Available" : "Unavailable"}
                  color={n8nStatus ? "success" : "warning"}
                  size="small"
                  variant="outlined"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>Reconnects:</Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  {parsedMetrics?.ws_reconnects_total || 0}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom>
              ⏱️ Performance
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>Uptime:</Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  {parsedMetrics ? Math.floor(parsedMetrics.uptime_seconds / 3600) + 'h ' +
                    Math.floor((parsedMetrics.uptime_seconds % 3600) / 60) + 'm' : 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>Memory Usage:</Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  {parsedMetrics?.memory_usage_mb || 'N/A'} MB
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>CPU Usage:</Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  {parsedMetrics?.cpu_usage_percent || 'N/A'}%
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography>Avg Response Time:</Typography>
                <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                  {parsedMetrics?.response_time_avg || 'N/A'}ms
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Box>
      </Box>

      {/* Raw Metrics */}
      <Accordion sx={{
        bgcolor: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        '&:before': { display: 'none' }
      }}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: '#00d4ff' }} />}
          sx={{ color: 'white' }}
        >
          <Typography variant="h6">📋 Raw Prometheus Metrics</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{
            bgcolor: 'rgba(0, 0, 0, 0.3)',
            p: 2,
            borderRadius: 1,
            fontFamily: 'monospace',
            fontSize: '0.875rem',
            color: '#b8c5d6',
            maxHeight: 400,
            overflow: 'auto'
          }}>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
              {metrics || 'Keine Metriken verfügbar...'}
            </pre>
          </Box>
        </AccordionDetails>
      </Accordion>
    </Container>
  );
};

export default Metrics;
