import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Button,
  Alert,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
} from 'recharts';
import { useMLStore } from '../stores/mlStore';
import { LoadingSpinner, ErrorDisplay, MetricCard } from '../components';

const COLORS = ['#00d4ff', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#2196f3'];

const Metrics: React.FC = () => {
  const {
    health,
    jobs,
    models,
    isLoading,
    error,
    fetchHealth,
    fetchJobs,
    fetchModels,
    fetchMetrics,
  } = useMLStore();

  const [metrics, setMetrics] = useState<string>('');
  const [parsedMetrics, setParsedMetrics] = useState<Record<string, number>>({});
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      setRefreshing(true);
      await Promise.all([
        fetchHealth(),
        fetchJobs(),
        fetchModels(),
      ]);

      // Load Prometheus metrics
      const metricsData = await fetchMetrics();
      setMetrics(metricsData);
      setParsedMetrics(parsePrometheusMetrics(metricsData));
    } catch (error) {
      console.error('Failed to load metrics:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const parsePrometheusMetrics = (rawMetrics: string): Record<string, number> => {
    const lines = rawMetrics.split('\n');
    const parsed: Record<string, number> = {};

    lines.forEach(line => {
      if (line.startsWith('#') || !line.trim()) return;

      const [name, value] = line.split(' ');
      const numValue = parseFloat(value);

      // Parse different metric types
      if (name.includes('jobs_total')) parsed.jobs_total = numValue;
      if (name.includes('jobs_active')) parsed.jobs_active = numValue;
      if (name.includes('models_total')) parsed.models_total = numValue;
      if (name.includes('training_duration_seconds')) parsed.training_duration = numValue;
      if (name.includes('memory_usage_mb')) parsed.memory_usage = numValue;
      if (name.includes('cpu_usage_percent')) parsed.cpu_usage = numValue;
    });

    return parsed;
  };

  // Mock data for charts (in a real app, this would come from historical data)
  const chartData = [
    { time: '00:00', jobs: 2, models: 15, memory: 245 },
    { time: '04:00', jobs: 3, models: 18, memory: 267 },
    { time: '08:00', jobs: 5, models: 22, memory: 289 },
    { time: '12:00', jobs: 4, models: 25, memory: 301 },
    { time: '16:00', jobs: 6, models: 28, memory: 278 },
    { time: '20:00', jobs: 3, models: 26, memory: 256 },
  ];

  const jobStatusData = [
    { name: 'Completed', value: jobs.filter(j => j.status === 'completed').length, color: '#4caf50' },
    { name: 'Running', value: jobs.filter(j => j.status === 'running').length, color: '#2196f3' },
    { name: 'Failed', value: jobs.filter(j => j.status === 'failed').length, color: '#f44336' },
    { name: 'Pending', value: jobs.filter(j => j.status === 'pending').length, color: '#ff9800' },
  ];

  const modelTypeData = [
    { name: 'Random Forest', value: models.filter(m => m.model_type === 'random_forest').length, color: '#4caf50' },
    { name: 'XGBoost', value: models.filter(m => m.model_type === 'xgboost').length, color: '#00d4ff' },
  ];

  return (
    <Box sx={{ py: 4, px: { xs: 2, md: 4 } }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ color: '#00d4ff', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
          <AnalyticsIcon /> Service-Metriken & Analytics
        </Typography>

        <Button
          startIcon={<RefreshIcon />}
          onClick={loadData}
          variant="outlined"
          disabled={refreshing}
        >
          {refreshing ? 'Aktualisiere...' : 'Aktualisieren'}
        </Button>
      </Box>

      {error && <ErrorDisplay error={error} onRetry={loadData} />}

      {/* Key Metrics Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6} lg={3}>
          <MetricCard
            title="Aktive Jobs"
            value={jobs.filter(j => j.status === 'running').length.toString()}
            subtitle="Zurzeit in Ausführung"
            icon={<SpeedIcon />}
            color="#2196f3"
          />
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <MetricCard
            title="Verarbeitete Jobs"
            value={health?.total_jobs_processed?.toLocaleString() || '0'}
            subtitle="Gesamt seit Start"
            icon={<TrendingUpIcon />}
            color="#4caf50"
            trend={{ value: 12, direction: 'up' }}
          />
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <MetricCard
            title="Trainierte Modelle"
            value={models.length.toString()}
            subtitle="In Datenbank verfügbar"
            icon={<AnalyticsIcon />}
            color="#9c27b0"
          />
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <MetricCard
            title="System Uptime"
            value={health ? `${Math.floor(health.uptime_seconds / 3600)}h` : 'N/A'}
            subtitle="Seit letztem Neustart"
            icon={<TimelineIcon />}
            color="#ff9800"
          />
        </Grid>
      </Grid>

      {/* Performance Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
              📈 System-Performance über Zeit
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
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
                <Area
                  type="monotone"
                  dataKey="jobs"
                  stroke="#00d4ff"
                  fill="rgba(0, 212, 255, 0.2)"
                  name="Aktive Jobs"
                />
                <Area
                  type="monotone"
                  dataKey="models"
                  stroke="#4caf50"
                  fill="rgba(76, 175, 80, 0.2)"
                  name="Modelle"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            height: 'fit-content'
          }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
              🥧 Job-Status Verteilung
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={jobStatusData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {jobStatusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Model Analytics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={6}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
              📊 Modell-Typ Verteilung
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={modelTypeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="name" stroke="#b8c5d6" />
                <YAxis stroke="#b8c5d6" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(26, 26, 46, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="value" fill="#00d4ff" name="Anzahl Modelle" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Paper sx={{
            p: 3,
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
              🎯 Modell-Performance Metriken
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Durchschnittliche Accuracy:</Typography>
                <Chip
                  label={`${(models.reduce((acc, m) => acc + (m.training_accuracy || 0), 0) / Math.max(models.length, 1) * 100).toFixed(1)}%`}
                  color="success"
                  size="small"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Durchschnittliche F1-Score:</Typography>
                <Chip
                  label={`${(models.reduce((acc, m) => acc + (m.training_f1 || 0), 0) / Math.max(models.length, 1) * 100).toFixed(1)}%`}
                  color="primary"
                  size="small"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Beste Accuracy:</Typography>
                <Chip
                  label={`${(Math.max(...models.map(m => m.training_accuracy || 0)) * 100).toFixed(1)}%`}
                  color="secondary"
                  size="small"
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2">Modelle mit ROC-AUC &gt; 0.8:</Typography>
                <Chip
                  label={models.filter(m => (m.roc_auc || 0) > 0.8).length.toString()}
                  color="info"
                  size="small"
                />
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* System Resources */}
      <Paper sx={{
        p: 3,
        bgcolor: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(10px)',
        mb: 4
      }}>
        <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
          💾 System-Ressourcen
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ color: '#00d4ff', mb: 1 }}>
                {parsedMetrics.memory_usage || 'N/A'} MB
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Speicherverbrauch
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ color: '#ff9800', mb: 1 }}>
                {parsedMetrics.cpu_usage?.toFixed(1) || 'N/A'}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                CPU-Auslastung
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" sx={{ color: '#4caf50', mb: 1 }}>
                {parsedMetrics.training_duration?.toFixed(0) || 'N/A'}s
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Letzte Trainingsdauer
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

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
    </Box>
  );
};

export default Metrics;
