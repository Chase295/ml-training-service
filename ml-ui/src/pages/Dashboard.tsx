import React, { useEffect } from 'react';
import {
  Container, Typography, Box, Alert, Card, CardContent
} from '@mui/material';
import {
  Work as JobsIcon,
  Science as ModelsIcon,
  Timeline as UptimeIcon,
  Storage as DatabaseIcon,
  Speed as PerformanceIcon,
  CheckCircle as SuccessIcon,
} from '@mui/icons-material';
import { useMLStore } from '../stores/mlStore';
import { MetricCard, StatusChip, LoadingSpinner, ErrorDisplay } from '../components';

const Dashboard: React.FC = () => {
  const {
    health,
    jobs,
    models,
    isLoading,
    error,
    fetchHealth,
    fetchJobs,
    fetchModels,
    startPolling,
    stopPolling,
  } = useMLStore();

  useEffect(() => {
    // Initial data load
    fetchHealth();
    fetchJobs();
    fetchModels();

    // Start polling for live updates
    startPolling();

    // Cleanup
    return () => {
      stopPolling();
    };
  }, [fetchHealth, fetchJobs, fetchModels, startPolling, stopPolling]);

  // Calculate statistics
  const activeJobs = jobs.filter(job => job.status === 'running').length;
  const pendingJobs = jobs.filter(job => job.status === 'pending').length;
  const completedJobs = jobs.filter(job => job.status === 'completed').length;
  const failedJobs = jobs.filter(job => job.status === 'failed').length;

  const totalModels = models.length;
  const readyModels = models.filter(m => m.status === 'READY').length;
  const trainingModels = models.filter(m => m.status === 'TRAINING').length;

  // Format uptime
  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  // Calculate trends (mock data for now)
  const jobTrend = completedJobs > 10 ? { value: 15, direction: 'up' as const } : { value: 0, direction: 'neutral' as const };
  const modelTrend = totalModels > 5 ? { value: 25, direction: 'up' as const } : { value: 0, direction: 'neutral' as const };

  if (isLoading && !health) {
    return <LoadingSpinner message="Lade Dashboard..." size={50} height="60vh" />;
  }

  return (
    <Box sx={{ py: 4, px: { xs: 2, md: 4 } }}>
      <Typography variant="h4" gutterBottom sx={{ color: '#00d4ff', fontWeight: 'bold' }}>
        🤖 ML Training Service Dashboard
      </Typography>

      <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
        Live-Systemübersicht und Status-Monitoring
      </Typography>

      {error && <ErrorDisplay error={error} onRetry={() => {
        fetchHealth();
        fetchJobs();
        fetchModels();
      }} />}

      {/* System Health Overview */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 4, width: '100%' }}>
        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
          <MetricCard
            title="System Status"
            value={health?.status === 'healthy' ? 'Gesund' : 'Problem'}
            subtitle="Service-Verfügbarkeit"
            icon={<SuccessIcon />}
            color={health?.status === 'healthy' ? '#4caf50' : '#f44336'}
          />
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
          <MetricCard
            title="Uptime"
            value={health ? formatUptime(health.uptime_seconds) : 'N/A'}
            subtitle="Seit letztem Neustart"
            icon={<UptimeIcon />}
            color="#00d4ff"
          />
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
          <MetricCard
            title="Datenbank"
            value={health?.db_connected ? 'Verbunden' : 'Getrennt'}
            subtitle="PostgreSQL Status"
            icon={<DatabaseIcon />}
            color={health?.db_connected ? '#4caf50' : '#f44336'}
          />
        </Box>

        <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
          <MetricCard
            title="Verarbeitete Jobs"
            value={health?.total_jobs_processed?.toLocaleString() || '0'}
            subtitle="Gesamt seit Start"
            icon={<PerformanceIcon />}
            color="#ff9800"
            trend={{ value: 8, direction: 'up' }}
          />
        </Box>
      </Box>

      {/* Jobs Overview */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
          <JobsIcon /> Job-Management Übersicht
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, width: '100%' }}>
          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
            <MetricCard
              title="Aktive Jobs"
              value={activeJobs.toString()}
              subtitle="Zurzeit laufend"
              icon={<JobsIcon />}
              color="#2196f3"
            />
          </Box>

          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
            <MetricCard
              title="Wartende Jobs"
              value={pendingJobs.toString()}
              subtitle="In der Queue"
              icon={<JobsIcon />}
              color="#ff9800"
            />
          </Box>

          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
            <MetricCard
              title="Abgeschlossen"
              value={completedJobs.toString()}
              subtitle="Erfolgreich fertig"
              icon={<JobsIcon />}
              color="#4caf50"
              trend={jobTrend}
            />
          </Box>

          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(25% - 18px)' }, minWidth: 0 }}>
            <MetricCard
              title="Fehlgeschlagen"
              value={failedJobs.toString()}
              subtitle="Mit Fehlern"
              icon={<JobsIcon />}
              color="#f44336"
            />
          </Box>
        </Box>
      </Box>

      {/* Models Overview */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
          <ModelsIcon /> Modell-Management Übersicht
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, width: '100%' }}>
          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(33.333% - 16px)' }, minWidth: 0 }}>
            <MetricCard
              title="Gesamt Modelle"
              value={totalModels.toString()}
              subtitle="In der Datenbank"
              icon={<ModelsIcon />}
              color="#9c27b0"
              trend={modelTrend}
            />
          </Box>

          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(33.333% - 16px)' }, minWidth: 0 }}>
            <MetricCard
              title="Bereite Modelle"
              value={readyModels.toString()}
              subtitle="Verwendungsbereit"
              icon={<ModelsIcon />}
              color="#4caf50"
            />
          </Box>

          <Box sx={{ flex: { xs: '1 1 100%', sm: '1 1 calc(50% - 12px)', lg: '1 1 calc(33.333% - 16px)' }, minWidth: 0 }}>
            <MetricCard
              title="In Training"
              value={trainingModels.toString()}
              subtitle="Zurzeit aktiv"
              icon={<ModelsIcon />}
              color="#ff9800"
            />
          </Box>
        </Box>
      </Box>

      {/* Recent Activity */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ color: '#00d4ff' }}>
          📋 Letzte Aktivitäten
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, width: '100%' }}>
          <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 calc(50% - 12px)' }, minWidth: 0 }}>
            <Card sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
                  🔄 Letzte Jobs
                </Typography>
                {jobs.slice(0, 5).length > 0 ? (
                  jobs.slice(0, 5).map((job) => (
                    <Box key={job.id} sx={{ mb: 2, pb: 2, borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          Job #{job.id}
                        </Typography>
                        <StatusChip status={job.status} size="small" />
                      </Box>
                      <Typography variant="caption" color="textSecondary">
                        {job.job_type} • {new Date(job.created_at).toLocaleString('de-DE')}
                      </Typography>
                      {job.progress > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" color="textSecondary">
                            Fortschritt: {job.progress}%
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  ))
                ) : (
                  <Typography variant="body2" color="textSecondary">
                    Keine Jobs vorhanden
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Box>

          <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 calc(50% - 12px)' }, minWidth: 0 }}>
            <Card sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
                  🧠 Letzte Modelle
                </Typography>
                {models.slice(0, 5).length > 0 ? (
                  models.slice(0, 5).map((model) => (
                    <Box key={model.id} sx={{ mb: 2, pb: 2, borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          {model.name}
                        </Typography>
                        <StatusChip status={model.status} size="small" />
                      </Box>
                      <Typography variant="caption" color="textSecondary">
                        {model.model_type} • Accuracy: {model.training_accuracy?.toFixed(3) || 'N/A'}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Erstellt: {new Date(model.created_at).toLocaleDateString('de-DE')}
                      </Typography>
                    </Box>
                  ))
                ) : (
                  <Typography variant="body2" color="textSecondary">
                    Keine Modelle vorhanden
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>

      {/* System Info */}
      <Alert severity="info" sx={{ bgcolor: 'rgba(33, 150, 243, 0.1)', border: '1px solid rgba(33, 150, 243, 0.3)' }}>
        <Typography variant="body2">
          <strong>💡 Hinweis:</strong> Alle Daten werden automatisch alle 30 Sekunden aktualisiert.
          Das Dashboard zeigt Live-Status und Echtzeit-Metriken des ML Training Service.
        </Typography>
      </Alert>
    </Box>
  );
};

export default Dashboard;
