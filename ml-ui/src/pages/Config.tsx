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
  Tab,
  Tabs,
  Divider,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Dataset as DatabaseIcon,
  Science as MLIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useMLStore } from '../stores/mlStore';
import { ValidatedTextField, ValidatedSelect, LoadingSpinner, ErrorDisplay } from '../components';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div role="tabpanel" hidden={value !== index}>
    {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
  </div>
);

const Config: React.FC = () => {
  const {
    config,
    isLoading,
    error,
    fetchConfig,
    updateConfig,
    reloadConfig,
    reconnectDb,
  } = useMLStore();

  const [tabValue, setTabValue] = useState(0);
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [testDbLoading, setTestDbLoading] = useState(false);
  const [dbTestResult, setDbTestResult] = useState<{success: boolean, message: string} | null>(null);
  const [reconnectDbLoading, setReconnectDbLoading] = useState(false);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  useEffect(() => {
    if (config) {
      setFormData({ ...config });
    }
  }, [config]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleFieldChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setSaveLoading(true);
    try {
      await updateConfig(formData);
      setHasChanges(false);
      // Clear any previous errors
      if (error) {
        // Reset error state in store by calling fetchConfig
        fetchConfig();
      }
    } catch (error) {
      console.error('Config save failed:', error);
      // Error is handled by the store and displayed by ErrorDisplay
    } finally {
      setSaveLoading(false);
    }
  };

  const handleReload = async () => {
    await reloadConfig();
    setHasChanges(false);
  };

  const handleReconnectDb = async () => {
    setReconnectDbLoading(true);
    try {
      const result = await reconnectDb();
      setDbTestResult({
        success: result.db_connected,
        message: result.message
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unbekannter Fehler';
      setDbTestResult({
        success: false,
        message: `❌ DB-Reconnect fehlgeschlagen: ${errorMessage}`
      });
    } finally {
      setReconnectDbLoading(false);
    }
  };

  const handleTestDbConnection = async () => {
    setTestDbLoading(true);
    setDbTestResult(null);

    try {
      // Teste die Health-API - wenn DB funktioniert, sollte sie db_connected: true zurückgeben
      const response = await fetch('/api/health');

      // 503 ist OK für Health-Check (bedeutet DB nicht verfügbar, aber Service läuft)
      if (!response.ok && response.status !== 503) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const healthData = await response.json();

      if (healthData.db_connected) {
        setDbTestResult({
          success: true,
          message: '✅ Datenbankverbindung erfolgreich! Alle APIs sind verfügbar.'
        });
      } else {
        setDbTestResult({
          success: false,
          message: '❌ Datenbank nicht verbunden. Überprüfen Sie die DSN und Zugangsdaten.'
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unbekannter Fehler';
      setDbTestResult({
        success: false,
        message: `❌ Verbindungsfehler: ${errorMessage}`
      });
    } finally {
      setTestDbLoading(false);
    }
  };

  if (isLoading && !config) {
    return <LoadingSpinner message="Lade Konfiguration..." height="60vh" />;
  }

  return (
    <Box sx={{ py: 4, px: { xs: 2, md: 4 } }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ color: '#00d4ff', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsIcon /> ML Training Service Konfiguration
        </Typography>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            startIcon={<RefreshIcon />}
            onClick={handleReload}
            variant="outlined"
            disabled={isLoading}
          >
            Neu laden
          </Button>
          <Button
            startIcon={<SaveIcon />}
            onClick={handleSave}
            variant="contained"
            disabled={!hasChanges || saveLoading}
            color="primary"
          >
            {saveLoading ? 'Speichere...' : 'Speichern'}
          </Button>
        </Box>
      </Box>

      {error && <ErrorDisplay error={error} onRetry={() => updateConfig(formData)} />}

      {hasChanges && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Sie haben ungespeicherte Änderungen. Klicken Sie auf "Speichern", um diese zu übernehmen.
        </Alert>
      )}

      <Card sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
        <CardContent sx={{ p: 0 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              '& .MuiTab-root': {
                color: 'text.secondary',
                '&.Mui-selected': {
                  color: '#00d4ff',
                },
              },
              '& .MuiTabs-indicator': {
                backgroundColor: '#00d4ff',
              },
            }}
          >
            <Tab icon={<DatabaseIcon />} label="Datenbank" />
            <Tab icon={<MLIcon />} label="ML-Training" />
          </Tabs>

          {/* Database Configuration */}
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
              <DatabaseIcon /> Datenbank-Konfiguration
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12}>
                <ValidatedTextField
                  label="DB DSN"
                  value={formData.db_dsn || ''}
                  onChange={(value) => handleFieldChange('db_dsn', value)}
                  placeholder="postgresql://user:password@host:5432/database"
                  helperText="PostgreSQL-Verbindungsstring (Passwort wird maskiert angezeigt)"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="DB Refresh Intervall"
                  value={formData.db_refresh_interval?.toString() || ''}
                  onChange={(value) => handleFieldChange('db_refresh_interval', parseInt(value) || 10)}
                  type="number"
                  helperText="Sekunden zwischen DB-Abfragen aktiver Streams (5-300)"
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(255, 255, 255, 0.05)', borderRadius: 1 }}>
              <Typography variant="subtitle2" gutterBottom sx={{ color: '#00d4ff' }}>
                🔍 Datenbank-Verbindung testen
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
                <Button
                  variant="outlined"
                  onClick={handleTestDbConnection}
                  disabled={testDbLoading || reconnectDbLoading}
                  sx={{ borderColor: '#00d4ff', color: '#00d4ff' }}
                >
                  {testDbLoading ? 'Teste...' : 'Verbindung testen'}
                </Button>
                <Button
                  variant="contained"
                  onClick={handleReconnectDb}
                  disabled={testDbLoading || reconnectDbLoading}
                  sx={{ bgcolor: '#4caf50', '&:hover': { bgcolor: '#45a049' } }}
                >
                  {reconnectDbLoading ? 'Verbinde...' : 'DB neu verbinden'}
                </Button>
              </Box>

              {dbTestResult && (
                <Alert
                  severity={dbTestResult.success ? 'success' : 'error'}
                  icon={dbTestResult.success ? <CheckIcon /> : <ErrorIcon />}
                  sx={{
                    bgcolor: dbTestResult.success
                      ? 'rgba(76, 175, 80, 0.1)'
                      : 'rgba(244, 67, 54, 0.1)',
                    border: dbTestResult.success
                      ? '1px solid rgba(76, 175, 80, 0.3)'
                      : '1px solid rgba(244, 67, 54, 0.3)'
                  }}
                >
                  <Typography variant="body2">
                    {dbTestResult.message}
                  </Typography>
                </Alert>
              )}
            </Box>
          </TabPanel>

          {/* ML Training Configuration */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
              <MLIcon /> ML-Training Konfiguration
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Modell-Speicherpfad"
                  value={formData.model_storage_path || ''}
                  onChange={(value) => handleFieldChange('model_storage_path', value)}
                  placeholder="/app/models"
                  helperText="Pfad wo trainierte Modelle gespeichert werden"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Max. parallele Jobs"
                  value={formData.max_concurrent_jobs?.toString() || ''}
                  onChange={(value) => handleFieldChange('max_concurrent_jobs', parseInt(value) || 2)}
                  type="number"
                  helperText="Maximale Anzahl gleichzeitig laufender Training-Jobs (1-10)"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Job Poll Intervall"
                  value={formData.job_poll_interval?.toString() || ''}
                  onChange={(value) => handleFieldChange('job_poll_interval', parseInt(value) || 5)}
                  type="number"
                  helperText="Sekunden zwischen Job-Status-Überprüfungen (1-60)"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Standard Training Stunden"
                  value={formData.default_training_hours?.toString() || ''}
                  onChange={(value) => handleFieldChange('default_training_hours', parseInt(value) || 24)}
                  type="number"
                  helperText="Standard-Trainingszeit in Stunden für neue Jobs (1-168)"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Max. Training Stunden"
                  value={formData.max_training_hours?.toString() || ''}
                  onChange={(value) => handleFieldChange('max_training_hours', parseInt(value) || 168)}
                  type="number"
                  helperText="Maximale erlaubte Trainingszeit in Stunden (24-168)"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Min. Training Stunden"
                  value={formData.min_training_hours?.toString() || ''}
                  onChange={(value) => handleFieldChange('min_training_hours', parseInt(value) || 1)}
                  type="number"
                  helperText="Minimale erlaubte Trainingszeit in Stunden (1-24)"
                />
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </Card>

      <Divider sx={{ my: 4, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

      <Alert severity="info" sx={{ bgcolor: 'rgba(33, 150, 243, 0.1)', border: '1px solid rgba(33, 150, 243, 0.3)' }}>
        <Typography variant="body2">
          <strong>⚠️ Wichtig:</strong> Änderungen werden sofort wirksam. Die meisten Einstellungen erfordern keinen Neustart.
          Änderungen an parallelen Jobs und Training-Zeitlimits werden beim nächsten Job angewendet.
        </Typography>
      </Alert>
    </Box>
  );
};

export default Config;
