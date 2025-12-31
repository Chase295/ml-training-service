import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Alert,
  Card,
  CardContent,
  CardHeader,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Save as SaveIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import { usePumpStore } from '../stores/pumpStore';
import type { ConfigUpdateRequest } from '../types/api';

const Config: React.FC = () => {
  const { config, updateConfig, fetchConfig, isLoading, error } = usePumpStore();

  const [formData, setFormData] = useState<ConfigUpdateRequest>({
    n8n_webhook_url: '',
    n8n_webhook_method: 'POST',
    db_dsn: '',
    coin_cache_seconds: 120,
    db_refresh_interval: 10,
    batch_size: 10,
    batch_timeout: 30,
    bad_names_pattern: 'test|bot|rug|scam|cant|honey|faucet',
    spam_burst_window: 30,
  });

  const [originalData, setOriginalData] = useState<ConfigUpdateRequest | null>(null);
  const [successMessage, setSuccessMessage] = useState<string>('');

  React.useEffect(() => {
    if (config) {
      const newFormData = {
        n8n_webhook_url: config.n8n_webhook_url || '',
        n8n_webhook_method: config.n8n_webhook_method || 'POST',
        db_dsn: config.db_dsn || '',
        coin_cache_seconds: config.coin_cache_seconds || 120,
        db_refresh_interval: config.db_refresh_interval || 10,
        batch_size: config.batch_size || 10,
        batch_timeout: config.batch_timeout || 30,
        bad_names_pattern: config.bad_names_pattern || 'test|bot|rug|scam|cant|honey|faucet',
        spam_burst_window: config.spam_burst_window || 30,
      };
      setFormData(newFormData);
      setOriginalData(newFormData);
    }
  }, [config]);

  const handleInputChange = (field: keyof ConfigUpdateRequest, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (!originalData) return;

      // Erstelle ein Objekt nur mit geänderten Feldern
      const updateData: Partial<ConfigUpdateRequest> = {};

      // Vergleiche jedes Feld mit den Originaldaten
      Object.keys(formData).forEach((key) => {
        const formKey = key as keyof ConfigUpdateRequest;
        if (formData[formKey] !== originalData[formKey]) {
          // Feld wurde geändert
          if (formKey === 'db_dsn' && formData.db_dsn && formData.db_dsn.includes('***')) {
            // Zensierte Passwörter nicht senden
            return;
          }
          (updateData as any)[formKey] = formData[formKey];
        }
      });

      // Nur senden, wenn etwas geändert wurde
      if (Object.keys(updateData).length === 0) {
        setSuccessMessage('Keine Änderungen erkannt.');
        setTimeout(() => setSuccessMessage(''), 3000);
        return;
      }

      await updateConfig(updateData);
      setSuccessMessage('Konfiguration erfolgreich aktualisiert!');
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      // Error wird bereits im Store behandelt
    }
  };

  const handleRefresh = () => {
    fetchConfig();
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        ⚙️ Service Konfiguration
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {successMessage}
        </Alert>
      )}

      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* n8n Konfiguration */}
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, gap: 3 }}>
            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="🔗 n8n Integration"
                  subheader="Webhook-Einstellungen für Coin-Discovery"
                />
                <CardContent>
                  <Box sx={{ mb: 2 }}>
                    <TextField
                      fullWidth
                      label="Webhook URL"
                      value={formData.n8n_webhook_url}
                      onChange={(e) => handleInputChange('n8n_webhook_url', e.target.value)}
                      placeholder="https://n8n.example.com/webhook/xyz"
                      helperText="Vollständige URL zum n8n Webhook-Endpoint"
                    />
                  </Box>

                  <FormControl fullWidth>
                    <InputLabel>HTTP Methode</InputLabel>
                    <Select
                      value={formData.n8n_webhook_method}
                      onChange={(e) => handleInputChange('n8n_webhook_method', e.target.value)}
                      label="HTTP Methode"
                    >
                      <MenuItem value="GET">GET</MenuItem>
                      <MenuItem value="POST">POST</MenuItem>
                    </Select>
                  </FormControl>
                </CardContent>
              </Card>
            </Box>

            {/* Datenbank Konfiguration */}
            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="🗄️ Datenbank"
                  subheader="PostgreSQL Connection String"
                />
                <CardContent>
                  <TextField
                    fullWidth
                    label="Database DSN"
                    value={formData.db_dsn}
                    onChange={(e) => handleInputChange('db_dsn', e.target.value)}
                    placeholder="postgresql://user:pass@host:port/database"
                    helperText="PostgreSQL Connection String (Passwort wird automatisch versteckt)"
                    multiline
                    rows={2}
                  />
                </CardContent>
              </Card>
            </Box>
          </Box>

          {/* Performance Einstellungen */}
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="💾 Cache"
                  subheader="Coin-Cache Einstellungen"
                />
                <CardContent>
                  <TextField
                    fullWidth
                    type="number"
                    label="Cache-Zeit (Sekunden)"
                    value={formData.coin_cache_seconds}
                    onChange={(e) => handleInputChange('coin_cache_seconds', parseInt(e.target.value))}
                    inputProps={{ min: 10, max: 3600 }}
                    helperText="Wie lange neue Coins gecacht werden (10-3600s)"
                  />
                </CardContent>
              </Card>
            </Box>

            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="🔄 Refresh"
                  subheader="Datenbank-Abfrage Interval"
                />
                <CardContent>
                  <TextField
                    fullWidth
                    type="number"
                    label="Refresh-Intervall (Sekunden)"
                    value={formData.db_refresh_interval}
                    onChange={(e) => handleInputChange('db_refresh_interval', parseInt(e.target.value))}
                    inputProps={{ min: 5, max: 300 }}
                    helperText="Wie oft die DB nach neuen Streams abgefragt wird"
                  />
                </CardContent>
              </Card>
            </Box>

            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="📦 Batch"
                  subheader="n8n Batch-Größe"
                />
                <CardContent>
                  <TextField
                    fullWidth
                    type="number"
                    label="Batch-Größe"
                    value={formData.batch_size}
                    onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}
                    inputProps={{ min: 1, max: 100 }}
                    helperText="Wie viele Coins pro n8n Batch gesendet werden"
                  />
                  <TextField
                    fullWidth
                    type="number"
                    label="Batch-Timeout (Sekunden)"
                    value={formData.batch_timeout}
                    onChange={(e) => handleInputChange('batch_timeout', parseInt(e.target.value))}
                    inputProps={{ min: 10, max: 300 }}
                    helperText="Maximale Zeit bevor ein unvollständiger Batch gesendet wird"
                    sx={{ mt: 2 }}
                  />
                </CardContent>
              </Card>
            </Box>
          </Box>

          {/* Filter Einstellungen */}
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, gap: 3 }}>
            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="🛡️ Coin Filter"
                  subheader="Filter für Namen und Wiederholungen"
                />
                <CardContent>
                  <TextField
                    fullWidth
                    label="Bad Names Pattern"
                    value={formData.bad_names_pattern}
                    onChange={(e) => handleInputChange('bad_names_pattern', e.target.value)}
                    placeholder="test|bot|rug|scam|cant|honey|faucet"
                    helperText="Regex-Pattern für schlechte Coin-Namen (mit | getrennt)"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    type="number"
                    label="Spam-Burst Fenster (Sekunden)"
                    value={formData.spam_burst_window}
                    onChange={(e) => handleInputChange('spam_burst_window', parseInt(e.target.value))}
                    inputProps={{ min: 5, max: 300 }}
                    helperText="Zeitfenster für Spam-Burst-Erkennung (5-300s)"
                  />
                </CardContent>
              </Card>
            </Box>

            <Box sx={{ flex: 1 }}>
              <Card>
                <CardHeader
                  title="📊 Filter Statistiken"
                  subheader="Live-Filter-Ergebnisse"
                />
                <CardContent>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Gefilterte Coins werden in den Prometheus-Metriken angezeigt
                  </Typography>
                  <Typography variant="body2">
                    • Bad Name Filter: Entfernt Coins mit verdächtigen Namen
                  </Typography>
                  <Typography variant="body2">
                    • Spam-Burst Filter: Verhindert Coin-Spam in kurzer Zeit
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </Box>

          {/* Aktionen */}
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleRefresh}
              disabled={isLoading}
            >
              Aktualisieren
            </Button>

            <Button
              type="submit"
              variant="contained"
              startIcon={<SaveIcon />}
              disabled={isLoading}
              size="large"
            >
              {isLoading ? 'Speichere...' : 'Speichern'}
            </Button>
          </Box>
        </Box>
      </form>

      {/* Aktuelle Konfiguration Anzeige */}
      {config && (
        <Card sx={{ mt: 3 }}>
          <CardHeader
            title="📋 Aktuelle Konfiguration"
            subheader="Live-Werte aus dem Service"
          />
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2 }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="textSecondary">n8n Webhook</Typography>
                  <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                    {config.n8n_webhook_url || 'Nicht gesetzt'}
                  </Typography>
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="textSecondary">HTTP Methode</Typography>
                  <Typography variant="body1">{config.n8n_webhook_method}</Typography>
                </Box>
              </Box>
              <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2 }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="textSecondary">Cache-Zeit</Typography>
                  <Typography variant="body1">{config.coin_cache_seconds} Sekunden</Typography>
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="textSecondary">Batch-Größe</Typography>
                  <Typography variant="body1">{config.batch_size} Coins</Typography>
                </Box>
              </Box>
              <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2, mt: 2 }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="textSecondary">Bad Names Pattern</Typography>
                  <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                    {config.bad_names_pattern || 'test|bot|rug|scam|cant|honey|faucet'}
                  </Typography>
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" color="textSecondary">Spam-Burst Fenster</Typography>
                  <Typography variant="body1">{config.spam_burst_window || 30} Sekunden</Typography>
                </Box>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default Config;
