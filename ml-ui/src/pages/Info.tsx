import React from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Info as InfoIcon,
  Timeline as TimelineIcon,
  Analytics as AnalyticsIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
  ExpandMore as ExpandMoreIcon,
  Science as ScienceIcon,
  Storage as StorageIcon,
} from '@mui/icons-material';
import { useMLStore } from '../stores/mlStore';

const Info: React.FC = () => {
  const { health, config } = useMLStore();

  // Live-Statistiken aus health-Daten berechnen
  const totalJobs = health?.total_jobs_processed || 0;
  const dbConnected = health?.db_connected || false;
  const uptimeSeconds = health?.uptime_seconds || 0;

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" gutterBottom sx={{ color: '#00d4ff', fontWeight: 'bold' }}>
          🤖 ML Training Service - Dokumentation
        </Typography>
        <Typography variant="h6" sx={{ color: 'text.secondary', mb: 2 }}>
          Vollständige Übersicht über das Machine Learning Training System
        </Typography>
        <Alert severity="info" sx={{ maxWidth: 800, mx: 'auto' }}>
          <Typography variant="body2">
            Dieses Dokument erklärt <strong>alle Features</strong>, <strong>API-Endpunkte</strong>
            und <strong>die Architektur</strong> des ML Training Service.
          </Typography>
        </Alert>
      </Box>

      {/* System Status Overview */}
      <Card sx={{ mb: 4, bgcolor: 'rgba(0, 212, 255, 0.1)', border: '1px solid rgba(0, 212, 255, 0.3)' }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <SpeedIcon sx={{ color: '#00d4ff', fontSize: 30 }} />
            <Typography variant="h4" sx={{ color: '#00d4ff', fontWeight: 'bold' }}>
              🚀 System Status Übersicht
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card sx={{ bgcolor: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
                <CardContent>
                  <Box textAlign="center">
                    <Typography variant="h5" sx={{ color: '#4caf50', mb: 1 }}>
                      {dbConnected ? '🟢 Verbunden' : '🔴 Getrennt'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Datenbank Status
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ bgcolor: 'rgba(255, 152, 0, 0.1)', border: '1px solid rgba(255, 152, 0, 0.3)' }}>
                <CardContent>
                  <Box textAlign="center">
                    <Typography variant="h5" sx={{ color: '#ff9800', mb: 1 }}>
                      {formatUptime(uptimeSeconds)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      System Uptime
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ bgcolor: 'rgba(156, 39, 176, 0.1)', border: '1px solid rgba(156, 39, 176, 0.3)' }}>
                <CardContent>
                  <Box textAlign="center">
                    <Typography variant="h5" sx={{ color: '#9c27b0', mb: 1 }}>
                      {totalJobs.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Verarbeitete Jobs
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* 1. System-Architektur */}
      <Card sx={{ mb: 4, bgcolor: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <InfoIcon sx={{ color: '#4caf50', fontSize: 30 }} />
            <Typography variant="h4" sx={{ color: '#4caf50', fontWeight: 'bold' }}>
              🏗️ 1. System-Architektur
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ color: '#4caf50' }}>
            ML Training Service - Architektur Übersicht
          </Typography>

          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3, mt: 2 }}>
            <Card sx={{ bgcolor: 'rgba(0, 212, 255, 0.1)', border: '1px solid rgba(0, 212, 255, 0.3)' }}>
              <CardContent>
                <Typography variant="h6" sx={{ color: '#00d4ff', mb: 2 }}>
                  🎯 ML Training Engine
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Funktion:</strong> Modell-Training mit Scikit-learn/XGBoost
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Datenquelle:</strong> PostgreSQL coin_metrics Tabelle
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Modelle:</strong> Random Forest, XGBoost Classifier
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Ausgabe:</strong> ml_models Tabelle + Pickle Dateien
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ bgcolor: 'rgba(255, 152, 0, 0.1)', border: '1px solid rgba(255, 152, 0, 0.3)' }}>
              <CardContent>
                <Typography variant="h6" sx={{ color: '#ff9800', mb: 2 }}>
                  🔄 Job Queue System
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Funktion:</strong> Asynchrone Job-Verarbeitung
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Backend:</strong> RQ (Redis Queue) + Worker
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Jobs:</strong> Training, Testing, Comparison
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Status:</strong> Live-Tracking aller Jobs
                </Typography>
              </CardContent>
            </Card>
          </Box>
        </CardContent>
      </Card>

      {/* 2. Verfügbare Features */}
      <Card sx={{ mb: 4, bgcolor: 'rgba(255, 152, 0, 0.1)', border: '1px solid rgba(255, 152, 0, 0.3)' }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <ScienceIcon sx={{ color: '#ff9800', fontSize: 30 }} />
            <Typography variant="h4" sx={{ color: '#ff9800', fontWeight: 'bold' }}>
              🧪 2. Verfügbare Features
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ color: '#ff9800' }}>
            ML Training Service Features:
          </Typography>

          <Grid container spacing={2}>
            <Grid xs={12} md={6}>
              <Card sx={{ bgcolor: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#4caf50', mb: 2 }}>
                    🚀 Modell Training
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Random Forest & XGBoost Classifier
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Feature Engineering (Rolling Statistics)
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • SMOTE für Imbalanced Data
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Zeitbasierte Vorhersage
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid xs={12} md={6}>
              <Card sx={{ bgcolor: 'rgba(33, 150, 243, 0.1)', border: '1px solid rgba(33, 150, 243, 0.3)' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#2196f3', mb: 2 }}>
                    🧪 Modell Testing
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Out-of-Sample Testing
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Vollständige Metriken (Accuracy, F1, ROC-AUC, MCC)
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Confusion Matrix
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Overfitting Detection
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid xs={12} md={6}>
              <Card sx={{ bgcolor: 'rgba(156, 39, 176, 0.1)', border: '1px solid rgba(156, 39, 176, 0.3)' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#9c27b0', mb: 2 }}>
                    ⚔️ Modell Vergleich
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Side-by-Side Vergleich
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Statistische Signifikanz
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Performance-Metriken
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Winner Detection
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid xs={12} md={6}>
              <Card sx={{ bgcolor: 'rgba(255, 193, 7, 0.1)', border: '1px solid rgba(255, 193, 7, 0.3)' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ color: '#ffc107', mb: 2 }}>
                    📊 Monitoring & Analytics
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Live Job Status Tracking
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • System Health Monitoring
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Performance Metrics
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Prometheus Integration
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* 3. API Dokumentation */}
      <Card sx={{ mb: 4, bgcolor: 'rgba(156, 39, 176, 0.1)', border: '1px solid rgba(156, 39, 176, 0.3)' }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <AnalyticsIcon sx={{ color: '#9c27b0', fontSize: 30 }} />
            <Typography variant="h4" sx={{ color: '#9c27b0', fontWeight: 'bold' }}>
              📚 3. API Dokumentation
            </Typography>
          </Box>

          <Typography variant="h6" sx={{ mb: 2, color: '#9c27b0' }}>
            🌐 Basis-Informationen
          </Typography>
          <Box sx={{ bgcolor: 'rgba(0,0,0,0.2)', p: 2, borderRadius: 1, mb: 3 }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
              <strong>Base-URL:</strong> /api<br/>
              <strong>Protokoll:</strong> HTTP/1.1 + RESTful API<br/>
              <strong>Content-Type:</strong> application/json<br/>
              <strong>Authentication:</strong> Keine erforderlich<br/>
              <strong>Rate-Limiting:</strong> Keines implementiert
            </Typography>
          </Box>

          <Typography variant="h6" sx={{ mb: 2, color: '#9c27b0' }}>
            📋 Wichtige Endpunkte
          </Typography>

          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">🚀 Modell-Management</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer component={Box} sx={{ bgcolor: 'rgba(0,0,0,0.2)', borderRadius: 1 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: '#9c27b0', fontWeight: 'bold' }}>Endpoint</TableCell>
                      <TableCell sx={{ color: '#9c27b0', fontWeight: 'bold' }}>Methode</TableCell>
                      <TableCell sx={{ color: '#9c27b0', fontWeight: 'bold' }}>Beschreibung</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>/api/models</TableCell>
                      <TableCell>GET</TableCell>
                      <TableCell>Alle Modelle auflisten</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>/api/models/create/simple</TableCell>
                      <TableCell>POST</TableCell>
                      <TableCell>Einfaches Modell-Training</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>/api/models/{'{id}'}/test</TableCell>
                      <TableCell>POST</TableCell>
                      <TableCell>Modell testen</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>/api/models/compare</TableCell>
                      <TableCell>POST</TableCell>
                      <TableCell>Modelle vergleichen</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>

          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">📊 Monitoring & Health</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer component={Box} sx={{ bgcolor: 'rgba(0,0,0,0.2)', borderRadius: 1 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: '#9c27b0', fontWeight: 'bold' }}>Endpoint</TableCell>
                      <TableCell sx={{ color: '#9c27b0', fontWeight: 'bold' }}>Methode</TableCell>
                      <TableCell sx={{ color: '#9c27b0', fontWeight: 'bold' }}>Beschreibung</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>/api/health</TableCell>
                      <TableCell>GET</TableCell>
                      <TableCell>System-Status & Live-Daten</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>/api/queue</TableCell>
                      <TableCell>GET</TableCell>
                      <TableCell>Job-Status Übersicht</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>/api/metrics</TableCell>
                      <TableCell>GET</TableCell>
                      <TableCell>Prometheus-Metriken</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>/api/test-results</TableCell>
                      <TableCell>GET</TableCell>
                      <TableCell>Test-Ergebnisse auflisten</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        </CardContent>
      </Card>

      {/* 4. Technische Architektur */}
      <Card sx={{ mb: 4, bgcolor: 'rgba(0, 212, 255, 0.1)', border: '1px solid rgba(0, 212, 255, 0.3)' }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <StorageIcon sx={{ color: '#00d4ff', fontSize: 30 }} />
            <Typography variant="h4" sx={{ color: '#00d4ff', fontWeight: 'bold' }}>
              🔧 4. Technische Architektur
            </Typography>
          </Box>

          <Typography variant="h6" sx={{ mb: 2, color: '#00d4ff' }}>
            Frontend-Technologie-Stack
          </Typography>
          <Box sx={{ bgcolor: 'rgba(0,0,0,0.2)', p: 2, borderRadius: 1, mb: 3 }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
              <strong>Framework:</strong> React 18 + TypeScript<br/>
              <strong>Build-Tool:</strong> Vite 5.4.8<br/>
              <strong>UI-Library:</strong> Material-UI (MUI) v7<br/>
              <strong>State-Management:</strong> Zustand v5.0.9<br/>
              <strong>HTTP-Client:</strong> Axios v1.13.2<br/>
              <strong>Charts:</strong> Recharts v3.6.0<br/>
              <strong>Styling:</strong> Emotion CSS-in-JS
            </Typography>
          </Box>

          <Typography variant="h6" sx={{ mb: 2, color: '#00d4ff' }}>
            Backend-Technologie-Stack
          </Typography>
          <Box sx={{ bgcolor: 'rgba(0,0,0,0.2)', p: 2, borderRadius: 1, mb: 3 }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
              <strong>Framework:</strong> FastAPI (Python)<br/>
              <strong>ML-Libraries:</strong> Scikit-learn, XGBoost<br/>
              <strong>Job Queue:</strong> RQ (Redis Queue)<br/>
              <strong>Datenbank:</strong> PostgreSQL<br/>
              <strong>ORM:</strong> SQLAlchemy<br/>
              <strong>Validierung:</strong> Pydantic
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Footer */}
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          🔄 System läuft seit {formatUptime(uptimeSeconds)} | Verarbeitete Jobs: {totalJobs.toLocaleString()}
          | Datenbank: {dbConnected ? '🟢 Verbunden' : '🔴 Getrennt'}
        </Typography>
      </Box>
    </Container>
  );
};

export default Info;
