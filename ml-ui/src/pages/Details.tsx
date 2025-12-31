import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Container, Typography, Paper, Box, Button, Chip,
  Grid, Card, CardContent, Alert, CircularProgress
} from '@mui/material'
import { ArrowBack } from '@mui/icons-material'
import { useMLStore } from '../stores/mlStore'

const Details: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [jobDetails, setJobDetails] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchJobDetails = async () => {
      if (!id) return
      
      try {
        // Mock data - ersetze mit echter API
        const mockJob = {
          id: id,
          status: 'completed',
          created_at: '2024-01-01T10:00:00Z',
          config: {
            model_type: 'xgboost',
            target_coin: 'SOL',
            features: ['price', 'volume'],
            epochs: 100
          },
          results: {
            model_id: 'model_' + id,
            metrics: {
              accuracy: 0.85,
              precision: 0.88,
              recall: 0.82,
              f1_score: 0.85
            },
            training_time: '45m 30s'
          }
        }
        setJobDetails(mockJob)
      } catch (err) {
        setError('Fehler beim Laden der Job-Details')
      } finally {
        setLoading(false)
      }
    }

    fetchJobDetails()
  }, [id])

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    )
  }

  if (error || !jobDetails) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error || 'Job nicht gefunden'}</Alert>
        <Button startIcon={<ArrowBack />} onClick={() => navigate('/jobs')} sx={{ mt: 2 }}>
          Zurück zu Jobs
        </Button>
      </Container>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'primary'
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'pending': return 'warning'
      default: return 'default'
    }
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Button startIcon={<ArrowBack />} onClick={() => navigate('/jobs')}>
          Zurück zu Jobs
        </Button>
      </Box>

      <Typography variant="h4" gutterBottom>
        Job Details: {jobDetails.id.slice(0, 8)}
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Job Informationen
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography><strong>Status:</strong> 
                <Chip 
                  label={jobDetails.status} 
                  color={getStatusColor(jobDetails.status) as any}
                  size="small"
                  sx={{ ml: 1 }}
                />
              </Typography>
            </Box>
            <Typography><strong>Erstellt:</strong> {new Date(jobDetails.created_at).toLocaleString()}</Typography>
            <Typography><strong>Modell-Typ:</strong> {jobDetails.config.model_type}</Typography>
            <Typography><strong>Target Coin:</strong> {jobDetails.config.target_coin}</Typography>
            <Typography><strong>Epochen:</strong> {jobDetails.config.epochs}</Typography>
          </Paper>
        </Grid>

        {jobDetails.results && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Ergebnisse
              </Typography>
              <Typography><strong>Training Zeit:</strong> {jobDetails.results.training_time}</Typography>
              <Typography><strong>Modell ID:</strong> {jobDetails.results.model_id.slice(0, 8)}</Typography>
              
              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>
                Metriken:
              </Typography>
              {Object.entries(jobDetails.results.metrics).map(([key, value]) => (
                <Typography key={key}>
                  <strong>{key.replace('_', ' ').toUpperCase()}:</strong> {(Number(value) * 100).toFixed(1)}%
                </Typography>
              ))}
            </Paper>
          </Grid>
        )}

        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Konfiguration Details
            </Typography>
            <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1, fontFamily: 'monospace' }}>
              <pre>{JSON.stringify(jobDetails.config, null, 2)}</pre>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}

export default Details
