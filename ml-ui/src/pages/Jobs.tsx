import React, { useEffect } from 'react'
import {
  Container, Typography, Paper, Table, TableBody, 
  TableCell, TableContainer, TableHead, TableRow,
  Button, Chip, Box, Alert
} from '@mui/material'
import { useMLStore } from '../stores/mlStore'
import { useNavigate } from 'react-router-dom'

const Jobs: React.FC = () => {
  const { jobs, loading, error, fetchJobs } = useMLStore()
  const navigate = useNavigate()

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Jobs</Typography>
        <Button variant="contained" onClick={() => navigate('/training')}>
          Neuer Job
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Job ID</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Modell-Typ</TableCell>
              <TableCell>Erstellt</TableCell>
              <TableCell>Aktionen</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobs.map((job) => (
              <TableRow key={job.id}>
                <TableCell>{job.id.slice(0, 8)}</TableCell>
                <TableCell>
                  <Chip 
                    label={job.status} 
                    color={getStatusColor(job.status) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>{job.config?.model_type || 'N/A'}</TableCell>
                <TableCell>
                  {new Date(job.created_at).toLocaleString()}
                </TableCell>
                <TableCell>
                  <Button 
                    size="small" 
                    onClick={() => navigate(`/details/${job.id}`)}
                  >
                    Details
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Container>
  )
}

export default Jobs
