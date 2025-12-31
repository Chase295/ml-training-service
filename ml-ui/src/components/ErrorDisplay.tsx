import React from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Typography,
  Paper,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

interface ErrorDisplayProps {
  error: string | Error;
  title?: string;
  severity?: 'error' | 'warning' | 'info';
  onRetry?: () => void;
  onClose?: () => void;
  showDetails?: boolean;
  compact?: boolean;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  title,
  severity = 'error',
  onRetry,
  onClose,
  showDetails = false,
  compact = false,
}) => {
  const [showStackTrace, setShowStackTrace] = React.useState(false);

  const errorMessage = typeof error === 'string' ? error : error.message;
  const errorDetails = typeof error === 'object' && error.stack ? error.stack : null;

  const getIcon = () => {
    switch (severity) {
      case 'error':
        return <ErrorIcon />;
      case 'warning':
        return <WarningIcon />;
      case 'info':
        return <InfoIcon />;
      default:
        return <ErrorIcon />;
    }
  };

  if (compact) {
    return (
      <Alert
        severity={severity}
        action={
          <Box display="flex" gap={1}>
            {onRetry && (
              <IconButton size="small" onClick={onRetry} color="inherit">
                <RefreshIcon fontSize="small" />
              </IconButton>
            )}
            {onClose && (
              <IconButton size="small" onClick={onClose} color="inherit">
                <CloseIcon fontSize="small" />
              </IconButton>
            )}
          </Box>
        }
        sx={{ mb: 2 }}
      >
        <Typography variant="body2">{errorMessage}</Typography>
      </Alert>
    );
  }

  return (
    <Paper
      sx={{
        p: 3,
        mb: 2,
        border: `1px solid ${severity === 'error' ? '#f44336' : severity === 'warning' ? '#ff9800' : '#2196f3'}`,
        backgroundColor: severity === 'error' ? 'rgba(244, 67, 54, 0.05)' :
                        severity === 'warning' ? 'rgba(255, 152, 0, 0.05)' :
                        'rgba(33, 150, 243, 0.05)',
      }}
    >
      <Box display="flex" alignItems="center" mb={2}>
        <Box sx={{ mr: 2, color: severity === 'error' ? '#f44336' : severity === 'warning' ? '#ff9800' : '#2196f3' }}>
          {getIcon()}
        </Box>
        <Box flex={1}>
          <Typography variant="h6" color="textPrimary">
            {title || (severity === 'error' ? 'Fehler' : severity === 'warning' ? 'Warnung' : 'Information')}
          </Typography>
          <Typography variant="body1" color="textSecondary">
            {errorMessage}
          </Typography>
        </Box>
        <Box display="flex" gap={1}>
          {onRetry && (
            <Button
              startIcon={<RefreshIcon />}
              onClick={onRetry}
              variant="outlined"
              size="small"
            >
              Erneut versuchen
            </Button>
          )}
          {onClose && (
            <IconButton onClick={onClose} size="small">
              <CloseIcon />
            </IconButton>
          )}
        </Box>
      </Box>

      {errorDetails && showDetails && (
        <Box>
          <Button
            onClick={() => setShowStackTrace(!showStackTrace)}
            size="small"
            sx={{ mb: 1 }}
          >
            {showStackTrace ? 'Details ausblenden' : 'Technische Details anzeigen'}
          </Button>
          <Collapse in={showStackTrace}>
            <Paper
              sx={{
                p: 2,
                backgroundColor: 'grey.900',
                color: 'grey.100',
                fontFamily: 'monospace',
                fontSize: '0.75rem',
                maxHeight: 300,
                overflow: 'auto',
              }}
            >
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {errorDetails}
              </pre>
            </Paper>
          </Collapse>
        </Box>
      )}
    </Paper>
  );
};

// Specialized error components
export const ApiError: React.FC<{ error: string | Error; onRetry?: () => void }> = ({ error, onRetry }) => (
  <ErrorDisplay
    error={error}
    title="API-Fehler"
    severity="error"
    onRetry={onRetry}
  />
);

export const NetworkError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => (
  <ErrorDisplay
    error="Netzwerkverbindung fehlgeschlagen. Bitte überprüfen Sie Ihre Internetverbindung."
    title="Verbindungsfehler"
    severity="warning"
    onRetry={onRetry}
  />
);

export const ValidationError: React.FC<{ errors: Record<string, string[]> }> = ({ errors }) => (
  <ErrorDisplay
    error={`Validierungsfehler: ${Object.values(errors).flat().join(', ')}`}
    title="Validierungsfehler"
    severity="warning"
  />
);

export default ErrorDisplay;
