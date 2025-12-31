import React from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  LinearProgress,
  Skeleton,
} from '@mui/material';

interface LoadingSpinnerProps {
  message?: string;
  size?: number;
  variant?: 'circular' | 'linear' | 'skeleton';
  height?: string | number;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Lädt...',
  size = 40,
  variant = 'circular',
  height = 100,
}) => {
  if (variant === 'linear') {
    return (
      <Box sx={{ width: '100%', mb: 2 }}>
        <LinearProgress />
        {message && (
          <Typography variant="body2" color="textSecondary" sx={{ mt: 1, textAlign: 'center' }}>
            {message}
          </Typography>
        )}
      </Box>
    );
  }

  if (variant === 'skeleton') {
    return (
      <Box sx={{ width: '100%', p: 2 }}>
        <Skeleton variant="rectangular" height={height} sx={{ mb: 2, borderRadius: 1 }} />
        <Skeleton variant="text" sx={{ mb: 1 }} />
        <Skeleton variant="text" width="60%" />
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: height,
        py: 4,
      }}
    >
      <CircularProgress size={size} sx={{ color: '#00d4ff', mb: 2 }} />
      {message && (
        <Typography variant="body1" color="textSecondary">
          {message}
        </Typography>
      )}
    </Box>
  );
};

// Specialized loading components
export const PageLoading: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingSpinner
    message={message || 'Seite wird geladen...'}
    variant="circular"
    height="60vh"
  />
);

export const CardLoading: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingSpinner
    message={message || 'Daten werden geladen...'}
    variant="circular"
    height={200}
    size={30}
  />
);

export const TableLoading: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingSpinner
    message={message || 'Tabelle wird geladen...'}
    variant="linear"
  />
);

export default LoadingSpinner;
