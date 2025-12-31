import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
} from '@mui/material';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  color?: string;
  loading?: boolean;
  helpText?: string;
  trend?: {
    value: number;
    direction: 'up' | 'down' | 'neutral';
  };
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color = '#00d4ff',
  loading = false,
  helpText,
  trend,
}) => {
  const getTrendIcon = () => {
    if (!trend) return null;

    const { direction, value: trendValue } = trend;
    const color = direction === 'up' ? '#4caf50' : direction === 'down' ? '#f44336' : '#9e9e9e';
    const arrow = direction === 'up' ? '↑' : direction === 'down' ? '↓' : '→';

    return (
      <Typography variant="caption" sx={{ color, ml: 1 }}>
        {arrow} {Math.abs(trendValue)}%
      </Typography>
    );
  };

  return (
    <Card
      sx={{
        background: `linear-gradient(135deg, ${color}15 0%, ${color}08 100%)`,
        border: `1px solid ${color}30`,
        backdropFilter: 'blur(10px)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {loading && <LinearProgress sx={{ mb: 1 }} />}

        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 500 }}>
            {title}
          </Typography>
          {icon && (
            <Box sx={{ color, opacity: 0.8 }}>
              {icon}
            </Box>
          )}
        </Box>

        <Box flex={1} display="flex" flexDirection="column" justifyContent="center">
          <Box display="flex" alignItems="baseline" mb={0.5}>
            <Typography
              variant="h4"
              sx={{
                color,
                fontWeight: 'bold',
                fontSize: '1.5rem',
              }}
            >
              {typeof value === 'number' ? value.toLocaleString() : value}
            </Typography>
            {getTrendIcon()}
          </Box>

          {subtitle && (
            <Typography
              variant="body2"
              color="textSecondary"
              sx={{ mt: 0.5, fontSize: '0.75rem' }}
            >
              {subtitle}
            </Typography>
          )}

          {helpText && (
            <Typography
              variant="caption"
              color="textSecondary"
              sx={{
                mt: 1,
                fontSize: '0.7rem',
                opacity: 0.7,
                fontStyle: 'italic'
              }}
            >
              {helpText}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default MetricCard;
