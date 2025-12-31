import React from 'react';
import { Chip } from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Schedule as PendingIcon,
  PlayArrow as RunningIcon,
  Stop as StoppedIcon,
} from '@mui/icons-material';

interface StatusChipProps {
  status: string;
  size?: 'small' | 'medium';
  variant?: 'filled' | 'outlined';
  showIcon?: boolean;
}

export const StatusChip: React.FC<StatusChipProps> = ({
  status,
  size = 'small',
  variant = 'filled',
  showIcon = true,
}) => {
  const getStatusConfig = (status: string) => {
    const statusLower = status.toLowerCase();

    switch (statusLower) {
      case 'ready':
      case 'healthy':
      case 'completed':
      case 'success':
        return {
          color: 'success' as const,
          icon: showIcon ? <SuccessIcon /> : undefined,
          label: status,
        };

      case 'training':
      case 'running':
      case 'pending':
        return {
          color: 'warning' as const,
          icon: showIcon ? (statusLower === 'training' || statusLower === 'running' ? <RunningIcon /> : <PendingIcon />) : undefined,
          label: status,
        };

      case 'failed':
      case 'error':
      case 'unhealthy':
        return {
          color: 'error' as const,
          icon: showIcon ? <ErrorIcon /> : undefined,
          label: status,
        };

      case 'stopped':
        return {
          color: 'default' as const,
          icon: showIcon ? <StoppedIcon /> : undefined,
          label: status,
        };

      case 'degraded':
      case 'warning':
        return {
          color: 'warning' as const,
          icon: showIcon ? <WarningIcon /> : undefined,
          label: status,
        };

      default:
        return {
          color: 'default' as const,
          icon: undefined,
          label: status,
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <Chip
      label={config.label}
      color={config.color}
      size={size}
      variant={variant}
      icon={config.icon}
      sx={{
        fontWeight: 'medium',
        textTransform: 'capitalize',
      }}
    />
  );
};

export default StatusChip;
