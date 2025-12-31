import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  Checkbox,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Alert,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Science as TestIcon,
  Compare as CompareIcon,
  Visibility as DetailsIcon,
} from '@mui/icons-material';
import { ModelResponse } from '../types/api';
import { useMLStore } from '../stores/mlStore';

interface ModelCardProps {
  model: ModelResponse;
  isSelected: boolean;
  onSelect: (modelId: number) => void;
  onDetails: (modelId: number) => void;
  onEdit: (modelId: number) => void;
  onDelete: (modelId: number) => void;
  onDownload: (modelId: number) => void;
  onTest: (modelId: number) => void;
  compact?: boolean;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  isSelected,
  onSelect,
  onDetails,
  onEdit,
  onDelete,
  onDownload,
  onTest,
  compact = false,
}) => {
  const [editDialogOpen, setEditDialogOpen] = React.useState(false);
  const [newName, setNewName] = React.useState(model.name);
  const [newDescription, setNewDescription] = React.useState(model.description || '');
  const [editing, setEditing] = React.useState(false);

  const handleEdit = () => {
    setEditDialogOpen(true);
  };

  const handleEditSave = async () => {
    // TODO: Implement edit functionality
    setEditDialogOpen(false);
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'ready': return 'success';
      case 'training': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getModelTypeEmoji = (modelType: string) => {
    return modelType === 'xgboost' ? '🚀' : '🌲';
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('de-DE', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  const formatDuration = (start: string, end: string) => {
    try {
      const startDate = new Date(start);
      const endDate = new Date(end);
      const diffDays = Math.round((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
      return `${diffDays} Tage`;
    } catch {
      return 'N/A';
    }
  };

  return (
    <>
      <Card
        sx={{
          border: isSelected ? '2px solid #00d4ff' : '1px solid #e0e0e0',
          backgroundColor: isSelected ? 'rgba(0, 212, 255, 0.05)' : 'white',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: 2,
            transform: 'translateY(-2px)',
          },
        }}
      >
        <CardContent sx={{ p: compact ? 2 : 3 }}>
          {/* Header Row */}
          <Box display="flex" alignItems="center" mb={2}>
            <Checkbox
              checked={isSelected}
              onChange={() => onSelect(model.id)}
              sx={{ mr: 1 }}
            />
            <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
              {model.name}
            </Typography>
            <Box display="flex" gap={1}>
              <IconButton size="small" onClick={() => onDetails(model.id)} title="Details">
                <DetailsIcon />
              </IconButton>
              <IconButton size="small" onClick={handleEdit} title="Bearbeiten">
                <EditIcon />
              </IconButton>
              <IconButton size="small" onClick={() => onDownload(model.id)} title="Download">
                <DownloadIcon />
              </IconButton>
            </Box>
          </Box>

          {/* Status and Type */}
          <Box display="flex" alignItems="center" gap={2} mb={2}>
            <Chip
              label={model.status}
              color={getStatusColor(model.status)}
              size="small"
            />
            <Typography variant="body2" color="textSecondary">
              {getModelTypeEmoji(model.model_type)} {model.model_type}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              ID: {model.id}
            </Typography>
          </Box>

          {/* Metrics Grid */}
          {!compact && (
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Box sx={{ flex: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {model.training_accuracy?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Accuracy
                </Typography>
              </Box>
              <Box sx={{ flex: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {model.training_f1?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  F1-Score
                </Typography>
              </Box>
              <Box sx={{ flex: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {model.roc_auc?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  ROC-AUC
                </Typography>
              </Box>
              <Box sx={{ flex: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {model.mcc?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  MCC
                </Typography>
              </Box>
            </Box>
          )}

          {/* Target Information */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              🎯 Ziel: {model.target_variable}
              {model.target_operator && model.target_value !== undefined &&
                ` ${model.target_operator} ${model.target_value}`
              }
            </Typography>

            {/* Time-based prediction info */}
            {(model.future_minutes && model.min_percent_change) && (
              <Typography variant="body2" color="primary" sx={{ fontWeight: 'bold' }}>
                ⏰ Zeitbasierte Vorhersage: {model.future_minutes}min,
                {model.min_percent_change}% {model.target_direction === 'up' ? 'steigt' : model.target_direction === 'down' ? 'fällt' : ''}
              </Typography>
            )}
          </Box>

          {/* Training Period */}
          <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="caption" color="textSecondary">
              📅 Training: {formatDate(model.train_start)} → {formatDate(model.train_end)}
            </Typography>
            <Typography variant="caption" color="textSecondary">
              ({formatDuration(model.train_start, model.train_end)})
            </Typography>
          </Box>

          {/* Features and Created */}
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="caption" color="textSecondary">
              📊 {model.features?.length || 0} Features
            </Typography>
            <Typography variant="caption" color="textSecondary">
              🕐 {formatDate(model.created_at)}
            </Typography>
          </Box>

          {/* Actions Row */}
          <Box display="flex" gap={1} sx={{ mt: 2 }}>
            <Button
              size="small"
              variant="outlined"
              startIcon={<TestIcon />}
              onClick={() => onTest(model.id)}
              disabled={model.status !== 'READY'}
            >
              Testen
            </Button>
            <Button
              size="small"
              variant="outlined"
              startIcon={<CompareIcon />}
              onClick={() => onTest(model.id)} // TODO: Implement compare
              disabled={model.status !== 'READY'}
            >
              Vergleichen
            </Button>
            <Button
              size="small"
              color="error"
              variant="outlined"
              startIcon={<DeleteIcon />}
              onClick={() => onDelete(model.id)}
            >
              Löschen
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Modell bearbeiten</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            sx={{ mt: 2 }}
          />
          <TextField
            fullWidth
            label="Beschreibung"
            multiline
            rows={3}
            value={newDescription}
            onChange={(e) => setNewDescription(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Abbrechen</Button>
          <Button onClick={handleEditSave} variant="contained">
            Speichern
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ModelCard;
