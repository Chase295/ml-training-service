import React, { useEffect, useState, useMemo } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ViewList as ModelsIcon,
  FilterList as FilterIcon,
  Search as SearchIcon,
  Clear as ClearIcon,
  SelectAll as SelectAllIcon,
  Delete as DeleteIcon,
  Science as TestIcon,
  Compare as CompareIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useMLStore } from '../stores/mlStore';
import { ModelCard, LoadingSpinner, ErrorDisplay } from '../components';
import type { ModelResponse } from '../types/api';

const Models: React.FC = () => {
  const {
    models,
    selectedModelIds,
    isLoading,
    error,
    fetchModels,
    selectModel,
    deselectModel,
    clearModelSelection,
    deleteModel,
    downloadModel,
  } = useMLStore();

  const [statusFilter, setStatusFilter] = useState<string>('Alle');
  const [typeFilter, setTypeFilter] = useState<string>('Alle');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<ModelResponse | null>(null);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Filter and sort models
  const filteredAndSortedModels = useMemo(() => {
    let filtered = models;

    // Status filter
    if (statusFilter !== 'Alle') {
      filtered = filtered.filter(model => model.status === statusFilter);
    }

    // Type filter
    if (typeFilter !== 'Alle') {
      filtered = filtered.filter(model => model.model_type === typeFilter);
    }

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(model =>
        model.name.toLowerCase().includes(query) ||
        model.model_type.toLowerCase().includes(query) ||
        model.target_variable.toLowerCase().includes(query)
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let aValue: any = a[sortBy as keyof ModelResponse];
      let bValue: any = b[sortBy as keyof ModelResponse];

      // Handle dates
      if (sortBy.includes('_at')) {
        aValue = new Date(aValue).getTime();
        bValue = new Date(bValue).getTime();
      }

      // Handle numbers
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }

      // Handle strings
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      return 0;
    });

    return filtered;
  }, [models, statusFilter, typeFilter, searchQuery, sortBy, sortOrder]);

  const selectedModels = models.filter(model => selectedModelIds.includes(model.id));
  const hasSelection = selectedModelIds.length > 0;

  const handleSelectAll = () => {
    if (selectedModelIds.length === filteredAndSortedModels.length) {
      // Deselect all
      filteredAndSortedModels.forEach(model => deselectModel(model.id));
    } else {
      // Select all filtered
      filteredAndSortedModels.forEach(model => selectModel(model.id));
    }
  };

  const handleBulkDelete = () => {
    selectedModelIds.forEach(id => deleteModel(id));
    clearModelSelection();
  };

  const handleModelClick = (modelId: number) => {
    // Navigate to model details
    window.location.href = `/details/${modelId}`;
  };

  const handleModelTest = (modelId: number) => {
    // Navigate to test page with selected model
    window.location.href = `/test?model=${modelId}`;
  };

  const handleModelCompare = (modelId: number) => {
    // Navigate to compare page with selected model
    window.location.href = `/compare?model_a=${modelId}`;
  };

  const handleModelDownload = async (modelId: number) => {
    try {
      const blob = await downloadModel(modelId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${modelId}.pkl`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  };

  const handleDeleteClick = (model: ModelResponse) => {
    setModelToDelete(model);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (modelToDelete) {
      deleteModel(modelToDelete.id);
      setDeleteDialogOpen(false);
      setModelToDelete(null);
    }
  };

  const getStatusStats = () => {
    const stats = {
      total: models.length,
      ready: models.filter(m => m.status === 'READY').length,
      training: models.filter(m => m.status === 'TRAINING').length,
      failed: models.filter(m => m.status === 'FAILED').length,
    };
    return stats;
  };

  const stats = getStatusStats();

  if (isLoading && models.length === 0) {
    return <LoadingSpinner message="Lade Modelle..." height="60vh" />;
  }

  return (
    <Box sx={{ py: 4, px: { xs: 2, md: 4 } }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ color: '#00d4ff', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
          <ModelsIcon /> Modell-Übersicht
        </Typography>

        <Button
          startIcon={<RefreshIcon />}
          onClick={() => fetchModels()}
          variant="outlined"
          disabled={isLoading}
        >
          Aktualisieren
        </Button>
      </Box>

      {error && <ErrorDisplay error={error} onRetry={() => fetchModels()} />}

      {/* Statistics */}
      <Grid container spacing={2} sx={{ mb: 4 }}>
        <Grid item xs={6} md={3}>
          <Box sx={{
            p: 2,
            bgcolor: 'rgba(0, 212, 255, 0.1)',
            border: '1px solid rgba(0, 212, 255, 0.3)',
            borderRadius: 1,
            textAlign: 'center'
          }}>
            <Typography variant="h4" sx={{ color: '#00d4ff' }}>{stats.total}</Typography>
            <Typography variant="body2" color="textSecondary">Gesamt</Typography>
          </Box>
        </Grid>
        <Grid item xs={6} md={3}>
          <Box sx={{
            p: 2,
            bgcolor: 'rgba(76, 175, 80, 0.1)',
            border: '1px solid rgba(76, 175, 80, 0.3)',
            borderRadius: 1,
            textAlign: 'center'
          }}>
            <Typography variant="h4" sx={{ color: '#4caf50' }}>{stats.ready}</Typography>
            <Typography variant="body2" color="textSecondary">Bereit</Typography>
          </Box>
        </Grid>
        <Grid item xs={6} md={3}>
          <Box sx={{
            p: 2,
            bgcolor: 'rgba(255, 152, 0, 0.1)',
            border: '1px solid rgba(255, 152, 0, 0.3)',
            borderRadius: 1,
            textAlign: 'center'
          }}>
            <Typography variant="h4" sx={{ color: '#ff9800' }}>{stats.training}</Typography>
            <Typography variant="body2" color="textSecondary">Training</Typography>
          </Box>
        </Grid>
        <Grid item xs={6} md={3}>
          <Box sx={{
            p: 2,
            bgcolor: 'rgba(244, 67, 54, 0.1)',
            border: '1px solid rgba(244, 67, 54, 0.3)',
            borderRadius: 1,
            textAlign: 'center'
          }}>
            <Typography variant="h4" sx={{ color: '#f44336' }}>{stats.failed}</Typography>
            <Typography variant="body2" color="textSecondary">Fehlgeschlagen</Typography>
          </Box>
        </Grid>
      </Grid>

      {/* Filters */}
      <Box sx={{
        p: 3,
        mb: 4,
        bgcolor: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 2
      }}>
        <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
          <FilterIcon /> Filter & Suche
        </Typography>

        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              placeholder="Suche nach Name, Typ oder Ziel..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                endAdornment: searchQuery && (
                  <IconButton size="small" onClick={() => setSearchQuery('')}>
                    <ClearIcon />
                  </IconButton>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  '&.Mui-focused fieldset': {
                    borderColor: '#00d4ff',
                  },
                },
              }}
            />
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                label="Status"
              >
                <MenuItem value="Alle">Alle</MenuItem>
                <MenuItem value="READY">Bereit</MenuItem>
                <MenuItem value="TRAINING">Training</MenuItem>
                <MenuItem value="FAILED">Fehlgeschlagen</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Modell-Typ</InputLabel>
              <Select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                label="Modell-Typ"
              >
                <MenuItem value="Alle">Alle</MenuItem>
                <MenuItem value="random_forest">Random Forest</MenuItem>
                <MenuItem value="xgboost">XGBoost</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Sortieren nach</InputLabel>
              <Select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                label="Sortieren nach"
              >
                <MenuItem value="created_at">Erstellt am</MenuItem>
                <MenuItem value="name">Name</MenuItem>
                <MenuItem value="training_accuracy">Accuracy</MenuItem>
                <MenuItem value="model_type">Typ</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={1}>
            <Button
              fullWidth
              variant="outlined"
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </Button>
          </Grid>

          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<ClearIcon />}
              onClick={() => {
                setStatusFilter('Alle');
                setTypeFilter('Alle');
                setSearchQuery('');
                setSortBy('created_at');
                setSortOrder('desc');
              }}
            >
              Zurücksetzen
            </Button>
          </Grid>
        </Grid>
      </Box>

      {/* Bulk Actions */}
      {hasSelection && (
        <Box sx={{
          p: 2,
          mb: 3,
          bgcolor: 'rgba(0, 212, 255, 0.1)',
          border: '1px solid rgba(0, 212, 255, 0.3)',
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}>
          <Typography variant="body1">
            {selectedModelIds.length} Modell(e) ausgewählt:
          </Typography>

          <Button
            startIcon={<CompareIcon />}
            variant="contained"
            color="primary"
            disabled={selectedModelIds.length !== 2}
          >
            Vergleichen
          </Button>

          <Button
            startIcon={<DeleteIcon />}
            variant="contained"
            color="error"
            onClick={handleBulkDelete}
          >
            Alle löschen
          </Button>

          <Button
            variant="outlined"
            onClick={clearModelSelection}
          >
            Auswahl aufheben
          </Button>
        </Box>
      )}

      {/* Results Info */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body1" color="textSecondary">
          {filteredAndSortedModels.length} von {models.length} Modellen angezeigt
        </Typography>

        {filteredAndSortedModels.length > 0 && (
          <Button
            startIcon={<SelectAllIcon />}
            variant="outlined"
            onClick={handleSelectAll}
            size="small"
          >
            {selectedModelIds.length === filteredAndSortedModels.length ? 'Alle abwählen' : 'Alle auswählen'}
          </Button>
        )}
      </Box>

      {/* Models Grid */}
      {filteredAndSortedModels.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <ModelsIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="textSecondary">
            {searchQuery || statusFilter !== 'Alle' || typeFilter !== 'Alle'
              ? 'Keine Modelle entsprechen den Filterkriterien'
              : 'Keine Modelle vorhanden'
            }
          </Typography>
          <Typography variant="body2" color="textSecondary">
            {searchQuery || statusFilter !== 'Alle' || typeFilter !== 'Alle'
              ? 'Versuchen Sie andere Filter oder passen Sie die Suche an.'
              : 'Erstellen Sie Ihr erstes Modell im Training-Bereich.'
            }
          </Typography>
        </Box>
      ) : (
        <Grid container spacing={3}>
          {filteredAndSortedModels.map((model) => (
            <Grid item xs={12} md={6} lg={4} key={model.id}>
              <ModelCard
                model={model}
                isSelected={selectedModelIds.includes(model.id)}
                onSelect={() => selectedModelIds.includes(model.id)
                  ? deselectModel(model.id)
                  : selectModel(model.id)
                }
                onDetails={() => handleModelClick(model.id)}
                onEdit={() => {/* TODO: Implement edit */}}
                onDelete={() => handleDeleteClick(model)}
                onTest={() => handleModelTest(model.id)}
                onDownload={() => handleModelDownload(model.id)}
              />
            </Grid>
          ))}
        </Grid>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Modell löschen</DialogTitle>
        <DialogContent>
          <Typography>
            Sind Sie sicher, dass Sie das Modell "{modelToDelete?.name}" löschen möchten?
            Diese Aktion kann nicht rückgängig gemacht werden.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Abbrechen</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Löschen
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Models;
