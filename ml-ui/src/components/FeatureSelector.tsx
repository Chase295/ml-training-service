import React from 'react';
import {
  FormControl,
  FormHelperText,
  Checkbox,
  FormControlLabel,
  Typography,
  Box,
  Button,
  Chip,
} from '@mui/material';

// Feature Selector für Modell-Features
interface FeatureCategory {
  name: string;
  features: Array<{
    id: string;
    label: string;
    description?: string;
  }>;
}

interface FeatureSelectorProps {
  label: string;
  categories: FeatureCategory[];
  selectedFeatures: string[];
  onChange: (features: string[]) => void;
  error?: string;
  helperText?: string;
  disabled?: boolean;
}

export const FeatureSelector: React.FC<FeatureSelectorProps> = ({
  label,
  categories,
  selectedFeatures,
  onChange,
  error,
  helperText,
  disabled = false,
}) => {
  const handleFeatureToggle = (featureId: string) => {
    if (disabled) return;

    const newSelected = selectedFeatures.includes(featureId)
      ? selectedFeatures.filter(id => id !== featureId)
      : [...selectedFeatures, featureId];

    onChange(newSelected);
  };

  const handleCategoryToggle = (categoryFeatures: Array<{id: string}>) => {
    if (disabled) return;

    const categoryIds = categoryFeatures.map(f => f.id);
    const allSelected = categoryIds.every(id => selectedFeatures.includes(id));
    const noneSelected = categoryIds.every(id => !selectedFeatures.includes(id));

    let newSelected: string[];

    if (allSelected) {
      // Alle abwählen
      newSelected = selectedFeatures.filter(id => !categoryIds.includes(id));
    } else {
      // Alle auswählen (überschreibt einzelne Selektionen)
      newSelected = [...selectedFeatures.filter(id => !categoryIds.includes(id)), ...categoryIds];
    }

    onChange(newSelected);
  };

  const handleSelectAll = () => {
    if (disabled) return;

    const allFeatureIds = categories.flatMap(cat => cat.features.map(f => f.id));
    const allSelected = allFeatureIds.every(id => selectedFeatures.includes(id));

    if (allSelected) {
      onChange([]);
    } else {
      onChange(allFeatureIds);
    }
  };

  const handleSelectAuto = () => {
    if (disabled) return;
    onChange(['auto']);
  };

  return (
    <FormControl fullWidth error={!!error} disabled={disabled}>
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="subtitle1" sx={{ color: 'text.primary', fontWeight: 600 }}>
            {label}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              variant="outlined"
              onClick={handleSelectAuto}
              disabled={disabled}
              sx={{
                minWidth: 'auto',
                px: 1.5,
                py: 0.5,
                fontSize: '0.75rem',
                borderColor: selectedFeatures.includes('auto') ? '#00d4ff' : 'grey.400',
                color: selectedFeatures.includes('auto') ? '#00d4ff' : 'text.secondary',
              }}
            >
              Auto
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={handleSelectAll}
              disabled={disabled}
              sx={{
                minWidth: 'auto',
                px: 1.5,
                py: 0.5,
                fontSize: '0.75rem',
                borderColor: selectedFeatures.length > 0 && !selectedFeatures.includes('auto') ? '#00d4ff' : 'grey.400',
                color: selectedFeatures.length > 0 && !selectedFeatures.includes('auto') ? '#00d4ff' : 'text.secondary',
              }}
            >
              Alle
            </Button>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {categories.map((category) => {
            const categoryFeatures = category.features;
            const categoryIds = categoryFeatures.map(f => f.id);
            const allSelected = categoryIds.every(id => selectedFeatures.includes(id));
            const someSelected = categoryIds.some(id => selectedFeatures.includes(id));

            return (
              <Box key={category.name} sx={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 1, p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={allSelected}
                        indeterminate={someSelected && !allSelected}
                        onChange={() => handleCategoryToggle(categoryFeatures)}
                        disabled={disabled}
                        sx={{
                          color: 'grey.400',
                          '&.Mui-checked': { color: '#00d4ff' },
                          '&.MuiCheckbox-indeterminate': { color: '#00d4ff' },
                        }}
                      />
                    }
                    label={
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                        {category.name}
                      </Typography>
                    }
                    sx={{ margin: 0 }}
                  />
                </Box>

                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 1, ml: 4 }}>
                  {categoryFeatures.map((feature) => (
                    <FormControlLabel
                      key={feature.id}
                      control={
                        <Checkbox
                          checked={selectedFeatures.includes(feature.id)}
                          onChange={() => handleFeatureToggle(feature.id)}
                          disabled={disabled}
                          sx={{
                            color: 'grey.400',
                            '&.Mui-checked': { color: '#00d4ff' },
                          }}
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2" sx={{ color: 'text.primary' }}>
                            {feature.label}
                          </Typography>
                          {feature.description && (
                            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
                              {feature.description}
                            </Typography>
                          )}
                        </Box>
                      }
                      sx={{ alignItems: 'flex-start', margin: 0 }}
                    />
                  ))}
                </Box>
              </Box>
            );
          })}
        </Box>

        {selectedFeatures.length > 0 && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'rgba(0, 212, 255, 0.1)', borderRadius: 1 }}>
            <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 500, mb: 1 }}>
              Ausgewählte Features ({selectedFeatures.length}):
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {selectedFeatures.map((featureId) => {
                let featureLabel = featureId;
                for (const category of categories) {
                  const feature = category.features.find(f => f.id === featureId);
                  if (feature) {
                    featureLabel = feature.label;
                    break;
                  }
                }
                return (
                  <Chip
                    key={featureId}
                    label={featureLabel}
                    size="small"
                    onDelete={() => handleFeatureToggle(featureId)}
                    sx={{
                      bgcolor: 'rgba(0, 212, 255, 0.2)',
                      color: '#00d4ff',
                      border: '1px solid rgba(0, 212, 255, 0.3)',
                      '& .MuiChip-deleteIcon': {
                        color: '#00d4ff',
                      },
                    }}
                  />
                );
              })}
            </Box>
          </Box>
        )}

        {(error || helperText) && (
          <FormHelperText error={!!error}>
            {error || helperText}
          </FormHelperText>
        )}
      </Box>
    </FormControl>
  );
};

export default FeatureSelector;
