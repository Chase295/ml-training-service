import React, { useState, useEffect, useMemo } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Button,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Divider,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  FormControlLabel,
  Checkbox,
  Grid,
} from '@mui/material';
import {
  Add as AddIcon,
  ExpandMore as ExpandMoreIcon,
  Timeline as TimelineIcon,
  Science as ScienceIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useMLStore } from '../stores/mlStore';
import {
  ValidatedTextField,
  ValidatedSelect,
  FeatureSelector,
  ValidatedDateTimePicker,
  DateProvider,
} from '../components';
import { TrainModelRequest, SimpleTrainModelRequest } from '../types/api';

const steps = ['Basis-Konfiguration', 'Ziel-Definition', 'Features & Engineering', 'Erweiterte Optionen'];

const Training: React.FC = () => {
  const {
    createSimpleModel,
    createTimeBasedModel,
    createModel,
    dataAvailability,
    isLoading,
    error,
    fetchDataAvailability,
    fetchPhases,
  } = useMLStore();

  const [activeStep, setActiveStep] = useState(0);
  const [useSimpleMode, setUseSimpleMode] = useState(true);
  const [formData, setFormData] = useState<Partial<TrainModelRequest>>({
    name: '',
    model_type: 'xgboost',
    use_time_based_prediction: false,
    features: ['price_open', 'price_high', 'price_low', 'price_close', 'volume_sol', 'market_cap_close'],
    train_start: undefined,
    train_end: undefined,
    description: '',
    use_engineered_features: false,
    use_smote: true,
    use_timeseries_split: true,
    cv_splits: 5,
    // Default values for time-based prediction
    future_minutes: 15,
    min_percent_change: 0.05,
    direction: 'up',
  });

  // Feature-Kategorien
  const featureCategories = useMemo(() => ({
    'Preis-Daten': [
      'price_open', 'price_high', 'price_low', 'price_close',
      'price_change_1h', 'price_change_24h', 'price_volatility'
    ],
    'Handels-Volumen': [
      'volume_sol', 'volume_usd', 'volume_change_1h', 'volume_change_24h'
    ],
    'Market Cap': [
      'market_cap_close', 'market_cap_change_1h', 'market_cap_change_24h'
    ],
    'Technische Indikatoren': [
      'rsi_14', 'macd', 'bollinger_upper', 'bollinger_lower',
      'ema_12', 'ema_26', 'sma_20', 'sma_50'
    ],
    'Pump-Detection': [
      'pump_probability', 'social_mentions', 'whale_transactions',
      'liquidity_ratio', 'bonding_curve_progress'
    ],
    'Zeitbasierte Features': [
      'hour_of_day', 'day_of_week', 'age_seconds', 'time_since_launch'
    ],
  }), []);

  useEffect(() => {
    fetchDataAvailability();
    fetchPhases();
  }, [fetchDataAvailability, fetchPhases]);

  const handleFieldChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleFeaturesChange = (features: string[]) => {
    setFormData(prev => ({ ...prev, features }));
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleSubmit = async () => {
    try {
      if (useSimpleMode) {
        // Simple mode
        const simpleRequest: SimpleTrainModelRequest = {
          name: formData.name!,
          model_type: formData.model_type!,
          target: `${formData.target_var} ${formData.operator} ${formData.target_value}`,
          features: formData.features!,
          train_start: formData.train_start!,
          train_end: formData.train_end!,
          description: formData.description,
        };
        await createSimpleModel(simpleRequest);
      } else if (formData.use_time_based_prediction) {
        // Time-based prediction
        await createTimeBasedModel(formData);
      } else {
        // Full training
        const fullRequest: TrainModelRequest = formData as TrainModelRequest;
        await createModel(fullRequest);
      }
    } catch (error) {
      console.error('Training submission failed:', error);
    }
  };

  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 0: // Basis-Konfiguration
        return !!(formData.name && formData.model_type);
      case 1: // Ziel-Definition
        if (formData.use_time_based_prediction) {
          // Für zeitbasierte Vorhersage reichen die Default-Werte
          const isValid = !!(formData.future_minutes !== undefined && formData.min_percent_change !== undefined && formData.direction);
          console.log('Time-based validation:', {
            future_minutes: formData.future_minutes,
            min_percent_change: formData.min_percent_change,
            direction: formData.direction,
            isValid
          });
          return isValid;
        } else {
          return !!(formData.target_var && formData.operator && formData.target_value !== undefined);
        }
      case 2: // Features
        return !!(formData.features && formData.features.length > 0 && formData.train_start && formData.train_end);
      case 3: // Erweiterte Optionen
        return true;
      default:
        return false;
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0: // Basis-Konfiguration
        return (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
              <SettingsIcon /> Basis-Konfiguration
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <ValidatedTextField
                  label="Modell-Name"
                  value={formData.name || ''}
                  onChange={(value) => handleFieldChange('name', value)}
                  required
                  placeholder="z.B. PumpDetector_v1"
                  helperText="Eindeutiger Name für Ihr Modell"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ValidatedSelect
                  label="Modell-Typ"
                  value={formData.model_type || 'xgboost'}
                  onChange={(value) => handleFieldChange('model_type', value)}
                  options={[
                    { value: 'random_forest', label: '🌲 Random Forest (Robust)' },
                    { value: 'xgboost', label: '🚀 XGBoost (Beste Performance)' },
                  ]}
                  helperText="Random Forest: Stabil, XGBoost: Höhere Accuracy möglich"
                />
              </Grid>

              <Grid item xs={12}>
                <ValidatedTextField
                  label="Beschreibung (optional)"
                  value={formData.description || ''}
                  onChange={(value) => handleFieldChange('description', value)}
                  multiline
                  rows={2}
                  placeholder="Kurze Beschreibung des Modells..."
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 3 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={useSimpleMode}
                    onChange={(e) => setUseSimpleMode(e.target.checked)}
                    color="primary"
                  />
                }
                label="Vereinfachten Modus verwenden (empfohlen für Anfänger)"
              />
            </Box>
          </Box>
        );

      case 1: // Ziel-Definition
        return (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
              <TimelineIcon /> Ziel-Definition
            </Typography>

            <FormControlLabel
              control={
                <Checkbox
                  checked={formData.use_time_based_prediction || false}
                  onChange={(e) => handleFieldChange('use_time_based_prediction', (e.target as any).checked)}
                  color="primary"
                />
              }
              label="Zeitbasierte Vorhersage aktivieren (empfohlen)"
            />

            {formData.use_time_based_prediction ? (
              // Zeitbasierte Vorhersage
              <Box sx={{ mt: 3 }}>
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Typography variant="body2">
                    <strong>Zeitbasierte Vorhersage:</strong> Das Modell lernt, wann ein Coin in der Zukunft
                    einen bestimmten prozentualen Anstieg haben wird.
                  </Typography>
                </Alert>

                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <ValidatedTextField
                      label="Minuten in die Zukunft"
                      value={formData.future_minutes?.toString() || '15'}
                      onChange={(value) => handleFieldChange('future_minutes', parseInt(value) || 15)}
                      type="number"
                      helperText="Zeitraum für die Vorhersage (z.B. 15 Minuten)"
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <ValidatedTextField
                      label="Mindest-Änderung (%)"
                      value={formData.min_percent_change?.toString() || '5.0'}
                      onChange={(value) => handleFieldChange('min_percent_change', parseFloat(value) || 0.05)}
                      type="number"
                      step="0.01"
                      helperText="Prozentuale Änderung (z.B. 5.0 für 5%)"
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <ValidatedSelect
                      label="Richtung"
                      value={formData.direction || 'up'}
                      onChange={(value) => handleFieldChange('direction', value)}
                      options={[
                        { value: 'up', label: '📈 Steigt an' },
                        { value: 'down', label: '📉 Fällt ab' },
                        { value: 'both', label: '↕️ Beide Richtungen' },
                      ]}
                    />
                  </Grid>
                </Grid>
              </Box>
            ) : (
              // Regelbasierte Vorhersage
              <Box sx={{ mt: 3 }}>
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Typography variant="body2">
                    <strong>Regelbasierte Vorhersage:</strong> Das Modell lernt einfache Regeln wie
                    "price_close &gt; 0.05" (Preis steigt um mehr als 5%).
                  </Typography>
                </Alert>

                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <ValidatedSelect
                      label="Ziel-Variable"
                      value={formData.target_var || 'price_close'}
                      onChange={(value) => handleFieldChange('target_var', value)}
                      options={[
                        { value: 'price_close', label: 'Schlusskurs' },
                        { value: 'price_change_1h', label: '1h Preisänderung' },
                        { value: 'market_cap_close', label: 'Market Cap' },
                        { value: 'volume_sol', label: 'Handelsvolumen' },
                      ]}
                      required
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <ValidatedSelect
                      label="Operator"
                      value={formData.operator || '>'}
                      onChange={(value) => handleFieldChange('operator', value)}
                      options={[
                        { value: '>', label: 'größer als' },
                        { value: '<', label: 'kleiner als' },
                        { value: '>=', label: 'größer/gleich' },
                        { value: '<=', label: 'kleiner/gleich' },
                        { value: '=', label: 'gleich' },
                      ]}
                      required
                    />
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <ValidatedTextField
                      label="Schwellwert"
                      value={formData.target_value?.toString() || ''}
                      onChange={(value) => handleFieldChange('target_value', parseFloat(value) || 0)}
                      type="number"
                      step="0.01"
                      required
                      helperText="Zahlenwert für die Regel (z.B. 0.05 für 5%)"
                    />
                  </Grid>
                </Grid>

                <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
                  Regel: <strong>{formData.target_var} {formData.operator} {formData.target_value}</strong>
                </Typography>
              </Box>
            )}
          </Box>
        );

      case 2: // Features & Engineering
        return (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: 1 }}>
              <ScienceIcon /> Features & Engineering
            </Typography>

            {/* Zeitraum-Auswahl */}
            <Card sx={{ mb: 3, bgcolor: 'rgba(0, 212, 255, 0.1)', border: '1px solid rgba(0, 212, 255, 0.3)' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>📅 Training-Zeitraum</Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <DateProvider>
                      <ValidatedDateTimePicker
                        label="Start-Zeitpunkt"
                        value={formData.train_start ? new Date(formData.train_start) : null}
                        onChange={(date) => handleFieldChange('train_start', date?.toISOString())}
                        required
                        minDate={dataAvailability?.min_timestamp ? new Date(dataAvailability.min_timestamp) : undefined}
                        maxDate={dataAvailability?.max_timestamp ? new Date(dataAvailability.max_timestamp) : undefined}
                      />
                    </DateProvider>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <DateProvider>
                      <ValidatedDateTimePicker
                        label="End-Zeitpunkt"
                        value={formData.train_end ? new Date(formData.train_end) : null}
                        onChange={(date) => handleFieldChange('train_end', date?.toISOString())}
                        required
                        minDate={dataAvailability?.min_timestamp ? new Date(dataAvailability.min_timestamp) : undefined}
                        maxDate={dataAvailability?.max_timestamp ? new Date(dataAvailability.max_timestamp) : undefined}
                      />
                    </DateProvider>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Feature-Auswahl */}
            <FeatureSelector
              label="📊 Features auswählen"
              categories={featureCategories}
              selectedFeatures={formData.features || []}
              onChange={handleFeaturesChange}
              helperText="Wählen Sie die Features aus, die Ihr Modell verwenden soll"
            />

            {/* Feature Engineering */}
            <Box sx={{ mt: 3 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData.use_engineered_features || false}
                    onChange={(e) => handleFieldChange('use_engineered_features', (e.target as any).checked)}
                    color="primary"
                  />
                }
                label="Erweiterte Pump-Detection Features verwenden"
              />

              {formData.use_engineered_features && (
                <Box sx={{ ml: 4, mt: 2 }}>
                  <ValidatedTextField
                    label="Rolling Window Größen"
                    value={formData.feature_engineering_windows?.join(', ') || '5, 10, 15'}
                    onChange={(value) => handleFieldChange('feature_engineering_windows', value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)))}
                    helperText="Komma-separierte Werte für Rolling Statistics (z.B. 5, 10, 15)"
                  />
                </Box>
              )}
            </Box>
          </Box>
        );

      case 3: // Erweiterte Optionen
        return (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ color: '#00d4ff' }}>
              ⚙️ Erweiterte Optionen
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.use_smote || true}
                      onChange={(e) => handleFieldChange('use_smote', (e.target as any).checked)}
                      color="primary"
                    />
                  }
                  label="SMOTE für Imbalanced Data (empfohlen)"
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.use_timeseries_split || true}
                      onChange={(e) => handleFieldChange('use_timeseries_split', (e.target as any).checked)}
                      color="primary"
                    />
                  }
                  label="TimeSeriesSplit für Cross-Validation (empfohlen für Zeitreihen)"
                />
              </Grid>

              {formData.use_timeseries_split && (
                <Grid item xs={12} md={6}>
                  <ValidatedTextField
                    label="CV Splits"
                    value={formData.cv_splits?.toString() || '5'}
                    onChange={(value) => handleFieldChange('cv_splits', parseInt(value) || 5)}
                    type="number"
                    helperText="Anzahl der Cross-Validation Splits"
                  />
                </Grid>
              )}

              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={formData.use_market_context || false}
                      onChange={(e) => handleFieldChange('use_market_context', (e.target as any).checked)}
                      color="primary"
                    />
                  }
                  label="Marktstimmung einbeziehen (SOL-Preis, etc.)"
                />
              </Grid>
            </Grid>

            {/* Hyperparameter */}
            <Accordion sx={{ mt: 3 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>🔧 Hyperparameter (optional)</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <ValidatedTextField
                      label="Max Depth (XGBoost)"
                      value={formData.params?.max_depth?.toString() || ''}
                      onChange={(value) => handleFieldChange('params', { ...formData.params, max_depth: parseInt(value) || 6 })}
                      type="number"
                      helperText="Maximale Baumtiefe (3-10, Default: 6)"
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <ValidatedTextField
                      label="Learning Rate (XGBoost)"
                      value={formData.params?.learning_rate?.toString() || ''}
                      onChange={(value) => handleFieldChange('params', { ...formData.params, learning_rate: parseFloat(value) || 0.1 })}
                      type="number"
                      step="0.01"
                      helperText="Lernrate (0.01-0.3, Default: 0.1)"
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <ValidatedTextField
                      label="N Estimators"
                      value={formData.params?.n_estimators?.toString() || ''}
                      onChange={(value) => handleFieldChange('params', { ...formData.params, n_estimators: parseInt(value) || 100 })}
                      type="number"
                      helperText="Anzahl Bäume (50-500, Default: 100)"
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <DateProvider>
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4" sx={{ color: '#00d4ff', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
            <AddIcon /> Neues Modell erstellen
          </Typography>
        </Box>

        <Alert severity="info" sx={{ mb: 4 }}>
          <Typography variant="body2">
            Verwenden Sie den <strong>Stepper</strong> unten, um Ihr Modell Schritt für Schritt zu konfigurieren.
            Für Anfänger wird der <strong>vereinfachte Modus</strong> empfohlen.
          </Typography>
        </Alert>

        {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

        <Paper sx={{ p: 3, bgcolor: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
            {steps.map((label, index) => (
              <Step key={label} completed={index < activeStep}>
                <StepLabel
                  sx={{
                    '& .MuiStepLabel-label': {
                      color: index <= activeStep ? '#00d4ff' : 'text.secondary',
                    },
                  }}
                >
                  {label}
                </StepLabel>
              </Step>
            ))}
          </Stepper>

          <Divider sx={{ mb: 3 }} />

          {renderStepContent(activeStep)}

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
              variant="outlined"
            >
              Zurück
            </Button>

            <Box>
              {activeStep === steps.length - 1 ? (
                <Button
                  variant="contained"
                  onClick={handleSubmit}
                  disabled={isLoading || !isStepValid(activeStep)}
                  sx={{ minWidth: 150 }}
                >
                  {isLoading ? 'Erstelle Modell...' : 'Modell erstellen'}
                </Button>
              ) : (
                <Button
                  variant="contained"
                  onClick={handleNext}
                  disabled={!isStepValid(activeStep)}
                >
                  Weiter
                </Button>
              )}
            </Box>
          </Box>
        </Paper>
      </Container>
    </DateProvider>
  );
};

export default Training;
