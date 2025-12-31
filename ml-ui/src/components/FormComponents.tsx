import React from 'react';
import {
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  FormHelperText,
  Checkbox,
  FormControlLabel,
  RadioGroup,
  Radio,
  Autocomplete,
  Chip,
  Box,
  Typography,
  InputAdornment,
  IconButton,
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from '@mui/icons-material';
import { DatePicker, TimePicker, DateTimePicker } from '@mui/x-date-pickers';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { de } from 'date-fns/locale';

// Wrapper für Date Picker mit deutscher Lokalisierung
export const DateProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <LocalizationProvider dateAdapter={AdapterDateFns} adapterLocale={de}>
    {children}
  </LocalizationProvider>
);

// Verbesserte TextField mit Validation
interface ValidatedTextFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  error?: string;
  required?: boolean;
  type?: string;
  multiline?: boolean;
  rows?: number;
  helperText?: string;
  placeholder?: string;
  disabled?: boolean;
  startAdornment?: React.ReactNode;
  endAdornment?: React.ReactNode;
  step?: string | number;
}

export const ValidatedTextField: React.FC<ValidatedTextFieldProps> = ({
  label,
  value,
  onChange,
  error,
  required = false,
  type = 'text',
  multiline = false,
  rows = 3,
  helperText,
  placeholder,
  disabled = false,
  startAdornment,
  endAdornment,
  step,
}) => {
  const [showPassword, setShowPassword] = React.useState(false);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  const inputProps: any = {};
  if (startAdornment) {
    inputProps.startAdornment = <InputAdornment position="start">{startAdornment}</InputAdornment>;
  }
  if (endAdornment) {
    inputProps.endAdornment = <InputAdornment position="end">{endAdornment}</InputAdornment>;
  }
  if (step !== undefined) {
    inputProps.step = step;
  }

  // Password field special handling
  if (type === 'password') {
    inputProps.endAdornment = (
      <InputAdornment position="end">
        <IconButton
          onClick={() => setShowPassword(!showPassword)}
          edge="end"
        >
          {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
        </IconButton>
      </InputAdornment>
    );
  }

  return (
    <TextField
      fullWidth
      label={label}
      value={value}
      onChange={handleChange}
      error={!!error}
      helperText={error || helperText}
      required={required}
      type={type === 'password' ? (showPassword ? 'text' : 'password') : type}
      multiline={multiline}
      rows={rows}
      placeholder={placeholder}
      disabled={disabled}
      InputProps={inputProps}
      sx={{
        '& .MuiOutlinedInput-root': {
          '&.Mui-focused fieldset': {
            borderColor: '#00d4ff',
          },
        },
        '& .MuiInputLabel-root.Mui-focused': {
          color: '#00d4ff',
        },
      }}
    />
  );
};

// Verbesserte Select mit Validation
interface ValidatedSelectProps {
  label: string;
  value: string | number;
  onChange: (value: string | number) => void;
  options: Array<{ value: string | number; label: string; disabled?: boolean }>;
  error?: string;
  required?: boolean;
  helperText?: string;
  disabled?: boolean;
  placeholder?: string;
}

export const ValidatedSelect: React.FC<ValidatedSelectProps> = ({
  label,
  value,
  onChange,
  options,
  error,
  required = false,
  helperText,
  disabled = false,
  placeholder,
}) => {
  const handleChange = (event: any) => {
    onChange(event.target.value);
  };

  return (
    <FormControl fullWidth error={!!error} disabled={disabled}>
      <InputLabel required={required}>{label}</InputLabel>
      <Select
        value={value}
        onChange={handleChange}
        label={label}
        displayEmpty={!!placeholder}
        sx={{
          '& .MuiOutlinedInput-root': {
            '&.Mui-focused fieldset': {
              borderColor: '#00d4ff',
            },
          },
          '& .MuiInputLabel-root.Mui-focused': {
            color: '#00d4ff',
          },
        }}
      >
        {placeholder && (
          <MenuItem value="" disabled>
            <Typography color="textSecondary">{placeholder}</Typography>
          </MenuItem>
        )}
        {options.map((option) => (
          <MenuItem
            key={option.value}
            value={option.value}
            disabled={option.disabled}
          >
            {option.label}
          </MenuItem>
        ))}
      </Select>
      {(error || helperText) && (
        <FormHelperText>{error || helperText}</FormHelperText>
      )}
    </FormControl>
  );
};

// Autocomplete für Multi-Select
interface ValidatedAutocompleteProps {
  label: string;
  value: string[];
  onChange: (value: string[]) => void;
  options: string[];
  error?: string;
  required?: boolean;
  helperText?: string;
  disabled?: boolean;
  placeholder?: string;
  maxTags?: number;
}

export const ValidatedAutocomplete: React.FC<ValidatedAutocompleteProps> = ({
  label,
  value,
  onChange,
  options,
  error,
  required = false,
  helperText,
  disabled = false,
  placeholder,
  maxTags = 10,
}) => {
  return (
    <Autocomplete
      multiple
      options={options}
      value={value}
      onChange={(_, newValue) => onChange(newValue)}
      disabled={disabled}
      renderTags={(tagValue, getTagProps) =>
        tagValue.slice(0, maxTags).map((option, index) => (
          <Chip
            {...getTagProps({ index })}
            key={option}
            label={option}
            size="small"
            sx={{
              backgroundColor: 'rgba(0, 212, 255, 0.1)',
              color: '#00d4ff',
              '& .MuiChip-deleteIcon': {
                color: '#00d4ff',
              },
            }}
          />
        ))
      }
      renderInput={(params) => (
        <TextField
          {...params}
          label={label}
          placeholder={placeholder}
          error={!!error}
          helperText={error || helperText}
          required={required}
          sx={{
            '& .MuiOutlinedInput-root': {
              '&.Mui-focused fieldset': {
                borderColor: '#00d4ff',
              },
            },
            '& .MuiInputLabel-root.Mui-focused': {
              color: '#00d4ff',
            },
          }}
        />
      )}
      sx={{ width: '100%' }}
    />
  );
};

// Checkbox Group für Feature Auswahl
interface FeatureSelectorProps {
  label: string;
  categories: Record<string, string[]>;
  selectedFeatures: string[];
  onChange: (features: string[]) => void;
  error?: string;
  helperText?: string;
  maxHeight?: string;
}

export const FeatureSelector: React.FC<FeatureSelectorProps> = ({
  label,
  categories,
  selectedFeatures,
  onChange,
  error,
  helperText,
  maxHeight = '300px',
}) => {
  const handleCategoryToggle = (categoryFeatures: string[], checked: boolean) => {
    const newSelected = checked
      ? [...selectedFeatures, ...categoryFeatures.filter(f => !selectedFeatures.includes(f))]
      : selectedFeatures.filter(f => !categoryFeatures.includes(f));
    onChange(newSelected);
  };

  const handleFeatureToggle = (feature: string, checked: boolean) => {
    const newSelected = checked
      ? [...selectedFeatures, feature]
      : selectedFeatures.filter(f => f !== feature);
    onChange(newSelected);
  };

  const isCategoryFullySelected = (categoryFeatures: string[]) => {
    return categoryFeatures.every(f => selectedFeatures.includes(f));
  };

  const isCategoryPartiallySelected = (categoryFeatures: string[]) => {
    return categoryFeatures.some(f => selectedFeatures.includes(f)) &&
           !isCategoryFullySelected(categoryFeatures);
  };

  return (
    <FormControl fullWidth error={!!error}>
      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
        {label}
      </Typography>

      <Box sx={{
        maxHeight,
        overflow: 'auto',
        border: '1px solid rgba(0, 0, 0, 0.23)',
        borderRadius: 1,
        p: 1,
      }}>
        {Object.entries(categories).map(([category, features]) => (
          <Box key={category} sx={{ mb: 2 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={isCategoryFullySelected(features)}
                  indeterminate={isCategoryPartiallySelected(features)}
                  onChange={(e) => handleCategoryToggle(features, e.target.checked)}
                />
              }
              label={
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                  {category} ({features.length})
                </Typography>
              }
            />

            <Box sx={{ ml: 3, mt: 1 }}>
              {features.map((feature) => (
                <FormControlLabel
                  key={feature}
                  control={
                    <Checkbox
                      size="small"
                      checked={selectedFeatures.includes(feature)}
                      onChange={(e) => handleFeatureToggle(feature, e.target.checked)}
                    />
                  }
                  label={
                    <Typography variant="body2" color="textSecondary">
                      {feature}
                    </Typography>
                  }
                />
              ))}
            </Box>
          </Box>
        ))}
      </Box>

      {(error || helperText) && (
        <FormHelperText error={!!error}>{error || helperText}</FormHelperText>
      )}

      <Typography variant="caption" color="textSecondary" sx={{ mt: 1 }}>
        Ausgewählt: {selectedFeatures.length} Features
      </Typography>
    </FormControl>
  );
};

// Date/Time Picker Components
interface ValidatedDateTimePickerProps {
  label: string;
  value: Date | null;
  onChange: (date: Date | null) => void;
  error?: string;
  required?: boolean;
  helperText?: string;
  disabled?: boolean;
  minDate?: Date;
  maxDate?: Date;
}

export const ValidatedDateTimePicker: React.FC<ValidatedDateTimePickerProps> = ({
  label,
  value,
  onChange,
  error,
  required = false,
  helperText,
  disabled = false,
  minDate,
  maxDate,
}) => {
  return (
    <DateProvider>
      <DateTimePicker
        label={label}
        value={value}
        onChange={onChange}
        disabled={disabled}
        minDateTime={minDate}
        maxDateTime={maxDate}
        slotProps={{
          textField: {
            fullWidth: true,
            error: !!error,
            helperText: error || helperText,
            required,
            sx: {
              '& .MuiOutlinedInput-root': {
                '&.Mui-focused fieldset': {
                  borderColor: '#00d4ff',
                },
              },
              '& .MuiInputLabel-root.Mui-focused': {
                color: '#00d4ff',
              },
            },
          },
        }}
      />
    </DateProvider>
  );
};

export const ValidatedDatePicker: React.FC<ValidatedDateTimePickerProps> = ({
  label,
  value,
  onChange,
  error,
  required = false,
  helperText,
  disabled = false,
  minDate,
  maxDate,
}) => {
  return (
    <DateProvider>
      <DatePicker
        label={label}
        value={value}
        onChange={onChange}
        disabled={disabled}
        minDate={minDate}
        maxDate={maxDate}
        slotProps={{
          textField: {
            fullWidth: true,
            error: !!error,
            helperText: error || helperText,
            required,
            sx: {
              '& .MuiOutlinedInput-root': {
                '&.Mui-focused fieldset': {
                  borderColor: '#00d4ff',
                },
              },
              '& .MuiInputLabel-root.Mui-focused': {
                color: '#00d4ff',
              },
            },
          },
        }}
      />
    </DateProvider>
  );
};

// Export aller Komponenten
