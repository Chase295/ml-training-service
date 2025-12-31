import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Paper,
  Checkbox,
  IconButton,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  KeyboardArrowUp as SortAscIcon,
  KeyboardArrowDown as SortDescIcon,
} from '@mui/icons-material';

export interface Column<T = any> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  width?: string | number;
  align?: 'left' | 'center' | 'right';
  render?: (value: any, row: T, index: number) => React.ReactNode;
  format?: (value: any) => string;
}

interface DataTableProps<T = any> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  emptyMessage?: string;
  selectable?: boolean;
  selectedRows?: (keyof T | string)[];
  onRowSelect?: (row: T, selected: boolean) => void;
  onSelectAll?: (selected: boolean) => void;
  onRowClick?: (row: T) => void;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  onSort?: (column: string) => void;
  maxHeight?: string | number;
  dense?: boolean;
}

export function DataTable<T extends Record<string, any>>({
  data,
  columns,
  loading = false,
  emptyMessage = 'Keine Daten verfügbar',
  selectable = false,
  selectedRows = [],
  onRowSelect,
  onSelectAll,
  onRowClick,
  sortBy,
  sortDirection,
  onSort,
  maxHeight,
  dense = false,
}: DataTableProps<T>) {
  const allSelected = data.length > 0 && selectedRows.length === data.length;
  const someSelected = selectedRows.length > 0 && selectedRows.length < data.length;

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    onSelectAll?.(event.target.checked);
  };

  const handleRowSelect = (row: T, checked: boolean) => {
    onRowSelect?.(row, checked);
  };

  const getValue = (row: T, column: Column<T>) => {
    if (typeof column.key === 'string' && column.key.includes('.')) {
      // Handle nested properties like 'model.name'
      return column.key.split('.').reduce((obj, key) => obj?.[key], row);
    }
    return row[column.key as keyof T];
  };

  const renderCell = (row: T, column: Column<T>, index: number) => {
    const value = getValue(row, column);

    if (column.render) {
      return column.render(value, row, index);
    }

    if (column.format) {
      return column.format(value);
    }

    // Default rendering
    if (value === null || value === undefined) {
      return <Typography variant="body2" color="textSecondary">N/A</Typography>;
    }

    if (typeof value === 'boolean') {
      return (
        <Chip
          label={value ? 'Ja' : 'Nein'}
          color={value ? 'success' : 'default'}
          size="small"
        />
      );
    }

    if (typeof value === 'number') {
      return value.toLocaleString();
    }

    if (typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)) {
      // ISO Date string
      try {
        return new Date(value).toLocaleString('de-DE');
      } catch {
        return value;
      }
    }

    return String(value);
  };

  if (loading) {
    return (
      <Paper sx={{ p: 2 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center' }}>
          Daten werden geladen...
        </Typography>
      </Paper>
    );
  }

  return (
    <TableContainer
      component={Paper}
      sx={{
        maxHeight,
        '& .MuiTableCell-root': {
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        },
      }}
    >
      <Table size={dense ? 'small' : 'medium'} stickyHeader={!!maxHeight}>
        <TableHead>
          <TableRow sx={{ backgroundColor: 'rgba(0, 0, 0, 0.04)' }}>
            {selectable && (
              <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={someSelected}
                  checked={allSelected}
                  onChange={handleSelectAll}
                />
              </TableCell>
            )}

            {columns.map((column) => (
              <TableCell
                key={String(column.key)}
                align={column.align || 'left'}
                sx={{
                  fontWeight: 'bold',
                  backgroundColor: 'background.paper',
                  width: column.width,
                  minWidth: column.width,
                }}
                sortDirection={sortBy === column.key ? sortDirection : false}
              >
                {column.sortable && onSort ? (
                  <TableSortLabel
                    active={sortBy === column.key}
                    direction={sortBy === column.key ? sortDirection : 'asc'}
                    onClick={() => onSort(String(column.key))}
                  >
                    {column.label}
                  </TableSortLabel>
                ) : (
                  column.label
                )}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>

        <TableBody>
          {data.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={columns.length + (selectable ? 1 : 0)}
                sx={{ textAlign: 'center', py: 4 }}
              >
                <Typography variant="body1" color="textSecondary">
                  {emptyMessage}
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            data.map((row, index) => {
              const isSelected = selectable && selectedRows.some(selectedId => {
                const rowId = typeof row.id !== 'undefined' ? row.id : index;
                return selectedId === rowId;
              });

              return (
                <TableRow
                  key={row.id || index}
                  hover
                  selected={isSelected}
                  onClick={() => onRowClick?.(row)}
                  sx={{
                    cursor: onRowClick ? 'pointer' : 'default',
                    '&:hover': {
                      backgroundColor: 'rgba(0, 212, 255, 0.08)',
                    },
                    '&.Mui-selected': {
                      backgroundColor: 'rgba(0, 212, 255, 0.15)',
                      '&:hover': {
                        backgroundColor: 'rgba(0, 212, 255, 0.2)',
                      },
                    },
                  }}
                >
                  {selectable && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={isSelected}
                        onChange={(e) => handleRowSelect(row, e.target.checked)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </TableCell>
                  )}

                  {columns.map((column) => (
                    <TableCell
                      key={String(column.key)}
                      align={column.align || 'left'}
                      sx={{
                        maxWidth: column.width,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {renderCell(row, column, index)}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default DataTable;
