// ============================================================
// ML Training Service API Types
// Basierend auf Pydantic Schemas aus app/api/schemas.py
// ============================================================

// ============================================================
// Request Types
// ============================================================

export interface SimpleTrainModelRequest {
  name: string;
  model_type: 'random_forest' | 'xgboost';
  target: string; // z.B. "price_close > 0.05"
  features: string[] | 'auto';
  train_start: string; // ISO datetime
  train_end: string; // ISO datetime
  description?: string;
  hyperparameters?: Record<string, any>;
  validation_split?: number;
}

export interface TrainModelRequest {
  name: string;
  model_type: 'random_forest' | 'xgboost';
  features: string[];
  phases?: number[];
  params?: Record<string, any>;
  train_start: string; // ISO datetime
  train_end: string; // ISO datetime
  description?: string;

  // Zeitbasierte Vorhersage (optional)
  use_time_based_prediction?: boolean;
  future_minutes?: number;
  min_percent_change?: number;
  direction?: 'up' | 'down' | 'both';

  // Feature Engineering
  use_engineered_features?: boolean;
  feature_engineering_windows?: number[];

  // SMOTE
  use_smote?: boolean;

  // TimeSeriesSplit
  use_timeseries_split?: boolean;
  cv_splits?: number;

  // Markt-Kontext
  use_market_context?: boolean;

  // Feature Ausschluss
  exclude_features?: string[];

  // Ziel-Variablen (optional bei zeitbasierter Vorhersage)
  target_var?: string;
  operator?: '>' | '<' | '>=' | '<=' | '=';
  target_value?: number;
}

export interface TestModelRequest {
  test_start: string; // ISO datetime
  test_end: string; // ISO datetime
}

export interface CompareModelsRequest {
  model_a_id: number;
  model_b_id: number;
  test_start: string; // ISO datetime
  test_end: string; // ISO datetime
}

export interface UpdateModelRequest {
  name?: string;
  description?: string;
}

export interface ConfigUpdateRequest {
  // Database
  db_dsn?: string;
  db_refresh_interval?: number;

  // ML Training
  model_storage_path?: string;
  max_concurrent_jobs?: number;
  job_poll_interval?: number;
  default_training_hours?: number;
  max_training_hours?: number;
  min_training_hours?: number;
}

// ============================================================
// Response Types
// ============================================================

export interface ModelResponse {
  id: number;
  name: string;
  model_type: string;
  status: string;
  created_at: string;
  updated_at: string;
  is_deleted: boolean;
  target_variable: string;
  target_operator?: string;
  target_value?: number;
  train_start: string;
  train_end: string;
  features: string[];
  phases?: number[];
  params?: Record<string, any>;
  feature_importance?: Record<string, number>;
  training_accuracy?: number;
  training_f1?: number;
  training_precision?: number;
  training_recall?: number;
  roc_auc?: number;
  mcc?: number;
  fpr?: number;
  fnr?: number;
  confusion_matrix?: Record<string, number>;
  simulated_profit_pct?: number;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  cv_scores?: Record<string, any>;
  cv_overfitting_gap?: number;
  model_file_path?: string;
  description?: string;

  // Zeitbasierte Vorhersage Felder
  future_minutes?: number;
  price_change_percent?: number;
  min_percent_change?: number;
  target_direction?: string;
}

export interface TestResultResponse {
  id: number;
  model_id: number;
  created_at: string;
  test_start: string;
  test_end: string;
  accuracy?: number;
  f1_score?: number;
  precision_score?: number;
  recall?: number;
  roc_auc?: number;
  mcc?: number;
  fpr?: number;
  fnr?: number;
  confusion_matrix?: Record<string, number>;
  simulated_profit_pct?: number;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  num_samples?: number;
  num_positive?: number;
  num_negative?: number;
  has_overlap: boolean;
  overlap_note?: string;
  feature_importance?: Record<string, number>;
  train_accuracy?: number;
  train_f1?: number;
  train_precision?: number;
  train_recall?: number;
  accuracy_degradation?: number;
  f1_degradation?: number;
  is_overfitted?: boolean;
  test_duration_days?: number;
}

export interface ComparisonResponse {
  id: number;
  created_at: string;
  model_a_id: number;
  model_b_id: number;
  test_start: string;
  test_end: string;
  num_samples?: number;

  // Modell A Metriken
  a_accuracy?: number;
  a_f1?: number;
  a_precision?: number;
  a_recall?: number;
  a_mcc?: number;
  a_fpr?: number;
  a_fnr?: number;
  a_simulated_profit_pct?: number;
  a_confusion_matrix?: Record<string, number>;
  a_train_accuracy?: number;
  a_train_f1?: number;
  a_accuracy_degradation?: number;
  a_f1_degradation?: number;
  a_is_overfitted?: boolean;
  a_test_duration_days?: number;

  // Modell B Metriken
  b_accuracy?: number;
  b_f1?: number;
  b_precision?: number;
  b_recall?: number;
  b_mcc?: number;
  b_fpr?: number;
  b_fnr?: number;
  b_simulated_profit_pct?: number;
  b_confusion_matrix?: Record<string, number>;
  b_train_accuracy?: number;
  b_train_f1?: number;
  b_accuracy_degradation?: number;
  b_f1_degradation?: number;
  b_is_overfitted?: boolean;
  b_test_duration_days?: number;

  winner_id?: number;
}

export interface JobResponse {
  id: number;
  job_type: string;
  status: string;
  priority: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  progress_msg?: string;
  error_msg?: string;
  worker_id?: string;

  // Training spezifisch
  train_model_type?: string;
  train_target_var?: string;
  train_operator?: string;
  train_value?: number;
  train_start?: string;
  train_end?: string;
  train_features?: string[];
  train_phases?: number[];
  train_params?: Record<string, any>;

  // Ergebnisse
  result_model_id?: number;
  test_model_id?: number;
  test_start?: string;
  test_end?: string;
  result_test_id?: number;
  compare_model_a_id?: number;
  compare_model_b_id?: number;
  compare_start?: string;
  compare_end?: string;
  result_comparison_id?: number;

  // Eingebettete Ergebnisse
  result_model?: ModelResponse;
  result_test?: TestResultResponse;
  result_comparison?: ComparisonResponse;
}

export interface CreateJobResponse {
  job_id: number;
  message: string;
  status: string;
}

export interface HealthResponse {
  status: string;
  db_connected: boolean;
  uptime_seconds: number;
  start_time?: number;
  total_jobs_processed: number;
  last_error?: string;
}

export interface ConfigResponse {
  // Database
  db_dsn: string;
  db_refresh_interval: number;

  // ML Training
  model_storage_path: string;
  max_concurrent_jobs: number;
  job_poll_interval: number;
  default_training_hours: number;
  max_training_hours: number;
  min_training_hours: number;
}

export interface ConfigUpdateResponse {
  status: string;
  message: string;
  updated_fields: string[];
  new_config: Partial<ConfigUpdateRequest>;
}

// ============================================================
// Utility Types
// ============================================================

export interface DataAvailabilityResponse {
  min_timestamp: string;
  max_timestamp: string;
  total_records: number;
}

export interface PhasesResponse {
  phases: Array<{
    id: number;
    name: string;
    description?: string;
    min_age_seconds?: number;
    max_age_seconds?: number;
  }>;
}

// ============================================================
// Legacy Types (für Rückwärtskompatibilität)
// ============================================================

export interface Job {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at?: string
  config: JobConfig
  results?: JobResults
  error?: string
}

export interface JobConfig {
  model_type: string
  target_coin: string
  features: string[]
  training_params: Record<string, any>
}

export interface JobResults {
  model_id: string
  metrics: ModelMetrics
  charts: ChartData[]
}

export interface Model {
  id: string
  name: string
  created_at: string
  config: ModelConfig
  metrics: ModelMetrics
  status: 'active' | 'archived'
}

export interface ModelConfig {
  type: string
  features: string[]
  parameters: Record<string, any>
}

export interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  [key: string]: number
}

export interface ChartData {
  name: string
  value: number
  timestamp?: string
}

export interface ApiResponse<T> {
  data: T
  success: boolean
  message?: string
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy'
  uptime_seconds: number
  db_connected: boolean
  active_jobs: number
}
