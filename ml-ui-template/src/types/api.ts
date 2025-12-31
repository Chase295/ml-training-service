// API Type Definitions for Pump Service

export interface CacheStats {
  total_coins: number;
  activated_coins: number;
  expired_coins: number;
  oldest_age_seconds: number;
  newest_age_seconds: number;
}

export interface TrackingStats {
  active_coins: number;
  total_trades: number;
  total_metrics_saved: number;
}

export interface DiscoveryStats {
  total_coins_discovered: number;
  n8n_available: boolean;
  n8n_buffer_size: number;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'error';
  ws_connected: boolean;
  db_connected: boolean;
  uptime_seconds: number;
  last_message_ago: number | null;
  reconnect_count: number;
  last_error: string | null;
  cache_stats: CacheStats;
  tracking_stats: TrackingStats;
  discovery_stats: DiscoveryStats;
}

export interface ConfigResponse {
  n8n_webhook_url: string;
  n8n_webhook_method: string;
  db_dsn: string;
  coin_cache_seconds: number;
  db_refresh_interval: number;
  batch_size: number;
  batch_timeout: number;
  bad_names_pattern: string;
  spam_burst_window: number;
  sol_reserves_full: number;
  whale_threshold_sol: number;
  age_calculation_offset_min: number;
  trade_buffer_seconds: number;
  ath_flush_interval: number;
}

export interface ConfigUpdateRequest {
  n8n_webhook_url?: string;
  n8n_webhook_method?: string;
  db_dsn?: string;
  coin_cache_seconds?: number;
  db_refresh_interval?: number;
  batch_size?: number;
  batch_timeout?: number;
  bad_names_pattern?: string;
  spam_burst_window?: number;
}

export interface ConfigUpdateResponse {
  status: string;
  message: string;
  updated_fields: string[];
  new_config: Partial<ConfigUpdateRequest>;
}

// Service Status Types
export type ServiceStatus = 'running' | 'stopped' | 'error' | 'unknown';

// Metrics Types
export interface MetricData {
  timestamp: string;
  value: number;
}

export interface SystemMetrics {
  coins_received: MetricData[];
  cache_size: MetricData[];
  active_coins: MetricData[];
  trades_received: MetricData[];
}
