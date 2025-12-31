import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  HealthResponse,
  ConfigResponse,
  ConfigUpdateRequest
} from '../types/api';
import { pumpApi } from '../services/api';

interface PumpStore {
  // State
  health: HealthResponse | null;
  config: ConfigResponse | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;

  // Actions
  fetchHealth: () => Promise<void>;
  fetchConfig: () => Promise<void>;
  updateConfig: (config: ConfigUpdateRequest) => Promise<void>;
  startPolling: () => void;
  stopPolling: () => void;

  // Computed
  isServiceHealthy: boolean;
}

export const usePumpStore = create<PumpStore>()(
  devtools(
    (set, get) => ({
      // Initial State
      health: null,
      config: null,
      isLoading: false,
      error: null,
      lastUpdated: null,

      // Actions
      fetchHealth: async () => {
        try {
          set({ isLoading: true, error: null });

          const health = await pumpApi.getHealth();

          set({
            health,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          // Bei API-Fehlern: Setze Status auf "error" (API unreachable)
          const disconnectedHealth = {
            status: 'error' as const,
            ws_connected: false,
            db_connected: false,
            uptime_seconds: 0,
            last_message_ago: null,
            reconnect_count: 0,
            last_error: error instanceof Error ? error.message : 'API unreachable',
            cache_stats: { total_coins: 0, activated_coins: 0, expired_coins: 0, oldest_age_seconds: 0, newest_age_seconds: 0 },
            tracking_stats: { active_coins: 0, total_trades: 0, total_metrics_saved: 0 },
            discovery_stats: { total_coins_discovered: 0, n8n_available: false, n8n_buffer_size: 0 }
          };

          set({
            health: disconnectedHealth,
            error: error instanceof Error ? error.message : 'Failed to fetch health',
            lastUpdated: new Date(),
            isLoading: false
          });
        }
      },

      fetchConfig: async () => {
        try {
          set({ isLoading: true, error: null });

          const config = await pumpApi.getConfig();

          set({
            config,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to fetch config',
            isLoading: false
          });
        }
      },

      updateConfig: async (configUpdate: ConfigUpdateRequest) => {
        try {
          set({ isLoading: true, error: null });
          await pumpApi.updateConfig(configUpdate);

          // Refresh config after update
          await get().fetchConfig();
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to update config',
            isLoading: false
          });
          throw error; // Re-throw for UI handling
        }
      },

      startPolling: () => {
        // Auto-refresh every 5 seconds for faster status updates
        const interval = setInterval(() => {
          get().fetchHealth();
          get().fetchConfig();
        }, 5000);

        // Store interval ID for cleanup
        (get() as any)._pollingInterval = interval;
      },

      stopPolling: () => {
        const interval = (get() as any)._pollingInterval;
        if (interval) {
          clearInterval(interval);
        }
      },

      // Computed Properties
      get isServiceHealthy(): boolean {
        const health = get().health;
        return health ? health.status === 'healthy' : false;
      },

    }),
    {
      name: 'pump-store',
    }
  )
);
