import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  // Model Types
  ModelResponse,
  TrainModelRequest,
  SimpleTrainModelRequest,
  TestModelRequest,
  CompareModelsRequest,
  TestResultResponse,
  ComparisonResponse,
  // Job Types
  JobResponse,
  CreateJobResponse,
  // System Types
  HealthResponse,
  // Config Types
  ConfigResponse,
  ConfigUpdateRequest,
  // Utility Types
  DataAvailabilityResponse,
  PhasesResponse,
} from '../types/api';
import { mlApi } from '../services/api';

interface MLStore {
  // ============================================================
  // State
  // ============================================================

  // Models
  models: ModelResponse[];
  selectedModelIds: number[];
  currentModel: ModelResponse | null;

  // Jobs & Queue
  jobs: JobResponse[];
  currentJob: JobResponse | null;

  // Test Results
  testResults: TestResultResponse[];
  currentTestResult: TestResultResponse | null;

  // Comparisons
  comparisons: ComparisonResponse[];
  currentComparison: ComparisonResponse | null;

  // System & Health
  health: HealthResponse | null;
  config: ConfigResponse | null;
  dataAvailability: DataAvailabilityResponse | null;
  phases: PhasesResponse['phases'];

  // UI State
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;

  // Page Navigation (für komplexe Flows)
  currentPage: string | null;
  pageParams: Record<string, any>;

  // ============================================================
  // Actions - Models
  // ============================================================

  fetchModels: () => Promise<void>;
  fetchModel: (modelId: number) => Promise<void>;
  createSimpleModel: (request: SimpleTrainModelRequest) => Promise<CreateJobResponse>;
  createTimeBasedModel: (request: any) => Promise<CreateJobResponse>;
  createModel: (request: TrainModelRequest) => Promise<CreateJobResponse>;
  testModel: (modelId: number, request: TestModelRequest) => Promise<CreateJobResponse>;
  compareModels: (request: CompareModelsRequest) => Promise<CreateJobResponse>;
  deleteModel: (modelId: number) => Promise<void>;
  downloadModel: (modelId: number) => Promise<Blob>;
  selectModel: (modelId: number) => void;
  deselectModel: (modelId: number) => void;
  clearModelSelection: () => void;

  // ============================================================
  // Actions - Jobs
  // ============================================================

  fetchJobs: () => Promise<void>;
  fetchJob: (jobId: string) => Promise<void>;

  // ============================================================
  // Actions - Test Results
  // ============================================================

  fetchTestResults: () => Promise<void>;
  fetchTestResult: (testId: number) => Promise<void>;
  deleteTestResult: (testId: number) => Promise<void>;

  // ============================================================
  // Actions - Comparisons
  // ============================================================

  fetchComparisons: () => Promise<void>;
  fetchComparison: (comparisonId: string) => Promise<void>;
  deleteComparison: (comparisonId: number) => Promise<void>;

  // ============================================================
  // Actions - System
  // ============================================================

  fetchHealth: () => Promise<void>;
  fetchConfig: () => Promise<void>;
  updateConfig: (config: ConfigUpdateRequest) => Promise<void>;
  reloadConfig: () => Promise<void>;
  reconnectDb: () => Promise<any>;
  fetchDataAvailability: () => Promise<void>;
  fetchPhases: () => Promise<void>;
  fetchMetrics: () => Promise<string>;

  // ============================================================
  // Actions - UI/Navigation
  // ============================================================

  setCurrentPage: (page: string, params?: Record<string, any>) => void;
  clearError: () => void;
  startPolling: () => void;
  stopPolling: () => void;

  // ============================================================
  // Computed Properties
  // ============================================================

  selectedModels: ModelResponse[];
  isServiceHealthy: boolean;
}

export const useMLStore = create<MLStore>()(
  devtools(
    (set, get) => ({
      // ============================================================
      // Initial State
      // ============================================================

      models: [],
      selectedModelIds: [],
      currentModel: null,

      jobs: [],
      currentJob: null,

      testResults: [],
      currentTestResult: null,

      comparisons: [],
      currentComparison: null,

      health: null,
      config: null,
      dataAvailability: null,
      phases: [],

      isLoading: false,
      error: null,
      lastUpdated: null,

      currentPage: null,
      pageParams: {},

      // ============================================================
      // Models Actions
      // ============================================================

      fetchModels: async () => {
        try {
          set({ isLoading: true, error: null });
          const models = await mlApi.getModels();
          set({
            models,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch models';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      fetchModel: async (modelId: number) => {
        try {
          set({ isLoading: true, error: null });
          const currentModel = await mlApi.getModel(modelId.toString());
          set({
            currentModel,
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch model';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      createSimpleModel: async (request: SimpleTrainModelRequest) => {
        try {
          set({ isLoading: true, error: null });
          const response = await mlApi.createSimpleModel(request);
          set({ isLoading: false });
          // Refresh jobs list
          await get().fetchJobs();
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to create model';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      createTimeBasedModel: async (request: any) => {
        try {
          set({ isLoading: true, error: null });
          const response = await mlApi.createTimeBasedModel(request);
          set({ isLoading: false });
          await get().fetchJobs();
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to create time-based model';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      createModel: async (request: TrainModelRequest) => {
        try {
          set({ isLoading: true, error: null });
          const response = await mlApi.createModel(request);
          set({ isLoading: false });
          await get().fetchJobs();
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to create model';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      testModel: async (modelId: number, request: TestModelRequest) => {
        try {
          set({ isLoading: true, error: null });
          const response = await mlApi.testModel(modelId.toString(), request);
          set({ isLoading: false });
          await get().fetchJobs();
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to test model';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      compareModels: async (request: CompareModelsRequest) => {
        try {
          set({ isLoading: true, error: null });
          const response = await mlApi.compareModels(request);
          set({ isLoading: false });
          await get().fetchJobs();
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to compare models';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      deleteModel: async (modelId: number) => {
        try {
          set({ isLoading: true, error: null });
          await mlApi.deleteModel(modelId.toString());
          set({ isLoading: false });
          // Remove from local state and refresh
          const { models } = get();
          set({
            models: models.filter(m => m.id !== modelId),
            selectedModelIds: get().selectedModelIds.filter(id => id !== modelId)
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete model';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      downloadModel: async (modelId: number) => {
        try {
          const blob = await mlApi.downloadModel(modelId.toString());
          return blob;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to download model';
          set({ error: errorMessage });
          throw error;
        }
      },

      selectModel: (modelId: number) => {
        const { selectedModelIds } = get();
        if (!selectedModelIds.includes(modelId)) {
          set({ selectedModelIds: [...selectedModelIds, modelId] });
        }
      },

      deselectModel: (modelId: number) => {
        const { selectedModelIds } = get();
        set({ selectedModelIds: selectedModelIds.filter(id => id !== modelId) });
      },

      clearModelSelection: () => {
        set({ selectedModelIds: [] });
      },

      // ============================================================
      // Jobs Actions
      // ============================================================

      fetchJobs: async () => {
        try {
          set({ isLoading: true, error: null });
          const jobs = await mlApi.getJobs();
          set({
            jobs,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch jobs';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      fetchJob: async (jobId: string) => {
        try {
          set({ isLoading: true, error: null });
          const currentJob = await mlApi.getJob(jobId.toString());
          set({
            currentJob,
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch job';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      // ============================================================
      // Test Results Actions
      // ============================================================

      fetchTestResults: async () => {
        try {
          set({ isLoading: true, error: null });
          const testResults = await mlApi.getTestResults();
          set({
            testResults,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch test results';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      fetchTestResult: async (testId: number) => {
        try {
          set({ isLoading: true, error: null });
          const currentTestResult = await mlApi.getTestResult(testId.toString());
          set({
            currentTestResult,
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch test result';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      deleteTestResult: async (testId: number) => {
        try {
          set({ isLoading: true, error: null });
          await mlApi.deleteTestResult(testId.toString());
          set({ isLoading: false });
          // Remove from local state
          const { testResults } = get();
          set({ testResults: testResults.filter(tr => tr.id !== testId) });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete test result';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      // ============================================================
      // Comparisons Actions
      // ============================================================

      fetchComparisons: async () => {
        try {
          set({ isLoading: true, error: null });
          const comparisons = await mlApi.getComparisons();
          set({
            comparisons,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch comparisons';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      fetchComparison: async (comparisonId: string) => {
        try {
          set({ isLoading: true, error: null });
          const currentComparison = await mlApi.getComparison(comparisonId.toString());
          set({
            currentComparison,
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch comparison';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      deleteComparison: async (comparisonId: number) => {
        try {
          set({ isLoading: true, error: null });
          await mlApi.deleteComparison(comparisonId.toString());
          set({ isLoading: false });
          // Remove from local state
          const { comparisons } = get();
          set({ comparisons: comparisons.filter(c => c.id !== comparisonId) });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete comparison';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      // ============================================================
      // System Actions
      // ============================================================

      fetchHealth: async () => {
        try {
          const health = await mlApi.getHealth();
          set({
            health,
            lastUpdated: new Date(),
            error: null
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch health status';
          set({
            error: errorMessage,
            health: {
              status: 'error',
              db_connected: false,
              uptime_seconds: 0,
              total_jobs_processed: 0,
              last_error: errorMessage
            }
          });
        }
      },

      fetchConfig: async () => {
        try {
          set({ isLoading: true, error: null });
          const config = await mlApi.getConfig();
          set({
            config,
            lastUpdated: new Date(),
            isLoading: false
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch config';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      updateConfig: async (configUpdate: ConfigUpdateRequest) => {
        try {
          set({ isLoading: true, error: null });
          await mlApi.updateConfig(configUpdate);
          // Refresh config after update
          await get().fetchConfig();
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to update config';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      reloadConfig: async () => {
        try {
          set({ isLoading: true, error: null });
          await mlApi.reloadConfig();
          // Refresh config after reload
          await get().fetchConfig();
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to reload config';
          set({
            error: errorMessage,
            isLoading: false
          });
        }
      },

      reconnectDb: async () => {
        try {
          set({ isLoading: true, error: null });
          const result = await mlApi.reconnectDb();

          // Refresh health status after reconnect
          await get().fetchHealth();

          return result;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to reconnect database';
          set({
            error: errorMessage,
            isLoading: false
          });
          throw error;
        }
      },

      fetchDataAvailability: async () => {
        try {
          const dataAvailability = await mlApi.getDataAvailability();
          set({ dataAvailability });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch data availability';
          set({ error: errorMessage });
        }
      },

      fetchPhases: async () => {
        try {
          const phasesData = await mlApi.getPhases();
          set({ phases: phasesData.phases || [] });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch phases';
          set({ error: errorMessage });
        }
      },

      fetchMetrics: async () => {
        try {
          const metrics = await mlApi.getMetrics();
          return metrics;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch metrics';
          set({ error: errorMessage });
          throw error;
        }
      },

      // ============================================================
      // UI Actions
      // ============================================================

      setCurrentPage: (page: string, params: Record<string, any> = {}) => {
        set({ currentPage: page, pageParams: params });
      },

      clearError: () => {
        set({ error: null });
      },

      startPolling: () => {
        // Auto-refresh every 30 seconds for health and jobs
        const interval = setInterval(() => {
          get().fetchHealth();
          get().fetchJobs();
        }, 30000);

        // Store interval ID for cleanup
        (get() as any)._pollingInterval = interval;
      },

      stopPolling: () => {
        const interval = (get() as any)._pollingInterval;
        if (interval) {
          clearInterval(interval);
        }
      },

      // ============================================================
      // Computed Properties
      // ============================================================

      get selectedModels(): ModelResponse[] {
        const { models, selectedModelIds } = get();
        return models.filter(model => selectedModelIds.includes(model.id));
      },

      get isServiceHealthy(): boolean {
        const { health } = get();
        return health ? health.status === 'healthy' : false;
      },
    }),
    {
      name: 'ml-store',
    }
  )
);
