import axios from 'axios';
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
  ConfigUpdateResponse,
} from '../types/api';

// API Base URL - IMMER window.location.origin verwenden (wie pump-find)
// Das ermöglicht es nginx/Vite proxy, die /api/* Anfragen abzufangen
const getApiBaseUrl = (): string => {
  return window.location.origin;
};

// API_BASE_URL wird dynamisch zur Laufzeit berechnet

const api = axios.create({
  timeout: 10000,
});

// Interceptor um baseURL dynamisch zu setzen
api.interceptors.request.use((config) => {
  if (!config.url?.startsWith('http')) {
    // Wenn keine absolute URL, dann baseURL hinzufügen
    config.baseURL = getApiBaseUrl();
  }
  return config;
});

// Request Interceptor für Error Handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const mlApi = {
  // ============================================================
  // Models Management
  // ============================================================

  // Get all models
  async getModels(): Promise<ModelResponse[]> {
    const response = await api.get('/api/models');
    return response.data;
  },

  // Get single model
  async getModel(modelId: string): Promise<ModelResponse> {
    const response = await api.get(`/api/models/${modelId}`);
    return response.data;
  },

  // Create model - simple rule-based
  async createSimpleModel(request: SimpleTrainModelRequest): Promise<CreateJobResponse> {
    const response = await api.post('/api/models/create/simple', request);
    return response.data;
  },

  // Create model - time-based prediction
  async createTimeBasedModel(request: any): Promise<CreateJobResponse> {
    const response = await api.post('/api/models/create/time-based', request);
    return response.data;
  },

  // Create model - full training
  async createModel(request: TrainModelRequest): Promise<CreateJobResponse> {
    const response = await api.post('/api/models/create', request);
    return response.data;
  },

  // Test model
  async testModel(modelId: string, request: TestModelRequest): Promise<CreateJobResponse> {
    const response = await api.post(`/api/models/${modelId}/test`, request);
    return response.data;
  },

  // Compare models
  async compareModels(request: CompareModelsRequest): Promise<CreateJobResponse> {
    const response = await api.post('/api/models/compare', request);
    return response.data;
  },

  // Delete model
  async deleteModel(modelId: string): Promise<void> {
    await api.delete(`/api/models/${modelId}`);
  },

  // Download model
  async downloadModel(modelId: string): Promise<Blob> {
    const response = await api.get(`/api/models/${modelId}/download`, {
      responseType: 'blob'
    });
    return response.data;
  },

  // ============================================================
  // Jobs & Queue Management
  // ============================================================

  // Get all jobs
  async getJobs(): Promise<JobResponse[]> {
    const response = await api.get('/api/queue');
    return response.data;
  },

  // Get single job
  async getJob(jobId: string): Promise<JobResponse> {
    const response = await api.get(`/api/queue/${jobId}`);
    return response.data;
  },

  // ============================================================
  // Test Results
  // ============================================================

  // Get all test results
  async getTestResults(): Promise<TestResultResponse[]> {
    const response = await api.get('/api/test-results');
    return response.data;
  },

  // Get single test result
  async getTestResult(testId: string): Promise<TestResultResponse> {
    const response = await api.get(`/api/test-results/${testId}`);
    return response.data;
  },

  // Delete test result
  async deleteTestResult(testId: string): Promise<void> {
    await api.delete(`/api/test-results/${testId}`);
  },

  // ============================================================
  // Comparisons
  // ============================================================

  // Get all comparisons
  async getComparisons(): Promise<ComparisonResponse[]> {
    const response = await api.get('/api/comparisons');
    return response.data;
  },

  // Get single comparison
  async getComparison(comparisonId: string): Promise<ComparisonResponse> {
    const response = await api.get(`/api/comparisons/${comparisonId}`);
    return response.data;
  },

  // Delete comparison
  async deleteComparison(comparisonId: string): Promise<void> {
    await api.delete(`/api/comparisons/${comparisonId}`);
  },

  // ============================================================
  // System & Monitoring
  // ============================================================

  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get('/api/health');
    return response.data;
  },

  // Prometheus metrics
  async getMetrics(): Promise<string> {
    const response = await api.get('/api/metrics', {
      headers: { 'Accept': 'text/plain' },
      responseType: 'text'
    });
    return response.data;
  },

  // Data availability
  async getDataAvailability(): Promise<any> {
    const response = await api.get('/api/data-availability');
    return response.data;
  },

  // Phases
  async getPhases(): Promise<any> {
    const response = await api.get('/api/phases');
    return response.data;
  },

  // ============================================================
  // Configuration
  // ============================================================

  // Get configuration
  async getConfig(): Promise<ConfigResponse> {
    const response = await api.get('/api/config');
    return response.data;
  },

  // Update configuration
  async updateConfig(config: ConfigUpdateRequest): Promise<ConfigUpdateResponse> {
    const response = await api.put('/api/config', config);
    return response.data;
  },

  // Reload configuration
  async reloadConfig(): Promise<any> {
    const response = await api.post('/api/reload-config');
    return response.data;
  },

  // Reconnect database
  async reconnectDb(): Promise<any> {
    const response = await api.post('/api/reconnect-db');
    return response.data;
  },

  // ============================================================
  // Utility Functions
  // ============================================================

  getApiUrl(): string {
    return getApiBaseUrl();
  },

  // Health check with timeout for UI
  async checkServiceHealth(): Promise<boolean> {
    try {
      const response = await api.get('/api/health', { timeout: 5000 });
      return response.status === 200;
    } catch {
      return false;
    }
  }
};

export default api;
