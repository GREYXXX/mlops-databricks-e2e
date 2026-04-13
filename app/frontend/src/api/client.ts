const BASE = "";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    let detail = res.statusText || `status ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* ignore parse errors */ }
    throw new Error(detail);
  }
  return res.json();
}

export interface TriggerPipelineResponse {
  run_id: number;
  job_id: number;
}

export interface PipelineTask {
  task_key: string;
  status: string;
  result: string | null;
  start_time: number | null;
  end_time: number | null;
  duration_ms: number | null;
  attempt_number: number | null;
}

export interface PipelineStatus {
  job_name: string;
  job_id?: number;
  run_id?: number;
  run_name?: string;
  status: string;
  result: string | null;
  start_time?: number;
  end_time?: number;
  tasks: PipelineTask[];
}

export interface PipelineRun {
  run_id: number;
  run_name: string | null;
  status: string;
  result: string | null;
  start_time: number | null;
  end_time: number | null;
  duration_ms: number | null;
}

export interface ExperimentRun {
  run_id: string;
  run_name: string | null;
  status: string;
  start_time: number | null;
  end_time: number | null;
  duration_ms: number | null;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
}

export interface RunDetail extends ExperimentRun {
  artifacts: { path: string; is_dir: boolean; file_size: number | null }[];
  artifact_uri: string;
}

export interface ModelVersion {
  version: string;
  name: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  status: string;
  source: string;
  run_id: string;
  aliases: string[];
}

export interface ModelDetail extends ModelVersion {
  run_details: RunDetail | null;
}

export interface ComparisonSide {
  version: string | null;
  aliases: string[];
  metrics: Record<string, number>;
  run_id: string | null;
}

export interface MetricDelta {
  champion: number;
  challenger: number;
  delta: number;
  improved: boolean;
}

export interface ModelComparison {
  champion: ComparisonSide;
  challenger: ComparisonSide;
  deltas: Record<string, MetricDelta>;
  promotion_status: string;
  promotion_reason?: string;
}

export interface VersionHistoryItem {
  version: string;
  aliases: string[];
  role: "past_champion" | "past_challenger" | "none";
  role_timestamp: string;
  run_id: string | null;
  creation_timestamp: number;
  metrics: Record<string, number>;
}

export interface PaginatedVersionHistory {
  items: VersionHistoryItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface TrainingHistoryItem {
  version: string;
  aliases: string[];
  run_id: string;
  run_name: string | null;
  creation_timestamp: number;
  training_start_time: number | null;
  training_end_time: number | null;
  metrics: Record<string, number>;
  params: Record<string, string>;
}

export interface PromoteResult {
  version: string;
  alias: string;
}

export interface DeleteResult {
  version: string;
  deleted: boolean;
}

export interface WorkspaceConfig {
  workspace_url: string;
  model_name: string;
  model_url: string | null;
  experiment_id: string | null;
  experiment_url: string | null;
  notebook_urls: Record<string, string>;
  /** Omitted by older backends; UI defaults to regression. */
  metrics_profile?: "regression" | "classification";
}

export const api = {
  getConfig: () => fetchJson<WorkspaceConfig>("/api/config"),
  getPipelineStatus: () => fetchJson<PipelineStatus>("/api/pipeline/status"),
  triggerPipeline: () =>
    fetchJson<TriggerPipelineResponse>("/api/pipeline/run", { method: "POST" }),
  getPipelineHistory: (limit = 10) =>
    fetchJson<PipelineRun[]>(`/api/pipeline/history?limit=${limit}`),
  getExperimentRuns: (maxResults = 50, orderBy?: string) => {
    let url = `/api/experiments/runs?max_results=${maxResults}`;
    if (orderBy) url += `&order_by=${encodeURIComponent(orderBy)}`;
    return fetchJson<ExperimentRun[]>(url);
  },
  getRunDetail: (runId: string) =>
    fetchJson<RunDetail>(`/api/experiments/runs/${runId}`),
  getArtifactUrl: (runId: string, path: string) =>
    `${BASE}/api/experiments/runs/${runId}/artifacts/${path}`,
  getTrainingHistory: () =>
    fetchJson<TrainingHistoryItem[]>("/api/models/training-history"),
  getModelVersions: () => fetchJson<ModelVersion[]>("/api/models/versions"),
  getChampion: () => fetchJson<ModelDetail>("/api/models/champion"),
  getModelComparison: () =>
    fetchJson<ModelComparison>("/api/models/comparison"),
  getVersionHistory: (page = 1, pageSize = 10) =>
    fetchJson<PaginatedVersionHistory>(
      `/api/models/history?page=${page}&page_size=${pageSize}`
    ),
  promoteToChampion: (version: string) =>
    fetchJson<PromoteResult>(`/api/models/promote/${version}`, {
      method: "POST",
    }),
  deleteModelVersion: (version: string) =>
    fetchJson<DeleteResult>(`/api/models/versions/${version}`, {
      method: "DELETE",
    }),
};
