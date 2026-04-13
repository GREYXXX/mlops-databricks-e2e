import { useState } from "react";
import { api } from "../api/client";
import type { TrainingHistoryItem, RunDetail, WorkspaceConfig } from "../api/client";
import { useApi } from "../hooks/useApi";
import { MetricsPanel } from "./MetricsPanel";
import { formatDate, formatMetric, cn } from "../lib/utils";
import {
  type MetricsProfile,
  HISTORY_TABLE_HEADERS,
  historyTableValues,
  KEY_PARAMS,
  KEY_PARAM_SHORT,
} from "../lib/metricsDisplay";

function AliasChips({ aliases }: { aliases: string[] }) {
  if (aliases.length === 0) return <span className="text-xs text-gray-400">—</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {aliases.map((alias) => (
        <span
          key={alias}
          className={cn(
            "inline-block rounded-full px-2 py-0.5 text-[11px] font-medium",
            alias.toLowerCase().startsWith("champion")
              ? "bg-amber-100 text-amber-700"
              : alias.toLowerCase().startsWith("challenger")
                ? "bg-blue-100 text-blue-700"
                : "bg-gray-100 text-gray-600"
          )}
        >
          {alias}
        </span>
      ))}
    </div>
  );
}

function TrainingDetailModal({
  item,
  runDetail,
  onClose,
  config,
  metricsProfile,
}: {
  item: TrainingHistoryItem;
  runDetail: RunDetail | null;
  onClose: () => void;
  config?: WorkspaceConfig;
  metricsProfile: MetricsProfile;
}) {
  const runUrl = config?.experiment_url && item.run_id
    ? `${config.experiment_url}/runs/${item.run_id}`
    : undefined;

  const detail = runDetail;
  const metrics = detail?.metrics ?? item.metrics;
  const params = detail?.params ?? item.params;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Training details"
    >
      <div
        className="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-2xl bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b px-6 py-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">
              Model v{item.version} — Training Details
            </h2>
            <p className="text-xs text-gray-500">
              {item.run_name || `Run ${item.run_id.substring(0, 8)}`}
              {item.training_start_time && (
                <> · Trained {formatDate(item.training_start_time)}</>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {runUrl && (
              <a
                href={runUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-md bg-blue-50 px-2 py-1 text-[11px] font-medium text-blue-700 ring-1 ring-blue-200 hover:bg-blue-100 transition-colors"
              >
                <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
                Open Run in MLflow
              </a>
            )}
            <button
              onClick={onClose}
              className="rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              aria-label="Close"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="space-y-6 p-6">
          {/* Aliases */}
          <div>
            <h3 className="mb-2 text-sm font-semibold text-gray-700">Aliases</h3>
            <AliasChips aliases={item.aliases} />
          </div>

          {/* Metrics */}
          <div>
            <h3 className="mb-3 text-sm font-semibold text-gray-700">Metrics</h3>
            <MetricsPanel metrics={metrics} metricsProfile={metricsProfile} />
          </div>

          {/* All Parameters */}
          {Object.keys(params).length > 0 && (
            <div>
              <h3 className="mb-3 text-sm font-semibold text-gray-700">
                Hyperparameters
              </h3>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {Object.entries(params).map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-gray-50 px-3 py-2 text-xs">
                    <span className="font-medium text-gray-500">{k}</span>
                    <p className="font-mono text-gray-800">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Artifacts */}
          {detail && detail.artifacts.length > 0 && (
            <div>
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Artifacts</h3>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {detail.artifacts
                  .filter(
                    (a) =>
                      !a.is_dir &&
                      (a.path.endsWith(".png") || a.path.endsWith(".jpg"))
                  )
                  .map((a) => (
                    <div
                      key={a.path}
                      className="overflow-hidden rounded-lg border border-gray-200"
                    >
                      <img
                        src={api.getArtifactUrl(detail.run_id, a.path)}
                        alt={a.path}
                        className="h-auto w-full object-contain"
                        loading="lazy"
                      />
                      <p className="bg-gray-50 px-2 py-1 text-[10px] text-gray-500 truncate">
                        {a.path}
                      </p>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="h-10 animate-pulse rounded bg-gray-200" />
      ))}
    </div>
  );
}

export function RunHistory() {
  const { data, isLoading, error } = useApi<TrainingHistoryItem[]>(
    "training-history",
    api.getTrainingHistory
  );
  const { data: config } = useApi<WorkspaceConfig>(
    "workspace-config",
    api.getConfig,
    { refreshInterval: 0 }
  );
  const [selectedItem, setSelectedItem] = useState<TrainingHistoryItem | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  async function handleRowClick(item: TrainingHistoryItem) {
    setSelectedItem(item);
    if (item.run_id) {
      setDetailLoading(true);
      try {
        const detail = await api.getRunDetail(item.run_id);
        setRunDetail(detail);
      } catch {
        setRunDetail(null);
      }
      setDetailLoading(false);
    }
  }

  if (isLoading) return <LoadingSkeleton />;
  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center text-sm text-red-600">
        Failed to load training history.
      </div>
    );
  }
  const metricsProfile: MetricsProfile =
    config?.metrics_profile ?? "regression";

  if (!data || data.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center text-sm text-gray-500">
        No model versions found. Run the pipeline to see results here.
      </div>
    );
  }

  return (
    <div>
      {/* Experiment link */}
      {config?.experiment_url && (
        <div className="mb-4 flex items-center justify-between">
          <p className="text-xs text-gray-500">
            {data.length} model version{data.length !== 1 ? "s" : ""} registered from experiment runs
          </p>
          <a
            href={config.experiment_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 ring-1 ring-blue-200 hover:bg-blue-100 transition-colors"
          >
            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            View Experiment in MLflow
          </a>
        </div>
      )}

      {/* Table */}
      <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <th className="px-4 py-3">Version</th>
              <th className="px-4 py-3">Aliases</th>
              <th className="px-4 py-3">Trained At</th>
              {HISTORY_TABLE_HEADERS[metricsProfile].map((h) => (
                <th key={h} className="px-4 py-3">
                  {h}
                </th>
              ))}
              <th className="px-4 py-3">Key Params</th>
              <th className="px-4 py-3 w-10"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.map((item) => {
              const metricCols = historyTableValues(item.metrics, metricsProfile);
              const paramKeys = KEY_PARAMS[metricsProfile];
              const paramSummary = paramKeys
                .filter((k) => item.params[k] != null)
                .map((k) => {
                  const v = item.params[k];
                  const short = KEY_PARAM_SHORT[k] ?? k;
                  return `${short}=${v}`;
                })
                .join(", ");

              return (
                <tr
                  key={item.version}
                  className="hover:bg-gray-50 transition-colors cursor-pointer"
                  onClick={() => handleRowClick(item)}
                >
                  <td className="px-4 py-3 font-medium text-gray-900">
                    v{item.version}
                  </td>
                  <td className="px-4 py-3">
                    <AliasChips aliases={item.aliases} />
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-600">
                    {item.training_start_time
                      ? formatDate(item.training_start_time)
                      : formatDate(item.creation_timestamp)}
                  </td>
                  {metricCols.map((val, i) => (
                    <td
                      key={i}
                      className="px-4 py-3 font-mono text-gray-700"
                    >
                      {formatMetric(val)}
                    </td>
                  ))}
                  <td className="px-4 py-3 text-xs text-gray-500 font-mono max-w-[200px] truncate">
                    {paramSummary || "—"}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {item.run_id && config?.experiment_url && (
                      <a
                        href={`${config.experiment_url}/runs/${item.run_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-gray-400 hover:text-blue-600 transition-colors"
                        title="Open in MLflow"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Detail modal */}
      {selectedItem && !detailLoading && (
        <TrainingDetailModal
          item={selectedItem}
          runDetail={runDetail}
          onClose={() => {
            setSelectedItem(null);
            setRunDetail(null);
          }}
          config={config ?? undefined}
          metricsProfile={metricsProfile}
        />
      )}

      {/* Loading overlay */}
      {detailLoading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20">
          <div className="rounded-xl bg-white px-8 py-6 shadow-xl">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-brand-600 border-t-transparent mx-auto" />
            <p className="mt-2 text-sm text-gray-600">Loading training details...</p>
          </div>
        </div>
      )}
    </div>
  );
}
