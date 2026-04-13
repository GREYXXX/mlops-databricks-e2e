import { useState, useCallback } from "react";
import { api } from "../api/client";
import type {
  ModelComparison as ModelComparisonType,
  WorkspaceConfig,
  PaginatedVersionHistory,
} from "../api/client";
import { useApi } from "../hooks/useApi";
import { Badge } from "./ui/Badge";
import { Card, CardHeader, CardBody } from "./ui/Card";
import { formatMetric } from "../lib/utils";
import { cn } from "../lib/utils";
import {
  type MetricsProfile,
  COMPARISON_LABELS,
  COMPARISON_ORDER,
  HISTORY_TABLE_HEADERS,
  historyTableValues,
  LOWER_IS_BETTER,
  PRIMARY_METRIC,
  PRIMARY_METRIC_LABEL,
} from "../lib/metricsDisplay";

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
      <div className="h-64 animate-pulse rounded-xl bg-gray-200" />
      <div className="h-64 animate-pulse rounded-xl bg-gray-200" />
    </div>
  );
}

const PROMOTION_BADGES: Record<string, { variant: "success" | "error" | "warning" | "neutral"; text: string }> = {
  promoted: { variant: "success", text: "Champion Promoted" },
  champion_wins: { variant: "success", text: "Champion Retains Title" },
  challenger_wins: { variant: "warning", text: "Challenger Wins" },
  rejected: { variant: "error", text: "Challenger Rejected" },
  pending: { variant: "warning", text: "Pending Evaluation" },
  no_models: { variant: "neutral", text: "No Models Found" },
};

function MetricRow({
  label,
  value,
  isWinner,
  delta,
  improved,
}: {
  label: string;
  value: number | undefined;
  isWinner?: boolean;
  delta?: number;
  improved?: boolean;
}) {
  if (value === undefined) return null;
  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-lg px-3 py-2",
        isWinner ? "bg-emerald-50" : "bg-gray-50"
      )}
    >
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-sm font-bold text-gray-900">
          {formatMetric(value)}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span
            className={cn(
              "text-xs font-semibold",
              improved ? "text-emerald-600" : "text-red-500"
            )}
          >
            {improved ? "\u2191" : "\u2193"} {formatMetric(Math.abs(delta))}
          </span>
        )}
        {isWinner && (
          <span className="text-xs text-emerald-600 font-semibold">Best</span>
        )}
      </div>
    </div>
  );
}


function ModelVersionHistory({
  onChange,
  metricsProfile,
}: {
  onChange: () => void;
  metricsProfile: MetricsProfile;
}) {
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 10;
  const fetcher = useCallback(
    () => api.getVersionHistory(page, PAGE_SIZE),
    [page]
  );
  const { data, isLoading, refresh } = useApi<PaginatedVersionHistory>(
    `version-history-${page}`,
    fetcher
  );
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<{
    version: string;
    type: "promote" | "delete";
  } | null>(null);

  if (isLoading) {
    return <div className="mt-6 h-32 animate-pulse rounded-xl bg-gray-200" />;
  }
  if (!data || data.total === 0) return null;

  const handlePromote = async (version: string) => {
    setActionInProgress(version);
    setActionError(null);
    setConfirmAction(null);
    try {
      await api.promoteToChampion(version);
      await refresh();
      onChange();
    } catch (err) {
      setActionError(
        `Promote v${version} failed: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setActionInProgress(null);
    }
  };

  const handleDelete = async (version: string) => {
    setActionInProgress(version);
    setActionError(null);
    setConfirmAction(null);
    try {
      await api.deleteModelVersion(version);
      await refresh();
      onChange();
    } catch (err) {
      setActionError(
        `Delete v${version} failed: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setActionInProgress(null);
    }
  };

  return (
    <div className="mt-8">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
          All Model Versions
        </h3>
        <span className="text-xs text-gray-400">{data.total} versions</span>
      </div>
      {actionError && (
        <div className="mb-3 flex items-center justify-between rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-700">
          <span>{actionError}</span>
          <button
            onClick={() => setActionError(null)}
            className="ml-3 text-red-400 hover:text-red-600"
          >
            ✕
          </button>
        </div>
      )}
      <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <th className="px-4 py-3">Version</th>
              <th className="px-4 py-3">Aliases</th>
              {HISTORY_TABLE_HEADERS[metricsProfile].map((h) => (
                <th key={h} className="px-4 py-3">
                  {h}
                </th>
              ))}
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.items.map((item) => {
              const metricCols = historyTableValues(
                item.metrics,
                metricsProfile
              );
              return (
                <tr
                  key={item.version}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <td className="px-4 py-3 font-medium text-gray-900">
                    v{item.version}
                  </td>
                  <td className="px-4 py-3">
                    {item.aliases.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {item.aliases.map((alias) => (
                          <span
                            key={alias}
                            className={cn(
                              "inline-block rounded-full px-2 py-0.5 text-xs font-medium",
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
                    ) : (
                      <span className="text-xs text-gray-400">—</span>
                    )}
                  </td>
                  {metricCols.map((val, i) => (
                    <td
                      key={i}
                      className="px-4 py-3 font-mono text-gray-700"
                    >
                      {formatMetric(val)}
                    </td>
                  ))}
                  <td className="px-4 py-3 text-right">
                    {confirmAction?.version === item.version ? (
                      <span className="inline-flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {confirmAction.type === "promote"
                            ? "Promote?"
                            : "Delete?"}
                        </span>
                        <button
                          onClick={() =>
                            confirmAction.type === "promote"
                              ? handlePromote(item.version)
                              : handleDelete(item.version)
                          }
                          disabled={actionInProgress !== null}
                          className={cn(
                            "rounded px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50 transition-colors",
                            confirmAction.type === "promote"
                              ? "bg-amber-500 hover:bg-amber-600"
                              : "bg-red-500 hover:bg-red-600"
                          )}
                        >
                          {actionInProgress === item.version
                            ? "..."
                            : "Yes"}
                        </button>
                        <button
                          onClick={() => setConfirmAction(null)}
                          className="rounded bg-gray-200 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-300 transition-colors"
                        >
                          No
                        </button>
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5">
                        <button
                          onClick={() =>
                            setConfirmAction({
                              version: item.version,
                              type: "promote",
                            })
                          }
                          disabled={actionInProgress !== null}
                          className="inline-flex items-center gap-1 rounded-md bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700 ring-1 ring-amber-200 hover:bg-amber-100 disabled:opacity-50 transition-colors"
                          title="Promote to Champion"
                        >
                          <svg
                            className="h-3 w-3"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M5 10l7-7m0 0l7 7m-7-7v18"
                            />
                          </svg>
                          Promote
                        </button>
                        <button
                          onClick={() =>
                            setConfirmAction({
                              version: item.version,
                              type: "delete",
                            })
                          }
                          disabled={actionInProgress !== null}
                          className="inline-flex items-center gap-1 rounded-md bg-red-50 px-2.5 py-1 text-xs font-medium text-red-600 ring-1 ring-red-200 hover:bg-red-100 disabled:opacity-50 transition-colors"
                          title="Delete version"
                        >
                          <svg
                            className="h-3 w-3"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                          Delete
                        </button>
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {/* Pagination */}
        {data.total_pages > 1 && (
          <div className="flex items-center justify-between border-t border-gray-100 bg-gray-50 px-4 py-3">
            <span className="text-xs text-gray-500">
              Page {data.page} of {data.total_pages}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={data.page <= 1}
                className="rounded-md bg-white px-3 py-1 text-xs font-medium text-gray-600 ring-1 ring-gray-200 hover:bg-gray-100 disabled:opacity-40 transition-colors"
              >
                Prev
              </button>
              <button
                onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
                disabled={data.page >= data.total_pages}
                className="rounded-md bg-white px-3 py-1 text-xs font-medium text-gray-600 ring-1 ring-gray-200 hover:bg-gray-100 disabled:opacity-40 transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function ModelComparison() {
  const { data, isLoading, error, refresh: refreshComparison } = useApi<ModelComparisonType>(
    "model-comparison",
    api.getModelComparison
  );
  const { data: config } = useApi<WorkspaceConfig>(
    "workspace-config",
    api.getConfig,
    { refreshInterval: 0 }
  );

  if (isLoading) return <LoadingSkeleton />;
  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center text-sm text-red-600">
        Failed to load model comparison.
      </div>
    );
  }
  if (!data) return null;

  const metricsProfile: MetricsProfile =
    config?.metrics_profile ?? "regression";
  const metricOrder = COMPARISON_ORDER[metricsProfile];
  const metricLabels = COMPARISON_LABELS[metricsProfile];
  const lowerIsBetter = LOWER_IS_BETTER[metricsProfile];
  const primaryKey = PRIMARY_METRIC[metricsProfile];
  const primaryLabel = PRIMARY_METRIC_LABEL[metricsProfile];

  const promotionBadge = PROMOTION_BADGES[data.promotion_status] || {
    variant: "neutral" as const,
    text: data.promotion_status,
  };

  // Determine which metrics to show: from deltas if available, else from individual metrics
  const deltas = data.deltas ?? {};
  const deltaKeys = Object.keys(deltas);
  const hasDeltas = deltaKeys.length > 0;

  // Gather all metric keys from both sides
  const allMetricKeys = metricOrder.filter(
    (k) =>
      deltas[k] !== undefined ||
      data.champion.metrics[k] !== undefined ||
      data.challenger.metrics[k] !== undefined
  );

  return (
    <div>
      {/* Model link */}
      {config?.model_url && (
        <div className="mb-4 flex justify-end">
          <a
            href={config.model_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 ring-1 ring-blue-200 hover:bg-blue-100 transition-colors"
          >
            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            View in Unity Catalog
          </a>
        </div>
      )}

      {/* Promotion banner + reason */}
      <div className="mb-4 flex flex-col items-center gap-2">
        <Badge variant={promotionBadge.variant} className="px-4 py-1.5 text-sm">
          {promotionBadge.text}
        </Badge>
        {data.promotion_reason && (
          <p className="text-xs text-gray-500 text-center max-w-lg">
            {data.promotion_reason}
          </p>
        )}
      </div>

      {/* Promotion decision explanation */}
      {hasDeltas && deltas[primaryKey] && (
        <div className="mb-4 rounded-lg border border-gray-200 bg-white p-4">
          <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-3">
            Promotion Decision Logic
          </h4>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-amber-400" />
              <span className="text-gray-700">
                Champion v{data.champion.version}
              </span>
              <span className="font-mono text-gray-500">
                {primaryLabel}{" "}
                {deltas[primaryKey]
                  ? formatMetric(deltas[primaryKey].champion)
                  : "N/A"}
              </span>
            </div>
            <span className="text-gray-400">vs</span>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-blue-400" />
              <span className="text-gray-700">
                Challenger v{data.challenger.version}
              </span>
              <span className="font-mono text-gray-500">
                {primaryLabel}{" "}
                {deltas[primaryKey]
                  ? formatMetric(deltas[primaryKey].challenger)
                  : "N/A"}
              </span>
            </div>
            <span className="ml-2 text-xs">
              {deltas[primaryKey]?.improved ? (
                <span className="rounded bg-emerald-100 px-2 py-0.5 text-emerald-700 font-medium">
                  {metricsProfile === "classification"
                    ? "Challenger is better (higher weighted F1)"
                    : "Challenger is better (lower RMSE)"}
                </span>
              ) : (
                <span className="rounded bg-amber-100 px-2 py-0.5 text-amber-700 font-medium">
                  {metricsProfile === "classification"
                    ? "Champion retains (higher weighted F1)"
                    : "Champion retains (lower RMSE)"}
                </span>
              )}
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* Champion Card */}
        <Card className="border-2 border-amber-200">
          <CardHeader className="bg-amber-50">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-amber-800">Champion</h3>
                <p className="text-xs text-amber-600">
                  {data.champion.version
                    ? `Version ${data.champion.version}`
                    : "Not assigned"}
                </p>
              </div>
              <span className="text-2xl" aria-label="trophy">
                &#x1F3C6;
              </span>
            </div>
          </CardHeader>
          <CardBody>
            {allMetricKeys.length > 0 ? (
              <div className="space-y-3">
                {allMetricKeys.map((key) => {
                  const delta = deltas[key];
                  const value = delta?.champion ?? data.champion.metrics[key];
                  const isWinner = delta
                    ? lowerIsBetter.has(key)
                      ? delta.champion <= delta.challenger
                      : delta.champion >= delta.challenger
                    : undefined;
                  return (
                    <MetricRow
                      key={key}
                      label={metricLabels[key] || key}
                      value={value}
                      isWinner={isWinner === true}
                    />
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-gray-400 text-center py-4">
                No metrics available
              </p>
            )}
          </CardBody>
        </Card>

        {/* Challenger Card */}
        <Card className="border-2 border-blue-200">
          <CardHeader className="bg-blue-50">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-blue-800">Challenger</h3>
                <p className="text-xs text-blue-600">
                  {data.challenger.version
                    ? `Version ${data.challenger.version}`
                    : "Not assigned"}
                </p>
              </div>
              <span className="text-2xl" aria-label="boxing glove">
                &#x1F94A;
              </span>
            </div>
          </CardHeader>
          <CardBody>
            {data.challenger.version ? (
              allMetricKeys.length > 0 ? (
                <div className="space-y-3">
                  {allMetricKeys.map((key) => {
                    const delta = deltas[key];
                    const value =
                      delta?.challenger ?? data.challenger.metrics[key];
                    const isWinner = delta
                      ? lowerIsBetter.has(key)
                        ? delta.challenger < delta.champion
                        : delta.challenger > delta.champion
                      : undefined;
                    return (
                      <MetricRow
                        key={key}
                        label={metricLabels[key] || key}
                        value={value}
                        isWinner={isWinner === true}
                        delta={delta?.delta}
                        improved={delta?.improved}
                      />
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-gray-400 text-center py-4">
                  No metrics available
                </p>
              )
            ) : (
              <div className="flex flex-col items-center gap-2 py-6 text-center">
                <div className="rounded-full bg-emerald-100 p-2">
                  <svg
                    className="h-5 w-5 text-emerald-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-700">
                  No active Challenger
                </p>
                <p className="text-xs text-gray-400 max-w-xs">
                  The last Challenger was promoted to Champion. Run a new
                  pipeline to create a new Challenger model.
                </p>
              </div>
            )}
          </CardBody>
        </Card>
      </div>

      {/* All Model Versions */}
      <ModelVersionHistory
        metricsProfile={metricsProfile}
        onChange={() => refreshComparison()}
      />
    </div>
  );
}
