import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { PipelineStatus, WorkspaceConfig } from "../api/client";
import { useApi } from "../hooks/useApi";
import { StageCard } from "./StageCard";
import { CodeViewer } from "./CodeViewer";
import { Badge } from "./ui/Badge";
import { formatDate, formatDuration } from "../lib/utils";
import { STAGE_CODE } from "../data/stageCode";

const STAGE_ORDER = [
  "data_preparation",
  "feature_engineering",
  "model_training",
  "model_evaluation",
  "model_registration",
  "champion_management",
];

const RUNNING_STATES = new Set(["RUNNING", "PENDING", "QUEUED", "BLOCKED"]);

function LoadingSkeleton() {
  return (
    <div className="flex items-center gap-4 overflow-x-auto py-6 px-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4">
          <div className="h-36 w-40 animate-pulse rounded-xl bg-gray-200" />
          {i < 5 && (
            <div className="h-0.5 w-8 animate-pulse bg-gray-200" />
          )}
        </div>
      ))}
    </div>
  );
}

function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={`${className} animate-spin`} viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

export function PipelineView() {
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [isTriggering, setIsTriggering] = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);

  const [fastPoll, setFastPoll] = useState(false);

  const { data: config } = useApi<WorkspaceConfig>(
    "workspace-config",
    api.getConfig,
    { refreshInterval: 0 }
  );

  const { data, isLoading, error, refresh } = useApi<PipelineStatus>(
    "pipeline-status",
    api.getPipelineStatus,
    { refreshInterval: fastPoll ? 5000 : 30000 }
  );

  const isPipelineRunning =
    data != null && RUNNING_STATES.has(data.result || data.status);

  // Sync fast-poll state with pipeline running state
  useEffect(() => {
    setFastPoll(isPipelineRunning);
  }, [isPipelineRunning]);

  const handleRunPipeline = async () => {
    setIsTriggering(true);
    setTriggerError(null);
    try {
      await api.triggerPipeline();
      setFastPoll(true);
      refresh();
      setTimeout(() => refresh(), 3000);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to trigger pipeline";
      setTriggerError(msg);
    } finally {
      setIsTriggering(false);
    }
  };

  if (isLoading) return <LoadingSkeleton />;

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center text-sm text-red-600">
        Failed to load pipeline status. The pipeline may not have run yet.
      </div>
    );
  }

  if (!data || data.status === "NOT_FOUND") {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center text-sm text-gray-500">
        No pipeline job found. Deploy and run the pipeline to see status here.
      </div>
    );
  }

  // Build ordered task list, filling in missing stages with placeholder
  const taskMap = new Map(data.tasks.map((t) => [t.task_key, t]));
  const orderedTasks = STAGE_ORDER.map((key) => {
    return (
      taskMap.get(key) || {
        task_key: key,
        status: "PENDING",
        result: null,
        start_time: null,
        end_time: null,
        duration_ms: null,
        attempt_number: null,
      }
    );
  });

  const overallResult = data.result || data.status;
  const badgeVariant =
    overallResult === "SUCCESS"
      ? "success"
      : RUNNING_STATES.has(overallResult)
        ? "info"
        : overallResult === "FAILED"
          ? "error"
          : "neutral";

  const totalDuration =
    data.end_time && data.start_time
      ? data.end_time - data.start_time
      : null;

  const runDisabled = isPipelineRunning || isTriggering;

  return (
    <div>
      {/* Run summary bar */}
      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-gray-600">
        <Badge variant={badgeVariant}>{overallResult}</Badge>
        {data.run_id && (
          <span>
            Run <span className="font-mono text-xs">#{data.run_id}</span>
          </span>
        )}
        {data.start_time && (
          <span>Started {formatDate(data.start_time)}</span>
        )}
        {totalDuration != null && (
          <span className="text-gray-400">
            Duration: {formatDuration(totalDuration)}
          </span>
        )}

        {/* Run Pipeline button */}
        <button
          onClick={handleRunPipeline}
          disabled={runDisabled}
          className={
            "ml-auto inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors " +
            (runDisabled
              ? "cursor-not-allowed bg-gray-400"
              : "bg-brand-700 hover:bg-brand-800 active:bg-brand-900")
          }
        >
          {isTriggering || isPipelineRunning ? (
            <Spinner />
          ) : (
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
          {isTriggering
            ? "Starting..."
            : isPipelineRunning
              ? "Running..."
              : "Run Pipeline"}
        </button>
      </div>

      {/* Trigger error message */}
      {triggerError && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-600">
          {triggerError}
        </div>
      )}

      {/* Stage pipeline visualization */}
      <div className="flex items-center gap-2 overflow-x-auto py-6 px-2">
        {orderedTasks.map((task, i) => (
          <div key={task.task_key} className="flex items-center gap-2">
            <StageCard
              task={task}
              stageNumber={i + 1}
              isSelected={selectedStage === task.task_key}
              onSelect={() =>
                setSelectedStage(
                  selectedStage === task.task_key ? null : task.task_key
                )
              }
            />
            {/* Connection arrow */}
            {i < orderedTasks.length - 1 && (
              <svg
                width="32"
                height="12"
                viewBox="0 0 32 12"
                className="flex-shrink-0 text-gray-300"
                aria-hidden="true"
              >
                <line
                  x1="0"
                  y1="6"
                  x2="24"
                  y2="6"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <polygon
                  points="24,2 32,6 24,10"
                  fill="currentColor"
                />
              </svg>
            )}
          </div>
        ))}
      </div>

      {/* Code viewer panel */}
      {selectedStage && STAGE_CODE[selectedStage] && (
        <CodeViewer
          key={selectedStage}
          stage={STAGE_CODE[selectedStage]}
          onClose={() => setSelectedStage(null)}
          notebookUrl={config?.notebook_urls[selectedStage]}
        />
      )}
    </div>
  );
}
