import { useEffect, useState } from "react";
import type { PipelineTask } from "../api/client";
import { StatusDot, mapTaskStatus } from "./ui/StatusDot";
import { formatDuration } from "../lib/utils";
import { cn } from "../lib/utils";

const STAGE_LABELS: Record<string, string> = {
  data_preparation: "Data Preparation",
  feature_engineering: "Feature Engineering",
  model_training: "Model Training",
  model_evaluation: "Model Evaluation",
  model_registration: "Model Registration",
  champion_management: "Champion Management",
};

interface StageCardProps {
  task: PipelineTask;
  stageNumber: number;
  isSelected?: boolean;
  onSelect?: () => void;
}

function ElapsedTimer({ startTime }: { startTime: number }) {
  const [elapsed, setElapsed] = useState(() => Date.now() - startTime);

  useEffect(() => {
    const id = setInterval(() => setElapsed(Date.now() - startTime), 1000);
    return () => clearInterval(id);
  }, [startTime]);

  return (
    <span className="tabular-nums text-blue-600 font-semibold">
      {formatDuration(elapsed)}
    </span>
  );
}

export function StageCard({ task, stageNumber, isSelected, onSelect }: StageCardProps) {
  const statusColor = mapTaskStatus(task.status, task.result);

  const statusLabel =
    task.result && task.result !== "UNKNOWN"
      ? task.result
      : task.status;

  const isRunning = statusColor === "running" && task.status === "RUNNING";

  const borderColor = {
    success: "border-emerald-300 bg-emerald-50/30",
    running: "border-blue-300 bg-blue-50/30",
    failed: "border-red-300 bg-red-50/30",
    pending: "border-gray-200 bg-white",
  }[statusColor];

  return (
    <div
      className={cn(
        "relative flex w-40 flex-col rounded-xl border-2 p-4 transition-all duration-200",
        "hover:shadow-lg cursor-pointer select-none",
        isSelected ? "border-brand-600 ring-2 ring-brand-200" : borderColor
      )}
      onClick={() => onSelect?.()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect?.();
      }}
      aria-expanded={isSelected}
      aria-label={`Stage ${stageNumber}: ${STAGE_LABELS[task.task_key] || task.task_key}`}
    >
      {/* Stage number badge */}
      <div className="absolute -left-3 -top-3 flex h-7 w-7 items-center justify-center rounded-full bg-brand-700 text-xs font-bold text-white shadow">
        {stageNumber}
      </div>

      {/* Status dot */}
      <div className="mb-2 flex justify-end">
        <StatusDot status={statusColor} />
      </div>

      {/* Stage name */}
      <h3 className="text-sm font-semibold text-gray-800 leading-tight">
        {STAGE_LABELS[task.task_key] || task.task_key}
      </h3>

      {/* Duration - live timer for running tasks */}
      <p className="mt-1 text-xs text-gray-500">
        {isRunning && task.start_time ? (
          <ElapsedTimer startTime={task.start_time} />
        ) : (
          formatDuration(task.duration_ms)
        )}
      </p>

      {/* Status label */}
      <p className="mt-1 text-[10px] font-medium uppercase tracking-wide text-gray-400">
        {statusLabel}
      </p>

      {/* Expanded details */}
      {isSelected && (
        <div className="mt-3 border-t border-gray-200 pt-3 text-xs text-gray-600 space-y-1">
          <div>
            <span className="font-medium">Attempt:</span>{" "}
            {task.attempt_number ?? "--"}
          </div>
          {task.start_time && (
            <div>
              <span className="font-medium">Started:</span>{" "}
              {new Date(task.start_time).toLocaleTimeString()}
            </div>
          )}
          {task.end_time && (
            <div>
              <span className="font-medium">Ended:</span>{" "}
              {new Date(task.end_time).toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
