import { cn } from "../../lib/utils";

type StatusColor = "success" | "running" | "failed" | "pending";

const colorMap: Record<StatusColor, string> = {
  success: "bg-emerald-500",
  running: "bg-blue-500",
  failed: "bg-red-500",
  pending: "bg-gray-400",
};

interface StatusDotProps {
  status: StatusColor;
  className?: string;
  label?: string;
}

export function StatusDot({ status, className, label }: StatusDotProps) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span
        className={cn(
          "inline-block h-2.5 w-2.5 rounded-full",
          colorMap[status],
          status === "running" && "animate-pulse-dot"
        )}
        aria-label={label || status}
      />
      {label && <span className="text-xs text-gray-600">{label}</span>}
    </span>
  );
}

export function mapTaskStatus(
  status: string,
  result: string | null
): StatusColor {
  const s = status?.toUpperCase() || "";
  const r = result?.toUpperCase() || "";

  if (s === "TERMINATED" || s === "COMPLETED" || s === "FINISHED") {
    if (r === "SUCCESS" || r === "SUCCEEDED") return "success";
    if (r === "FAILED" || r === "TIMEDOUT" || r === "CANCELED") return "failed";
    return "success";
  }
  if (s === "RUNNING" || s === "PENDING" || s === "QUEUED" || s === "BLOCKED") {
    return "running";
  }
  if (s === "SKIPPED" || s === "UPSTREAM_FAILED") return "failed";
  if (s === "INTERNAL_ERROR") return "failed";
  return "pending";
}
