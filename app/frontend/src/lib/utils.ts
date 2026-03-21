export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || ms <= 0) return "--";
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

export function formatDate(timestamp: number | null | undefined): string {
  if (timestamp == null) return "--";
  return new Date(timestamp).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function truncateRunId(runId: string): string {
  if (!runId) return "--";
  return runId.substring(0, 8);
}

export function formatMetric(value: number | null | undefined, decimals = 4): string {
  if (value == null) return "--";
  return value.toFixed(decimals);
}

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
