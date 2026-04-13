import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { formatMetric } from "../lib/utils";
import {
  type MetricsProfile,
  PANEL_METRICS,
  panelFindMetric,
} from "../lib/metricsDisplay";

interface MetricsPanelProps {
  metrics: Record<string, number>;
  metricsProfile?: MetricsProfile;
}

export function MetricsPanel({
  metrics,
  metricsProfile = "regression",
}: MetricsPanelProps) {
  const keyMetrics = PANEL_METRICS[metricsProfile];
  const keyValues = keyMetrics.map((m) => ({
    ...m,
    value: panelFindMetric(metrics, m.key),
  }));

  const chartData = keyValues
    .filter((m) => m.value != null)
    .map((m) => ({
      name: m.label,
      value: m.value!,
    }));

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {keyValues.map((m) => (
          <div
            key={m.key}
            className="rounded-lg border border-gray-100 bg-gray-50 px-4 py-3 text-center"
          >
            <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
              {m.label}
            </p>
            <p className="mt-1 text-xl font-bold text-brand-800">
              {m.value != null ? formatMetric(m.value) : "--"}
            </p>
          </div>
        ))}
      </div>

      {chartData.length > 0 && (
        <div className="mt-4 h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={metricsProfile === "classification" ? [0, 1] : undefined}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                  fontSize: "12px",
                }}
              />
              <Bar
                dataKey="value"
                fill="#2563eb"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
