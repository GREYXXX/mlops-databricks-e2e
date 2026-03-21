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

interface MetricsPanelProps {
  metrics: Record<string, number>;
}

const KEY_METRICS = [
  { key: "rmse", label: "RMSE", lower: true },
  { key: "mae", label: "MAE", lower: true },
  { key: "r2", label: "R\u00B2", lower: false },
  { key: "mape", label: "MAPE", lower: true },
];

function findMetric(
  metrics: Record<string, number>,
  search: string
): number | null {
  // Try exact match, then partial
  if (metrics[search] != null) return metrics[search];
  for (const [k, v] of Object.entries(metrics)) {
    if (k.toLowerCase().includes(search.toLowerCase())) return v;
  }
  return null;
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const keyValues = KEY_METRICS.map((m) => ({
    ...m,
    value: findMetric(metrics, m.key),
  }));

  const chartData = keyValues
    .filter((m) => m.value != null)
    .map((m) => ({
      name: m.label,
      value: m.value!,
    }));

  return (
    <div>
      {/* Metric cards */}
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

      {/* Chart */}
      {chartData.length > 0 && (
        <div className="mt-4 h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
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
