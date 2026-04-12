/** Dashboard metric columns: regression (housing) vs classification (newsgroups). */

export type MetricsProfile = "regression" | "classification";

/** MLflow keys to try in order (first hit wins). */
export const REGRESSION_HISTORY_KEYS = [
  ["test_rmse", "rmse"],
  ["test_mae", "mae"],
  ["test_r2", "r2"],
] as const;

export const CLASSIFICATION_HISTORY_KEYS = [
  ["eval_ensemble_f1_weighted", "ensemble_f1_weighted"],
  ["eval_ensemble_accuracy", "ensemble_accuracy"],
  ["eval_ensemble_precision", "ensemble_precision"],
] as const;

export const HISTORY_TABLE_HEADERS: Record<MetricsProfile, string[]> = {
  regression: ["RMSE", "MAE", "R\u00B2"],
  classification: ["F1 (weighted)", "Accuracy", "Precision"],
};

export function pickFirstMetric(
  metrics: Record<string, number>,
  candidates: readonly string[]
): number | null {
  for (const k of candidates) {
    if (metrics[k] != null) return metrics[k]!;
  }
  return null;
}

export function historyTableValues(
  metrics: Record<string, number>,
  profile: MetricsProfile
): (number | null)[] {
  const groups =
    profile === "classification"
      ? CLASSIFICATION_HISTORY_KEYS
      : REGRESSION_HISTORY_KEYS;
  return groups.map((candidates) => pickFirstMetric(metrics, candidates));
}

/** Comparison cards + deltas: API-normalized keys (match backend). */
export const COMPARISON_LABELS: Record<MetricsProfile, Record<string, string>> = {
  regression: {
    rmse: "RMSE",
    mae: "MAE",
    r2: "R\u00B2",
    mape: "MAPE",
    median_ae: "Median AE",
  },
  classification: {
    f1_weighted: "F1 (weighted)",
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
  },
};

export const COMPARISON_ORDER: Record<MetricsProfile, string[]> = {
  regression: ["rmse", "mae", "r2", "mape", "median_ae"],
  classification: ["f1_weighted", "accuracy", "precision", "recall"],
};

export const LOWER_IS_BETTER: Record<MetricsProfile, Set<string>> = {
  regression: new Set(["rmse", "mae", "mape", "median_ae"]),
  classification: new Set(),
};

export const PRIMARY_METRIC: Record<MetricsProfile, string> = {
  regression: "rmse",
  classification: "f1_weighted",
};

export const PRIMARY_METRIC_LABEL: Record<MetricsProfile, string> = {
  regression: "RMSE",
  classification: "Weighted F1",
};

/** Metric panel bar chart (top metrics). */
export const PANEL_METRICS: Record<
  MetricsProfile,
  { key: string; label: string; lower: boolean }[]
> = {
  regression: [
    { key: "rmse", label: "RMSE", lower: true },
    { key: "mae", label: "MAE", lower: true },
    { key: "r2", label: "R\u00B2", lower: false },
    { key: "mape", label: "MAPE", lower: true },
  ],
  classification: [
    { key: "f1_weighted", label: "F1 (weighted)", lower: false },
    { key: "accuracy", label: "Accuracy", lower: false },
    { key: "precision", label: "Precision", lower: false },
    { key: "recall", label: "Recall", lower: false },
  ],
};

export function panelFindMetric(
  metrics: Record<string, number>,
  search: string
): number | null {
  if (metrics[search] != null) return metrics[search];
  for (const [k, v] of Object.entries(metrics)) {
    if (k.toLowerCase().includes(search.toLowerCase())) return v;
  }
  return null;
}

/** Condensed param keys for training history row. */
export const KEY_PARAMS: Record<MetricsProfile, string[]> = {
  regression: ["num_leaves", "learning_rate", "n_estimators", "max_depth"],
  classification: ["textcnn_epochs", "embed_dim", "num_classes", "feature_run_id"],
};

export const KEY_PARAM_SHORT: Record<string, string> = {
  num_leaves: "leaves",
  learning_rate: "lr",
  n_estimators: "trees",
  max_depth: "depth",
  textcnn_epochs: "cnn_ep",
  embed_dim: "emb",
  num_classes: "classes",
  feature_run_id: "feat_run",
};
