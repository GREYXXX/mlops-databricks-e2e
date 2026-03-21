import { useState, useCallback } from "react";
import { PipelineView } from "./components/PipelineView";
import { ModelComparison } from "./components/ModelComparison";
import { RunHistory } from "./components/RunHistory";
import { cn } from "./lib/utils";
import { mutate, preload } from "swr";
import { api } from "./api/client";

type Tab = "pipeline" | "models" | "history";

const TABS: { key: Tab; label: string }[] = [
  { key: "pipeline", label: "Pipeline" },
  { key: "models", label: "Models" },
  { key: "history", label: "Training History" },
];

// Prefetch other tabs' data in the background on app load
preload("workspace-config", api.getConfig);
preload("model-comparison", api.getModelComparison);
preload("version-history-1", () => api.getVersionHistory(1, 10));
preload("training-history", api.getTrainingHistory);

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("pipeline");
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const handleRefresh = useCallback(() => {
    mutate(() => true, undefined, { revalidate: true });
    setLastRefresh(new Date());
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-700 text-lg font-bold text-white">
              M
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900">
                MLOps E2E Dashboard
              </h1>
              <p className="text-xs text-gray-400">
                California Housing Pipeline
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs text-gray-400">
              Last refresh:{" "}
              {lastRefresh.toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
            <button
              onClick={handleRefresh}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:bg-gray-50 active:bg-gray-100"
              aria-label="Refresh data"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Tab navigation */}
      <div className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <nav className="-mb-px flex gap-6" aria-label="Tabs">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={cn(
                  "whitespace-nowrap border-b-2 py-3 text-sm font-medium transition-colors",
                  activeTab === tab.key
                    ? "border-brand-600 text-brand-600"
                    : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700"
                )}
                aria-current={activeTab === tab.key ? "page" : undefined}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {activeTab === "pipeline" && (
          <section aria-label="Pipeline status">
            <h2 className="mb-4 text-base font-semibold text-gray-800">
              Pipeline Status
            </h2>
            <PipelineView />
          </section>
        )}

        {activeTab === "models" && (
          <section aria-label="Model comparison">
            <h2 className="mb-4 text-base font-semibold text-gray-800">
              Champion vs Challenger
            </h2>
            <ModelComparison />
          </section>
        )}

        {activeTab === "history" && (
          <section aria-label="Training history">
            <h2 className="mb-4 text-base font-semibold text-gray-800">
              Training History
            </h2>
            <RunHistory />
          </section>
        )}
      </main>
    </div>
  );
}
