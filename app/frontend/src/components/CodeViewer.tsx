import { useState } from "react";
import { Highlight, themes } from "prism-react-renderer";
import type { StageCodeEntry } from "../data/stageCode";
import { extractFunctionCode } from "../data/stageCode";
import { cn } from "../lib/utils";

interface CodeViewerProps {
  stage: StageCodeEntry;
  onClose: () => void;
  notebookUrl?: string;
}

type Tab = "source" | "notebook";

export function CodeViewer({ stage, onClose, notebookUrl }: CodeViewerProps) {
  const [activeTab, setActiveTab] = useState<Tab>("source");
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);

  const funcCode = selectedFunction
    ? extractFunctionCode(stage.sourceCode, selectedFunction)
    : null;

  const code = funcCode ?? (activeTab === "source" ? stage.sourceCode : stage.notebookCode);
  const fileName = selectedFunction
    ? `${stage.sourceFile} > ${selectedFunction}()`
    : activeTab === "source"
      ? stage.sourceFile
      : stage.notebookFile;

  const hasImports =
    stage.importedFunctions && stage.importedFunctions.length > 0;

  return (
    <div className="mt-4 rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-4 py-3">
        <div className="flex items-center gap-4">
          <h3 className="text-sm font-semibold text-gray-800">
            {stage.label}
          </h3>
          <p className="text-xs text-gray-500">{stage.description}</p>
          {notebookUrl && (
            <a
              href={notebookUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-md bg-blue-50 px-2 py-1 text-[11px] font-medium text-blue-700 ring-1 ring-blue-200 hover:bg-blue-100 transition-colors whitespace-nowrap"
              title="Open notebook in Databricks workspace"
            >
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              Open in Workspace
            </a>
          )}
        </div>
        <button
          onClick={onClose}
          className="flex h-6 w-6 items-center justify-center rounded text-gray-400 hover:bg-gray-200 hover:text-gray-600 transition-colors"
          aria-label="Close code viewer"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <path d="M1 1l12 12M13 1L1 13" />
          </svg>
        </button>
      </div>

      {/* Tabs + file name */}
      <div className="flex items-center gap-2 border-b border-gray-200 px-4">
        <button
          onClick={() => {
            setActiveTab("source");
            setSelectedFunction(null);
          }}
          className={cn(
            "px-3 py-2 text-xs font-medium border-b-2 transition-colors",
            activeTab === "source" && !selectedFunction
              ? "border-brand-600 text-brand-700"
              : "border-transparent text-gray-500 hover:text-gray-700"
          )}
        >
          Source Code
        </button>
        <button
          onClick={() => {
            setActiveTab("notebook");
            setSelectedFunction(null);
          }}
          className={cn(
            "px-3 py-2 text-xs font-medium border-b-2 transition-colors",
            activeTab === "notebook" && !selectedFunction
              ? "border-brand-600 text-brand-700"
              : "border-transparent text-gray-500 hover:text-gray-700"
          )}
        >
          Notebook
        </button>
        {selectedFunction && (
          <span className="border-b-2 border-brand-600 px-3 py-2 text-xs font-medium text-brand-700">
            {selectedFunction}()
          </span>
        )}
        <span className="ml-auto text-[11px] font-mono text-gray-400">
          {fileName}
        </span>
      </div>

      {/* Imported functions bar */}
      {hasImports && (
        <div className="flex items-center gap-2 border-b border-gray-100 bg-gray-50/50 px-4 py-2">
          <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wide">
            Imported:
          </span>
          {stage.importedFunctions!.map((fn) => (
            <button
              key={fn.name}
              onClick={() => setSelectedFunction(selectedFunction === fn.name ? null : fn.name)}
              className={cn(
                "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-mono transition-colors",
                selectedFunction === fn.name
                  ? "bg-brand-100 text-brand-800 ring-1 ring-brand-300"
                  : "bg-white text-blue-700 ring-1 ring-gray-200 hover:bg-blue-50 hover:ring-blue-300"
              )}
              title={`View ${fn.name} from ${fn.module}`}
            >
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
              {fn.name}
            </button>
          ))}
        </div>
      )}

      {/* Code block */}
      <div className="max-h-96 overflow-auto">
        <Highlight theme={themes.github} code={code} language="python">
          {({ style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className="text-xs leading-5 p-4 m-0"
              style={{ ...style, background: "transparent" }}
            >
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  <span className="inline-block w-10 text-right mr-4 text-gray-300 select-none">
                    {i + 1}
                  </span>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}
