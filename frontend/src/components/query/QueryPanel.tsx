"use client";

import { useState, useRef } from "react";
import { useQuery } from "@/hooks/useQuery";
import { useAppStore } from "@/store/appStore";
import { AnswerCard } from "./AnswerCard";

export function QueryPanel() {
  const [input, setInput] = useState("");
  const { results, isQuerying, error, runQuery } = useQuery();
  const { documents } = useAppStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = async () => {
    const q = input.trim();
    if (!q || isQuerying) return;
    setInput("");
    await runQuery(q);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-3 border-b border-[#1e2230] shrink-0">
        <p className="text-[10px] tracking-widest uppercase text-[#3a4466]">
          Audit Query
        </p>
        <h2 className="text-sm font-semibold text-[#c0ccdd] mt-0.5">
          Ask a question
        </h2>
      </div>

      {/* Results area */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
        {!isQuerying && results.length === 0 && !error && (
          <div className="text-xs text-[#2e3450] mt-8 text-center">
            <p className="text-2xl mb-2">🔍</p>
            <p>Type a question to extract verified spans from your documents.</p>
          </div>
        )}

        {isQuerying && (
          <div className="text-xs text-[#4f8ef7] animate-pulse">
            Auditing documents…
          </div>
        )}

        {error && (
          <div className="text-xs text-red-400 rounded border border-red-900 bg-red-950 px-3 py-2">
            {error}
          </div>
        )}

        {results.map((r, i) => (
          <AnswerCard key={r.span_hash} result={r} index={i} documents={documents} />
        ))}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-[#1e2230] px-4 py-3 flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder="e.g. What is the data retention policy?"
          className="flex-1 resize-none rounded bg-[#12141a] border border-[#1e2230]
                     px-3 py-2 text-xs text-[#c8d8f0] placeholder-[#2e3450]
                     focus:outline-none focus:border-[#4f8ef7] transition-colors"
        />
        <button
          onClick={submit}
          disabled={isQuerying || !input.trim()}
          className="shrink-0 rounded bg-[#4f8ef7] px-4 py-2 text-xs font-semibold
                     text-white transition-opacity disabled:opacity-30 hover:opacity-90"
        >
          Run
        </button>
      </div>
    </div>
  );
}