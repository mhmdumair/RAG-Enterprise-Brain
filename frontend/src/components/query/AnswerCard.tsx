"use client";

import { Answer, Document } from "@/lib/types";
import { useAppStore } from "@/store/appStore";
import { formatScore } from "@/lib/utils";

interface Props {
  result: Answer;
  index: number;
  documents: Document[];
}

export function AnswerCard({ result, index, documents }: Props) {
  const { activeAnswer, openAnswer } = useAppStore();
  const isActive = activeAnswer?.span_hash === result.span_hash;

  const handleClick = () => {
    openAnswer(result, documents);
  };

  const confidenceColor =
    result.span_score > 0.7
      ? "text-emerald-400"
      : result.span_score > 0.4
      ? "text-yellow-400"
      : "text-red-400";

  return (
    <div
      onClick={handleClick}
      className={`
        min-w-0 rounded border px-4 py-3 cursor-pointer text-xs transition-all duration-150
        ${isActive
          ? "border-[#4f8ef7] bg-[#0d1525]"
          : "border-[#1e2230] bg-[#12141a] hover:border-[#2a3050]"
        }
      `}
    >
      {/* Top row */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-[#3a4466] uppercase tracking-widest">
          #{index + 1} · p.{result.page_number}
        </span>
        <span className={`text-[10px] font-semibold ${confidenceColor}`}>
          {formatScore(result.span_score)}
        </span>
      </div>

      {/* Answer span */}
      <p className="text-[#c8d8f0] leading-relaxed mb-2 wrap-break-word">
        &ldquo;{result.text}&rdquo;
      </p>

      {/* Source badge */}
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center rounded bg-[#1a2035] px-2 py-0.5
                         text-[10px] text-[#4f8ef7] border border-[#2a3560] truncate max-w-full">
          📎 {result.filename}
        </span>
      </div>
    </div>
  );
}