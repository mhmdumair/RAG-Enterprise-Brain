"use client";

import { useState, useRef, useEffect } from "react";
import { useQuery } from "@/hooks/useQuery";
import { useAppStore } from "@/store/appStore";
import { AnswerCard } from "./AnswerCard";

interface Props {
  onToggleSidebar: () => void;
}

export function ChatPanel({ onToggleSidebar }: Props) {
  const [input, setInput] = useState("");
  const { results, currentResult, isQuerying, error, runQuery } = useQuery();
  const { queryHistory, documents } = useAppStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [queryHistory, results, isQuerying]);

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

  const recentHistory = queryHistory.slice().reverse();
  const allMessages = currentResult && queryHistory[0]?.query !== currentResult.query
    ? [...recentHistory, currentResult]
    : recentHistory;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-3 border-b border-[#1e2230] shrink-0 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-[#c0ccdd]">
            Enterprise Brain Chat
          </h2>
          <p className="text-[10px] text-[#3a4466]">
            Ask questions about your documents
          </p>
        </div>
        <button
          onClick={onToggleSidebar}
          className="rounded bg-[#171b2c] px-3 py-1 text-xs text-[#c8d8f0] transition hover:bg-[#1f2840] flex items-center gap-1"
          title="Toggle Sidebar"
        >
          ☰ <span className="hidden sm:inline">Menu</span>
        </button>
      </div>

      {/* Messages area */}
      <div className="flex-1 min-w-0 overflow-y-auto px-5 py-4 space-y-4">
        {allMessages.length === 0 && !isQuerying && (
          <div className="text-center text-[#2e3450] mt-8">
            <div className="text-4xl mb-4">🧠</div>
            <p className="text-sm">Welcome to Enterprise Brain</p>
            <p className="text-xs mt-2">Upload documents and ask questions to get verified answers.</p>
          </div>
        )}

        {allMessages.map((msg, idx) => (
          <div key={idx} className="space-y-3 min-w-0">
            {/* User message */}
            <div className="flex justify-end">
              <div className="w-full max-w-full sm:max-w-[70%] rounded-lg bg-[#4f8ef7] px-4 py-2 text-white text-sm wrap-break-word">
                {msg.query}
              </div>
            </div>

            {/* Bot response */}
            <div className="flex justify-start min-w-0">
              <div className="w-full max-w-full sm:max-w-[80%] rounded-lg bg-[#1e2230] px-4 py-3 space-y-3 min-w-0">
                {msg.answers && msg.answers.length > 0 ? (
                  <>
                    <div className="text-[10px] uppercase tracking-widest text-[#6070a0]">
                      {msg.answers.length} verified response{msg.answers.length > 1 ? "s" : ""}
                    </div>
                    <div className="space-y-2 min-w-0">
                      {msg.answers.map((answer, i) => (
                        <AnswerCard key={answer.span_hash} result={answer} index={i} documents={documents} />
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-[#c8d8f0]">No verified answer found in the documents.</p>
                )}
              </div>
            </div>
          </div>
        ))}

        {isQuerying && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg bg-[#1e2230] px-4 py-2 text-[#c8d8f0] text-sm">
              <div className="flex items-center space-x-2">
                <div className="animate-pulse">Thinking...</div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg bg-red-900 border border-red-700 px-4 py-2 text-red-200 text-sm">
              {error}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-[#1e2230] px-4 py-3">
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="Ask a question..."
            className="flex-1 resize-none rounded-lg bg-[#12141a] border border-[#1e2230]
                       px-3 py-2 text-sm text-[#c8d8f0] placeholder-[#2e3450]
                       focus:outline-none focus:border-[#4f8ef7] transition-colors"
          />
          <button
            onClick={submit}
            disabled={isQuerying || !input.trim()}
            className="shrink-0 rounded-lg bg-[#4f8ef7] px-4 py-2 text-sm font-semibold
                       text-white transition-opacity disabled:opacity-30 hover:opacity-90"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}