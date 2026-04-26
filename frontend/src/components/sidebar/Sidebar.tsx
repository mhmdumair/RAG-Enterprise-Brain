"use client";

import { useDocuments } from "@/hooks/useDocuments";
import { UploadZone } from "./UploadZone";
import { FileItem } from "./FileItem";

interface Props {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export function Sidebar({ isCollapsed, onToggleCollapse }: Props) {
  const { documents, isLoading } = useDocuments();

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#1e2230] flex items-center justify-between gap-2">
        <div>
          <p className="text-[10px] tracking-widest uppercase text-[#3a4466]">
            Enterprise Brain
          </p>
          {!isCollapsed && (
            <h1 className="text-sm font-semibold text-[#c0ccdd] mt-0.5">
              Documents
            </h1>
          )}
        </div>
        <button
          onClick={onToggleCollapse}
          className="rounded bg-[#171b2c] px-2 py-1 text-xs text-[#c8d8f0] transition hover:bg-[#1f2840]"
        >
          {isCollapsed ? "▶" : "◀"}
        </button>
      </div>

      {!isCollapsed && (
        <>
          {/* Upload */}
          <UploadZone />

          {/* Doc list */}
          <div className="flex-1 overflow-y-auto py-1 space-y-0.5">
            {isLoading && (
              <p className="px-4 text-xs text-[#3a4055]">Loading…</p>
            )}
            {!isLoading && documents.length === 0 && (
              <p className="px-4 text-xs text-[#2e3450]">No documents ingested</p>
            )}
            {documents.map((doc) => (
              <FileItem key={doc.document_id} doc={doc} />
            ))}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-[#1e2230] text-[10px] text-[#2a3050]">
            {documents.length} / 10 docs
          </div>
        </>
      )}
    </div>
  );
}