"use client";

import { Document } from "@/lib/types";
import { useDocuments } from "@/hooks/useDocuments";
import { useAppStore } from "@/store/appStore";

interface Props {
  doc: Document;
}

export function FileItem({ doc }: Props) {
  const { remove } = useDocuments();
  const { activeFile, setActiveFile } = useAppStore();
  const isActive = activeFile?.document_id === doc.document_id;

  return (
    <div
      onClick={() => setActiveFile(doc)}
      className={`
        group flex items-center justify-between gap-2 px-3 py-2 mx-2 rounded
        cursor-pointer text-xs transition-colors duration-100
        ${isActive
          ? "bg-[#1a2035] text-[#e2e8f0]"
          : "text-[#6070a0] hover:bg-[#141620] hover:text-[#c0ccdd]"
        }
      `}
    >
      <span className="truncate flex-1">{doc.filename}</span>
      <button
        onClick={(e) => {
          e.stopPropagation();
          remove(doc.document_id);
        }}
        className="opacity-0 group-hover:opacity-100 text-[#3a4055] hover:text-red-400
                   transition-opacity duration-100 shrink-0 leading-none"
        title="Delete"
      >
        ✕
      </button>
    </div>
  );
}