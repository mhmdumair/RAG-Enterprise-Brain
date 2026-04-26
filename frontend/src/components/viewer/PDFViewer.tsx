"use client";

import { useState, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { useAppStore } from "@/store/appStore";
import { HighlightLayer } from "./HighlightLayer";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

interface Props {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

// Point to the PDF.js worker — use local copy
pdfjs.GlobalWorkerOptions.workerSrc = `/pdf.worker.min.js`;

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function PDFViewer({ isCollapsed, onToggleCollapse }: Props) {
  const { activeFile, activePage, setActivePage } = useAppStore();
  const [numPages, setNumPages] = useState(0);
  const [pageDims, setPageDims] = useState<{ width: number; height: number } | null>(null);

  const onDocumentLoad = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    // Clamp activePage to valid range
    setActivePage(prev => Math.min(numPages, Math.max(1, prev)));
  }, [setActivePage]);

  const onPageLoad = useCallback((page: { width: number; height: number }) => {
    setPageDims({ width: page.width, height: page.height });
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    if (numPages <= 1) return;
    e.preventDefault();
    if (e.deltaY > 0) {
      // scroll down, next page
      setActivePage(prev => Math.min(numPages, prev + 1));
    } else {
      // scroll up, prev page
      setActivePage(prev => Math.max(1, prev - 1));
    }
  }, [numPages, setActivePage]);

  if (isCollapsed) {
    return (
      <div className="flex h-full flex-col border-l border-[#1e2230] bg-[#0b0c10]">
        <div className="flex items-center justify-between px-2 py-2 border-b border-[#1e2230]">
          <span className="text-[10px] uppercase tracking-widest text-[#6070a0]">
            Viewer
          </span>
          <button
            onClick={onToggleCollapse}
            className="rounded bg-[#171b2c] px-2 py-1 text-xs text-[#c8d8f0] transition hover:bg-[#1f2840]"
          >
            ▶
          </button>
        </div>
      </div>
    );
  }

  if (!activeFile) {
    return (
      <div className="flex flex-col h-full items-center justify-center px-6 text-center text-[#3a4055]">
        <div className="text-4xl mb-4">📄</div>
        <p className="text-sm font-semibold text-[#c8d8f0]">PDF viewer</p>
        <p className="text-xs mt-2 max-w-70">
          Select a document from the sidebar or click an answer in chat to open the PDF.
        </p>
      </div>
    );
  }

  const fileUrl = `${API_BASE}/documents/${activeFile.document_id}/file`;

  return (
    <div className="flex flex-col h-full">
      {/* Viewer header */}
      <div className="px-4 py-3 border-b border-[#1e2230] shrink-0 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleCollapse}
            className="rounded bg-[#171b2c] px-2 py-1 text-xs text-[#c8d8f0] transition hover:bg-[#1f2840]"
          >
            ◀
          </button>
          <p className="text-xs text-[#6070a0] truncate max-w-55">
            {activeFile.filename}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-[#3a4466]">
          <button
            onClick={() => setActivePage(Math.max(1, activePage - 1))}
            disabled={activePage <= 1}
            className="disabled:opacity-30 hover:text-[#c0ccdd] transition-colors"
          >
            ‹
          </button>
          <span>
            {activePage} / {numPages > 0 ? numPages : "—"}
          </span>
          <button
            onClick={() => setActivePage(Math.min(numPages, activePage + 1))}
            disabled={activePage >= numPages}
            className="disabled:opacity-30 hover:text-[#c0ccdd] transition-colors"
          >
            ›
          </button>
        </div>
      </div>

      {/* PDF canvas */}
      <div 
        className="flex-1 overflow-auto flex justify-center bg-[#0a0b0e] py-4"
      >
        <Document
          file={fileUrl}
          onLoadSuccess={onDocumentLoad}
          loading={
            <p className="text-xs text-[#3a4055] mt-12">Loading PDF…</p>
          }
          error={
            <p className="text-xs text-red-400 mt-12">Failed to load PDF.</p>
          }
        >
          {numPages > 0 && (
            <div className="relative" onWheel={onWheel}>
              <Page
                pageNumber={activePage}
                width={440}
                onLoadSuccess={onPageLoad}
                renderTextLayer
                renderAnnotationLayer={false}
              />
              {pageDims && (
                <HighlightLayer
                  pageWidth={pageDims.width}
                  pageHeight={pageDims.height}
                  pageNumber={activePage}
                />
              )}
            </div>
          )}
        </Document>
      </div>
    </div>
  );
}