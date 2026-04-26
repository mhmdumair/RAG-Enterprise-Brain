"use client";

import { useState, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { useAppStore } from "@/store/appStore";
import { HighlightLayer } from "./HighlightLayer";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Point to the PDF.js worker — use local copy
pdfjs.GlobalWorkerOptions.workerSrc = `/pdf.worker.min.js`;

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function PDFViewer() {
  const { activeFile, activePage, setActivePage } = useAppStore();
  const [numPages, setNumPages] = useState(0);
  const [pageDims, setPageDims] = useState<{ width: number; height: number } | null>(null);

  const onDocumentLoad = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  }, []);

  const onPageLoad = useCallback((page: { width: number; height: number }) => {
    setPageDims({ width: page.width, height: page.height });
  }, []);

  if (!activeFile) return null;

  const fileUrl = `${API_BASE}/documents/${activeFile.document_id}/file`;

  return (
    <div className="flex flex-col h-full">
      {/* Viewer header */}
      <div className="px-4 py-3 border-b border-[#1e2230] shrink-0 flex items-center justify-between">
        <p className="text-xs text-[#6070a0] truncate max-w-[260px]">
          {activeFile.filename}
        </p>
        <div className="flex items-center gap-2 text-xs text-[#3a4466]">
          <button
            onClick={() => setActivePage(Math.max(1, activePage - 1))}
            disabled={activePage <= 1}
            className="disabled:opacity-30 hover:text-[#c0ccdd] transition-colors"
          >
            ‹
          </button>
          <span>
            {activePage} / {numPages || "—"}
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
      <div className="flex-1 overflow-auto flex justify-center bg-[#0a0b0e] py-4">
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
          <div className="relative">
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
        </Document>
      </div>
    </div>
  );
}