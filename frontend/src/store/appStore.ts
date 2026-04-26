// src/store/appStore.ts
// Global state shared across all three panels.
// The three panels communicate through this store — never through props.

import { create } from "zustand"
import type { Answer, Document, QueryResponse } from "@/lib/types"

interface AppState {
  // ── Sidebar state ───────────────────────────────────────────────────────
  documents: Document[]
  setDocuments: (docs: Document[]) => void

  // ── PDF viewer state ────────────────────────────────────────────────────
  activeFile: Document | null           // which PDF is open
  activePage: number                    // current page number (1-based)
  activeAnswer: Answer | null           // which answer is highlighted
  pdfObjectUrl: string | null           // blob URL for react-pdf

  setActiveFile: (doc: Document | null) => void
  setActivePage: (page: number) => void
  setActiveAnswer: (answer: Answer | null) => void
  setPdfObjectUrl: (url: string | null) => void

  // Open a specific answer — sets file + page + highlight in one action
  openAnswer: (answer: Answer, documents: Document[]) => void

  // ── Query panel state ───────────────────────────────────────────────────
  queryHistory: QueryResponse[]
  addQueryResult: (result: QueryResponse) => void
  clearHistory: () => void
}

export const useAppStore = create<AppState>((set) => ({
  // ── Sidebar ─────────────────────────────────────────────────────────────
  documents: [],
  setDocuments: (docs) => set({ documents: docs }),

  // ── Viewer ──────────────────────────────────────────────────────────────
  activeFile: null,
  activePage: 1,
  activeAnswer: null,
  pdfObjectUrl: null,

  setActiveFile: (doc) => set({ activeFile: doc, activePage: 1, activeAnswer: null }),
  setActivePage: (page) => set({ activePage: page }),
  setActiveAnswer: (answer) => set({ activeAnswer: answer }),
  setPdfObjectUrl: (url) => set({ pdfObjectUrl: url }),

  openAnswer: (answer, documents) => {
    // Find the document matching this answer's filename
    const doc = documents.find((d) => d.filename === answer.filename) ?? null
    set({
      activeFile: doc,
      activePage: answer.page_number,
      activeAnswer: answer,
    })
  },

  // ── Query history ────────────────────────────────────────────────────────
  queryHistory: [],
  addQueryResult: (result) =>
    set((state) => ({
      queryHistory: [result, ...state.queryHistory].slice(0, 20), // keep last 20
    })),
  clearHistory: () => set({ queryHistory: [] }),
}))