// src/lib/types.ts
// TypeScript interfaces mirroring FastAPI Pydantic schemas exactly.

export interface BBox {
  x0: number
  y0: number
  x1: number
  y1: number
  page_width: number
  page_height: number
}

export interface Answer {
  text: string
  span_score: number
  null_score: number
  filename: string
  page_number: number
  bbox: BBox
  chunk_text: string
  span_hash: string
  rake_used: boolean
}

export interface QueryResponse {
  query: string
  answers: Answer[]
  total_answers: number
  total_chunks_searched: number
  rake_used: boolean
  processing_ms: number
}

export interface QueryRequest {
  query: string
  top_k?: number
}

export interface Document {
  document_id: string
  filename: string
  total_pages: number
  total_chunks: number
  file_size_mb: number
  status: string
  created_at: string
}

export interface DocumentListResponse {
  documents: Document[]
  total: number
}

export interface IngestResponse {
  document_id: string
  filename: string
  total_pages: number
  total_chunks: number
  total_vectors: number
  message: string
}

export interface HealthResponse {
  status: string
  version: string
  components: Record<string, { status: string; detail: string }>
}

export interface ErrorResponse {
  error: string
  message: string
  details: Record<string, unknown>
}

// UI-only types — not from API

export interface ActiveHighlight {
  answer: Answer
  queryText: string
}

export type PanelState = "idle" | "loading" | "results" | "error"