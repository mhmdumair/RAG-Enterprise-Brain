// src/lib/api.ts
// Typed API client — all fetch calls to FastAPI go through here.
// Import from this file everywhere — never call fetch() directly.

import type {
  QueryRequest,
  QueryResponse,
  DocumentListResponse,
  IngestResponse,
  HealthResponse,
} from "./types"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// ── Helpers ───────────────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.headers ?? {}),
    },
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }))
    throw new APIError(res.status, err.message ?? "Request failed", err)
  }

  // 204 No Content — return empty object
  if (res.status === 204) return {} as T

  return res.json() as Promise<T>
}

export class APIError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown
  ) {
    super(message)
    this.name = "APIError"
  }
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health")
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function getDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/documents")
}

export async function deleteDocument(document_id: string): Promise<void> {
  return request<void>(`/documents/${document_id}`, {
    method: "DELETE",
  })
}

// ── Ingest ────────────────────────────────────────────────────────────────────

export async function ingestPDF(file: File): Promise<IngestResponse> {
  const form = new FormData()
  form.append("file", file)
  return request<IngestResponse>("/ingest", {
    method: "POST",
    body: form,
    // Do NOT set Content-Type header — browser sets it with boundary automatically
  })
}

// ── Query ─────────────────────────────────────────────────────────────────────

export async function runQuery(payload: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}