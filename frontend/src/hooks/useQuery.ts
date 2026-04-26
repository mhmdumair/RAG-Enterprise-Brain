// src/hooks/useQuery.ts
// Query state — manages the current query, loading, and results.

import { useState, useCallback } from "react"
import { runQuery as runQueryApi, APIError } from "@/lib/api"
import { useAppStore } from "@/store/appStore"
import type { QueryResponse } from "@/lib/types"

export function useQuery() {
  const addQueryResult = useAppStore((s) => s.addQueryResult)
  const documents = useAppStore((s) => s.documents)
  const openAnswer = useAppStore((s) => s.openAnswer)
  const history = useAppStore((s) => s.queryHistory)

  const [isQuerying, setIsQuerying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<QueryResponse | null>(null)

  const runQuery = useCallback(
    async (text: string, topK: number = 5): Promise<void> => {
      if (!text.trim() || text.length < 3) return

      setIsQuerying(true)
      setError(null)
      setResult(null)

      try {
        const response = await runQueryApi({ query: text, top_k: topK })
        setResult(response)
        addQueryResult(response)

        // Auto-open first answer in PDF viewer
        if (response.answers.length > 0) {
          openAnswer(response.answers[0], documents)
        }
      } catch (err) {
        if (err instanceof APIError) {
          if (err.status === 404) {
            setError("No verified answer found in the ingested documents.")
          } else {
            setError(err.message)
          }
        } else {
          setError("Something went wrong. Please try again.")
        }
      } finally {
        setIsQuerying(false)
      }
    },
    [addQueryResult, documents, openAnswer]
  )

  return {
    results: result?.answers ?? [],
    runQuery,
    isQuerying,
    error,
    history,
    clearError: () => setError(null),
    clearResult: () => setResult(null),
  }
}