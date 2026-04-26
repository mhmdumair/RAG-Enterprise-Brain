// src/hooks/useDocuments.ts
// SWR hook — fetches document list and provides mutate for refresh.

import useSWR from "swr"
import { getDocuments, deleteDocument, ingestPDF, APIError } from "@/lib/api"
import { useAppStore } from "@/store/appStore"
import { useCallback, useState } from "react"

const DOCUMENTS_KEY = "/documents"

export function useDocuments() {
  const setDocuments = useAppStore((s) => s.setDocuments)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  const { data, error, isLoading, mutate } = useSWR(
    DOCUMENTS_KEY,
    getDocuments,
    {
      onSuccess: (data) => setDocuments(data.documents),
      refreshInterval: 0,
      revalidateOnFocus: false,
    }
  )

  const upload = useCallback(
    async (file: File): Promise<void> => {
      setUploading(true)
      setUploadError(null)
      try {
        await ingestPDF(file)
        await mutate() // refresh the list
      } catch (err) {
        const msg =
          err instanceof APIError
            ? err.message
            : "Upload failed. Please try again."
        setUploadError(msg)
        throw err
      } finally {
        setUploading(false)
      }
    },
    [mutate]
  )

  const remove = useCallback(
    async (document_id: string): Promise<void> => {
      setDeleting(document_id)
      try {
        await deleteDocument(document_id)
        await mutate() // refresh the list
      } finally {
        setDeleting(null)
      }
    },
    [mutate]
  )

  return {
    documents: data?.documents ?? [],
    total: data?.total ?? 0,
    isLoading,
    error,
    uploading,
    uploadError,
    deleting,
    upload,
    remove,
    refresh: mutate,
  }
}