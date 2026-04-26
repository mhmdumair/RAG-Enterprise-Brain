"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useDocuments } from "@/hooks/useDocuments";

export function UploadZone() {
  const { upload } = useDocuments();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      setError(null);
      setUploading(true);
      try {
        await upload(file);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [upload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    disabled: uploading,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        mx-3 my-3 rounded border border-dashed px-3 py-4 text-center text-xs cursor-pointer
        transition-colors duration-150
        ${isDragActive
          ? "border-[#4f8ef7] bg-[#4f8ef710] text-[#4f8ef7]"
          : "border-[#2a2f42] text-[#4a5070] hover:border-[#4f8ef7] hover:text-[#4f8ef7]"
        }
        ${uploading ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input {...getInputProps()} />
      {uploading ? (
        <p>Ingesting…</p>
      ) : (
        <>
          <p className="mb-1">Drop PDF here</p>
          <p className="text-[10px] text-[#2e3450]">or click to browse</p>
        </>
      )}
      {error && <p className="mt-2 text-red-400">{error}</p>}
    </div>
  );
}