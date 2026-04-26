// src/lib/utils.ts

import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import type { BBox } from "./types"

// ── Tailwind class merger (required by shadcn) ─────────────────────────────

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// ── BBox helpers ───────────────────────────────────────────────────────────

/**
 * Convert a normalized BBox (0.0–1.0) to pixel CSS values
 * relative to a rendered PDF page container.
 *
 * @param bbox      Normalized BBox from the API
 * @param pageWidth  Actual rendered width of the PDF page in pixels
 * @param pageHeight Actual rendered height of the PDF page in pixels
 * @returns CSS position values for absolute positioning
 */
export function bboxToPixels(
  bbox: BBox,
  pageWidth: number,
  pageHeight: number
): { left: number; top: number; width: number; height: number } {
  return {
    left: bbox.x0 * pageWidth,
    top: bbox.y0 * pageHeight,
    width: (bbox.x1 - bbox.x0) * pageWidth,
    height: (bbox.y1 - bbox.y0) * pageHeight,
  }
}

/**
 * Format a confidence score (raw logit) into a human-readable
 * percentage string for display in the UI.
 */
export function formatScore(score: number): string {
  // Clamp raw logit to a 0–100 display range
  const clamped = Math.min(Math.max(score / 5, 0), 1)
  return `${Math.round(clamped * 100)}%`
}

/**
 * Format file size in MB to a readable string.
 */
export function formatFileSize(mb: number): string {
  if (mb < 1) return `${Math.round(mb * 1024)} KB`
  return `${mb.toFixed(1)} MB`
}

/**
 * Truncate a string to maxLength characters.
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + "..."
}