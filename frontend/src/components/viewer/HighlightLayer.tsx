"use client";

import { useAppStore } from "@/store/appStore";
import { bboxToPixels } from "@/lib/utils";

interface Props {
  pageWidth: number;
  pageHeight: number;
  pageNumber: number;
}

export function HighlightLayer({ pageWidth, pageHeight, pageNumber }: Props) {
  const activeAnswer = useAppStore((s) => s.activeAnswer);

  if (
    !activeAnswer ||
    activeAnswer.page_number !== pageNumber ||
    !activeAnswer.bbox
  ) {
    return null;
  }

  const rect = bboxToPixels(activeAnswer.bbox, pageWidth, pageHeight);

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={pageWidth}
      height={pageHeight}
      style={{ zIndex: 10 }}
    >
      <rect
        x={rect.left}
        y={rect.top}
        width={rect.width}
        height={rect.height}
        fill="rgba(79, 142, 247, 0.18)"
        stroke="#4f8ef7"
        strokeWidth={1.5}
        rx={2}
      />
    </svg>
  );
}