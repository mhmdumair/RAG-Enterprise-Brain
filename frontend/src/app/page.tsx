"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatPanel } from "@/components/query/ChatPanel";

const PDFViewer = dynamic(
  () => import("@/components/viewer/PDFViewer").then((mod) => ({ default: mod.PDFViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="flex-1 flex items-center justify-center text-[#3a4055] text-sm">
        <div className="text-center space-y-2">
          <div className="text-3xl">📄</div>
          <p>Loading PDF viewer...</p>
        </div>
      </div>
    ),
  }
);

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [viewerCollapsed, setViewerCollapsed] = useState(false);

  const viewerClass = viewerCollapsed
    ? "w-16 min-w-[64px]"
    : "w-120 min-w-105";

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0e0f11] font-sans text-[#e2e8f0]">
      {sidebarCollapsed ? null : (
        <aside className="w-72 min-w-70 border-r border-[#1e2230] flex flex-col">
          <Sidebar
            isCollapsed={false}
            onToggleCollapse={() => setSidebarCollapsed((open) => !open)}
          />
        </aside>
      )}

      <main className="flex-1 min-w-0 flex">
        <section className="flex-1 min-w-0 flex flex-col border-r border-[#1e2230]">
          <ChatPanel onToggleSidebar={() => setSidebarCollapsed((open) => !open)} />
        </section>

        <aside className={`${viewerClass} flex flex-col`}>
          <PDFViewer
            isCollapsed={viewerCollapsed}
            onToggleCollapse={() => setViewerCollapsed((open) => !open)}
          />
        </aside>
      </main>
    </div>
  );
}