"use client";

import dynamic from "next/dynamic";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { QueryPanel } from "@/components/query/QueryPanel";
import { useAppStore } from "@/store/appStore";

const PDFViewer = dynamic(() => import("@/components/viewer/PDFViewer").then(mod => ({ default: mod.PDFViewer })), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center text-[#3a4055] text-sm">
      <div className="text-center space-y-2">
        <div className="text-3xl">📄</div>
        <p>Loading PDF viewer...</p>
      </div>
    </div>
  )
});

export default function Home() {
  const activeFile = useAppStore((s) => s.activeFile);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0e0f11] font-mono text-[#e2e8f0]">
      <aside className="w-64 min-w-[256px] border-r border-[#1e2230] flex flex-col">
        <Sidebar />
      </aside>

      <main className="flex-1 min-w-0 flex flex-col border-r border-[#1e2230]">
        <QueryPanel />
      </main>

      <aside className="w-[480px] min-w-[480px] flex flex-col">
        {activeFile ? (
          <PDFViewer />
        ) : (
          <div className="flex-1 flex items-center justify-center text-[#3a4055] text-sm">
            <div className="text-center space-y-2">
              <div className="text-3xl">📄</div>
              <p>Select a document to preview</p>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}