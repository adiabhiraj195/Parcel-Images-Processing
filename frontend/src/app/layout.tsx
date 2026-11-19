import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Parcel Intelligence System',
  description: 'Local-first parcel-image processing prototype for warehouse shipment operations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <header className="border-b border-slate-200 bg-white shadow-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow">
                PI
              </div>
              <span className="font-semibold text-lg text-slate-900 tracking-tight">
                Parcel Intelligence
              </span>
              <span className="text-xs bg-indigo-50 text-indigo-700 font-medium px-2 py-0.5 rounded border border-indigo-200">
                Local-First
              </span>
            </div>
            <div className="text-xs text-slate-500">
              Warehouse Scans & Shipment Review
            </div>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
