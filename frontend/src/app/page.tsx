'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Plus,
  Layers,
  ArrowRight,
  ShieldCheck,
  Cpu,
  Clock,
} from 'lucide-react';
import { fetchHealth, fetchBatches } from '@/lib/api';
import { BatchResponse, HealthResponse } from '@/types';
import { FileUploadModal } from '@/components/upload/FileUploadModal';

export default function HomePage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [batches, setBatches] = useState<BatchResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [healthData, batchList] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchBatches().catch(() => []),
      ]);
      setHealth(healthData);
      setBatches(batchList);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Warehouse Parcel Operations Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Local-first image intelligence, zero-shot parcel detection, local PaddleOCR & barcode verification.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            disabled={loading}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => setIsModalOpen(true)}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition shadow-sm"
          >
            <Plus className="w-4 h-4" />
            New Upload Batch
          </button>
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
              Backend Service
            </span>
            {health ? (
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Operational
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-700 bg-rose-50 px-2 py-0.5 rounded-full border border-rose-200">
                <AlertCircle className="w-3 h-3 text-rose-600" /> Offline
              </span>
            )}
          </div>
          <div className="mt-4">
            <div className="text-lg font-semibold text-slate-800">
              {health?.service || 'FastAPI Service'}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {health ? `v${health.version} • Python ${health.python_version}` : 'Connecting...'}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
              Local Hardware
            </span>
            <Cpu className="w-4 h-4 text-indigo-600" />
          </div>
          <div className="mt-4">
            <div className="text-lg font-semibold text-slate-800">
              Apple Silicon M1
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Zero paid APIs • CPU Zero-Shot & PaddleOCR
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
              Security Boundary
            </span>
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="mt-4">
            <div className="text-lg font-semibold text-slate-800">
              Private S3 Presigned
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Zero AWS credentials exposed to browser
            </div>
          </div>
        </div>
      </div>

      {/* Batches Table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-900">Intake Batches</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Select a batch to inspect images, review discrepancies, or download verified CSV export.
            </p>
          </div>
          <span className="text-xs font-semibold bg-slate-100 text-slate-600 px-2.5 py-1 rounded-full">
            {batches.length} {batches.length === 1 ? 'Batch' : 'Batches'}
          </span>
        </div>

        {batches.length === 0 ? (
          <div className="p-12 text-center">
            <div className="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-semibold text-slate-900">No batches created yet</h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1 mb-4">
              Upload scanned JPG/PNG parcel photographs to start local processing.
            </p>
            <button
              onClick={() => setIsModalOpen(true)}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition shadow-sm"
            >
              <Plus className="w-4 h-4" />
              Create First Batch
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-100">
                <tr>
                  <th className="px-6 py-3">Batch Name</th>
                  <th className="px-6 py-3">Total Images</th>
                  <th className="px-6 py-3">Uploaded</th>
                  <th className="px-6 py-3">Processed</th>
                  <th className="px-6 py-3">Created At</th>
                  <th className="px-6 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {batches.map((b) => (
                  <tr key={b.id} className="hover:bg-slate-50/80 transition">
                    <td className="px-6 py-4 font-semibold text-slate-900">
                      <Link
                        href={`/batches/${b.id}`}
                        className="hover:text-indigo-600 transition flex items-center gap-2"
                      >
                        {b.name}
                      </Link>
                    </td>
                    <td className="px-6 py-4 font-medium">{b.total_images}</td>
                    <td className="px-6 py-4">
                      <span className="text-indigo-600 font-semibold">{b.uploaded_images}</span>
                      <span className="text-slate-400"> / {b.total_images}</span>
                    </td>
                    <td className="px-6 py-4">
                      {b.processed_images + b.failed_images < b.total_images && b.total_images > 0 ? (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                          <span className="relative flex h-1.5 w-1.5">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-indigo-600"></span>
                          </span>
                          <span>Processing ({b.processed_images}/{b.total_images})</span>
                        </span>
                      ) : (
                        <div>
                          <span className="text-emerald-600 font-semibold">{b.processed_images}</span>
                          <span className="text-slate-400"> / {b.total_images}</span>
                          {b.failed_images > 0 && (
                            <span className="text-rose-600 ml-1.5 font-medium">
                              ({b.failed_images} failed)
                            </span>
                          )}
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4 text-slate-500">
                      <span className="inline-flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5 text-slate-400" />
                        {new Date(b.created_at).toLocaleString()}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link
                        href={`/batches/${b.id}`}
                        className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:text-indigo-800"
                      >
                        View Results <ArrowRight className="w-3.5 h-3.5" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      <FileUploadModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onBatchCreated={(newBatchId) => {
          loadData();
          window.location.href = `/batches/${newBatchId}`;
        }}
      />
    </div>
  );
}
