'use client';

import { useEffect, useState, useMemo } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import {
  ArrowLeft,
  Download,
  RefreshCw,
  Search,
  Filter,
  CheckCircle,
  AlertCircle,
  Edit,
  Clock,
  Layers,
  Box,
  PackageX,
  Boxes,
  AlertTriangle,
} from 'lucide-react';
import { fetchBatchResults, getBatchExportUrl, subscribeToBatchEvents } from '@/lib/api';
import { BatchResultsResponse, ExtractionResponse, ActiveProcessingItem } from '@/types';
import { StatusBadge, SceneFlagBadge, ParcelOutcomeBadge, ProcessingStageBadge } from '@/components/ui/StatusBadge';
import { ImageReviewModal } from '@/components/review/ImageReviewModal';
import { LiveProcessingCard } from '@/components/processing/LiveProcessingCard';

const getEvidence = (item: ExtractionResponse): any => {
  if (!item.evidence_json) return null;
  if (typeof item.evidence_json === 'object') return item.evidence_json;
  try {
    return JSON.parse(item.evidence_json);
  } catch {
    return null;
  }
};

export default function BatchDetailPage() {
  const params = useParams();
  const batchId = params.batchId as string;

  const [data, setData] = useState<BatchResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [selectedImage, setSelectedImage] = useState<ExtractionResponse | null>(null);
  const [activeProcessing, setActiveProcessing] = useState<ActiveProcessingItem | null>(null);
  const [isStreamConnected, setIsStreamConnected] = useState(false);

  const loadResults = async () => {
    try {
      const res = await fetchBatchResults(batchId);
      setData(res);
      if (res.active_processing) {
        setActiveProcessing(res.active_processing);
      } else if (
        res.batch.processed_images + res.batch.failed_images >=
        res.batch.total_images
      ) {
        setActiveProcessing(null);
      }
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch batch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadResults();

    // 1. Subscribe to real-time Server-Sent Events (SSE)
    const unsubscribe = subscribeToBatchEvents(
      batchId,
      (event) => {
        setIsStreamConnected(true);
        if (event.type === 'initial_state') {
          if (event.active_image) {
            setActiveProcessing(event.active_image);
          }
        } else if (event.type === 'stage_update') {
          setActiveProcessing({
            image_id: event.image_id,
            filename: event.filename,
            stage: event.stage,
            stage_name: event.stage_name,
            stage_message: event.stage_message,
            stage_progress: event.stage_progress,
            stage_index: event.stage_index,
            total_stages: event.total_stages || 6,
            stage_started_at: event.timestamp,
            preview_url: event.preview_url,
          });

          // Optimistically update matching item in the results list
          setData((prev) => {
            if (!prev) return prev;
            const updated = prev.results.map((r) => {
              if (r.image_id === event.image_id) {
                return {
                  ...r,
                  processing_status: 'PROCESSING',
                  current_stage: event.stage,
                  stage_message: event.stage_message,
                  stage_progress: event.stage_progress,
                };
              }
              return r;
            });
            return { ...prev, results: updated };
          });
        } else if (event.type === 'image_completed' || event.type === 'image_failed') {
          loadResults();
        } else if (event.type === 'batch_completed') {
          setActiveProcessing(null);
          loadResults();
        }
      },
      () => {
        setIsStreamConnected(false);
      }
    );

    // 2. Poll every 4 seconds while batch has active/uncompleted images
    const interval = setInterval(() => {
      loadResults();
    }, 4000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [batchId]);

  const totalParcelsDetected = useMemo(() => {
    if (!data) return 0;
    return data.results.filter(
      (r) => r.parcel_count !== null && r.parcel_count > 0 && !r.scene_flags?.includes('NO_PARCEL')
    ).length;
  }, [data]);

  const noParcelAnomalies = useMemo(() => {
    if (!data) return 0;
    return data.results.filter(
      (r) =>
        r.scene_flags?.includes('NO_PARCEL') ||
        r.parcel_count === null ||
        r.parcel_count === 0
    ).length;
  }, [data]);

  const queuedCount = useMemo(() => {
    if (!data) return 0;
    return data.results.filter(
      (r) =>
        r.processing_status === 'QUEUED' ||
        (r.id.startsWith('pending-') && r.image_id !== activeProcessing?.image_id)
    ).length;
  }, [data, activeProcessing]);

  const isBatchProcessing = useMemo(() => {
    if (!data) return false;
    return (
      activeProcessing !== null ||
      data.batch.processed_images + data.batch.failed_images < data.batch.total_images ||
      data.results.some(
        (r) => r.processing_status === 'PROCESSING' || r.processing_status === 'QUEUED'
      )
    );
  }, [data, activeProcessing]);

  const filteredResults = useMemo(() => {
    if (!data) return [];
    return data.results.filter((item) => {
      const matchesSearch =
        item.filename?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.tracking_number?.toLowerCase().includes(searchTerm.toLowerCase());
      const isActivelyProc =
        item.image_id === activeProcessing?.image_id ||
        item.processing_status === 'PROCESSING';
      const isQueuedItem =
        item.processing_status === 'QUEUED' ||
        (item.id.startsWith('pending-') && !isActivelyProc);

      const matchesStatus =
        statusFilter === 'ALL' ||
        (statusFilter === 'STATE:PROCESSING' && isActivelyProc) ||
        (statusFilter === 'STATE:QUEUED' && isQueuedItem) ||
        item.status === statusFilter ||
        (statusFilter.startsWith('FLAG:') &&
          item.scene_flags?.includes(statusFilter.replace('FLAG:', ''))) ||
        (statusFilter.startsWith('OUTCOME:') &&
          item.parcel_label_outcome === statusFilter.replace('OUTCOME:', ''));
      return matchesSearch && matchesStatus;
    });
  }, [data, searchTerm, statusFilter, activeProcessing]);

  if (loading && !data) {
    return (
      <div className="py-24 text-center">
        <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin mx-auto mb-3" />
        <div className="text-sm font-semibold text-slate-700">Loading batch results...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-16 text-center max-w-md mx-auto">
        <AlertCircle className="w-10 h-10 text-rose-500 mx-auto mb-3" />
        <h3 className="text-base font-bold text-slate-900">Error Loading Batch</h3>
        <p className="text-xs text-slate-500 mt-1 mb-6">{error || 'Batch not found'}</p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Batches
        </Link>
      </div>
    );
  }

  const { batch, results } = data;
  const progressPercent =
    batch.total_images > 0
      ? Math.round((batch.processed_images / batch.total_images) * 100)
      : 0;

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 mb-2 transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Back to Dashboard
          </Link>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">{batch.name}</h1>
            <span className="text-xs bg-slate-100 text-slate-600 font-mono px-2 py-0.5 rounded">
              ID: {batch.id.substring(0, 8)}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadResults}
            className="inline-flex items-center gap-2 px-3 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-lg transition shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
          <a
            href={getBatchExportUrl(batch.id)}
            download
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg transition shadow-sm"
          >
            <Download className="w-4 h-4" />
            Export CSV ({results.length} rows)
          </a>
        </div>
      </div>

      {/* Live Processing Card when batch is actively running or queued */}
      {isBatchProcessing && (
        <LiveProcessingCard
          batch={batch}
          activeImage={activeProcessing}
          totalImages={batch.total_images}
          processedCount={batch.processed_images}
          queuedCount={queuedCount}
          isStreamConnected={isStreamConnected}
        />
      )}

      {/* Progress & Stat Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3.5">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            Total Images
          </div>
          <div className="text-2xl font-bold text-slate-900 mt-1">{batch.total_images}</div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            Processed
          </div>
          <div className="text-2xl font-bold text-indigo-600 mt-1">
            {batch.processed_images}
          </div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-emerald-200 bg-emerald-50/20 shadow-sm">
          <div className="flex items-center justify-between text-[11px] font-semibold text-emerald-700 uppercase tracking-wider">
            <span>Physical Parcels</span>
            <Box className="w-3.5 h-3.5 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-emerald-700 mt-1">
            {totalParcelsDetected}
          </div>
          <div className="text-[10px] text-emerald-600 mt-0.5">Physical box verified</div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-rose-200 bg-rose-50/20 shadow-sm">
          <div className="flex items-center justify-between text-[11px] font-semibold text-rose-700 uppercase tracking-wider">
            <span>No Parcel Scans</span>
            <PackageX className="w-3.5 h-3.5 text-rose-600" />
          </div>
          <div className="text-2xl font-bold text-rose-700 mt-1">
            {noParcelAnomalies}
          </div>
          <div className="text-[10px] text-rose-600 mt-0.5">Empty belt / no parcel flag</div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            Pipeline Failures
          </div>
          <div className="text-2xl font-bold text-slate-600 mt-1">{batch.failed_images}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">Processing errors</div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-700 mb-2">
          <span>Processing Pipeline Progress</span>
          <span>{progressPercent}% Complete</span>
        </div>
        <div className="w-full bg-slate-100 rounded-full h-2">
          <div
            className="bg-indigo-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Filters & Search */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search filename or tracking number..."
            className="w-full text-xs pl-9 pr-3 py-2 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <Filter className="w-4 h-4 text-slate-400 shrink-0" />
          <span className="text-xs text-slate-500 font-semibold shrink-0">Filter By:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="text-xs px-3 py-2 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
          >
            <option value="ALL">All Results ({results.length})</option>
            <optgroup label="Live Processing Status">
              <option value="STATE:PROCESSING">⚡ Actively Processing</option>
              <option value="STATE:QUEUED">⏳ Waiting in Queue</option>
            </optgroup>
            <optgroup label="Parcel & Label Outcomes">
              <option value="OUTCOME:LABEL_FOUND">🎯 LABEL_FOUND</option>
              <option value="OUTCOME:PARCEL_DETECTED_LABEL_NOT_FOUND">📦 PARCEL_DETECTED_LABEL_NOT_FOUND</option>
              <option value="OUTCOME:LABEL_DETECTED_UNREADABLE">⚠️ LABEL_DETECTED_UNREADABLE</option>
              <option value="OUTCOME:MULTIPLE_PARCELS">📦📦 MULTIPLE_PARCELS</option>
              <option value="OUTCOME:NO_PARCEL_DETECTED">🚫 NO_PARCEL_DETECTED</option>
              <option value="OUTCOME:LOW_CONFIDENCE_REVIEW">🔍 LOW_CONFIDENCE_REVIEW</option>
            </optgroup>
            <optgroup label="Extraction Status">
              <option value="SUCCESS">SUCCESS</option>
              <option value="PARTIAL_EXTRACTION">PARTIAL_EXTRACTION</option>
              <option value="SOURCE_CONFLICT">SOURCE_CONFLICT</option>
              <option value="LOW_CONFIDENCE">LOW_CONFIDENCE</option>
              <option value="PROCESSING_FAILED">PROCESSING_FAILED</option>
            </optgroup>
            <optgroup label="Scene Evidence & Flags">
              <option value="FLAG:NO_PARCEL">🚩 NO_PARCEL (Empty Scanner Bed)</option>
              <option value="FLAG:LABEL_NOT_FOUND">🚩 LABEL_NOT_FOUND</option>
              <option value="FLAG:MULTIPLE_PARCELS">🚩 MULTIPLE_PARCELS</option>
              <option value="FLAG:PARTIAL_PARCEL">🚩 PARTIAL_PARCEL</option>
              <option value="FLAG:LABEL_OCCLUDED">🚩 LABEL_OCCLUDED</option>
              <option value="FLAG:LABEL_UNREADABLE">🚩 LABEL_UNREADABLE</option>
            </optgroup>
          </select>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-100">
              <tr>
                <th className="px-5 py-3">Image Filename</th>
                <th className="px-5 py-3">Status</th>
                <th className="px-5 py-3">Parcel Proof & Scene Flags</th>
                <th className="px-5 py-3">Tracking / AWB #</th>
                <th className="px-5 py-3">Weight</th>
                <th className="px-5 py-3">Dimensions (L×W×H)</th>
                <th className="px-5 py-3 text-center">Reviewed</th>
                <th className="px-5 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {filteredResults.length === 0 ? (
                <tr>
                  <td colSpan={8} className="p-8 text-center text-slate-400">
                    No results match your filter or search query.
                  </td>
                </tr>
              ) : (
                filteredResults.map((item) => {
                  const ev = getEvidence(item);
                  const isNoParcel =
                    item.scene_flags?.includes('NO_PARCEL') ||
                    item.parcel_count === null ||
                    item.parcel_count === 0;

                  const isActivelyProcessing =
                    item.image_id === activeProcessing?.image_id ||
                    item.processing_status === 'PROCESSING';
                  const isQueued =
                    item.processing_status === 'QUEUED' ||
                    (item.id.startsWith('pending-') && !isActivelyProcessing && isBatchProcessing);

                  return (
                    <tr
                      key={item.id}
                      className={`transition-all duration-300 ${
                        isActivelyProcessing
                          ? 'bg-indigo-50/70 border-l-4 border-l-indigo-600 shadow-xs ring-1 ring-inset ring-indigo-200'
                          : 'hover:bg-slate-50/80'
                      }`}
                    >
                      <td className="px-5 py-3.5 font-medium text-slate-900 truncate max-w-[180px]">
                        <div className="flex items-center gap-2">
                          {isActivelyProcessing && (
                            <span className="relative flex h-2 w-2 shrink-0">
                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-600"></span>
                            </span>
                          )}
                          <span className="truncate" title={item.filename}>
                            {item.filename}
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        {isActivelyProcessing || isQueued ? (
                          <ProcessingStageBadge
                            status={isActivelyProcessing ? 'PROCESSING' : 'QUEUED'}
                            stage={
                              isActivelyProcessing && activeProcessing
                                ? activeProcessing.stage
                                : item.current_stage
                            }
                            stageMessage={
                              isActivelyProcessing && activeProcessing
                                ? activeProcessing.stage_message
                                : item.stage_message
                            }
                            stageProgress={
                              isActivelyProcessing && activeProcessing
                                ? activeProcessing.stage_progress
                                : item.stage_progress
                            }
                          />
                        ) : (
                          <StatusBadge status={item.status} />
                        )}
                      </td>
                      {/* Parcel Proof & Scene Flags Column */}
                      <td className="px-5 py-3.5">
                        <div className="flex flex-col gap-1 items-start">
                          <div className="flex flex-wrap items-center gap-1.5">
                            {item.parcel_count !== null && item.parcel_count > 0 ? (
                              <span
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold ${
                                  item.parcel_count === 1
                                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                    : 'bg-purple-50 text-purple-700 border border-purple-200'
                                }`}
                              >
                                {item.parcel_count === 1 ? (
                                  <Box className="w-3 h-3 text-emerald-600" />
                                ) : (
                                  <Boxes className="w-3 h-3 text-purple-600" />
                                )}
                                {item.parcel_count} {item.parcel_count === 1 ? 'Parcel' : 'Parcels'}
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
                                <PackageX className="w-3 h-3 text-rose-600" />
                                0 Parcels (None)
                              </span>
                            )}
                            <ParcelOutcomeBadge outcome={item.parcel_label_outcome} />
                          </div>

                          {/* Scene Flags badges */}
                          {item.scene_flags && item.scene_flags.length > 0 ? (
                            <div className="flex flex-wrap gap-1 mt-0.5">
                              {item.scene_flags.map((flag) => (
                                <SceneFlagBadge key={flag} flag={flag} />
                              ))}
                            </div>
                          ) : (
                            !isNoParcel && (
                              <span className="text-[10px] text-slate-400 font-sans">
                                Clean scene
                              </span>
                            )
                          )}
                        </div>
                      </td>

                      {/* Tracking Number Column */}
                      <td className="px-5 py-3.5">
                        {item.tracking_number ? (
                          <div className="flex flex-col">
                            <span className="font-mono text-slate-900 font-medium">
                              {item.tracking_number}
                            </span>
                            {(() => {
                              const prov = ev?.tracking_provenance;
                              const src = prov?.source;
                              const conf = prov?.ocr_confidence;
                              const conflict = prov?.conflict_status;
                              if (!src && conf === undefined) return null;
                              return (
                                <div className="flex items-center gap-1 mt-0.5">
                                  {src && (
                                    <span className="text-[10px] text-slate-500 bg-slate-100 px-1.5 py-0.2 rounded font-sans truncate max-w-[130px]">
                                      {src}
                                    </span>
                                  )}
                                  {conf !== undefined && conf > 0 && (
                                    <span className="text-[10px] text-slate-500 font-sans">
                                      {Math.round(conf * 100)}%
                                    </span>
                                  )}
                                  {conflict === 'CONFLICT' && (
                                    <span className="text-[10px] text-rose-600 font-bold">
                                      ⚠️ Conflict
                                    </span>
                                  )}
                                </div>
                              );
                            })()}
                          </div>
                        ) : (
                          <span className="text-slate-300 font-sans">—</span>
                        )}
                      </td>

                      {/* Weight Column */}
                      <td className="px-5 py-3.5">
                        {item.weight_value !== null ? (
                          <div className="flex flex-col">
                            <span className="text-slate-900 font-medium">
                              {item.weight_value} {item.weight_unit || ''}
                            </span>
                            {(() => {
                              const src = ev?.weight_provenance?.source;
                              const vol = item.volumetric_weight ?? ev?.volumetric_weight;
                              return (
                                <div className="flex flex-col text-[10px] text-slate-500 mt-0.5">
                                  {src && (
                                    <span className="bg-slate-100 text-slate-500 px-1.5 py-0.2 rounded font-sans w-fit">
                                      {src}
                                    </span>
                                  )}
                                  {vol !== undefined && vol !== null && (
                                    <span className="text-slate-400 mt-0.5">
                                      Vol: {vol} {item.weight_unit || ''}
                                    </span>
                                  )}
                                </div>
                              );
                            })()}
                          </div>
                        ) : (
                          <span className="text-slate-300 font-sans">—</span>
                        )}
                      </td>

                      {/* Dimensions Column */}
                      <td className="px-5 py-3.5">
                        {item.length !== null && item.width !== null && item.height !== null ? (
                          <div className="flex flex-col">
                            <span className="text-slate-900 font-medium">
                              {item.length} × {item.width} × {item.height}{' '}
                              {item.dimension_unit || ''}
                            </span>
                            {(() => {
                              const src = ev?.dimensions_provenance?.source;
                              if (!src) return null;
                              return (
                                <span className="bg-slate-100 text-slate-500 text-[10px] px-1.5 py-0.2 rounded font-sans w-fit mt-0.5">
                                  {src}
                                </span>
                              );
                            })()}
                          </div>
                        ) : (
                          <span className="text-slate-300 font-sans">—</span>
                        )}
                      </td>

                      {/* Reviewed Column */}
                      <td className="px-5 py-3.5 text-center">
                        {item.reviewed ? (
                          <span className="inline-flex items-center text-emerald-600 font-medium">
                            <CheckCircle className="w-4 h-4" />
                          </span>
                        ) : (
                          <span className="text-slate-300 font-sans">—</span>
                        )}
                      </td>

                      {/* Action Column */}
                      <td className="px-5 py-3.5 text-right">
                        <button
                          onClick={() => setSelectedImage(item)}
                          className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:text-indigo-800"
                        >
                          <Edit className="w-3.5 h-3.5" /> Inspect / Edit
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Review Modal */}
      {selectedImage && (
        <ImageReviewModal
          image={selectedImage}
          onClose={() => setSelectedImage(null)}
          onUpdated={() => {
            loadResults();
          }}
        />
      )}
    </div>
  );
}
