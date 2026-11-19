import React from 'react';
import { ExtractionStatus } from '@/types';

interface StatusBadgeProps {
  status: ExtractionStatus | string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  let badgeStyle = 'bg-slate-100 text-slate-700 border-slate-200';

  switch (status) {
    case 'SUCCESS':
      badgeStyle = 'bg-emerald-50 text-emerald-700 border-emerald-200 font-medium';
      break;
    case 'PARTIAL_EXTRACTION':
      badgeStyle = 'bg-amber-50 text-amber-700 border-amber-200';
      break;
    case 'SOURCE_CONFLICT':
      badgeStyle = 'bg-rose-100 text-rose-800 border-rose-300 font-bold';
      break;
    case 'NO_PARCEL':
      badgeStyle = 'bg-purple-50 text-purple-700 border-purple-200 font-semibold';
      break;
    case 'MULTIPLE_PARCELS':
      badgeStyle = 'bg-blue-50 text-blue-700 border-blue-200';
      break;
    case 'PARTIAL_PARCEL':
      badgeStyle = 'bg-orange-50 text-orange-700 border-orange-200';
      break;
    case 'LABEL_NOT_FOUND':
    case 'LABEL_OCCLUDED':
    case 'LABEL_UNREADABLE':
      badgeStyle = 'bg-rose-50 text-rose-700 border-rose-200';
      break;
    case 'LABEL_FOUND':
      badgeStyle = 'bg-emerald-50 text-emerald-700 border-emerald-200 font-semibold';
      break;
    case 'PARCEL_DETECTED_LABEL_NOT_FOUND':
      badgeStyle = 'bg-amber-50 text-amber-700 border-amber-200 font-semibold';
      break;
    case 'NO_PARCEL_DETECTED':
      badgeStyle = 'bg-slate-100 text-slate-700 border-slate-300 font-semibold';
      break;
    case 'LOW_CONFIDENCE_REVIEW':
      badgeStyle = 'bg-yellow-50 text-yellow-800 border-yellow-200 font-semibold';
      break;
    case 'LOW_CONFIDENCE':
      badgeStyle = 'bg-yellow-50 text-yellow-800 border-yellow-200';
      break;
    case 'PROCESSING_FAILED':
      badgeStyle = 'bg-red-100 text-red-800 border-red-300 font-semibold';
      break;
    default:
      badgeStyle = 'bg-slate-100 text-slate-700 border-slate-200';
  }

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs border ${badgeStyle}`}
    >
      {status}
    </span>
  );
};

export const SceneFlagBadge: React.FC<{ flag: string }> = ({ flag }) => {
  let style = 'bg-slate-100 text-slate-700 border-slate-200';
  switch (flag) {
    case 'NO_PARCEL':
      style = 'bg-rose-50 text-rose-800 border-rose-300 font-bold shadow-xs';
      break;
    case 'LABEL_NOT_FOUND':
      style = 'bg-amber-50 text-amber-800 border-amber-300 font-medium';
      break;
    case 'MULTIPLE_PARCELS':
      style = 'bg-purple-50 text-purple-800 border-purple-300 font-semibold';
      break;
    case 'PARTIAL_PARCEL':
      style = 'bg-orange-50 text-orange-800 border-orange-300 font-medium';
      break;
    case 'LABEL_OCCLUDED':
    case 'LABEL_UNREADABLE':
      style = 'bg-red-50 text-red-700 border-red-200 font-medium';
      break;
    default:
      style = 'bg-slate-100 text-slate-700 border-slate-200';
  }

  return (
    <span
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono border ${style}`}
    >
      {flag}
    </span>
  );
};

export const ParcelOutcomeBadge: React.FC<{ outcome: string | null | undefined }> = ({ outcome }) => {
  if (!outcome) return null;

  let style = 'bg-slate-100 text-slate-700 border-slate-200';
  let label = outcome;

  switch (outcome) {
    case 'LABEL_FOUND':
      style = 'bg-emerald-50 text-emerald-800 border-emerald-300 font-semibold';
      label = 'Label Found';
      break;
    case 'LABEL_DETECTED_UNREADABLE':
      style = 'bg-rose-50 text-rose-800 border-rose-300 font-semibold';
      label = 'Label Unreadable';
      break;
    case 'PARCEL_DETECTED_LABEL_NOT_FOUND':
      style = 'bg-amber-50 text-amber-800 border-amber-300 font-semibold';
      label = 'No Label On Parcel';
      break;
    case 'NO_PARCEL_DETECTED':
      style = 'bg-slate-100 text-slate-700 border-slate-300 font-medium';
      label = 'No Parcel';
      break;
    case 'MULTIPLE_PARCELS':
      style = 'bg-purple-50 text-purple-800 border-purple-300 font-semibold';
      label = 'Multiple Parcels';
      break;
    case 'LOW_CONFIDENCE_REVIEW':
      style = 'bg-yellow-50 text-yellow-800 border-yellow-300 font-semibold';
      label = 'Needs Review';
      break;
  }

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono border ${style}`}
      title={`Detection Outcome: ${outcome}`}
    >
      {label}
    </span>
  );
};

export const STAGE_DISPLAY_NAMES: Record<string, { label: string; icon: string }> = {
  PENDING: { label: 'Pending Upload', icon: '⏳' },
  QUEUED: { label: 'In Queue', icon: '⏱️' },
  INGESTING: { label: 'Storage Ingest', icon: '📥' },
  PREPROCESSING: { label: 'Preprocessing', icon: '🖼️' },
  SCENE_DETECTION: { label: 'Scene Validation', icon: '🔍' },
  METADATA_EXTRACTION: { label: 'Machine Overlay', icon: '🏷️' },
  LABEL_EXTRACTION: { label: 'Label & Barcodes', icon: '📦' },
  RECONCILIATION: { label: 'Reconciliation', icon: '⚖️' },
  COMPLETED: { label: 'Completed', icon: '✅' },
  FAILED: { label: 'Failed', icon: '❌' },
};

export interface ProcessingStageBadgeProps {
  status?: string;
  stage?: string | null;
  stageMessage?: string | null;
  stageProgress?: number | null;
}

export const ProcessingStageBadge: React.FC<ProcessingStageBadgeProps> = ({
  status,
  stage,
  stageMessage,
  stageProgress,
}) => {
  const isQueued = status === 'QUEUED' || stage === 'QUEUED';
  const isProcessing = status === 'PROCESSING' || (stage && !['QUEUED', 'COMPLETED', 'FAILED'].includes(stage));

  if (isQueued) {
    return (
      <div className="inline-flex flex-col gap-0.5">
        <span
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200 shadow-xs"
          title={stageMessage || 'Waiting in processing queue...'}
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
          </span>
          <span>In Queue</span>
        </span>
      </div>
    );
  }

  if (isProcessing) {
    const stageInfo = stage ? STAGE_DISPLAY_NAMES[stage] : null;
    const label = stageInfo?.label || stage || 'Processing';
    const percent = stageProgress !== undefined && stageProgress !== null ? Math.min(100, Math.max(0, stageProgress)) : 50;

    return (
      <div className="inline-flex flex-col gap-1 w-full max-w-[170px]" title={stageMessage || `Stage: ${label}`}>
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 shadow-xs animate-pulse">
          <svg className="animate-spin h-3 w-3 text-indigo-600 shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span className="truncate">{label}</span>
          <span className="text-[10px] text-indigo-500 ml-auto font-mono">{percent}%</span>
        </span>
        <div className="w-full bg-indigo-100 rounded-full h-1 overflow-hidden">
          <div
            className="bg-indigo-600 h-1 rounded-full transition-all duration-300"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>
    );
  }

  return <StatusBadge status={status || 'SUCCESS'} />;
};
